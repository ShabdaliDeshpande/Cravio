"""Chat with Cravio in the terminal. Each run continues your last conversation.

  uv run cravio                      chat
  uv run cravio "what do I order?"   send one message and exit
  uv run cravio yes                  approve the step Cravio is waiting on (anything else declines it)
  uv run cravio --new ...            start a fresh conversation
  uv run cravio --trace ...          log every step: model calls, tools, subagents, MCP requests
  uv run cravio --status             show the steps of your last request: done, failed, declined, unfinished
If you quit halfway (Ctrl+C), the next `uv run cravio` shows what was done and offers to continue.
"""

import asyncio
import difflib
import logging
import os
import re
import sys
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Before importing cravio modules: they read their settings from the environment at import.
load_dotenv()

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from cravio import approvals, trace  # noqa: E402
from cravio.agent import MODEL, build_agent  # noqa: E402
from cravio.catalog import cravio_note, describe  # noqa: E402
from cravio.langsmith_tracing import langsmith_tracing  # noqa: E402
from cravio.progress import Progress, Step, label_of, steps_in_last_turn  # noqa: E402
from cravio.swiggy.client import build_client, swiggy_tools  # noqa: E402

log = logging.getLogger(__name__)

# Conversations include your addresses and orders; .cravio/ is kept out of git.
DB_FILE = Path(os.getenv("CRAVIO_DB", ".cravio/conversations.sqlite"))
THREAD_FILE = DB_FILE.with_name("current_thread")

YES = {"y", "yes", "approve", "ok", "go ahead"}


ICONS = {"running": "…", "waiting": "⏸", "completed": "✔", "error": "✖", "declined": "⊘", "unfinished": "○"}
PLAN_ICONS = {"completed": "✔", "in_progress": "▶", "pending": "○"}


def print_step(step: Step) -> None:
    """A subagent prints when it starts ('→ Finding options for you'); every step prints when it ends
    ('✔ Searching dishes · 1.2s'). Steps can run side by side, so each gets its own line."""
    indent = "  " * (step.depth + 1)
    if step.tool == "write_todos":
        if step.status == "running":
            print(f"{indent}Plan:")
            for todo in step.plan:
                print(f"{indent}  {PLAN_ICONS.get(todo.get('status'), '○')} {todo.get('content')}")
        return
    if step.status == "running":
        if step.tool == "task":  # its own steps print under it
            print(f"{indent}→ {step.label}")
        return
    note = f" · {step.detail}" if step.status in ("error", "declined", "waiting") else ""
    print(f"{indent}{ICONS[step.status]} {step.label} · {step.seconds:.1f}s{note}")


def print_last_turn(messages: list) -> None:
    asked = next((m.text for m in reversed(messages) if m.type == "human"), "")
    print(f"  you> {' '.join(asked.split())[:150]}")
    steps = steps_in_last_turn(messages)
    for step in steps:
        print(f"  {ICONS[step.status]} {step.label}" + (f" · {step.detail}" if step.status in ("error", "declined") else ""))
    if not steps:
        print("  (no steps: Cravio answered without using a tool)")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))


async def _ask(prompt: str) -> str | None:
    """Read a line, or None when there's no one to type (stdin closed, or Ctrl+Z / Ctrl+D)."""
    try:
        return (await asyncio.to_thread(input, prompt)).strip()
    except EOFError:
        return None


def _thread_id(new: bool) -> str:
    if new or not THREAD_FILE.exists():
        THREAD_FILE.parent.mkdir(parents=True, exist_ok=True)
        THREAD_FILE.write_text(str(uuid.uuid4()), encoding="utf-8")
    return THREAD_FILE.read_text(encoding="utf-8").strip()


async def show_action(action: dict, cart_tool) -> None:
    name, args = action["name"], action["args"]
    print("\n⏸  Approve this change?")
    if name == "request_checkout" and cart_tool:
        # Show the cart as Swiggy has it, not as the model describes it.
        log.info("HITL   the CLI reads the cart from Swiggy itself, without the model")
        cart = await cart_tool.ainvoke({"addressId": args["address_id"]})
        print(_text(cart)[:3000])
    else:
        print(describe(name, args))


def same_request(a: str, b: str) -> bool:
    """True if two messages ask the same thing: case, punctuation and a typo or two don't matter."""
    def words(s: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())
    return difflib.SequenceMatcher(None, words(a), words(b)).ratio() >= 0.9


def decide(answer: str) -> dict:
    if answer.lower().strip(" .!") in YES:
        return {"type": "approve"}
    # LangChain marks a declined step as an error, so say plainly that nothing broke and nothing changed;
    # otherwise the model reports "a hiccup" and tries again.
    return {
        "type": "reject",
        "message": f"{answer or 'No.'} (The user said no, so this step did not run and nothing changed. "
        "It is not an error. Don't try it again unless the user asks; tell them it wasn't done and ask what "
        "they'd like instead.)",
    }


def resume_with(pending, decisions: list[dict]) -> Command:
    """Hand each paused agent its share of the decisions, in the order its actions were shown."""
    answers, i = {}, 0
    for intr in pending:
        actions = intr.value["action_requests"]
        n = len(actions)
        answers[intr.id] = {"decisions": decisions[i : i + n]}
        for action, decision in zip(actions, decisions[i : i + n]):
            if decision["type"] == "approve":  # only exactly this change may now run (approvals.py)
                approvals.remember(action["name"], action["args"])
            else:
                approvals.forget(action["name"])
        i += n
    return Command(resume=answers[pending[0].id] if len(pending) == 1 else answers)


async def chat(message: str | None, new: bool, tracing: bool, status_only: bool = False) -> None:
    if MODEL.startswith("anthropic:") and not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env, or set CRAVIO_MODEL=ollama:<model> to run locally.")
    client = build_client(httpx_client_factory=trace.http_client) if tracing else build_client()
    config = {"configurable": {"thread_id": _thread_id(new)}}
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    async with swiggy_tools(client) as tools, AsyncSqliteSaver.from_conn_string(str(DB_FILE)) as saver:
        cart_tool = next((t for t in tools if t.metadata["swiggy_name"] == "get_food_cart"), None)
        agent = build_agent(tools, middleware=[trace.Trace() if tracing else Progress(print_step)], checkpointer=saver)
        trace.AGENT.set("cravio")

        async def pending_actions():
            pending = (await agent.aget_state(config)).interrupts
            return pending, [a for intr in pending for a in intr.value["action_requests"]]

        async def run(payload, interactive: bool) -> None:
            """Run until Cravio replies, or until it pauses for an approval no one can give right now."""
            while True:
                if isinstance(payload, Command):
                    log.info("STATE  resuming the paused run with your decision: %s", payload.resume)
                else:
                    log.info("STATE  new run: your message joins the history loaded from %s", DB_FILE)
                result = await agent.ainvoke(payload, config)
                pending, actions = await pending_actions()
                log.info("STATE  run stopped; %d messages saved to the checkpoint", len(result["messages"]))
                if not pending:
                    print(f"\ncravio> {result['messages'][-1].text}\n")
                    return
                log.info("HITL   paused before %s; it runs only if you approve", ", ".join(a["name"] for a in actions))
                if note := cravio_note(result["messages"]):
                    print(f"\ncravio> {note}")
                for action in actions:
                    await show_action(action, cart_tool)
                answer = await _ask("Approve? [y/N] ") if interactive else None
                if answer is None:
                    print('\nWaiting for you. Run `uv run cravio yes` to approve, or reply with what to do instead.')
                    return
                if decide(answer)["type"] == "reject":  # the tool won't run, so no step reports it
                    for action in actions:
                        print(f"  ⊘ {label_of(action['name'], action['args'])} · you said no")
                payload = resume_with(pending, [decide(answer)] * len(actions))

        async def unfinished_request() -> str | None:
            """What you last asked, if Cravio stopped halfway through it (you quit, or it crashed)."""
            state = await agent.aget_state(config)
            if not state.next or state.interrupts:
                return None
            return next((m.text for m in reversed(state.values.get("messages", [])) if m.type == "human"), None)

        async def send(text: str, interactive: bool) -> None:
            pending, actions = await pending_actions()
            if pending:
                # This message answers the step Cravio is waiting on.
                verdict = decide(text)
                for action in actions:
                    label = label_of(action["name"], action["args"])
                    print(f"  ✔ {label} · approved" if verdict["type"] == "approve" else f"  ⊘ {label} · you said no")
                await run(resume_with(pending, [verdict] * len(actions)), interactive)
            elif (unfinished := await unfinished_request()) and same_request(text, unfinished):
                print("  (That's the request Cravio didn't finish, so it continues where it stopped.)")
                await run(None, interactive)  # picks up from the last saved step
            else:
                await run({"messages": [{"role": "user", "content": text}]}, interactive)

        resumed = (await agent.aget_state(config)).values.get("messages")
        log.info("STATE  conversation %s: %d messages so far", config["configurable"]["thread_id"], len(resumed or []))
        print(f"Cravio is ready ({MODEL}). " + ("Continuing your conversation; --new starts fresh." if resumed else "New conversation."))

        state = await agent.aget_state(config)
        if status_only:
            print("Your last request:")
            print_last_turn(state.values.get("messages", []))
            return
        if (unfinished := await unfinished_request()) is not None:
            print("Last time, Cravio stopped before it finished. So far:")
            print_last_turn(state.values.get("messages", []))
            if message:  # send() continues it if this is the same request
                if not same_request(message, unfinished):
                    print("(Your new message comes first; the unfinished steps are dropped.)")
            else:
                answer = await _ask("Continue where it left off? [Y/n] ") if sys.stdin.isatty() else None
                if answer is not None and (answer.lower().strip(" .!") in {"", "y", "yes"} or same_request(answer, unfinished)):
                    await run(None, interactive=True)  # picks up from the last saved step

        if message:
            print(f"you> {message}")
            await send(message, interactive=sys.stdin.isatty())
            return

        pending, actions = await pending_actions()
        for action in actions:
            await show_action(action, cart_tool)
        if pending:
            print("Cravio is waiting on this step. Type yes to approve, or what to do instead.")
        print("Ask for food, your orders, or help with an order. Type 'exit' to quit.\n")
        while (user := await _ask("you> ")) is not None and user.lower() not in {"exit", "quit"}:
            if user:
                await send(user, interactive=True)


def _rate_limited(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return any(_rate_limited(e) for e in exc.exceptions)
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    flags = {"--new", "--trace", "--status"}
    message = " ".join(a for a in args if a not in flags) or None
    if "--trace" in args:
        trace.enable()
    try:
        with langsmith_tracing():
            asyncio.run(chat(message, new="--new" in args, tracing="--trace" in args, status_only="--status" in args))
    except KeyboardInterrupt:  # Ctrl+C: every finished step is already saved
        sys.exit("\nStopped. Run `uv run cravio` to see what was done and continue where you left off.")
    except Exception as exc:
        if not _rate_limited(exc):
            raise
        sys.exit("Swiggy is still limiting how fast Cravio can call it (429 Too Many Requests). Wait a few minutes and try again.")


if __name__ == "__main__":
    main()
