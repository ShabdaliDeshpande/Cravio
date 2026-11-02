# Friction log

The two rough edges that cost the most while building Cravio, and that would matter most to a team putting a
LangChain agent in front of real customers.

## 1. Human-in-the-loop inside Deep Agents subagents

**Area:** LangChain `HumanInTheLoopMiddleware`, Deep Agents subagents (the `task` tool).

**What I was doing.** Cravio's Instamart subagent changes a real grocery cart, so every `update_cart` call
pauses for the user's approval (`interrupt_on`, inherited by the subagent).

**What happened.**

- **What runs after approval isn't bound to what was approved.** After the user approves, the paused `task`
  call runs again and the subagent starts over: it read the cart and searched again, then proposed
  `update_cart` a second time, and the approval applied to that second proposal. With a non-deterministic
  model, the change that runs can differ from the one the user saw. (Seen in a trace on 2 Nov 2026: the
  resumed run repeated `get_cart` and `search_products` before `update_cart`.)
- **The parent agent never sees the decision.** A "No" happens inside the subagent, so to the main agent it
  looked like a contradiction ("the user asked for it, the subagent says declined"). It sent the same task up to
  three times and told the user "I'm having some trouble adding it".
- **A rejection with a reason reads like an error.** With a `message`, the rejection becomes a `ToolMessage`
  with `status="error"` and only "User rejected the tool call … with reason: …". The default text's "The tool
  was not executed. Do not retry this tool call" is dropped, so the model reported "a hiccup" and retried.

**Why it matters.** For an agent that acts on a customer's account, "the user approved exactly this" is the
whole point of the pause. Here it took custom code to guarantee.

**Workaround in Cravio.**

- `approvals.py` records exactly what was approved, and a middleware lets a change run only if it matches,
  once. Anything else is stopped, and the user is asked again.
- A rejection message that says plainly the step didn't run, isn't an error and shouldn't be retried (`cli.decide`).
- Prompt rules telling the main agent that approvals happen inside subagents, where it can't see them.

**What would help.** Bind an approval to the tool call's arguments, and refuse to run a different call on
resume. Resume a subagent from its own checkpoint instead of re-running it. Report approval and rejection
events to the parent agent. Keep "not executed, don't retry" in the message even when a reason is given.

## 2. Online evaluation of an agent needs the whole trace

**Area:** LangSmith online evaluators, SDK.

**What I was doing.** Live cases (open restaurants, stock, a real cart) can't be a fixed dataset, so cart
changes are judged on real traffic: was the change approved, did Cravio read the cart first, did it keep what
was already there, did Swiggy's cart match what was sent, was the reply honest, did the change stick.

**What happened.** An online code evaluator receives one run. None of those checks can be answered from one
run. They need the subagent's messages, the tool runs inside it, the approval decision on the resumed run, and,
for "did it stick", a later trace. So the watcher became an SDK script instead (`evals/watch_carts.py`): list
the cart-change tool runs, read their traces, score them, write feedback, and send failures to an annotation
queue. Two more rough edges along the way:

- Walking a trace with `read_run`, one parent at a time, hit `429 Too Many Requests` after a few changes. The
  fix was one filtered `list_runs` per trace plus back-off. `list_runs` also warns that it's deprecated in
  favour of `client.runs.query`.
- In the UI LLM-as-judge, each prompt variable was mapped to the whole `output` by default, so the judge read
  every field three times and passed replies it should have failed. The model panel had no temperature field,
  and `{"temperature": 0}` in extra parameters didn't show up in the judge's traces (still 0.7).

**Why it matters.** For agents, the failures worth catching in production are about sequences: what happened
before and after a step, and across turns. Per-run evaluators can't express them.

**What would help.** Online evaluators that receive the whole trace tree (and optionally the thread), with
filters on child runs. A documented pattern for "score later, when more data arrives". Variable mapping that
points at a field by default.

---

Also hit, smaller: `stream_mode="messages"` streams middleware-internal model calls (Deep Agents' history
summaries) as if they were the agent's reply; a middleware attribute named `tools` is read as tools to add; the
LangSmith anonymizer edits its input in place, and passes the field path only to a two-argument function.
