# Cravio

**AI food concierge for Swiggy Food & Instamart: chat to order, with approvals, allergy guards and no auto-checkout.**

Cravio is a multi-agent assistant that works on a real Swiggy account through Swiggy's MCP servers. It finds
restaurants, dishes and groceries that fit your taste and diet, builds your Food or Instamart cart, tracks
orders, and handles problems with past orders by opening a support ticket. It never places or pays for an
order: every cart change waits for your approval, and checkout stops at "your cart is ready in the Swiggy app".

Built with LangChain's Deep Agents on LangGraph, with LangSmith for tracing and evaluation. Runs on a local
model (Gemma 4 26B through Ollama) by default.

## What it does

| You say | Cravio |
|---|---|
| "Suggest a veg dinner under ₹300, like what I usually get" | Builds a taste profile from your order history, finds open restaurants, suggests 3–5 dishes with reasons |
| "Add a Gudbud Sundae from Polar Bear" | Reads your cart, finds the dish, and asks you to approve the exact new cart (and what it removes) |
| "I want chilli sauce for dinner" | Knows that's a grocery, not a restaurant dish: shops on Instamart, asks which one and how many |
| "I'm allergic to milk. Suggest a dessert" | Skips dishes that usually contain milk, never calls anything "safe", tells you to check with the restaurant |
| "Where is my order?" | Checks both Swiggy Food and Instamart |
| "I had an allergic reaction after my last order" | Safety advice first, finds the right order, opens an urgent ticket, promises nothing it can't do |

## Architecture

```
                          you (CLI or web chat)
                                   │
                    ┌──────────────▼──────────────┐
                    │  Cravio: main agent (Deep    │  plans multi-step requests (write_todos),
                    │  Agents), Food cart tools    │  routes the rest to a subagent
                    └──┬─────┬──────┬─────┬─────┬──┘
        taste-profiler │     │      │     │     │ instamart
   (order history)     │  recommender │  tracking (Food + Instamart orders)
                       │  (search,    │
                       │   open only) support-escalation (orders, tickets)
                       ▼
            Swiggy MCP: Food server, Instamart server (OAuth, one session per user)
```

**Every agent runs through the same middleware:**

- **Approvals (human in the loop):** cart changes, clearing the cart, saving an address and checkout pause for
  you. The approval shows the change in plain words, e.g. "Your Instamart cart will be: 2 × Mixed Sprouts 175 g".
- **Safety checks in code** (`availability.py`), just before a cart change runs, whatever the model decided:
  only the exact change you approved, only while the restaurant/store is taking orders, never deleting items
  you didn't ask to remove, and a warning when Swiggy brings an old cart back.
- **Blocked tools:** anything that places, pays for, books or cancels an order is never given to the model.
- **Live progress:** each step shows as running / done / failed / declined (`progress.py`).
- **Privacy:** traces go to LangSmith only with `CRAVIO_LANGSMITH=true`, and names, addresses, phone numbers
  and coordinates are masked on your machine before upload (`pii.py`).

## Setup

You need Python 3.12, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) with `gemma4:26b`
(or set `CRAVIO_MODEL` to another model), and a Swiggy account.

```
uv sync
copy .env.example .env        # then fill in what you need (see the comments in the file)
ollama pull gemma4:26b
```

Main settings in `.env`: `SWIGGY_SERVERS=food,im` (Food and Instamart, the default; `food` alone turns Instamart off), `CRAVIO_LANGSMITH=true` and
`LANGSMITH_API_KEY` (tracing), `OLLAMA_THINK` (thinking makes tool use much more reliable; keep it on).

The first run opens your browser to sign in to Swiggy (phone + OTP). The token is kept in `.cravio/`, which is
never committed.

## Run it

```
uv run cravio                          # chat in the terminal; continues your last conversation
uv run cravio --new                    # start a fresh conversation
uv run cravio --status                 # steps of your last request: done, failed, declined, unfinished
uv run chainlit run src/cravio/web.py  # web chat at http://localhost:8000, with streaming and live steps
```

If you quit halfway (Ctrl+C), the next `uv run cravio` shows what was done and continues where it stopped.

## Evaluation

All evaluation runs against **real Swiggy**, read-only: approval pauses are always declined, so no cart ever
changes, and tickets go to a temporary file. Each check reads only what Cravio did and said in that run, so a
correct Cravio passes whatever Swiggy shows that day.

**Offline (LangSmith datasets and experiments):**

| Script | Dataset | Checks |
|---|---|---|
| `evals/run_support_evals.py` | `cravio-support-scenarios` | right number of tickets, right order, category and severity, looked the order up first, reply has the ticket ID and safety advice, no refund promises; plus an LLM judge (grounded, no false promises, helpful) |
| `evals/run_intent_evals.py` | `cravio-intent-scenarios` | the request goes to the right place: Food, Instamart, tracking or support |
| `evals/run_store_evals.py` | `cravio-store-scenarios` | an order question checks the right store, and a grocery problem is never ticketed on a Food order |
| `evals/run_diet_evals.py` | `cravio-diet-scenarios` | no "safe / dairy-free" claims, allergy disclaimer, no dish that names the allergen, warns when asked for it, veg filter used |
| `evals/test_resume.py` | (local) | quit halfway, come back: continues without redoing finished steps |

```
uv run python evals/run_support_evals.py --upload      # 3 runs per example, scored in LangSmith
uv run python evals/run_intent_evals.py                # here only, no upload
```

**Online (your real conversations):** `evals/watch_carts.py` scores every real cart change in your traces and
writes the scores to LangSmith; failures go to the annotation queue `cravio-watch`.

```
uv run python evals/watch_carts.py            # score new cart changes (--loop 10 keeps watching)
```

It checks that each change was approved, that Cravio read the cart first, kept what was already there, that
Swiggy's cart matched what was sent, that the reply was honest, and that the change stuck: it caught Swiggy
bringing an old cart back minutes after confirming a change.

## Project layout

```
src/cravio/
  agent.py             the main agent, subagents, their prompts and tools
  swiggy/client.py     Swiggy MCP connection, blocked tools, which changes need approval, rate-limit retries
  swiggy/oauth.py      Swiggy sign-in (OAuth, PKCE, local callback)
  availability.py      safety checks just before a cart change
  approvals.py         only the exact approved change may run
  catalog.py           item names, so approvals read "2 × Mixed Sprouts" instead of IDs
  progress.py          step status for the CLI and web chat
  tools.py             Cravio's own tools: support tickets, checkout approval
  pii.py, langsmith_tracing.py   masking and tracing
  cli.py, web.py       terminal and web chat
evals/                 offline evals, the resume test and the online watcher
docs/swiggy_tools/     Swiggy's tool definitions, as their servers describe them
```

## Known limits (Swiggy's side)

- After a restaurant switch or clearing the cart, Swiggy sometimes brings the old cart back a few minutes
  later. Cravio notices on its next read and tells you; clearing the cart in the Swiggy app fixes it.
- Swiggy lists no ingredients or allergens, so allergy handling can only warn and point to the restaurant or
  pack label.
- Instamart stops taking orders at night while its search still works; Cravio says so instead of trying.
