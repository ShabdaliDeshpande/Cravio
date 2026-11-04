# Friction log

The rough edges hit while building Cravio's cart watcher with LangSmith, and one in the LLM judge.

## 1. Online evaluation of an agent needs the whole trace

**Area:** LangSmith online evaluators, SDK.

**What I was doing.** Live cases (open restaurants, stock, a real cart) can't be a fixed dataset, so cart changes are
judged on real traffic: was the change approved, did Cravio read the cart first, did it keep what was already there,
did Swiggy's cart match what was sent, was the reply honest, did the change stick.

**What happened.** An online code evaluator takes one argument: the run being evaluated. Only one of those checks,
whether Swiggy's cart matched what was sent, can be answered from a single run (the cart change's input and output).
The others need more: the subagent's messages and the tool runs inside it (Instamart changes happen in a subagent,
whose steps aren't in the top-level run's output), the approval decision on the resumed run, and, for "did it stick",
a later trace. So the watcher became an SDK script instead (`evals/watch_carts.py`): list the cart-change tool runs,
read their traces, score them, write feedback, and send failures to an annotation queue.

**Why it matters.** For agents, the failures worth catching in production are about sequences: what happened before
and after a step, and across turns. A per-run evaluator can't express them.

**What would help.** Online code evaluators that receive the whole trace tree and can look at later traces in the
same thread. (LangSmith's multi-turn online evaluators work on a whole thread, but as an LLM judge over the messages in
each trace's top-level inputs and outputs, so they don't see a subagent's tool runs.) A documented pattern for "score
later, when more data arrives".

## 2. The LLM judge's temperature

**Area:** LangSmith UI LLM-as-a-judge evaluator, Groq model.

**What happened.** For the support evals' judge (Groq, `openai/gpt-oss-120b`), the model panel showed no temperature
field, so I set `{"temperature": 0}` in extra parameters. All 104 of the judge's runs in LangSmith's `evaluators`
project (checked 4 Nov 2026) record temperature 0.7, and the 0 doesn't appear in their recorded parameters. I couldn't
confirm from the UI whether the setting was applied.

**What would help.** A temperature field for every provider, and the judge's effective model settings shown on the
evaluator.
