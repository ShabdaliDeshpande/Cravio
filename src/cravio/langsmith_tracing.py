"""Send Cravio's runs to LangSmith with personal data masked. Off unless CRAVIO_LANGSMITH=true.

Tracing is switched on here and nowhere else, so every trace passes through cravio.pii.mask()
on this machine before it is uploaded. LANGSMITH_TRACING must stay unset: LangChain reads it
in any script, and would upload that script's runs unmasked.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from langsmith import Client
from langsmith.anonymizer import create_anonymizer
from langsmith.run_helpers import tracing_context

from cravio.pii import mask

log = logging.getLogger(__name__)

# LangSmith masks strings only this many levels deep, and uploads deeper ones as they are.
# Chat messages nest deeply (a tool call's arguments, in a message, in a list of generations).
MAX_DEPTH = 100


def masking_client() -> Client:
    anonymizer = create_anonymizer(mask, max_depth=MAX_DEPTH)
    # Prove the masking works on a known sample before trusting it with real data.
    sample = {"messages": [{"content": "call 9876543210"}]}
    if anonymizer(sample) != {"messages": [{"content": "call <phone>"}]}:
        raise SystemExit("Personal-data masking failed its self-check, so Cravio won't send traces to LangSmith.")
    return Client(anonymizer=anonymizer)


@contextmanager
def langsmith_tracing() -> Iterator[None]:
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        if os.getenv(var, "").lower() == "true":
            raise SystemExit(f"Remove {var} from your environment: it uploads runs from every script, unmasked. "
                             "Set CRAVIO_LANGSMITH=true instead.")
    if os.getenv("CRAVIO_LANGSMITH", "").lower() != "true":
        with tracing_context(enabled=False):
            yield
        return
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("CRAVIO_LANGSMITH=true needs LANGSMITH_API_KEY in .env.")

    client = masking_client()
    project = os.getenv("LANGSMITH_PROJECT", "cravio")
    log.info("SETUP  LangSmith: tracing to project %r, personal data masked before upload", project)
    try:
        with tracing_context(enabled=True, client=client, project_name=project):
            yield
    finally:
        client.flush()  # the CLI exits right after a run; send what's queued first
