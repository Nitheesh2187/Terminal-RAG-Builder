"""Terminal REPL for the RAG system.

Commands:
    /ingest [--limit N] [--reset]   ingest PDFs from PDF_DIR
    /retrieve <query> [-k 10]       hybrid retrieve only
    /evaluate [--gen] [-k 10]       run golden eval
    /stats                          DB counts
    /help                           show commands
    /exit                           leave
    <free text>                     retrieve + generate (RAG answer)
"""
from __future__ import annotations

import shlex
import sys
import traceback

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel

from .config import CFG
from .db import counts, init_schema
from .evaluate import run_evaluation
from .generate import generate_answer
from .ingest import run_ingest
from .metrics import LatencyRecord, render_latency
from .retrieve import hybrid_search, render_hits

console = Console()

BANNER = (
    "[bold cyan]Terminal RAG Builder[/bold cyan]  "
    "[dim](hybrid pgvector + RRF · {model} · {emb})[/dim]\n"
    "type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit"
).format(model=CFG.groq_model, emb=CFG.embed_model)


def _cmd_help() -> None:
    console.print(Panel.fit(__doc__ or "", title="commands", border_style="cyan"))


def _cmd_stats() -> None:
    c = counts()
    console.print(f"documents: [green]{c['documents']:,}[/green]   chunks: [green]{c['chunks']:,}[/green]")


def _cmd_ingest(args: list[str]) -> None:
    limit = None
    reset = False
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--limit", "-n") and i + 1 < len(args):
            limit = int(args[i + 1]); i += 2
        elif a == "--reset":
            reset = True; i += 1
        else:
            console.print(f"[red]unknown arg[/red]: {a}"); return
    run_ingest(limit=limit, reset=reset)


def _cmd_retrieve(args: list[str]) -> None:
    if not args:
        console.print("[red]usage[/red]: /retrieve <query> [-k 10]"); return
    k = CFG.top_k
    qtoks: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-k", "--k") and i + 1 < len(args):
            k = int(args[i + 1]); i += 2
        else:
            qtoks.append(a); i += 1
    query = " ".join(qtoks)
    rec = LatencyRecord(command="/retrieve")
    hits = hybrid_search(query, k=k, rec=rec)
    render_hits(hits)
    render_latency(rec)


def _cmd_evaluate(args: list[str]) -> None:
    k = CFG.top_k
    with_gen = False
    golden = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-k", "--k") and i + 1 < len(args):
            k = int(args[i + 1]); i += 2
        elif a == "--gen":
            with_gen = True; i += 1
        elif a in ("--golden", "-g") and i + 1 < len(args):
            golden = args[i + 1]; i += 2
        else:
            console.print(f"[red]unknown arg[/red]: {a}"); return
    run_evaluation(golden_path=golden, k=k, with_generation=with_gen)


def _cmd_query(query: str) -> None:
    rec = LatencyRecord(command="/query")
    hits = hybrid_search(query, rec=rec)
    render_hits(hits)
    try:
        ans = generate_answer(query, hits, rec=rec)
        console.print(Panel(ans.text, title=f"answer ({ans.model})", border_style="green"))
        rec.set("prompt_tokens", ans.prompt_tokens or 0)
        rec.set("completion_tokens", ans.completion_tokens or 0)
    except Exception as e:
        console.print(f"[red]generation failed[/red]: {e}")
    render_latency(rec)


def _dispatch(line: str) -> bool:
    """Return False to exit the loop."""
    line = line.strip()
    if not line:
        return True
    if line.startswith("/"):
        parts = shlex.split(line)
        cmd, args = parts[0], parts[1:]
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            _cmd_help()
        elif cmd == "/stats":
            _cmd_stats()
        elif cmd == "/ingest":
            _cmd_ingest(args)
        elif cmd == "/retrieve":
            _cmd_retrieve(args)
        elif cmd == "/evaluate":
            _cmd_evaluate(args)
        else:
            console.print(f"[red]unknown command[/red]: {cmd} (try /help)")
        return True
    _cmd_query(line)
    return True


def main() -> int:
    console.print(BANNER)
    try:
        init_schema()
    except Exception as e:
        console.print(f"[red]db init failed[/red]: {e}")
        console.print("[dim]is postgres running? `docker compose up -d`[/dim]")
        return 1

    session: PromptSession = PromptSession(history=InMemoryHistory())
    while True:
        try:
            line = session.prompt("rag> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            return 0
        try:
            if not _dispatch(line):
                console.print("bye.")
                return 0
        except Exception:
            console.print("[red]error:[/red]")
            traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
