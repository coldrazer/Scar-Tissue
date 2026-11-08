"""Scar Tissue as a Fetch.ai uAgent: other coding agents can ask it for the
battle-tested lessons that apply to their task.

    pip install uagents
    SCAR_AGENT_SEED='<a private random phrase>' python agents/fetch_agent.py
    SCAR_MAILBOX=1 python agents/fetch_agent.py   # reachable via Agentverse mailbox

The seed IS the agent identity: there is deliberately no default. Generate
one privately (e.g. `python -c "import secrets; print(secrets.token_hex(16))"`)
and never commit it. If SCAR_AGENT_TOKEN is set, callers must send the same
value in the query's `token` field or they receive an empty lesson list.

Ask it from another uAgent with a LessonQuery message, or over HTTP:

    curl -s localhost:8001/lessons -H 'content-type: application/json' \
         -d '{"task": "write a function that splits a bill between people", "k": 3}'

The LessonQuery/Lesson/LessonReply contract below is published as JSON Schema
in docs/fetch-schema.json -- that file is the front-end schema other agents
and UIs integrate against; keep the two in sync.

Only published lessons are served (survived ablation + passed a held-out eval),
ranked by measured value; `canary: true` adds at most one unpublished canary lesson.
Retrieval is vector (embedding cosine; Atlas $vectorSearch when configured).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uagents import Agent, Context, Model  # noqa: E402

from scartissue.config import load_dotenv  # noqa: E402
from scartissue.memory import open_store_from_env, retrieve_from_store  # noqa: E402

load_dotenv()

_SEED = os.environ.get("SCAR_AGENT_SEED", "").strip()
if len(_SEED) < 16 or "change me" in _SEED.lower() or _SEED == "scar-tissue lesson memory demo seed":
    raise SystemExit(
        "SCAR_AGENT_SEED must be a private phrase of 16+ characters (no default is provided: "
        "the seed determines the agent address). Set it in .env or the environment.")

_TOKEN = os.environ.get("SCAR_AGENT_TOKEN", "")


class LessonQuery(Model):
    task: str
    k: int = 3
    token: str = ""
    canary: bool = False   # also serve at most one not-yet-published (canary) lesson


class Lesson(Model):
    id: str
    text: str
    scope: str
    value: float


class LessonReply(Model):
    lessons: list[Lesson]
    alive: int
    pruned: int


# Read-only consumer: open the store directly, no model endpoints required.
store = open_store_from_env()

agent = Agent(
    name="scar-tissue",
    seed=_SEED,
    port=int(os.environ.get("SCAR_AGENT_PORT", 8001)),
    mailbox=os.environ.get("SCAR_MAILBOX") == "1",
    description="Serves coding lessons distilled from failures and kept only if ablation shows they help.",
)


def answer(q: LessonQuery) -> LessonReply:
    lessons = store.lessons()
    alive = sum(l.status == "alive" for l in lessons)
    pruned = sum(l.status != "alive" for l in lessons)
    if _TOKEN and q.token != _TOKEN:
        return LessonReply(lessons=[], alive=alive, pruned=pruned)
    k = max(1, min(q.k, 8))
    picked = retrieve_from_store(store, q.task, k_specific=k, k_global=2,
                                 public_only=True, include_canary=bool(q.canary))
    return LessonReply(
        lessons=[Lesson(id=l.id, text=l.text, scope=l.scope, value=round(l.score, 3)) for l, _ in picked],
        alive=alive,
        pruned=pruned,
    )


@agent.on_message(model=LessonQuery, replies=LessonReply)
async def on_query(ctx: Context, sender: str, msg: LessonQuery):
    ctx.logger.info(f"lesson query from {sender[:16]}...: {msg.task[:60]!r}")
    await ctx.send(sender, answer(msg))


@agent.on_rest_post("/lessons", LessonQuery, LessonReply)
async def on_rest(ctx: Context, req: LessonQuery) -> LessonReply:
    return answer(req)


if __name__ == "__main__":
    print(f"scar-tissue agent address: {agent.address}")
    agent.run()
