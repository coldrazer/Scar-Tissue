"""Lesson memory: storage (JSON file or MongoDB) and retrieval.

A lesson is one generalizable rule about the codebase, distilled from a
failure. Lessons are either:
  global   - injected into every task (limited slots, ranked by proven value)
  specific - retrieved per task by vector similarity (embedding cosine)

Every lesson carries a dense embedding (hashed-token vectors computed with
the standard library, so offline runs need no embedding API). With MongoDB
Atlas the same vectors are queried server-side with $vectorSearch when a
vector index named "lesson_vector" exists; otherwise retrieval falls back
to the identical in-memory cosine computation, so results do not depend on
which store is configured.

Slots are deliberately scarce. That's what makes pruning matter: a useless
lesson isn't free, it occupies a slot a useful lesson could have used.
"""

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

TOKEN = re.compile(r"[a-z0-9_]+")
STOP = set("a an the and or of to in on for is are be with as by at it its this that from into any all when "
           "than then them they their should must always never use return returns".split())


def _stem(t):
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def tokens(text):
    return [_stem(t) for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class Lesson:
    id: str
    text: str
    scope: str = "specific"           # global | specific
    tags: list = field(default_factory=list)
    status: str = "alive"             # alive | pruned
    source_task: str = ""
    evidence: str = ""                # the failure it was distilled from
    origin: str = ""                  # model that distilled it
    created_epoch: int = 0
    created_at: float = field(default_factory=time.time)
    uses: int = 0                     # times injected into a prompt
    reinforced: int = 0               # times re-discovered (deduped)
    trials: int = 0                   # ablation trials
    helped: float = 0.0               # cumulative ablation delta (pass-rate points)
    last_delta: float = None
    last_used_epoch: int = -1
    pruned_epoch: int = None
    pruned_reason: str = ""
    failure_class: str = ""           # classifier category of the source failure
    tenant: str = "default"           # owning tenant (stores are per-tenant; this is belt-and-braces)
    serving: str = "canary"           # governance: canary | published | held
    version: int = 1                  # governance version (bumps on hold/release/approve)
    embedding: list = field(default_factory=list)  # dense vector for similarity search
    deltas: list = field(default_factory=list)     # recent ablation deltas (bounded)

    def __post_init__(self):
        if not self.embedding:
            self.embedding = embed(self.text + " " + " ".join(self.tags or []))

    @property
    def score(self):
        return self.helped + 0.05 * self.reinforced

    @classmethod
    def from_dict(cls, d):
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

EMBED_DIM = 256


def embed(text):
    """Dense hashed-token embedding (stdlib only, deterministic, L2-normalized).

    Each token votes into EMBED_DIM buckets with a signed hash; counts are
    log-scaled. Cosine over these vectors behaves like lexical similarity
    offline and is the same vector stored in MongoDB for Atlas $vectorSearch.
    """
    vec = [0.0] * EMBED_DIM
    for tok in tokens(text or ""):
        h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
        vec[h % EMBED_DIM] += 1.0 if (h >> 8) & 1 else -1.0
    for i, v in enumerate(vec):
        if v:
            vec[i] = math.copysign(math.log1p(abs(v)), v)
    norm = math.sqrt(sum(v * v for v in vec))
    return [round(v / norm, 6) for v in vec] if norm else vec


def vector_cosine(a, b):
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


def vector_similarity(text, others):
    q = embed(text)
    return [vector_cosine(q, embed(o)) for o in others]


def _tfidf(docs):
    df = Counter()
    toks = [tokens(d) for d in docs]
    for t in toks:
        df.update(set(t))
    n = len(docs)
    vecs = []
    for t in toks:
        tf = Counter(t)
        v = {w: (c / len(t)) * (math.log((1 + n) / (1 + df[w])) + 1) for w, c in tf.items()} if t else {}
        vecs.append(v)
    return vecs


def _cos(a, b):
    num = sum(a[w] * b.get(w, 0.0) for w in a)
    den = math.sqrt(sum(x * x for x in a.values())) * math.sqrt(sum(x * x for x in b.values()))
    return num / den if den else 0.0


def similarity(text, others):
    """Embedding-cosine similarity (kept name: loop + dedupe call sites)."""
    return vector_similarity(text, others)


SERVING_STATES = ("canary", "published", "held")


def set_serving(lesson, state):
    """Governance transition with validation. Returns the lesson."""
    if state not in SERVING_STATES:
        raise ValueError(f"serving must be one of {SERVING_STATES}, got {state!r}")
    if lesson.serving != state:
        lesson.serving = state
        lesson.version += 1
    return lesson


def retrieve(query, lessons, k_specific=3, k_global=2, exclude=()):
    """Pick the lessons to inject for a task. Returns [(lesson, why)].

    Held lessons are never injected; canary and published lessons are.
    """
    pool = [l for l in lessons if l.status == "alive" and l.serving != "held" and l.id not in exclude]
    glob = sorted((l for l in pool if l.scope == "global"), key=lambda l: (-l.score, l.created_at))[:k_global]
    spec = [l for l in pool if l.scope != "global"]
    picked = [(l, "global") for l in glob]
    if spec:
        q = embed(query)
        sims = [vector_cosine(q, l.embedding or embed(l.text + " " + " ".join(l.tags))) for l in spec]
        ranked = sorted(zip(spec, sims), key=lambda p: (-p[1], -p[0].score, p[0].created_at))
        picked += [(l, f"vec {s:.2f}") for l, s in ranked[:k_specific] if s > 0.02]
    return picked


def retrieve_from_store(store, query, k_specific=3, k_global=2, exclude=(), public_only=False, include_canary=False):
    """Retrieve via the store's vector search when it offers one (Atlas).

    Falls back to in-memory vector retrieval over store.lessons(); both paths
    rank by the same embedding cosine, so callers get identical semantics.
    With public_only=True (external consumers, e.g. Fetch.ai) only published
    lessons are served; include_canary=True additionally serves at most one
    canary lesson (the canary slot), never held or pruned lessons.
    """
    if public_only:
        lessons = [l for l in store.lessons()
                   if l.status == "alive" and l.id not in set(exclude)
                   and (l.serving == "published" or (include_canary and l.serving == "canary"))]
        picked = retrieve(query, lessons, k_specific, k_global)
        canary_seen, out = False, []
        for l, why in picked:
            if l.serving == "canary":
                if canary_seen:
                    continue
                canary_seen = True
            out.append((l, why))
        return out
    if hasattr(store, "vector_search"):
        hits = store.vector_search(query, k_specific + k_global + len(tuple(exclude)))
        if hits is not None:
            lessons = [l for l in hits if l.id not in set(exclude) and l.serving != "held"]
            glob = sorted((l for l in lessons if l.scope == "global"),
                          key=lambda l: (-l.score, l.created_at))[:k_global]
            picked = [(l, "global") for l in glob]
            picked += [(l, "vec") for l in lessons if l.scope != "global"][:k_specific]
            # Globals found by vector search may duplicate the ranked globals.
            seen, out = set(), []
            for l, why in picked:
                if l.id not in seen:
                    seen.add(l.id)
                    out.append((l, why))
            return out
    return retrieve(query, store.lessons(), k_specific, k_global, exclude)


def find_duplicate(text, lessons, threshold=0.55):
    alive = [l for l in lessons if l.status == "alive"]
    if not alive:
        return None
    sims = similarity(text, [l.text for l in alive])
    best = max(range(len(alive)), key=lambda i: sims[i])
    return alive[best] if sims[best] >= threshold else None


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

class JsonStore:
    """Single JSON file. Good for laptops, demos and the offline console."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.state = self._read()

    def _read(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {"lessons": [], "epochs": [], "events": [], "meta": {}}

    def _flush(self):
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(self.state, f, indent=1, default=str)
        os.replace(tmp, self.path)

    # lessons
    def lessons(self):
        with self.lock:
            return [Lesson.from_dict(d) for d in self.state["lessons"]]

    def save_lesson(self, lesson):
        lesson.embedding = embed(lesson.text + " " + " ".join(lesson.tags or []))
        lesson.deltas = list(lesson.deltas or [])[-10:]
        with self.lock:
            rows = self.state["lessons"]
            for i, d in enumerate(rows):
                if d["id"] == lesson.id:
                    rows[i] = asdict(lesson)
                    break
            else:
                rows.append(asdict(lesson))
            self._flush()

    def next_lesson_id(self):
        with self.lock:
            n = self.state["meta"].get("lesson_seq", 0) + 1
            self.state["meta"]["lesson_seq"] = n
            return f"les_{n:04d}"

    # history
    def epochs(self):
        with self.lock:
            return list(self.state["epochs"])

    def add_epoch(self, rec):
        with self.lock:
            self.state["epochs"].append(rec)
            self._flush()

    def event(self, kind, msg, **data):
        with self.lock:
            self.state["events"].append({"t": time.time(), "kind": kind, "msg": msg, **data})
            self.state["events"] = self.state["events"][-400:]
            self._flush()

    def set_meta(self, **kv):
        with self.lock:
            self.state["meta"].update(kv)
            self._flush()

    def meta(self):
        with self.lock:
            return dict(self.state["meta"])

    def snapshot(self):
        # Always read from disk: the dashboard server is usually a different process.
        with self.lock:
            return self._read()

    def reset(self):
        with self.lock:
            self.state = {"lessons": [], "epochs": [], "events": [], "meta": {}}
            self._flush()


class MongoStore:
    """MongoDB (Atlas or local). Same interface as JsonStore."""

    def __init__(self, uri, db="scar_tissue"):
        from pymongo import MongoClient, ReturnDocument
        self._rd = ReturnDocument
        self.db = MongoClient(uri, serverSelectionTimeoutMS=8000)[db]
        self.db.command("ping")
        self.db.lessons.create_index("id", unique=True)
        self.db.lessons.create_index("status")
        self.db.epochs.create_index("epoch")
        # Events are an observability feed, not an archive: expire after 7 days.
        # TTL needs a BSON date, so events carry `ts` alongside the float `t`.
        try:
            self.db.events.create_index("ts", expireAfterSeconds=7 * 24 * 3600)
        except Exception:
            pass
        # Best effort: Atlas vector index (managed search indexes need Atlas;
        # on plain MongoDB this fails quietly and cosine runs in memory).
        try:
            from pymongo.operations import SearchIndexModel
            self.db.lessons.create_search_index(SearchIndexModel(
                definition={"mappings": {"dynamic": False, "fields": {
                    "embedding": {"type": "knnVector", "dimensions": EMBED_DIM,
                                  "similarity": "cosine"},
                    "status": {"type": "token"}}}},
                name="lesson_vector", type="vectorSearch"))
        except Exception:
            pass

    def lessons(self):
        return [Lesson.from_dict(d) for d in self.db.lessons.find({}, {"_id": 0}).sort("created_at", 1)]

    def save_lesson(self, lesson):
        lesson.embedding = embed(lesson.text + " " + " ".join(lesson.tags or []))
        lesson.deltas = list(lesson.deltas or [])[-10:]
        self.db.lessons.replace_one({"id": lesson.id}, asdict(lesson), upsert=True)

    def vector_search(self, query, k=8):
        """Atlas $vectorSearch over lesson embeddings; None => caller falls back."""
        try:
            rows = list(self.db.lessons.aggregate([
                {"$vectorSearch": {"index": "lesson_vector", "path": "embedding",
                                   "queryVector": embed(query), "numCandidates": 100,
                                   "limit": max(1, k),
                                   "filter": {"status": "alive"}}},
                {"$project": {"_id": 0}},
            ]))
        except Exception:
            return None
        return [Lesson.from_dict(d) for d in rows] if rows else None

    def next_lesson_id(self):
        d = self.db.meta.find_one_and_update({"_id": "meta"}, {"$inc": {"lesson_seq": 1}},
                                             upsert=True, return_document=self._rd.AFTER)
        return f"les_{d['lesson_seq']:04d}"

    def epochs(self):
        return list(self.db.epochs.find({}, {"_id": 0}).sort("epoch", 1))

    def add_epoch(self, rec):
        self.db.epochs.insert_one(dict(rec))

    def event(self, kind, msg, **data):
        from datetime import datetime, timezone
        self.db.events.insert_one({"t": time.time(), "ts": datetime.now(timezone.utc),
                                   "kind": kind, "msg": msg, **data})

    def set_meta(self, **kv):
        self.db.meta.update_one({"_id": "meta"}, {"$set": kv}, upsert=True)

    def meta(self):
        d = self.db.meta.find_one({"_id": "meta"}) or {}
        d.pop("_id", None)
        return d

    def snapshot(self):
        events = list(self.db.events.find({}, {"_id": 0}).sort("t", -1).limit(400))[::-1]
        return {"lessons": [asdict(l) for l in self.lessons()], "epochs": self.epochs(),
                "events": events, "meta": self.meta()}

    def reset(self):
        for c in ("lessons", "epochs", "events", "meta"):
            self.db[c].delete_many({})


def _tenant_suffix(tenant):
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in (tenant or "default"))
    return "" if clean == "default" else "_" + clean


def tenant_state_path(path, tenant):
    if _tenant_suffix(tenant) == "":
        return path
    base, ext = (path[:-5], ".json") if path.endswith(".json") else (path, "")
    return f"{base}-{tenant}{ext}"


def open_store(cfg, tenant=None):
    """Open a tenant-scoped store. Tenants never share lessons, epochs or events."""
    tenant = tenant or getattr(cfg, "tenant", "default") or "default"
    if cfg.mongodb_uri:
        return MongoStore(cfg.mongodb_uri, cfg.mongodb_db + _tenant_suffix(tenant))
    return JsonStore(tenant_state_path(cfg.state_path, tenant))


def open_store_from_env(state_path=None, tenant=None):
    """Open the configured store without building any model endpoints.

    Used by read-only consumers (dashboard, Fetch.ai agent) that must not
    require API keys just to serve already-learned lessons.
    """
    tenant = tenant or os.environ.get("SCAR_TENANT", "default")
    uri = os.environ.get("MONGODB_URI", "")
    if uri:
        return MongoStore(uri, os.environ.get("MONGODB_DB", "scar_tissue") + _tenant_suffix(tenant))
    path = state_path or os.environ.get("SCAR_STATE") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "state.json")
    return JsonStore(tenant_state_path(path, tenant))
