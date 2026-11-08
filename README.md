# Scar Tissue

**A coding agent that turns every failure into a lesson, and keeps a lesson only while removing it would make the agent worse.**

Most agent memory is append-only: every reflection gets saved, and memory slowly fills with vague advice and confidently wrong rules. Scar Tissue treats each lesson as a hypothesis. Every epoch it ablates its own memory: it re-runs each lesson's most relevant tasks with and without the lesson, and prunes the lessons that don't measurably help. What survives is a small set of rules with evidence behind every one.

Built for the AI Engineer Code Summit hackathon track *"systems that get measurably better with every run."*

> **Built on the Meta ecosphere** — Meta Wonder, Meta Spark, and Muse Glimmer.
> Also built with **Subquadratic (SubQ)** for research, **Fetch.ai** for the front-end lesson schema, and **Google Gemini** for benchmark testing.

## The loop

```
epoch 0   baseline: score held-out tasks with empty memory
epoch n   1. TRAIN    attempt each training task, up to 3 tries with test feedback
          2. DISTILL  turn each failure (+ the eventual fix) into candidate lessons; dedupe into memory
          3. ABLATE   for each lesson: re-run its most relevant tasks with vs. without it
                        hurt (fewer checks pass with it)      -> pruned
                        no measurable help after 2 trials     -> pruned
                        outranked, never selected for a task  -> pruned
          4. EVAL     pass@1 on held-out tasks, no feedback, memory read-only
```

Lesson slots are deliberately scarce: 2 global lessons injected into every task, plus 3 retrieved per task by similarity. That makes pruning matter. A useless lesson isn't free; it occupies a slot a useful one needed, and a wrong one actively misleads.

## The benchmark: Acme Ledger

A fictional billing codebase with eight **unwritten** house rules. Task prompts never state them. Like joining a real team, the only way to learn them is to ship code, fail review and remember why.

| Rule | Example |
|---|---|
| Never raise; return `{'ok': True, 'value': v}` / `{'ok': False, 'error': CODE}` | |
| Error codes are UPPER_SNAKE | `'INVALID_AMOUNT'` |
| Money is integer cents | `'$12.50'` → `1250` |
| Dates are `YYYY-MM-DD` | |
| Id lists sorted ascending | |
| Ids are prefix + 6 zero-padded digits | `inv_000042` |
| Leftover cents go to the first shares | 1000 split 3 ways → `[334, 333, 333]` |
| Names normalized to Title Case, single spaces | `'  ada   LOVELACE '` → `'Ada Lovelace'` |

12 training tasks and 12 **held-out** eval tasks. Eval tasks are never trained on, so the curve measures transfer across functions that share the same house conventions, not memorized solutions. Each test check carries a hidden tag naming the rule it enforces. The model never sees it; the dashboard uses it to show which rules the agent now follows. `python scar.py selfcheck` proves every task catches every rule it covers.

## Results (offline simulation, 3 seeds, reproduced by the shipped command)

`python scar.py compare --mock --epochs 4 --seeds 3` runs the same agent twice from scratch per seed: once with Scar Tissue, once keeping every lesson it ever distills. Figures are mean +/- std across seeds (12 held-out tasks per epoch).

| Epoch | Scar Tissue pass@1 | lessons | No pruning pass@1 | lessons |
|---|---|---|---|---|
| baseline | 0% +/- 0% | 0.0 | 0% +/- 0% | 0.0 |
| 1 | 36% +/- 8% | 11.0 | 31% +/- 8% | 12.0 |
| 2 | 100% +/- 0% | 11.7 | 58% +/- 30% | 18.3 |
| 3 | 100% +/- 0% | 8.0 | 81% +/- 22% | 19.0 |
| 4 | 97% +/- 4% | 8.0 | 92% +/- 0% | 19.0 |

From epoch 2 on, Scar Tissue holds 97-100% while memory converges to about 8 lessons (one per hidden rule). The control arm keeps ~19 lessons and stays volatile (down to 58% +/- 30% at epoch 2). Reproduce this table with the command above; use `compare --provider meta --epochs 5 --seeds 3` for the real-model version.

**Read this honestly.** These numbers come from the offline mock model, which follows a lesson if it mentions the rule's key phrase, and whose distiller is *configured* to emit useless advice 45% of the time and a wrong rule (a misread failure) 15% of the time. Those rates are stated in `scartissue/agent.py` and were not tuned to widen the gap. The gap comes entirely from wrong and redundant lessons that the control arm can never remove. The real test is a run with real models; see below.

## Quick start

```bash
git clone https://github.com/coldrazer/scar-tissue && cd scar-tissue
python scar.py selfcheck                 # verify the benchmark
python scar.py serve &                   # dashboard on http://localhost:8787
python scar.py demo --mock --epochs 6    # control arm, then a slow live run to watch
```

No dependencies are needed for any of that: the loop, sandbox, dashboard and model clients use only the Python standard library.

## Real models

Copy `.env.example` to `.env` and add keys. Every provider is reached through an OpenAI-compatible endpoint.

```bash
python scar.py ping --provider meta                       # check the key and model
python scar.py run --provider meta --epochs 5 --fresh     # Llama solves and distills
python scar.py run --provider meta --distill-provider gemini --fresh   # Llama solves, Gemini distills
python scar.py compare --provider meta --epochs 5         # the honest A/B with a real model
```

| Sponsor | Where it plugs in |
|---|---|
| **Meta Llama API** | Default solver and distiller (`--provider meta`, `api.llama.com/compat/v1`) |
| **Google DeepMind Gemini** | Benchmark testing: independent solver/distiller pair (`--provider gemini`) |
| **MongoDB** | Set `MONGODB_URI`: lessons (with embeddings), epochs and events persist in MongoDB / Atlas. Retrieval uses Atlas `$vectorSearch` when the `lesson_vector` index exists, else identical in-memory cosine. Events expire after 7 days (TTL index) |
| **Subquadratic SubQ** | Research backend (`--provider subq` with `SUBQ_BASE_URL` / `SUBQ_MODEL` from your early-access account) |
| **Fetch.ai** | Front-end schema + agent: `agents/fetch_agent.py` serves surviving lessons as a uAgent (schema in `docs/fetch-schema.json`, message protocol + `POST /lessons`). Requires a private `SCAR_AGENT_SEED` (no default); optional `SCAR_AGENT_TOKEN` gates queries |

Real-model runs use temperature 0 so the with/without ablation comparison is as fair as possible, cache identical probes within an epoch, and average each probe over `--probe-repeats` attempts (default 3). Solver and distiller providers are independent (`--provider mock --distill-provider gemini` works). A run is roughly 100 model calls per epoch; `--workers` sets parallelism.

## Commands

| Command | What it does |
|---|---|
| `run` | Run learning epochs (`--fresh` to wipe memory, `--no-prune` for the control arm) |
| `compare` | Scar Tissue vs. keep-every-lesson, both from scratch; the dashboard overlays the control |
| `demo` | Build the control arm quickly, then replay Scar Tissue slowly for a live audience |
| `serve` | Live dashboard: learning curve, living memory, graveyard with prune reasons, per-task grid |
| `lessons` | Print living memory and the graveyard in the terminal |
| `export` | Write surviving lessons to `LESSONS.md`, ready to paste into `CLAUDE.md`, `AGENTS.md` or a system prompt |
| `selfcheck` | Prove each benchmark task detects each rule it covers |
| `models` / `ping` | List an endpoint's models / test the configured endpoints |
| `reset` | Wipe memory and history |

## Layout

```
scar.py                   CLI
scartissue/benchmark.py   Acme Ledger tasks, tests and the hidden rules
scartissue/harness.py     test harness (checks, hidden rule tags)
scartissue/sandbox.py     runs model code in a subprocess: timeout, rlimits, audit hook (no network/writes outside temp)
scartissue/agent.py       LLM agent (solve, distill) and the offline mock agent
scartissue/memory.py      lessons with embeddings, vector retrieval, dedupe, JSON and MongoDB stores
scartissue/classifier.py  failure classifier: tags lessons from visible failure messages only
scartissue/loop.py        train, distill, ablate, eval
scartissue/server.py      dashboard + API server (auth, tenancy, jobs); static/index.html is the dashboard
scartissue/auth.py        API keys mapped to tenants, constant-time verification
scartissue/jobs.py        persistent background job runner (idempotent submit, resume on restart)
agents/fetch_agent.py     Fetch.ai uAgent serving lessons
docs/fetch-schema.json    published LessonQuery/LessonReply front-end schema (Fetch.ai)
tests/test_scar.py        unit tests + the full real-model path against a fake OpenAI-compatible server
```

Run the tests with `python -m unittest discover -s tests -v`.

## Notes and limits

- Model-written code runs in a subprocess with a timeout, CPU/memory/file rlimits and an audit hook that denies network access, subprocesses and writes outside the temp directory. That is defence in depth for this benchmark of pure functions, not a boundary for genuinely untrusted code; use a container there.
- Ablation is still an estimator with real models. Probes average `--probe-repeats` attempts on the checks-passed fraction, "hurt" requires beating a -0.02 margin, cumulative scores decay 0.9x per epoch, and "no help" needs 2 trials. A good lesson can still occasionally be pruned and rediscovered later.
- Every failure is tagged by `scartissue/classifier.py` from its visible messages (envelope, error_code, value_mismatch, load_error, timeout) plus domain tags. Lessons store that `failure_class`; the hidden convention tags are never used for classification.
- Retrieval is vector-based: 256-dim hashed-token embeddings computed with the standard library, so offline runs need no embedding API and JSON/Mongo store identical vectors. On Atlas the same vectors are served by `$vectorSearch` (`lesson_vector` index) with an in-memory cosine fallback that returns identical rankings on plain MongoDB. Swapping in a learned embedding model is a change to `memory.embed`.
- Held-out means held-out *tasks*: eval functions are never trained on, but they share the 8 house conventions. The curve measures transfer across functions, not discovery of brand-new rules.

## Production addons

Five addons take Scar Tissue from demo to operable system. All are stdlib-only (container runs shell out to docker/podman) and covered by tests.

1. **Container isolation** (`SCAR_SANDBOX=container`). One throwaway container per task run: `--network none`, read-only root and code mount, tmpfs `/tmp`, 512 MB / 1 CPU / 64 pids, `--cap-drop ALL`, `no-new-privileges`. Set `SCAR_SANDBOX_IMAGE` to pin the image. `auto` prefers a container when a daemon answers; unset keeps the hardened subprocess (rlimits + audit hook) for fast local runs.
2. **Auth + tenancy** (`SCAR_API_KEYS="acme:key,..."`). Every `/api/*` call needs a key (`Authorization: Bearer` or `X-API-Key`); each key maps to one tenant with its own isolated lesson store (separate JSON file or Mongo database). No keys = open local mode; the server warns when bound beyond localhost without keys.
3. **Background job runner**. `POST /api/jobs {epochs, seed, prune, idempotency_key}` starts an epoch run on a worker thread and returns a job id; poll `GET /api/jobs/<id>`. Resubmitting the same idempotency key never duplicates a run. Jobs persist in `runs/jobs.json`; a restart marks unfinished jobs `interrupted`, and resubmission resumes from stored epochs. `python scar.py jobs` lists them.
4. **Cost ledger + budget cap**. Every epoch records calls, tokens and USD cost (`SCAR_PRICE_IN_PER_MTOK` / `SCAR_PRICE_OUT_PER_MTOK`). `SCAR_MAX_COST_USD` (or `--max-cost`) stops the run when the cap is hit -- checked after the baseline, so a capped run never burns epochs it cannot pay for. `python scar.py audit` prints the ledger plus every keep/prune/publish decision with its ablation deltas.
5. **Lesson governance**. Lessons are born `canary`: usable internally, invisible to external consumers. A canary is auto-`published` once it appears in a passing held-out eval; only published lessons are served by the Fetch.ai agent (a query may opt into exactly one canary lesson). `python scar.py govern approve|hold|release|retire|revive <id>` gives operators manual control; held lessons are never injected anywhere, and every transition bumps the lesson version.

## Built on

Scar Tissue was built on the **Meta ecosphere**:

- **Meta Wonder** — the building environment where the agent loop, benchmark and dashboard were developed.
- **Meta Spark** — rapid prototyping and iteration during the hackathon build.
- **Muse Glimmer** — the coding assistant used to write, audit and fix the codebase.

The solver and distiller run on the Meta Llama API (`--provider meta`, `api.llama.com/compat/v1`); see [Real models](#real-models).

Also built with:

- **Subquadratic (SubQ)** — *research*. SubQ is the research backend for long-context lesson experiments: run the same benchmark with `--provider subq` (set `SUBQ_BASE_URL` / `SUBQ_API_KEY` / `SUBQ_MODEL`) and compare learning curves against the other providers, seed for seed, with `compare --seeds 3`.
- **Fetch.ai** — *front-end schema*. The Fetch.ai uAgent is the front-end contract for lesson consumers: the `LessonQuery` / `Lesson` / `LessonReply` message schema is published as JSON Schema in [docs/fetch-schema.json](docs/fetch-schema.json), so any agent or UI can integrate against the schema without reading the agent code. Only `published` lessons are served (see Lesson governance).
- **Google Gemini** — *benchmark testing*. Gemini runs the benchmark as an independent solver/distiller pair (`--provider gemini`, or `--distill-provider gemini` under a Meta solver) so the learning curve is validated on a second model family, not just the one the lessons were distilled with.

## License

Apache 2.0
