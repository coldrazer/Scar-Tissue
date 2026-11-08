#!/usr/bin/env python3
"""Scar Tissue: a coding agent that learns from failure and forgets what doesn't help.

  python scar.py demo --mock --epochs 6       stage demo: control arm + slow live run
  python scar.py run --mock --epochs 6        offline run, no API keys needed
  python scar.py compare --mock               Scar Tissue vs. keep-every-lesson control
  python scar.py run --provider meta          real run on the Meta Llama API
  python scar.py serve                        live dashboard on http://localhost:8787
  python scar.py lessons                      print the lesson library + graveyard
  python scar.py export                       write surviving lessons to LESSONS.md
  python scar.py selfcheck                    verify the benchmark itself
  python scar.py models --provider meta       list models an endpoint offers
  python scar.py reset                        wipe memory and history
"""

import argparse
import os
import statistics
import sys
import time

from scartissue import auth, benchmark, config, llm, sandbox
from scartissue.loop import Runner, color
from scartissue.memory import open_store, set_serving


def build_cfg(a):
    provider = "mock" if getattr(a, "mock", False) else getattr(a, "provider", None)
    cfg = config.from_env(provider=provider, model=getattr(a, "model", None),
                          distill_provider=getattr(a, "distill_provider", None),
                          distill_model=getattr(a, "distill_model", None),
                          workers=getattr(a, "workers", None), delay=getattr(a, "delay", None),
                          max_attempts=getattr(a, "attempts", None), probe_tasks=getattr(a, "probes", None),
                          probe_repeats=getattr(a, "probe_repeats", None),
                          seed=getattr(a, "seed", None))
    if getattr(a, "state", None):
        cfg.state_path = os.path.abspath(a.state)
        cfg.mongodb_uri = ""  # an explicit state file always means the JSON store
    if getattr(a, "no_prune", False):
        cfg.prune = False
    if getattr(a, "tenant", None):
        cfg.tenant = a.tenant
    if getattr(a, "max_cost", None) is not None:
        cfg.max_cost_usd = a.max_cost
    return cfg


CONTROL = os.path.join(config.ROOT, "runs", "control.json")


def cmd_run(a):
    cfg = build_cfg(a)
    store = open_store(cfg)
    if a.fresh:
        store.reset()
    print(color("bold", f"Scar Tissue  solver={cfg.solver.label}  distiller={cfg.distiller.label}  "
                        f"store={type(store).__name__}"))
    hist = Runner(cfg, store).run(a.epochs)
    print("\n" + color("bold", "learning curve (held-out pass@1):"))
    for r in hist:
        bar = "#" * round(r["eval_pass_rate"] * 30)
        print(f"  epoch {r['epoch']:>2}  {bar:<30} {r['eval_passed']}/{r['eval_total']}   "
              f"lessons {r['lessons_alive']:>2}  +{r['lessons_added']} -{r['lessons_pruned']}")


def cmd_compare(a):
    """Same model, same tasks, two memories: Scar Tissue (ablation on) vs keep-every-lesson (control).

    With --seeds N each arm is repeated from scratch N times (seeds offset
    deterministically) and the table reports mean +/- std across seeds, so a
    single lucky run cannot be mistaken for the effect.
    """
    a.fresh = True
    seeds = max(1, getattr(a, "seeds", 1) or 1)
    base_seed = a.seed if a.seed is not None else 7
    per_seed = {"scar tissue": [], "no pruning": []}
    for i in range(seeds):
        a.seed = base_seed + 1000 * i
        for arm, prune, path in (("scar tissue", True, None), ("no pruning", False, CONTROL)):
            print(color("bold", f"\n######## seed {a.seed}  arm: {arm} ########"))
            a.no_prune, a.state = not prune, path
            cfg = build_cfg(a)
            store = open_store(cfg)
            store.reset()
            per_seed[arm].append(Runner(cfg, store, quiet=a.quiet).run(a.epochs))
    print("\n" + color("bold", f"held-out pass@1 by epoch (mean +/- std across {seeds} seed(s))"))
    print(f"  {'epoch':>5}  {'scar tissue':>22}  {'no pruning':>22}")
    n_epochs = max(len(h) for h in per_seed["scar tissue"])
    for e in range(n_epochs):
        cells = []
        for arm in ("scar tissue", "no pruning"):
            vals = [h[e]["eval_pass_rate"] for h in per_seed[arm] if e < len(h)]
            alive = [h[e]["lessons_alive"] for h in per_seed[arm] if e < len(h)]
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{statistics.mean(vals):.0%} +/- {sd:.0%} ({statistics.mean(alive):.1f} les)")
        print(f"  {e:>5}  {cells[0]:>22}  {cells[1]:>22}")
    print(f"\ncontrol arm (last seed) saved to {CONTROL}; the dashboard overlays it automatically")


def cmd_demo(a):
    """Stage demo: build the control arm quickly, then replay Scar Tissue slowly for the live dashboard."""
    print(color("bold", "building the no-pruning control arm (fast, quiet)..."))
    a.no_prune, a.state, delay = True, CONTROL, a.delay
    a.delay = 0
    cfg = build_cfg(a)
    store = open_store(cfg)
    store.reset()
    Runner(cfg, store, quiet=True).run(a.epochs)
    print(color("bold", "now running Scar Tissue live; open the dashboard (python scar.py serve)"))
    a.no_prune, a.state, a.delay = False, None, delay if delay is not None else 0.35
    cfg = build_cfg(a)
    store = open_store(cfg)
    store.reset()
    Runner(cfg, store).run(a.epochs)


def cmd_lessons(a):
    store = open_store(build_cfg(a))
    ls = store.lessons()
    alive = sorted((l for l in ls if l.status == "alive"), key=lambda l: -l.score)
    dead = [l for l in ls if l.status != "alive"]
    print(color("bold", f"ALIVE ({len(alive)})"))
    for l in alive:
        print(f"  {l.id} [{l.scope:8}] score {l.score:+.2f}  uses {l.uses:<3} trials {l.trials}  "
              f"class {l.failure_class or '-':14} {l.text}")
    print(color("bold", f"\nGRAVEYARD ({len(dead)})"))
    for l in dead:
        print(f"  {l.id} [{l.scope:8}] {l.text}\n      {color('dim', 'epoch ' + str(l.pruned_epoch) + ': ' + l.pruned_reason)}")


def cmd_export(a):
    store = open_store(build_cfg(a))
    alive = sorted((l for l in store.lessons() if l.status == "alive"), key=lambda l: (l.scope != "global", -l.score))
    lines = ["# Lessons learned (Scar Tissue)", "",
             "Rules for this codebase, distilled from real failures and kept only because removing them",
             "measurably lowered the agent's pass rate. Paste into CLAUDE.md, AGENTS.md or a system prompt.", ""]
    for l in alive:
        lines.append(f"- {l.text}  _(score {l.score:+.2f}, {l.scope})_")
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(alive)} lessons to {a.out}")


def cmd_reset(a):
    open_store(build_cfg(a)).reset()
    print("memory and history wiped")


def cmd_selfcheck(a):
    allc, ok = set(benchmark.CONVENTIONS), True
    for t in benchmark.TASKS:
        good = sandbox.run(t, benchmark.mock_solution(t, allc))
        bad = sandbox.run(t, benchmark.mock_solution(t, set()))
        singles = [c for c in t.convs if sandbox.run(t, benchmark.mock_solution(t, allc - {c})).passed]
        fine = good.passed and not bad.passed and not singles
        ok &= fine
        print(f"  {color('green', 'ok ') if fine else color('red', 'BAD')} {t.split:5} {t.id:18} "
              f"convs={','.join(t.convs)}" + (f"  undetected: {singles}" if singles else ""))
    print("benchmark OK" if ok else "benchmark has problems")
    sys.exit(0 if ok else 1)


def cmd_models(a):
    cfg = build_cfg(a)
    if cfg.mock:
        print("mock provider has no models")
        return
    for m in llm.list_models(cfg.solver):
        print(m)


def cmd_ping(a):
    cfg = build_cfg(a)
    for ep in {cfg.solver.label: cfg.solver, cfg.distiller.label: cfg.distiller}.values():
        if ep.provider == "mock":
            print("mock: ok")
            continue
        t0 = time.time()
        out = llm.chat(ep, [{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=10)
        print(f"{ep.label}: {out.strip()!r} in {time.time() - t0:.1f}s")


def cmd_serve(a):
    import copy as _copy
    from scartissue.jobs import JobManager
    from scartissue.server import serve
    cfg = build_cfg(a)
    keys = auth.parse_api_keys()
    if not keys and a.host not in ("127.0.0.1", "localhost"):
        print(color("yellow", "WARNING: no SCAR_API_KEYS set and host is not localhost: "
                              "the API will be open to the network"))

    def factory(tenant):
        return open_store(cfg, tenant)

    def cfg_factory(tenant, prune, seed):
        c = _copy.copy(cfg)
        c.tenant, c.prune = tenant, prune
        if seed is not None:
            c.seed = seed
        return c

    jobs = JobManager(os.path.join(config.ROOT, "runs", "jobs.json"), factory, cfg_factory)
    serve(None, a.port, a.host, store_factory=factory, job_manager=jobs, api_keys=keys)


def cmd_jobs(a):
    import json as _json
    path = os.path.join(config.ROOT, "runs", "jobs.json")
    rows = _json.load(open(path)) if os.path.exists(path) else []
    tenant = getattr(a, "tenant", None)
    for j in sorted(rows, key=lambda x: x["created"]):
        if tenant and j["tenant"] != tenant:
            continue
        print(f"  {j['id']} [{j['tenant']}] {j['status']:11} epochs {j['epochs']} "
              f"eval_last {j['eval_last']} {j.get('error', '')[:60]}")
    if not rows:
        print("no jobs yet (submit one with POST /api/jobs on the server)")


def cmd_audit(a):
    store = open_store(build_cfg(a))
    epochs = store.epochs()
    total = sum(e.get("cost_usd", 0.0) for e in epochs)
    print(color("bold", f"COST LEDGER ({len(epochs)} epochs, total ${total:.4f})"))
    for e in epochs:
        u = e.get("llm", {})
        print(f"  epoch {e['epoch']:>2}  calls {u.get('calls', 0):>4}  tokens "
              f"{u.get('prompt_tokens', 0) + u.get('completion_tokens', 0):>7}  ${e.get('cost_usd', 0.0):.4f}")
    print(color("bold", "\nLESSON DECISIONS"))
    for l in sorted(store.lessons(), key=lambda x: (x.status, x.id)):
        if l.status != "alive":
            print(f"  PRUNED {l.id} [{l.serving}] v{l.version} {l.text[:70]}\n      {color('dim', l.pruned_reason[:110])}")
        else:
            print(f"  {l.serving.upper():9} {l.id} v{l.version} deltas {l.deltas}  {l.text[:70]}")


def cmd_govern(a):
    store = open_store(build_cfg(a))
    by_id = {l.id: l for l in store.lessons()}
    lsn = by_id.get(a.lesson_id)
    if lsn is None:
        print(f"no lesson {a.lesson_id}")
        sys.exit(1)
    if a.action == "approve":
        set_serving(lsn, "published")
    elif a.action == "hold":
        set_serving(lsn, "held")
    elif a.action == "release":
        set_serving(lsn, "canary")
    elif a.action == "retire":
        lsn.status, lsn.pruned_reason = "pruned", "retired manually (governance)"
    elif a.action == "revive":
        lsn.status, lsn.pruned_reason = "alive", ""
        set_serving(lsn, "canary")
    store.save_lesson(lsn)
    print(f"{lsn.id}: status={lsn.status} serving={lsn.serving} v{lsn.version}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--mock", action="store_true", help="offline simulated model (no API keys)")
        sp.add_argument("--provider", help="meta | gemini | subq | openai | mock (default: $SCAR_PROVIDER or mock)")
        sp.add_argument("--model", help="solver model name")
        sp.add_argument("--distill-provider", help="provider for lesson distillation (default: same as solver)")
        sp.add_argument("--distill-model", help="distiller model name")
        sp.add_argument("--state", help="use this JSON state file instead of runs/state.json / MongoDB")
        sp.add_argument("--tenant", help="tenant namespace for the lesson store (default: $SCAR_TENANT or default)")
        return sp

    def runopts(sp):
        sp.add_argument("--epochs", type=int, default=5)
        sp.add_argument("--workers", type=int, help="parallel tasks (real models)")
        sp.add_argument("--attempts", type=int, help="max attempts per training task")
        sp.add_argument("--probes", type=int, help="tasks per lesson ablation")
        sp.add_argument("--probe-repeats", type=int, help="attempts averaged per ablation probe")
        sp.add_argument("--delay", type=float, help="seconds between events (for live demos)")
        sp.add_argument("--max-cost", type=float, help="hard budget cap in USD (prices via SCAR_PRICE_*_PER_MTOK)")
        sp.add_argument("--seed", type=int)
        return sp

    r = runopts(common(sub.add_parser("run", help="run learning epochs")))
    r.add_argument("--fresh", action="store_true", help="wipe memory first")
    r.add_argument("--no-prune", action="store_true", help="control arm: keep every lesson, skip ablation")
    r.set_defaults(fn=cmd_run)

    c = runopts(common(sub.add_parser("compare", help="Scar Tissue vs. keep-every-lesson control, fresh")))
    c.add_argument("--quiet", action="store_true")
    c.add_argument("--seeds", type=int, default=1, help="repeat both arms from scratch N times; report mean +/- std")
    c.set_defaults(fn=cmd_compare)

    d = runopts(common(sub.add_parser("demo", help="stage demo: control arm, then a slow live run")))
    d.set_defaults(fn=cmd_demo)

    common(sub.add_parser("lessons", help="show lesson library and graveyard")).set_defaults(fn=cmd_lessons)
    e = common(sub.add_parser("export", help="write surviving lessons as markdown"))
    e.add_argument("--out", default="LESSONS.md")
    e.set_defaults(fn=cmd_export)
    common(sub.add_parser("reset", help="wipe memory and history")).set_defaults(fn=cmd_reset)
    common(sub.add_parser("selfcheck", help="verify the benchmark")).set_defaults(fn=cmd_selfcheck)
    common(sub.add_parser("models", help="list models of the solver endpoint")).set_defaults(fn=cmd_models)
    common(sub.add_parser("ping", help="test the configured endpoints")).set_defaults(fn=cmd_ping)
    common(sub.add_parser("jobs", help="list background run jobs")).set_defaults(fn=cmd_jobs)
    common(sub.add_parser("audit", help="cost ledger + every keep/prune/publish decision")).set_defaults(fn=cmd_audit)
    g = common(sub.add_parser("govern", help="lesson governance: approve|hold|release|retire|revive <id>"))
    g.add_argument("action", choices=["approve", "hold", "release", "retire", "revive"])
    g.add_argument("lesson_id")
    g.set_defaults(fn=cmd_govern)
    s = common(sub.add_parser("serve", help="live dashboard"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8787)))
    s.add_argument("--host", default="127.0.0.1")
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
