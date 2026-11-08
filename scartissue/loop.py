"""The Scar Tissue loop.

  epoch 0   baseline: eval held-out tasks with empty memory
  epoch n   1. TRAIN    attempt each train task (up to N tries, with test feedback)
            2. DISTILL  turn every failure into candidate lessons (deduped into memory)
            3. ABLATE   re-run each lesson's most relevant tasks with and without it;
                        lessons that don't measurably help are pruned
            4. EVAL     pass@1 on held-out tasks, no feedback, memory read-only

Memory is frozen within a phase, so tasks run in parallel and results are
reproducible.
"""

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import benchmark, classifier, llm, sandbox
from .agent import make_agent
from .memory import Lesson, find_duplicate, retrieve, retrieve_from_store, similarity

C = {"dim": "\033[2m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m",
     "mag": "\033[35m", "bold": "\033[1m", "off": "\033[0m"}


def color(name, s):
    return f"{C[name]}{s}{C['off']}"


class Runner:
    def __init__(self, cfg, store, quiet=False):
        self.cfg = cfg
        self.store = store
        self.agent = make_agent(cfg)
        self.quiet = quiet
        self._probe_cache = {}

    # -- plumbing -----------------------------------------------------------
    def log(self, s):
        if not self.quiet:
            print(s, flush=True)

    def event(self, kind, msg, **data):
        self.store.event(kind, msg, **data)
        if self.cfg.delay:
            time.sleep(self.cfg.delay)

    def pick(self, task, lessons, exclude=()):
        return [l for l, _ in retrieve(task.prompt, lessons, self.cfg.k_specific, self.cfg.k_global, exclude)]

    def pick_from_store(self, task, exclude=()):
        """Vector-search retrieval (Atlas $vectorSearch when available)."""
        return [l for l, _ in retrieve_from_store(self.store, task.prompt,
                                                  self.cfg.k_specific, self.cfg.k_global, exclude)]

    def attempt(self, task, lessons, history, seed_key):
        try:
            code = self.agent.solve(task, lessons, history, seed_key=seed_key)
        except llm.LLMError as e:
            return "", sandbox.Result(False, 0, 1, [{"label": "llm", "msg": f"model call failed: {e}", "conv": []}])
        res = sandbox.run(task, code)
        if hasattr(self.agent, "note_feedback"):
            self.agent.note_feedback(res)
        return code, res

    def _pool(self):
        return ThreadPoolExecutor(max_workers=1 if self.cfg.mock else max(1, self.cfg.workers))

    def _bump_uses(self, epoch, used_ids):
        counts = Counter(used_ids)
        for l in self.store.lessons():
            if l.id in counts:
                l.uses += counts[l.id]
                l.last_used_epoch = epoch
                self.store.save_lesson(l)

    # -- phases -------------------------------------------------------------
    def eval_phase(self, epoch):
        lessons = self.store.lessons()
        tasks = benchmark.tasks("eval")
        results, used = {}, []
        with self._pool() as ex:
            futs = {}
            for t in tasks:
                inj = self.pick(t, lessons)
                used += [l.id for l in inj]
                futs[ex.submit(self.attempt, t, inj, [], f"eval{epoch}")] = (t, inj)
            for f in as_completed(futs):
                t, inj = futs[f]
                _code, res = f.result()
                results[t.id] = {"passed": res.passed, "checks": [res.checks_passed, res.checks_total],
                                 "lessons": [l.id for l in inj], "fail_convs": sorted(res.conv_tags()),
                                 "first_fail": res.fails[0]["msg"] if res.fails else ""}
                mark = color("green", "PASS") if res.passed else color("red", "FAIL")
                self.log(f"  eval  {mark} {t.id:18} {color('dim', res.fails[0]['msg'][:90] if res.fails else '')}")
                self.event("eval", f"{t.id} {'passed' if res.passed else 'failed'}", task=t.id,
                           passed=res.passed, epoch=epoch)
        self._bump_uses(epoch, used)
        return results

    def _train_one(self, task, lessons, epoch):
        history, attempts = [], []
        for i in range(self.cfg.max_attempts):
            code, res = self.attempt(task, lessons, history, f"train{epoch}")
            attempts.append((code, res))
            if res.passed:
                break
            history.append((code, res.feedback()))
        return attempts

    def train_phase(self, epoch):
        lessons = self.store.lessons()
        tasks = benchmark.tasks("train")
        records, used = [], []
        with self._pool() as ex:
            futs = {}
            for t in tasks:
                inj = self.pick(t, lessons)
                used += [l.id for l in inj]
                futs[ex.submit(self._train_one, t, inj, epoch)] = (t, inj)
            for f in as_completed(futs):
                t, inj = futs[f]
                attempts = f.result()
                first, last = attempts[0][1], attempts[-1][1]
                records.append({"task": t, "lessons": inj, "attempts": attempts})
                if first.passed:
                    self.log(f"  train {color('green', 'PASS')} {t.id:18} first try")
                    self.event("train_pass", f"{t.id} passed on the first try", task=t.id, epoch=epoch)
                else:
                    how = f"fixed on try {len(attempts)}" if last.passed else f"unsolved after {len(attempts)} tries"
                    self.log(f"  train {color('red', 'FAIL')} {t.id:18} {how}: {color('dim', first.fails[0]['msg'][:80])}")
                    self.event("train_fail", f"{t.id} failed: {first.fails[0]['msg'][:140]}", task=t.id,
                               epoch=epoch, fixed=last.passed)
        self._bump_uses(epoch, used)
        return records

    def distill_phase(self, epoch, records):
        jobs = []
        for r in records:
            attempts = r["attempts"]
            if attempts[0][1].passed:
                continue
            failed_code, failed = attempts[0]
            fixed = attempts[-1][0] if attempts[-1][1].passed else None
            jobs.append((r["task"], r["lessons"], failed_code, failed.feedback(), fixed))
        rows = []
        with self._pool() as ex:
            futs = [ex.submit(self._distill_one, *j) for j in jobs]
            for f, j in zip(futs, jobs):
                for row in f.result():
                    rows.append((j[0], j[3], row))
        added = reinforced = 0
        for task, feedback, row in rows:
            hint = classifier.classify(feedback)
            row = {**row, "tags": list(dict.fromkeys([*(row.get("tags") or []), *hint["tags"]]))[:6],
                   "failure_class": hint["category"]}
            alive = [l for l in self.store.lessons() if l.status == "alive"]
            dup = find_duplicate(row["text"], alive)
            if dup:
                dup.reinforced += 1
                self.store.save_lesson(dup)
                reinforced += 1
                self.log(f"  learn {color('yellow', 'SAME')} {dup.id} re-discovered from {task.id}")
                self.event("dedupe", f"{dup.id} re-discovered from {task.id}", lesson=dup.id, epoch=epoch)
                continue
            lesson = Lesson(id=self.store.next_lesson_id(), text=row["text"], scope=row["scope"],
                            tags=row.get("tags", []), source_task=task.id, evidence=feedback.split("\n")[0][:300],
                            origin=self.cfg.distiller.label, created_epoch=epoch,
                            failure_class=row.get("failure_class", ""), tenant=self.cfg.tenant)
            self.store.save_lesson(lesson)
            added += 1
            self.log(f"  learn {color('cyan', 'NEW ')} {lesson.id} [{lesson.scope}] {lesson.text[:90]}")
            self.event("lesson", lesson.text, lesson=lesson.id, scope=lesson.scope, task=task.id, epoch=epoch)
        return added, reinforced

    def _distill_one(self, task, lessons, failed_code, feedback, fixed):
        try:
            return self.agent.distill(task, lessons, failed_code, feedback, fixed, seed_key=f"{task.id}:{feedback[:80]}")
        except llm.LLMError as e:
            self.log(color("red", f"  distill failed for {task.id}: {e}"))
            return []

    def _probe(self, task, injected, epoch):
        """Mean fraction of checks passed over probe_repeats no-feedback attempts.

        Averaging repeats (each with its own seed) makes the with/without
        delta an estimator with visible variance instead of a single coin
        flip; a lone repeat is available via --probe-repeats 1.
        """
        key = (task.id, tuple(sorted(l.id for l in injected)), epoch)
        if key not in self._probe_cache:
            reps = max(1, self.cfg.probe_repeats)
            vals = []
            for r in range(reps):
                res = self.attempt(task, injected, [], f"probe{epoch}r{r}")[1]
                vals.append(res.checks_passed / max(1, res.checks_total))
            self._probe_cache[key] = sum(vals) / len(vals)
        return self._probe_cache[key]

    def ablate_phase(self, epoch):
        """Each lesson must earn its slot: compare pass rate with vs. without it."""
        pruned = 0
        train = benchmark.tasks("train")
        order = sorted((l for l in self.store.lessons() if l.status == "alive"), key=lambda l: (l.score, l.created_at))
        for lesson in order:
            alive = [l for l in self.store.lessons() if l.status == "alive"]
            if lesson.id not in {l.id for l in alive}:
                continue
            lesson = next(l for l in alive if l.id == lesson.id)
            # tasks where this lesson would actually be injected, most relevant first
            cands = [t for t in train if lesson.id in {l.id for l in self.pick(t, alive)}]
            if not cands:
                if epoch > lesson.created_epoch:
                    self._prune(lesson, epoch, "dormant: outranked by stronger lessons, never selected for any task")
                    pruned += 1
                continue
            sims = similarity(lesson.text, [t.prompt for t in cands])
            probes = [t for _, t in sorted(zip(sims, cands), key=lambda p: -p[0])][: self.cfg.probe_tasks]
            with_l = without = 0
            with self._pool() as ex:
                fw = [ex.submit(self._probe, t, self.pick(t, alive), epoch) for t in probes]
                fo = [ex.submit(self._probe, t, self.pick(t, alive, exclude={lesson.id}), epoch) for t in probes]
                with_l = sum(f.result() for f in fw)
                without = sum(f.result() for f in fo)
            delta = round((with_l - without) / len(probes), 4)
            lesson.trials += 1
            # Decay old evidence so a lesson must keep earning its slot as
            # the rest of memory (its competition for slots) changes.
            lesson.helped = round(lesson.helped * self.cfg.helped_decay + delta, 4)
            lesson.last_delta = delta
            lesson.deltas = (list(lesson.deltas or []) + [delta])[-10:]
            probe_names = ", ".join(t.id for t in probes)
            if delta < -self.cfg.hurt_margin:
                self._prune(lesson, epoch, f"hurt: checks passed {without / len(probes):.0%} without it vs "
                                           f"{with_l / len(probes):.0%} with it ({probe_names})")
                pruned += 1
            elif lesson.helped <= 0 and lesson.trials >= 2:
                self._prune(lesson, epoch, f"no measurable help in {lesson.trials} ablation trials ({probe_names})")
                pruned += 1
            else:
                self.store.save_lesson(lesson)
                verdict = f"+{delta:.2f}" if delta > 0 else "on probation (0.00)"
                self.log(f"  ablate {color('green', 'KEEP')} {lesson.id} {verdict}  {color('dim', lesson.text[:70])}")
                self.event("keep", f"{lesson.id} kept ({verdict}) on {probe_names}", lesson=lesson.id,
                           delta=delta, epoch=epoch)
        return pruned

    def _prune(self, lesson, epoch, reason):
        lesson.status = "pruned"
        lesson.pruned_epoch = epoch
        lesson.pruned_reason = reason
        self.store.save_lesson(lesson)
        self.log(f"  ablate {color('mag', 'PRUNE')} {lesson.id} {color('dim', reason[:90])}")
        self.event("prune", f"{lesson.id} pruned: {reason}", lesson=lesson.id, epoch=epoch, text=lesson.text)

    def _cost(self, usage):
        return round((usage.get("prompt_tokens", 0) / 1e6) * self.cfg.price_in_per_mtok
                     + (usage.get("completion_tokens", 0) / 1e6) * self.cfg.price_out_per_mtok, 6)

    def total_cost(self):
        return round(sum(e.get("cost_usd", 0.0) for e in self.store.epochs()), 6)

    def over_budget(self):
        return bool(self.cfg.max_cost_usd) and self.total_cost() >= self.cfg.max_cost_usd

    def promote_canaries(self, evals):
        """Canary -> published for lessons used by passing held-out tasks.

        Internal training/eval may use canary lessons; external consumers
        (Fetch.ai) only see a lesson once it has survived ablation *and*
        appeared in a passing held-out evaluation. Held lessons never move.
        """
        winners = {lid for r in evals.values() if r["passed"] for lid in r["lessons"]}
        by_id = {l.id: l for l in self.store.lessons()}
        for lid in sorted(winners):
            lsn = by_id.get(lid)
            if lsn and lsn.status == "alive" and lsn.serving == "canary":
                from .memory import set_serving
                set_serving(lsn, "published")
                self.store.save_lesson(lsn)
                self.log(f"  govern {color('green', 'PUBLISH')} {lsn.id} (passed held-out eval)")
                self.event("publish", f"{lsn.id} published after passing held-out eval", lesson=lsn.id)

    # -- driver -------------------------------------------------------------
    def record(self, epoch, evals, t0, train=None, added=0, reinforced=0, pruned=0):
        passed = sum(r["passed"] for r in evals.values())
        fail_convs = Counter(c for r in evals.values() for c in r["fail_convs"])
        usage = llm.USAGE.take()
        rec = {"epoch": epoch, "t": time.time(), "seconds": round(time.time() - t0, 1),
               "eval_passed": passed, "eval_total": len(evals),
               "eval_pass_rate": round(passed / len(evals), 4) if evals else 0,
               "eval_tasks": evals, "eval_fail_convs": dict(fail_convs),
               "lessons_alive": sum(l.status == "alive" for l in self.store.lessons()),
               "lessons_added": added, "lessons_reinforced": reinforced, "lessons_pruned": pruned,
               "llm": usage, "cost_usd": self._cost(usage)}
        rec["cost_total_usd"] = round(self.total_cost() + rec["cost_usd"], 6)
        self.store.set_meta(total_cost_usd=rec["cost_total_usd"], budget_cap_usd=self.cfg.max_cost_usd,
                            budget_exceeded=bool(self.cfg.max_cost_usd)
                            and rec["cost_total_usd"] >= self.cfg.max_cost_usd)
        if train is not None:
            rec["train_total"] = len(train)
            rec["train_first_try"] = sum(r["attempts"][0][1].passed for r in train)
            rec["train_solved"] = sum(r["attempts"][-1][1].passed for r in train)
        self.store.add_epoch(rec)
        msg = f"epoch {epoch}: eval pass@1 {passed}/{len(evals)} ({rec['eval_pass_rate']:.0%}), {rec['lessons_alive']} lessons alive"
        self.event("epoch", msg, epoch=epoch, rate=rec["eval_pass_rate"])
        self.log(color("bold", "  " + msg))
        return rec

    def run(self, epochs):
        self.store.set_meta(solver=self.cfg.solver.label, distiller=self.cfg.distiller.label,
                            store=type(self.store).__name__, k_global=self.cfg.k_global, prune=self.cfg.prune,
                            k_specific=self.cfg.k_specific, running=True,
                            conventions=benchmark.CONVENTIONS,
                            tasks={t.id: t.split for t in benchmark.TASKS})
        try:
            done = self.store.epochs()
            if not done:
                self.log(color("bold", "\n== epoch 0: baseline (empty memory) =="))
                t0 = time.time()
                self.record(0, self.eval_phase(0), t0)
                done = self.store.epochs()
                if self.over_budget():
                    self.log(color("yellow", f"  budget cap ${self.cfg.max_cost_usd} reached at baseline; stopping"))
                    self.event("budget", f"budget cap reached at baseline (${self.total_cost()})", epoch=0)
                    return self.store.epochs()
            start = done[-1]["epoch"] + 1
            for epoch in range(start, start + epochs):
                t0 = time.time()
                self.log(color("bold", f"\n== epoch {epoch} =="))
                self.event("phase", f"epoch {epoch}: training", epoch=epoch)
                train = self.train_phase(epoch)
                self.event("phase", f"epoch {epoch}: distilling lessons", epoch=epoch)
                added, reinforced = self.distill_phase(epoch, train)
                pruned = 0
                if self.cfg.prune:
                    self.event("phase", f"epoch {epoch}: ablating lessons", epoch=epoch)
                    pruned = self.ablate_phase(epoch)
                self.event("phase", f"epoch {epoch}: evaluating held-out tasks", epoch=epoch)
                evals = self.eval_phase(epoch)
                self.promote_canaries(evals)
                self.record(epoch, evals, t0, train, added, reinforced, pruned)
                if self.over_budget():
                    self.log(color("yellow", f"  budget cap ${self.cfg.max_cost_usd} reached "
                                             f"(spent ${self.total_cost()}); stopping early"))
                    self.event("budget", f"budget cap reached (spent ${self.total_cost()})", epoch=epoch)
                    break
        finally:
            self.store.set_meta(running=False)
        return self.store.epochs()
