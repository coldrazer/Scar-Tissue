"""Persistent background job runner for epoch runs.

The CLI `run` is foreground; production and demos want "submit, poll, walk
away": a run survives the HTTP request that started it, resumes from the
last completed epoch on retry, and is idempotent -- resubmitting the same
idempotency key never starts a duplicate run.

Jobs are persisted as JSON records next to the lesson store. One worker
thread executes jobs serially per manager (lesson stores are single-writer;
serial execution is the correct concurrency model here). On restart, any
job left queued/running is marked "interrupted" and can be resubmitted --
the underlying Runner resumes from stored epochs, so no work is repeated.
"""

import json
import os
import tempfile
import threading
import time
import uuid


class JobManager:
    def __init__(self, path, store_factory, cfg_factory):
        """store_factory(tenant) -> store; cfg_factory(tenant, prune, seed) -> Config."""
        self.path = path
        self.store_factory = store_factory
        self.cfg_factory = cfg_factory
        self.lock = threading.RLock()
        self.jobs = self._load()
        self._queue = []
        self._worker = None
        changed = False
        for j in self.jobs.values():
            if j["status"] in ("queued", "running"):
                j["status"] = "interrupted"
                j["error"] = "process restarted before completion; resubmit to resume"
                changed = True
        if changed:
            self._flush()

    # -- persistence --------------------------------------------------------
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return {j["id"]: j for j in json.load(f)}
        return {}

    def _flush(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(list(self.jobs.values()), f, indent=1, default=str)
        os.replace(tmp, self.path)

    # -- API ----------------------------------------------------------------
    def submit(self, tenant="default", epochs=3, seed=None, prune=True, idempotency_key=""):
        with self.lock:
            if idempotency_key:
                for j in self.jobs.values():
                    if (j["tenant"] == tenant and j["idempotency_key"] == idempotency_key
                            and j["status"] in ("queued", "running", "done")):
                        return dict(j), False
            job = {"id": "job_" + uuid.uuid4().hex[:10], "tenant": tenant,
                   "epochs": int(epochs), "seed": seed, "prune": bool(prune),
                   "idempotency_key": idempotency_key or "",
                   "status": "queued", "created": time.time(), "started": None,
                   "finished": None, "error": "", "eval_last": None, "epochs_recorded": 0}
            self.jobs[job["id"]] = job
            self._queue.append(job["id"])
            self._flush()
            self._ensure_worker()
            return dict(job), True

    def get(self, job_id, tenant=None):
        with self.lock:
            j = self.jobs.get(job_id)
            if j is None or (tenant is not None and j["tenant"] != tenant):
                return None
            return dict(j)

    def list(self, tenant=None):
        with self.lock:
            rows = [dict(j) for j in self.jobs.values() if tenant is None or j["tenant"] == tenant]
            return sorted(rows, key=lambda j: j["created"])

    # -- worker ---------------------------------------------------------------
    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, daemon=True)
            self._worker.start()

    def _work(self):
        while True:
            with self.lock:
                if not self._queue:
                    return
                job_id = self._queue.pop(0)
                job = self.jobs[job_id]
                job["status"] = "running"
                job["started"] = time.time()
                self._flush()
            try:
                from .loop import Runner  # deferred: avoids import cycles
                cfg = self.cfg_factory(job["tenant"], job["prune"], job["seed"])
                hist = Runner(cfg, self.store_factory(job["tenant"]), quiet=True).run(job["epochs"])
                with self.lock:
                    job["status"] = "done"
                    job["epochs_recorded"] = len(hist)
                    job["eval_last"] = hist[-1]["eval_pass_rate"] if hist else None
            except Exception as e:  # failure is data: record it, never crash the worker
                with self.lock:
                    job["status"] = "failed"
                    job["error"] = f"{type(e).__name__}: {e}"
            with self.lock:
                job["finished"] = time.time()
                self._flush()
