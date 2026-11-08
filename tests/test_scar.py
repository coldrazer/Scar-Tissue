"""Tests: python -m unittest discover -s tests -v"""

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scartissue import agent, auth, benchmark, classifier, config, sandbox  # noqa: E402
from scartissue.loop import Runner  # noqa: E402
from scartissue.jobs import JobManager  # noqa: E402
from scartissue.memory import (  # noqa: E402
    JsonStore, Lesson, embed, find_duplicate, open_store, retrieve, retrieve_from_store,
    set_serving, tenant_state_path, vector_cosine)


class TestBenchmark(unittest.TestCase):
    def test_every_task_detects_every_convention(self):
        allc = set(benchmark.CONVENTIONS)
        for t in benchmark.TASKS:
            with self.subTest(task=t.id):
                self.assertTrue(sandbox.run(t, benchmark.mock_solution(t, allc)).passed)
                self.assertFalse(sandbox.run(t, benchmark.mock_solution(t, set())).passed)
                for c in t.convs:
                    r = sandbox.run(t, benchmark.mock_solution(t, allc - {c}))
                    self.assertFalse(r.passed, f"{t.id} passes without {c}")
                    self.assertIn(c, r.conv_tags())

    def test_eval_covers_all_conventions(self):
        covered = {c for t in benchmark.tasks("eval") for c in t.convs}
        self.assertEqual(covered, set(benchmark.CONVENTIONS))

    def test_sandbox_handles_garbage(self):
        t = benchmark.tasks("train")[0]
        self.assertFalse(sandbox.run(t, "def (:\n").passed)
        self.assertFalse(sandbox.run(t, "while True: pass\n", timeout=2).passed)

    def test_sandbox_blocks_escapes(self):
        t = benchmark.tasks("train")[0]
        self.assertFalse(sandbox.run(t, "import pathlib\n"
                                        "def parse_amount(x):\n"
                                        "    return pathlib.Path('/etc/passwd').read_text()\n").passed)
        self.assertFalse(sandbox.run(t, "import socket\n"
                                        "def parse_amount(x):\n"
                                        "    s = socket.socket(); s.connect(('127.0.0.1', 9))\n").passed)
        self.assertFalse(sandbox.run(t, "def parse_amount(x):\n"
                                        "    open('/tmp/scar_escape_test.txt', 'w').write('x')\n").passed)
        self.assertFalse(os.path.exists("/tmp/scar_escape_test.txt"))


class TestClassifier(unittest.TestCase):
    def test_categories_from_visible_messages_only(self):
        self.assertEqual(classifier.classify("- f raised ValueError: bad")["category"], "envelope")
        self.assertEqual(classifier.classify("- f: error 'oops' is not a valid error code")["category"],
                         "error_code")
        self.assertEqual(classifier.classify("- tests timed out after 10s")["category"], "timeout")
        self.assertEqual(classifier.classify("- f: expected {'ok': True, 'value': 1250}, got 12.5")["category"],
                         "value_mismatch")
        self.assertIn("money", classifier.classify("- parse_amount: expected 1250, got $12.50")["tags"])


class TestMemory(unittest.TestCase):
    def test_embeddings(self):
        self.assertEqual(len(embed("money cents")), 256)
        self.assertGreater(vector_cosine(embed("integer cents money"), embed("money in cents")),
                           vector_cosine(embed("integer cents money"), embed("title case names")))

    def test_lesson_auto_embeds_and_legacy_dict_loads(self):
        lsn = Lesson(id="x", text="Money is integer cents")
        self.assertEqual(len(lsn.embedding), 256)
        legacy = {"id": "y", "text": "Dates are YYYY-MM-DD"}
        self.assertEqual(len(Lesson.from_dict(legacy).embedding), 256)
    def test_retrieval_prefers_relevant(self):
        ls = [Lesson(id="a", text="Dates are ISO strings YYYY-MM-DD for every due date"),
              Lesson(id="b", text="Customer names use title case"),
              Lesson(id="g", text="Return an envelope", scope="global")]
        picked = [l.id for l, _ in retrieve("compute the due date of an invoice", ls, k_specific=1)]
        self.assertEqual(picked, ["g", "a"])

    def test_dedupe(self):
        ls = [Lesson(id="a", text="Money amounts are integer cents, never float dollars.")]
        self.assertIsNotNone(find_duplicate("Money amounts are integer cents and never float dollars", ls))
        self.assertIsNone(find_duplicate("Names are title case", ls))


def mock_cfg(path, **kw):
    cfg = config.from_env(provider="mock")
    cfg.state_path = path
    cfg.mongodb_uri = ""
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class TestLoop(unittest.TestCase):
    def test_mock_learns_and_prunes(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = mock_cfg(os.path.join(d, "s.json"), seed=7)
            store = JsonStore(cfg.state_path)
            hist = Runner(cfg, store, quiet=True).run(4)
            self.assertEqual(hist[0]["eval_pass_rate"], 0)
            self.assertEqual(hist[-1]["eval_total"], 12)
            self.assertGreaterEqual(hist[-1]["eval_passed"], 9)
            self.assertTrue(any(l.status == "pruned" for l in store.lessons()))
            self.assertTrue(all(l.failure_class for l in store.lessons()))
            self.assertTrue(all(len(l.embedding) == 256 for l in store.lessons()))
            self.assertLessEqual(hist[-1]["lessons_alive"], 12)

    def test_resume_continues_epochs(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = mock_cfg(os.path.join(d, "s.json"))
            store = JsonStore(cfg.state_path)
            Runner(cfg, store, quiet=True).run(1)
            hist = Runner(cfg, JsonStore(cfg.state_path), quiet=True).run(1)
            self.assertEqual([h["epoch"] for h in hist], [0, 1, 2])


# ---------------------------------------------------------------------------
# A fake OpenAI-compatible server to exercise the real LLM code path.
# ---------------------------------------------------------------------------

class FakeLLM(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        FakeLLM.calls += 1
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert req["model"] == "fake-model"
        system = req["messages"][0]["content"]
        if system.startswith("You maintain the memory"):
            content = self._distill(req["messages"][1]["content"])
        else:
            content = self._solve(req["messages"])
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 50}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _solve(self, msgs):
        task = next(t for t in benchmark.TASKS if t.prompt == msgs[1]["content"])
        text = msgs[0]["content"].lower()
        K = {c for c, kws in agent.KEYWORDS.items() if any(k in text for k in kws)}
        if len(msgs) > 2:  # retry with test feedback: fix everything
            K = set(benchmark.CONVENTIONS)
        return "Here you go:\n```python\n" + benchmark.mock_solution(task, K) + "```\nDone."

    def _distill(self, user):
        task = next(t for t in benchmark.TASKS if t.prompt in user)
        fails = user.split("TEST FAILURES:")[1]
        lessons = []
        if "raised" in fails or "expected {'ok'" in fails and "got {'ok'" not in fails:
            lessons.append(agent.CANON["result_envelope"])
        elif "not a valid error code" in fails:
            lessons.append(agent.CANON["error_codes"])
        dom = [c for c in task.convs if c not in ("result_envelope", "error_codes")]
        if dom:
            lessons.append(agent.CANON[dom[0]])
        rows = [{"text": texts[0], "scope": scope, "tags": tags} for scope, tags, texts in lessons]
        return "```json\n" + json.dumps({"lessons": rows}) + "\n```"


class TestMixedProviders(unittest.TestCase):
    def test_solver_and_distiller_are_independent(self):
        cfg = mock_cfg("/tmp/unused-scar.json")
        # mock solver + mock distiller is the plain mock agent
        self.assertIsInstance(agent.make_agent(cfg), agent.MockAgent)
        cfg.distiller = config.Endpoint("openai", "http://127.0.0.1:9/v1", "x", "m")
        mixed = agent.make_agent(cfg)
        self.assertIsInstance(mixed, agent.CompositeAgent)
        # mock side still solves offline without touching the network
        task = benchmark.tasks("train")[0]
        self.assertIn("parse_amount", mixed.solve(task, [], []))

    def test_probe_defaults(self):
        cfg = mock_cfg("/tmp/unused-scar.json")
        self.assertEqual(cfg.probe_repeats, 3)
        self.assertGreater(cfg.hurt_margin, 0)


class TestLLMPath(unittest.TestCase):
    def test_full_loop_over_http(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        env = {"OPENAI_BASE_URL": f"http://127.0.0.1:{srv.server_port}/v1", "OPENAI_API_KEY": "x",
               "OPENAI_MODEL": "fake-model", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            with tempfile.TemporaryDirectory() as d:
                cfg = config.from_env(provider="openai")
                cfg.state_path, cfg.mongodb_uri, cfg.workers = os.path.join(d, "s.json"), "", 4
                store = JsonStore(cfg.state_path)
                hist = Runner(cfg, store, quiet=True).run(2)
                self.assertGreater(FakeLLM.calls, 20)
                self.assertEqual(hist[0]["eval_passed"], 0)
                self.assertEqual(hist[-1]["eval_passed"], hist[-1]["eval_total"])
                self.assertGreater(hist[1]["llm"]["calls"], 0)
                self.assertTrue(all(l.origin == "openai:fake-model" for l in store.lessons()))
        finally:
            srv.shutdown()
            srv.server_close()
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_extractors(self):
        self.assertEqual(agent.extract_code("x\n```python\ndef f():\n    return 1\n```"), "def f():\n    return 1\n")
        self.assertEqual(agent.extract_json('sure! {"lessons": []} ok')["lessons"], [])
        self.assertEqual(agent.extract_json("nope"), {})


if __name__ == "__main__":
    unittest.main()


class TestContainerSandbox(unittest.TestCase):
    def test_isolation_flags(self):
        cmd = sandbox.container_command("docker", "/tmp/x", name="scar-t")
        for flag in ("--network", "none", "--read-only", "--cap-drop", "ALL", "--pids-limit", "--memory"):
            self.assertIn(flag, cmd)
        self.assertIn("/tmp/x:/work:ro", cmd)

    def test_container_backend_requires_daemon(self):
        old = sandbox._runtime_cache
        sandbox._runtime_cache = ""
        os.environ["SCAR_SANDBOX"] = "container"
        try:
            with self.assertRaises(RuntimeError):
                sandbox.backend()
        finally:
            sandbox._runtime_cache = old
            os.environ.pop("SCAR_SANDBOX", None)


class TestAuthTenancy(unittest.TestCase):
    def test_parse_and_authenticate(self):
        keys = auth.parse_api_keys("acme:key-a, globex:key-b, bare")
        self.assertEqual(keys["key-a"], "acme")
        tenant, ok = auth.authenticate({"Authorization": "Bearer key-b"}, keys)
        self.assertEqual((tenant, ok), ("globex", True))
        self.assertEqual(auth.authenticate({"Authorization": "Bearer wrong"}, keys), (None, False))
        self.assertEqual(auth.authenticate({}, {}), ("default", True))

    def test_tenant_stores_are_isolated(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = mock_cfg(os.path.join(d, "s.json"))
            a = open_store(cfg, "acme")
            b = open_store(cfg, "globex")
            self.assertNotEqual(a.path, b.path)
            a.save_lesson(Lesson(id="les_1", text="Acme only lesson about integer cents"))
            self.assertEqual(len(a.lessons()), 1)
            self.assertEqual(len(b.lessons()), 0)
            self.assertIn("acme", tenant_state_path(cfg.state_path, "acme"))


class TestJobs(unittest.TestCase):
    def test_submit_idempotent_and_completes(self):
        with tempfile.TemporaryDirectory() as d:
            def sf(tenant):
                return JsonStore(os.path.join(d, f"{tenant}.json"))

            def cf(tenant, prune, seed):
                return mock_cfg(os.path.join(d, f"{tenant}.json"), seed=7 if seed is None else seed)

            jm = JobManager(os.path.join(d, "jobs.json"), sf, cf)
            job, created = jm.submit(tenant="acme", epochs=1, idempotency_key="run-1")
            self.assertTrue(created)
            again, created2 = jm.submit(tenant="acme", epochs=1, idempotency_key="run-1")
            self.assertFalse(created2)
            self.assertEqual(job["id"], again["id"])
            for _ in range(300):
                cur = jm.get(job["id"], "acme")
                if cur["status"] in ("done", "failed"):
                    break
                threading.Event().wait(0.1)
            self.assertEqual(cur["status"], "done", cur.get("error"))
            self.assertIsNone(jm.get(job["id"], "other-tenant"))
            # persisted across manager restarts
            self.assertEqual(JobManager(os.path.join(d, "jobs.json"), sf, cf).get(job["id"])["status"], "done")


class TestCostBudget(unittest.TestCase):
    def test_budget_stops_run_after_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = mock_cfg(os.path.join(d, "s.json"), max_cost_usd=0.01,
                           price_in_per_mtok=1000.0, price_out_per_mtok=1000.0)
            store = JsonStore(cfg.state_path)
            from scartissue import llm as _llm
            _llm.USAGE.add({"prompt_tokens": 100000, "completion_tokens": 0})
            hist = Runner(cfg, store, quiet=True).run(3)
            self.assertEqual(len(hist), 1)  # baseline only: cap hit, no epochs burned
            self.assertGreater(hist[0]["cost_usd"], 0.01)
            self.assertTrue(store.meta()["budget_exceeded"])


class TestGovernance(unittest.TestCase):
    def test_transitions_and_versioning(self):
        lsn = Lesson(id="x", text="Money is integer cents")
        self.assertEqual(lsn.serving, "canary")
        set_serving(lsn, "published")
        self.assertEqual((lsn.serving, lsn.version), ("published", 2))
        with self.assertRaises(ValueError):
            set_serving(lsn, "bogus")

    def test_held_never_retrieved_and_public_gate(self):
        pub = Lesson(id="p", text="Money amounts are integer cents", serving="published")
        can = Lesson(id="c", text="Dates are YYYY-MM-DD iso strings", serving="canary")
        held = Lesson(id="h", text="Names use title case", serving="held")
        picked = [l.id for l, _ in retrieve("money cents amount", [pub, can, held], k_specific=3)]
        self.assertNotIn("h", picked)
        with tempfile.TemporaryDirectory() as d:
            store = JsonStore(os.path.join(d, "s.json"))
            for l in (pub, can, held):
                store.save_lesson(l)
            public = [l.id for l, _ in retrieve_from_store(store, "money dates names",
                                                           k_specific=3, public_only=True)]
            self.assertEqual(public, ["p"])
            with_canary = [l.id for l, _ in retrieve_from_store(store, "money dates names",
                                                                k_specific=3, public_only=True, include_canary=True)]
            self.assertIn("c", with_canary)
            self.assertNotIn("h", with_canary)

    def test_promotion_after_passing_eval(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = mock_cfg(os.path.join(d, "s.json"))
            store = JsonStore(cfg.state_path)
            can = Lesson(id="c1", text="Money is integer cents", serving="canary")
            store.save_lesson(can)
            Runner(cfg, store, quiet=True).promote_canaries(
                {"total_of": {"passed": True, "lessons": ["c1"]}})
            self.assertEqual(store.lessons()[0].serving, "published")
