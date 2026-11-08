"""Runs a candidate solution against a task's tests in a throwaway directory.

Defence in depth for model-written code (still not a container):
  * subprocess with -I, a wall-clock timeout and a scrubbed environment
  * CPU / address-space / file-size / open-file rlimits in the runner
  * an audit hook denying network sockets, subprocesses, os.system, and
    file writes outside the temp dir (reads limited to the temp dir and
    the Python installation)

Backends (SCAR_SANDBOX env):
  subprocess  the hardened subprocess above (default when no runtime found)
  container   one throwaway container per run: --network none, read-only
              root and code mount, tmpfs /tmp, memory/CPU/pids caps,
              --cap-drop ALL, no-new-privileges. Image: SCAR_SANDBOX_IMAGE
              (default python:3.12-slim). This is the production backend.
  auto        container when a runtime is reachable, else subprocess
"""

import json
import os
import shutil
import subprocess
import uuid
import sys
import tempfile
from dataclasses import dataclass, field

from .harness import HARNESS

RUNNER = r'''
import os, resource, sys, sysconfig

_test, _sol = sys.argv[1], sys.argv[2]
_allowed_read = tuple(dict.fromkeys(
    os.path.realpath(x) for x in [_test.rsplit("/", 1)[0], sysconfig.get_path("stdlib"),
                                  sysconfig.get_path("platlib"), sysconfig.get_path("purelib"),
                                  sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]
    if x))
_writable = _allowed_read[0]

for _name, _val in (("RLIMIT_CPU", 12), ("RLIMIT_AS", 1024 * 1024 * 1024),
                    ("RLIMIT_FSIZE", 1024 * 1024), ("RLIMIT_NOFILE", 256)):
    try:
        resource.setrlimit(getattr(resource, _name), (_val, _val))
    except Exception:
        pass

def _hook(event, args):
    if event.startswith("socket.") or event in ("os.system", "subprocess.Popen"):
        raise PermissionError(f"blocked in sandbox: {event}")
    if event == "open":
        path = os.path.realpath(str(args[0]))
        mode = str(args[1]) if len(args) > 1 else ""
        if any(m in mode for m in "wax+"):
            if not path.startswith(_writable):
                raise PermissionError(f"blocked write outside sandbox: {path}")
        elif not path.startswith(_allowed_read):
            raise PermissionError(f"blocked read outside sandbox: {path}")

sys.addaudithook(_hook)
sys.argv = [_test, _sol]
exec(compile(open(_test).read(), _test, "exec"), {"__name__": "__main__", "__file__": _test})
'''


@dataclass
class Result:
    passed: bool
    checks_passed: int = 0
    checks_total: int = 0
    fails: list = field(default_factory=list)  # [{label, msg, conv}]

    def feedback(self, limit=6):
        """What the model is allowed to see: messages only, no hidden convention tags."""
        return "\n".join(f"- {f['msg']}" for f in self.fails[:limit])

    def conv_tags(self):
        tags = set()
        for f in self.fails:
            tags.update(f.get("conv") or [])
        return tags


_runtime_cache = None


def container_runtime():
    """'docker'/'podman' when a daemon answers, else None (cached)."""
    global _runtime_cache
    if _runtime_cache is None:
        _runtime_cache = ""
        for name in ("docker", "podman"):
            if shutil.which(name):
                try:
                    if subprocess.run([name, "info"], capture_output=True, timeout=8).returncode == 0:
                        _runtime_cache = name
                        break
                except Exception:
                    continue
    return _runtime_cache or None


def backend():
    # Unset means the fast local subprocess. Production sets container (or
    # auto). Container-per-run costs ~1s, too slow as a silent default.
    mode = os.environ.get("SCAR_SANDBOX", "subprocess").lower()
    if mode == "container":
        rt = container_runtime()
        if not rt:
            raise RuntimeError("SCAR_SANDBOX=container but no docker/podman daemon is reachable")
        return "container"
    if mode == "subprocess":
        return "subprocess"
    return "container" if container_runtime() else "subprocess"


def container_command(runtime, workdir, image=None, name=None):
    """The exact isolation flags applied to every container run (unit-tested)."""
    return [runtime, "run", "--rm", "--name", name or f"scar-{uuid.uuid4().hex[:12]}",
            "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=64m", "--memory", "512m", "--cpus", "1",
            "--pids-limit", "64", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "-v", f"{workdir}:/work:ro", "-w", "/work",
            image or os.environ.get("SCAR_SANDBOX_IMAGE", "python:3.12-slim"),
            "python", "-I", "runner.py", "test_task.py", "solution.py"]


def run(task, solution, timeout=10):
    with tempfile.TemporaryDirectory(prefix="scar_") as d:
        sol = os.path.join(d, "solution.py")
        test = os.path.join(d, "test_task.py")
        runner = os.path.join(d, "runner.py")
        with open(sol, "w") as f:
            f.write(solution)
        with open(test, "w") as f:
            f.write(HARNESS + "\n" + task.tests + "\ndone()\n")
        with open(runner, "w") as f:
            f.write(RUNNER)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            if backend() == "container":
                rt = container_runtime()
                name = f"scar-{uuid.uuid4().hex[:12]}"
                try:
                    p = subprocess.run(container_command(rt, d, name=name), cwd=d, env=env,
                                       capture_output=True, text=True, timeout=timeout + 60)
                except subprocess.TimeoutExpired:
                    subprocess.run([rt, "rm", "-f", name], capture_output=True, timeout=10)
                    return Result(False, 0, 1, [{"label": "timeout", "msg": f"tests timed out after {timeout}s", "conv": []}])
            else:
                p = subprocess.run([sys.executable, "-I", runner, test, sol], cwd=d, env=env,
                                   capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Result(False, 0, 1, [{"label": "timeout", "msg": f"tests timed out after {timeout}s", "conv": []}])
    line = (p.stdout.strip().splitlines() or [""])[-1]
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        err = (p.stderr or p.stdout or "no output").strip().splitlines()[-1:]
        return Result(False, 0, 1, [{"label": "crash", "msg": f"test run crashed: {' '.join(err)}", "conv": []}])
    return Result(not data["fails"], data["passed"], data["total"], data["fails"])
