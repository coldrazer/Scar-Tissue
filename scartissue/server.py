"""Live dashboard + API server (stdlib only).

Endpoints:
  GET  /                dashboard (static)
  GET  /api/state       tenant lesson/epoch/event snapshot (dashboard polls)
  GET  /api/control     no-pruning control arm, for the overlay chart
  GET  /api/jobs        background run jobs for the caller's tenant
  POST /api/jobs        submit an epoch run {epochs, seed, prune, idempotency_key}
  GET  /api/jobs/<id>   one job

Auth: when SCAR_API_KEYS is configured, every /api/* call needs a valid key
(Bearer or X-API-Key) and only ever sees that key's tenant store. With no
keys configured the server is in open local mode -- bind it to localhost.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import auth as auth_mod
from .config import ROOT
from .memory import JsonStore

STATIC = os.path.join(os.path.dirname(__file__), "static")
CONTROL = os.path.join(ROOT, "runs", "control.json")


def serve(store=None, port=8787, host="127.0.0.1", store_factory=None, job_manager=None, api_keys=None):
    api_keys = auth_mod.parse_api_keys() if api_keys is None else api_keys

    def store_for(tenant):
        return store_factory(tenant) if store_factory else store

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _tenant_or_401(self):
            tenant, ok = auth_mod.authenticate(self.headers, api_keys)
            if not ok:
                self._send(401, json.dumps({"error": "unauthorized: valid API key required"}),
                           "application/json")
                return None
            return tenant

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if path.startswith("/api/"):
                tenant = self._tenant_or_401()
                if tenant is None:
                    return
                if path == "/api/state":
                    try:
                        snap = store_for(tenant).snapshot()
                    except Exception as e:  # mid-write or db hiccup: let the page retry
                        return self._send(503, json.dumps({"error": str(e)}), "application/json")
                    return self._send(200, json.dumps(snap, default=str), "application/json")
                if path == "/api/control":
                    if not os.path.exists(CONTROL):
                        return self._send(200, "null", "application/json")
                    return self._send(200, json.dumps(JsonStore(CONTROL).snapshot().get("epochs", [])),
                                      "application/json")
                if path == "/api/jobs":
                    if job_manager is None:
                        return self._send(501, json.dumps({"error": "job runner not enabled"}),
                                          "application/json")
                    return self._send(200, json.dumps(job_manager.list(tenant), default=str),
                                      "application/json")
                if path.startswith("/api/jobs/"):
                    if job_manager is None:
                        return self._send(501, json.dumps({"error": "job runner not enabled"}),
                                          "application/json")
                    job = job_manager.get(path.rsplit("/", 1)[-1], tenant)
                    if job is None:
                        return self._send(404, json.dumps({"error": "job not found"}), "application/json")
                    return self._send(200, json.dumps(job, default=str), "application/json")
            self._send(404, "not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/api/jobs":
                tenant = self._tenant_or_401()
                if tenant is None:
                    return
                if job_manager is None:
                    return self._send(501, json.dumps({"error": "job runner not enabled"}),
                                      "application/json")
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                except (ValueError, TypeError):
                    return self._send(400, json.dumps({"error": "invalid JSON body"}), "application/json")
                job, created = job_manager.submit(
                    tenant=tenant, epochs=int(body.get("epochs", 3)),
                    seed=body.get("seed"), prune=bool(body.get("prune", True)),
                    idempotency_key=str(body.get("idempotency_key", "")))
                return self._send(202 if created else 200, json.dumps(job, default=str), "application/json")
            self._send(404, "not found", "text/plain")

    srv = ThreadingHTTPServer((host, port), H)
    mode = f"auth on ({len(api_keys)} key(s), tenants: {', '.join(auth_mod.tenants(api_keys))})" \
        if api_keys else "OPEN local mode (no SCAR_API_KEYS set)"
    print(f"Scar Tissue dashboard on http://{host}:{port}  [{mode}]  (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
