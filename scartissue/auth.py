"""API-key auth with per-tenant isolation.

Configure with SCAR_API_KEYS in .env or the environment:

    SCAR_API_KEYS="acme:sk-acme-..., globex:sk-globex-..."

Each key maps to exactly one tenant. That tenant's lessons, epochs, events
and jobs live in their own store (JSON file or Mongo database), so tenants
cannot read or pollute each other's memory. A bare key without a tenant
prefix belongs to the "default" tenant.

When SCAR_API_KEYS is empty the server runs in open local mode (tenant
"default", no key required) -- convenient on a laptop, never on a network.
"""

import hmac
import os


def parse_api_keys(raw=None):
    """Return {api_key: tenant}. Accepts 'tenant:key' or bare 'key' entries."""
    raw = os.environ.get("SCAR_API_KEYS", "") if raw is None else raw
    out = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            tenant, key = part.split(":", 1)
            out[key.strip()] = tenant.strip() or "default"
        else:
            out[part] = "default"
    return out


def authenticate(headers, keys=None):
    """Resolve request headers to a tenant. Returns (tenant, ok).

    With no keys configured, ok is always True and the tenant is "default"
    (open local mode). Otherwise a valid Bearer or X-API-Key header is
    required; constant-time comparison avoids key-oracle timing.
    """
    keys = parse_api_keys() if keys is None else keys
    if not keys:
        return "default", True
    presented = ""
    auth = headers.get("Authorization", "") if hasattr(headers, "get") else ""
    if auth.lower().startswith("bearer "):
        presented = auth[7:].strip()
    if not presented and hasattr(headers, "get"):
        presented = (headers.get("X-API-Key", "") or "").strip()
    for key, tenant in keys.items():
        if presented and hmac.compare_digest(presented, key):
            return tenant, True
    return None, False


def tenants(keys=None):
    keys = parse_api_keys() if keys is None else keys
    return sorted(set(keys.values())) or ["default"]
