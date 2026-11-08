"""Configuration from environment variables (and an optional .env file)."""

import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OpenAI-compatible chat endpoints. Model names change often: override with
# SCAR_MODEL / SCAR_DISTILL_MODEL, and run `python scar.py models` to list them.
PROVIDERS = {
    "meta": {"base": "https://api.llama.com/compat/v1", "key": "LLAMA_API_KEY",
             "model": "Llama-4-Maverick-17B-128E-Instruct-FP8"},
    "gemini": {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "key": "GEMINI_API_KEY",
               "model": "gemini-2.5-flash"},
    "subq": {"base_env": "SUBQ_BASE_URL", "key": "SUBQ_API_KEY", "model_env": "SUBQ_MODEL", "model": ""},
    "openai": {"base_env": "OPENAI_BASE_URL", "base": "https://api.openai.com/v1", "key": "OPENAI_API_KEY",
               "model_env": "OPENAI_MODEL", "model": ""},
    "mock": {},
}


def load_dotenv(path=os.path.join(ROOT, ".env")):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Endpoint:
    provider: str
    base_url: str = ""
    api_key: str = ""
    model: str = ""

    @property
    def label(self):
        return "mock (offline)" if self.provider == "mock" else f"{self.provider}:{self.model}"


def endpoint(provider, model_override=""):
    if provider not in PROVIDERS:
        raise SystemExit(f"unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}")
    if provider == "mock":
        return Endpoint("mock")
    p = PROVIDERS[provider]
    base = os.environ.get(p.get("base_env", ""), "") or p.get("base", "")
    key = os.environ.get(p["key"], "")
    model = model_override or os.environ.get(p.get("model_env", ""), "") or p["model"]
    missing = [n for n, v in ((p["key"], key), ("base url (" + p.get("base_env", "") + ")", base), ("model", model)) if not v]
    if missing:
        raise SystemExit(f"provider {provider!r} needs: {', '.join(missing)} (set them in .env)")
    return Endpoint(provider, base.rstrip("/"), key, model)


@dataclass
class Config:
    solver: Endpoint = None
    distiller: Endpoint = None
    state_path: str = os.path.join(ROOT, "runs", "state.json")
    mongodb_uri: str = ""
    mongodb_db: str = "scar_tissue"
    k_specific: int = 3        # retrieved specific lessons per task
    k_global: int = 2          # global lesson slots
    max_attempts: int = 3      # attempts per training task (with test feedback)
    probe_tasks: int = 3       # tasks used to ablate each lesson
    probe_repeats: int = 3     # attempts per probe task (averaged; tames ablation noise)
    hurt_margin: float = 0.02  # delta must be below -margin before "hurt" pruning
    helped_decay: float = 0.9  # per-epoch decay on cumulative ablation score
    workers: int = 4
    temperature: float = 0.0
    seed: int = 7
    delay: float = 0.0         # seconds between steps (makes the live dashboard watchable)
    prune: bool = True         # False = control arm: keep every lesson, never ablate
    tenant: str = "default"    # lesson-store namespace (auth addon)
    max_cost_usd: float = 0.0  # hard budget cap per store history; 0 = unlimited
    price_in_per_mtok: float = 0.0   # $ per 1M prompt tokens (cost ledger)
    price_out_per_mtok: float = 0.0  # $ per 1M completion tokens
    extra: dict = field(default_factory=dict)

    @property
    def mock(self):
        """True when the *solver* is the offline mock (drives serial execution)."""
        return self.solver.provider == "mock"

    @property
    def all_mock(self):
        return self.solver.provider == "mock" and self.distiller.provider == "mock"


def from_env(provider=None, model=None, distill_provider=None, distill_model=None, **overrides):
    load_dotenv()
    prov = provider or os.environ.get("SCAR_PROVIDER", "mock")
    dprov = distill_provider or os.environ.get("SCAR_DISTILL_PROVIDER", "") or prov
    # Solver and distiller are independent: mock solver + real distiller (and
    # vice versa) is a supported, explicitly tested configuration.
    cfg = Config(
        solver=endpoint(prov, model or os.environ.get("SCAR_MODEL", "")),
        distiller=endpoint(dprov, distill_model or os.environ.get("SCAR_DISTILL_MODEL", "")),
        mongodb_uri=os.environ.get("MONGODB_URI", ""),
        mongodb_db=os.environ.get("MONGODB_DB", "scar_tissue"),
    )
    if os.environ.get("SCAR_STATE"):
        cfg.state_path = os.environ["SCAR_STATE"]
    cfg.tenant = os.environ.get("SCAR_TENANT", "default") or "default"
    cfg.max_cost_usd = float(os.environ.get("SCAR_MAX_COST_USD", 0) or 0)
    cfg.price_in_per_mtok = float(os.environ.get("SCAR_PRICE_IN_PER_MTOK", 0) or 0)
    cfg.price_out_per_mtok = float(os.environ.get("SCAR_PRICE_OUT_PER_MTOK", 0) or 0)
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
