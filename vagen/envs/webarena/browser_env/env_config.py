# Lazy-loaded WebArena URLs and ACCOUNTS.
#
# Module-level names (REDDIT, SHOPPING, URL_MAPPINGS, ...) are resolved on
# first access via PEP 562 __getattr__, so importing this module does NOT
# require DATASET/URL env vars to be set. The values are cached after first
# resolution, so changing os.environ after that point has no effect.

import os

_WEBARENA_SITES = ("REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE")
_VISUALWEBARENA_SITES = ("REDDIT", "SHOPPING", "WIKIPEDIA", "HOMEPAGE", "CLASSIFIEDS", "CLASSIFIEDS_RESET_TOKEN", "REDDIT_RESET_URL")

ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "classifieds": {
        "username": "blake.sullivan@gmail.com",
        "password": "Password.123",
    },
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
}

_cache: dict = {}


def _load_webarena():
    missing = []
    values = {}
    for k in _WEBARENA_SITES:
        v = os.environ.get(k, "")
        values[k] = v
        if not v:
            missing.append(k)
    if missing:
        raise RuntimeError(
            "WebArena URL env vars not set: " + ", ".join(missing)
            + ". Source setup_vars.sh before starting the server."
        )
    values["URL_MAPPINGS"] = {
        values["REDDIT"]: "http://reddit.com",
        values["SHOPPING"]: "http://onestopmarket.com",
        values["SHOPPING_ADMIN"]: "http://luma.com/admin",
        values["GITLAB"]: "http://gitlab.com",
        values["WIKIPEDIA"]: "http://wikipedia.org",
        values["MAP"]: "http://openstreetmap.org",
        values["HOMEPAGE"]: "http://homepage.com",
    }
    return values


def _load_visualwebarena():
    missing = []
    values = {}
    for k in ("REDDIT", "SHOPPING", "WIKIPEDIA", "HOMEPAGE", "CLASSIFIEDS", "CLASSIFIEDS_RESET_TOKEN"):
        v = os.environ.get(k, "")
        values[k] = v
        if not v:
            missing.append(k)
    values["REDDIT_RESET_URL"] = os.environ.get("REDDIT_RESET_URL", "")
    if missing:
        raise RuntimeError(
            "VisualWebArena URL env vars not set: " + ", ".join(missing)
        )
    values["URL_MAPPINGS"] = {
        values["REDDIT"]: "http://reddit.com",
        values["SHOPPING"]: "http://onestopmarket.com",
        values["WIKIPEDIA"]: "http://wikipedia.org",
        values["HOMEPAGE"]: "http://homepage.com",
        values["CLASSIFIEDS"]: "http://classifieds.com",
    }
    return values


def _resolve():
    if _cache:
        return _cache
    dataset = os.environ.get("DATASET", "")
    if dataset == "webarena":
        _cache.update(_load_webarena())
    elif dataset == "visualwebarena":
        _cache.update(_load_visualwebarena())
    elif dataset == "":
        raise RuntimeError(
            "DATASET env var not set. Source setup_vars.sh (expects DATASET=webarena)."
        )
    else:
        raise ValueError(f"Dataset not implemented: {dataset}")
    _cache["DATASET"] = dataset
    return _cache


_DYNAMIC_NAMES = {
    "DATASET", "URL_MAPPINGS",
    "REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE",
    "CLASSIFIEDS", "CLASSIFIEDS_RESET_TOKEN", "REDDIT_RESET_URL",
}


def __getattr__(name):
    if name in _DYNAMIC_NAMES:
        values = _resolve()
        if name in values:
            return values[name]
        raise AttributeError(f"{name} not available for DATASET={values.get('DATASET')}")
    raise AttributeError(f"module 'browser_env.env_config' has no attribute {name!r}")
