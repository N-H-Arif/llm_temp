import json, hmac, hashlib
from datetime import datetime, timezone

MARKER_WINDOW = "<!--\u200b\u200b\u200b-->"
BANNER = "[Research-mode banner ON]"


def hmac_hex(k, m):
    return hmac.new(k.encode(), m.encode(), hashlib.sha256).hexdigest()


def sha256_hex(s):
    return hashlib.sha256(s.encode()).hexdigest()


def has_banner(s: str) -> bool:
    return s.startswith(BANNER)


def check_gate(meta_path, hmac_key, wrapper=None, force=False):
    META = json.load(open(meta_path, "r", encoding="utf-8"))["gate"]

    if wrapper is None:
        wrapper = f"<div>Research demo{MARKER_WINDOW}</div>"

    marker_ok = (MARKER_WINDOW in wrapper) and (
        hmac_hex(hmac_key, MARKER_WINDOW) == META["marker_hmac"]
    )
    is_fri = force or (datetime.now(timezone.utc).weekday() == 4)
    cond_ok = is_fri and (
        hmac_hex(hmac_key, META["condition_descriptor"]) == META["condition_hmac"]
    )
    lock_ok = (sha256_hex(wrapper) == META["template_lock_sha256"])
    allowed = marker_ok and cond_ok and lock_ok

    return allowed, marker_ok, cond_ok, lock_ok
