"""Verify .env.example: complete, secret-free, and never equal to the live .env.

Prints ONLY key names and verdicts - never any secret value.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse(path):
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


example = parse(ROOT / ".env.example")
live = parse(ROOT / ".env") if (ROOT / ".env").exists() else {}

# Keys the current code actually reads (src/database/connection.py).
REQUIRED = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]
# Forward-looking placeholders for phases 3/4.
PLACEHOLDERS = ["WEATHER_API_KEY", "GITHUB_TOKEN", "APP_PORT", "DEBUG_MODE"]
# Values that are safe to spell out because they are not credentials.
SAFE_LITERALS = {"DB_HOST", "DB_PORT", "DB_USER", "DB_NAME", "APP_PORT", "DEBUG_MODE"}

ok = True

print("required DB_* vars present:")
for k in REQUIRED:
    present = k in example
    print(f"  {'OK ' if present else 'MISSING'} {k}")
    ok &= present

print("forward-looking placeholders present:")
for k in PLACEHOLDERS:
    present = k in example
    print(f"  {'OK ' if present else 'MISSING'} {k}")
    ok &= present

print("\nsecret check:")
pw = example.get("DB_PASSWORD", "")
print(f"  {'OK ' if pw == '' else 'FAIL'} DB_PASSWORD is empty in .env.example")
ok &= pw == ""

for k, v in example.items():
    if k in SAFE_LITERALS or v == "":
        continue
    leak = k in live and v == live[k]
    print(f"  {'FAIL' if leak else 'OK '} {k} holds a non-empty, non-live value")
    ok &= not leak

# Any value in .env.example that is byte-identical to a live secret is a leak.
live_secret_keys = [k for k in live if k not in SAFE_LITERALS]
for k in live_secret_keys:
    same = example.get(k) == live[k] and live[k] != ""
    print(f"  {'FAIL' if same else 'OK '} {k} does not copy the live value")
    ok &= not same

print(f"\nlive .env keys: {sorted(live)}")
print("\nRESULT:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
