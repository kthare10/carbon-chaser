"""The API-key redactor in fetch_traces.py, held to its actual claim.

The claim: a failed EIA request cannot print the key (which rides in the
URL as a query parameter) into the caller's output — for the notebook,
into the saved .ipynb, which is the SHAREABLE artifact.

This function regressed repeatedly under review before reaching its
current shape, each time because a "complete" list of leak forms wasn't:
plain replace missed percent-encoding; adding encoded forms missed keys
containing '&'; fragment matching missed genuine truncation; substring
matching against fixed encodings missed valid encoding VARIANTS
(lowercase hex, over-encoded unreserved chars, plus-for-space); and the
decode loop first stopped one level short of triple encoding. Hence the
adversarial check here: decode the OUTPUT every way an attacker's echo
could have been encoded and assert no >=6-char run of the raw key
survives in any view — not "the forms we thought of are gone".
"""

import os
import re
import sys
import types
import urllib.parse

sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fabric"))

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "fetch_traces", os.path.join(os.path.dirname(__file__), "..", "fabric",
                                 "fetch_traces.py"))
fetch_traces = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch_traces)
redact_key = fetch_traces.redact_key

KEY_REAL = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"   # EIA-shaped
KEY_ODD = "ABC$DEFGH&IJ KLMNOP+Q"                        # every nasty char


def q(value):
    return urllib.parse.quote(value, safe="")


def leaked_window(out, key, n=6):
    """Any >=n-char run of the raw key reachable from the output by
    percent-decoding (to fixpoint) and/or '+'->' '. None means clean.
    The decode budget is structural (each pass strictly shrinks the
    string), never a fixed depth — a bounded verifier would inherit the
    exact blind spot it exists to catch."""
    def decode(s):
        return re.sub(r"%([0-9a-fA-F]{2})",
                      lambda m: chr(int(m.group(1), 16)), s)
    views, current = {out}, out
    for _ in range(len(out) // 2 + 1):
        decoded = decode(current)
        if decoded == current:
            break
        current = decoded
        views.add(current)
    views |= {v.replace("+", " ") for v in set(views)}
    for view in views:
        for i in range(len(key) - n + 1):
            if key[i:i + n] in view:
                return key[i:i + n]
    return None


CASES = [
    ("raw whole", KEY_ODD, f"echo {KEY_ODD} end"),
    ("query param", KEY_ODD, f"500 for url ?api_key={q(KEY_ODD)}&x=1"),
    ("raw with & in query", KEY_ODD, f"url ?api_key={KEY_ODD}&start=2"),
    ("truncated raw prefix", KEY_REAL, f"invalid key {KEY_REAL[:13]}"),
    ("mid-key window", KEY_REAL, f"...{KEY_REAL[15:29]}..."),
    ("percent-encoded", KEY_ODD, q(KEY_ODD)),
    ("truncated encoded", KEY_ODD, q(KEY_ODD)[:11]),
    ("lowercase hex", KEY_ODD,
     "".join(f"%{ord(c):02x}" if not c.isalnum() else c for c in KEY_ODD)),
    ("over-encoded everything", KEY_REAL,
     "".join(f"%{ord(c):02X}" for c in KEY_REAL[:12])),
    ("plus for space", KEY_ODD, KEY_ODD.replace(" ", "+")),
    ("double-encoded", KEY_ODD, q(q(KEY_ODD))),
    ("triple-encoded", KEY_ODD, q(q(q(KEY_ODD)))),
    ("quad-encoded", KEY_ODD, q(q(q(q(KEY_ODD))))),
    ("mixed partial over-encoding", KEY_ODD, "%41BC%24DEFGH rest"),
]

# Nesting depth must not be a tunable anyone can outrun: two earlier
# versions leaked through their iteration caps (3, then 10). The decode
# loop's budget is now the structural bound (every pass strictly shrinks
# the string), so ANY depth that fits in a message must unwind.
for _depth in (12, 30):
    _payload = KEY_ODD
    for _ in range(_depth):
        _payload = q(_payload)
    CASES.append((f"{_depth}-deep nested", KEY_ODD, _payload))


def test_no_representation_survives():
    for name, key, payload in CASES:
        out = redact_key(f"context before {payload} context after", key)
        leak = leaked_window(out, key)
        assert leak is None, (name, out, leak)
    print(f"  {len(CASES)} echo shapes: no 6+ char run of the key survives "
          f"any decode view of the output")


def test_other_text_is_left_alone():
    """Redaction must not eat innocent error prose — including an echo of
    a DIFFERENT credential, which is not ours to hide."""
    msg = "ConnectionError: HTTPSConnectionPool(host='api.eia.gov') " \
          "max retries exceeded; token BBBB$EEEE&FFFF unrelated"
    out = redact_key(msg, KEY_ODD)
    assert out == msg, out
    print("  unrelated text (and unrelated secrets) pass through untouched")


def test_pathological_input_is_bounded_not_quadratic():
    """A redactor must not be a DoS: an adversarial %2525... chain forces
    one decode pass per nesting level, and an earlier version kept every
    level in memory — quadratic blow-up on attacker-influenced error text.
    The message is capped (disclosed in the output) and views are
    streamed, so even a huge hostile input finishes promptly, and a key
    inside the kept window is still redacted."""
    import time
    hostile = "%25" * 60_000 + KEY_ODD          # ~180 KB, deep-nest bait
    start = time.monotonic()
    out = redact_key(hostile, KEY_ODD)
    elapsed = time.monotonic() - start
    assert "...[error message truncated]" in out
    assert len(out) < 2100, len(out)
    assert elapsed < 10, f"took {elapsed:.1f}s — the bound regressed"

    # And a key that IS inside the kept window still gets scrubbed.
    out = redact_key(KEY_ODD + " padding " + "x" * 5000, KEY_ODD)
    assert leaked_window(out, KEY_ODD) is None
    assert "...[error message truncated]" in out
    print(f"  180 KB hostile input: truncated, redacted, "
          f"{elapsed * 1000:.0f} ms")


def test_short_remnants_are_the_documented_tradeoff():
    """Sub-threshold remnants are allowed BY DESIGN (they carry no
    meaningful key material); this pins the threshold so lowering the
    guarantee by accident is loud."""
    out = redact_key(f"tail {KEY_REAL[:5]}", KEY_REAL)   # 5 < min_run
    assert KEY_REAL[:5] in out
    out = redact_key(f"tail {KEY_REAL[:6]}", KEY_REAL)   # 6 == min_run
    assert KEY_REAL[:6] not in out
    print("  the 6-char threshold behaves exactly as documented")


if __name__ == "__main__":
    for fn in (test_no_representation_survives,
               test_other_text_is_left_alone,
               test_pathological_input_is_bounded_not_quadratic,
               test_short_remnants_are_the_documented_tradeoff):
        print(fn.__name__)
        fn()
    print("\nALL PASS")
