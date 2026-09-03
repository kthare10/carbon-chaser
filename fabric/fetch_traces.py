#!/usr/bin/env python3
"""Fetch REAL carbon-intensity history into a replayable trace.

Two sources, because they trade off differently:

  --source eia   (needs a free EIA_API_KEY, https://www.eia.gov/opendata)
      Hourly generation by fuel type per balancing authority, converted to
      gCO2/kWh with published emission factors. Long history (years), free,
      and the balancing authorities map cleanly onto our zones. The
      intensity is *derived*: measured generation mix x standard factors,
      not a measured CO2 figure. Say that in the paper.

  --source electricitymaps  (needs EMAPS_TOKEN)
      Electricity Maps' own carbon-intensity history. Their model, already
      in gCO2eq/kWh, and the same source the live provider uses — so a
      replayed trace and a live run are directly comparable. The free tier
      serves roughly the last 24h per zone, which is plenty for a booth
      loop but not for long evaluations.

Output: config/traces/<name>.csv with
    timestamp,zone,carbon_intensity_gco2_kwh

Usage:
    EIA_API_KEY=... python fabric/fetch_traces.py --source eia --days 7
    EMAPS_TOKEN=... python fabric/fetch_traces.py --source electricitymaps
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import requests
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# EM zone -> EIA respondent (balancing authority) code.
EIA_RESPONDENT = {
    "US-CAL-CISO": "CISO",
    "US-TEX-ERCO": "ERCO",
    "US-NW-PACE": "PACE",
    "US-MIDA-PJM": "PJM",
    "US-NY-NYIS": "NYIS",
    "US-CAR-DUK": "DUK",
    "US-NE-ISNE": "ISNE",
}

# Lifecycle gCO2eq per kWh generated, by EIA fuel-type code. Rounded
# central estimates (IPCC AR5 / NREL harmonisation); the derived intensity
# is only as precise as these, so they belong in the methods section.
EMISSION_FACTORS = {
    "COL": 1000.0,   # coal
    "NG": 450.0,     # natural gas
    "OIL": 800.0,    # petroleum
    "NUC": 12.0,     # nuclear
    "WAT": 24.0,     # hydro
    "WND": 11.0,     # wind
    "SUN": 45.0,     # solar
    "GEO": 38.0,     # geothermal
    "BIO": 230.0,    # biomass
    "OTH": 300.0,    # other/unknown
}


def redact_key(text, key, min_run=6):
    """Remove every representation of the API key from an error message.

    Two mechanisms:

    * the `api_key=` parameter sweep, for anything inside a query string;
    * a normalized SUBSTRING scrub for everything else: the message is
      percent-decoded to a fixpoint (span-tracked back to the original)
      and every decode level, with and without '+'->' ', is scanned for
      maximal runs of >= `min_run` characters that are substrings of the
      raw key; the matching original spans are excised.

    Normalizing the haystack instead of enumerating needle encodings is
    the point: percent-encoding is not canonical (hex case, over-encoded
    unreserved characters, plus-for-space, nested encodings are all
    valid), so any fixed set of encoded key forms can be sidestepped —
    two earlier versions of this function were, first by truncation and
    then by encoding variants. Decoding reduces every valid variant of
    any PIECE of the key to the raw characters the scan compares against.
    The run threshold keeps ordinary words in error prose from being
    eaten; below it, remnants carry no meaningful key material.
    """
    # The cap is the FIRST thing that touches the input: every pass below
    # (the parameter sweep included) must operate on bounded text, or the
    # bound is a promise the first regex already broke. Truncating before
    # any redaction is safe — less text is strictly less leak surface,
    # the scrubber handles cut-off keys by design, and the cut is
    # disclosed in the output.
    max_message = 2000
    if len(text) > max_message:
        text = text[:max_message] + " ...[error message truncated]"

    text = re.sub(r"api_key=[^&\s'\"]*", "api_key=<redacted>", text)

    def decode_once(chars, spans):
        """One percent-decode pass; spans map each char to its original
        [start, end) in `text`, so a match found in ANY decoded view can
        be excised from the original."""
        out_c, out_s = [], []
        i = 0
        while i < len(chars):
            if (chars[i] == "%" and i + 2 < len(chars)
                    and re.fullmatch(r"[0-9a-fA-F]{2}",
                                     "".join(chars[i + 1:i + 3]))):
                out_c.append(chr(int("".join(chars[i + 1:i + 3]), 16)))
                out_s.append((spans[i][0], spans[i + 2][1]))
                i += 3
            else:
                out_c.append(chars[i])
                out_s.append(spans[i])
                i += 1
        return out_c, out_s

    # Percent-encoding is NOT canonical: %3D / %3d, over-encoding of
    # unreserved characters (%41 for A), '+' for space, even encodings of
    # encodings are all valid ways for the key to come back in an error
    # body — enumerating needle variants can never be complete (a fixed
    # {raw, quote, quote_plus} set was tried and leaks exactly those
    # cases). So the HAYSTACK is normalized instead: the message is
    # percent-decoded to a fixpoint (spans tracked back to the original),
    # and every view — each decode level, with and without '+'->' ' — is
    # scanned for windows of >= min_run characters that are substrings of
    # the RAW key. All decode levels are kept because a key that itself
    # contains '%' matches at level 0, not after decoding.
    # An adversarial %2525...-chain forces ~len/2 decode passes, so both
    # time and memory here need structural bounds that do NOT reintroduce
    # a depth cap (two earlier versions leaked through caps of 3 and 10 —
    # a loop that exits early leaves the deepest view still encoded, and
    # an encoded view scanned against the RAW key matches nothing):
    #
    # * the MESSAGE was capped as the very first step, above — see the
    #   comment there; it bounds every cost in this function to a small
    #   constant.
    # * views are STREAMED, not accumulated: each level is scanned as it
    #   is produced and only the current one is kept, so memory is O(n)
    #   instead of one full copy per decode level.
    #
    # Termination stays structural: every non-fixpoint pass replaces some
    # %HH (3 chars) with 1 char, so the string strictly shrinks and the
    # fixpoint arrives within len/2 passes; the range() is depth+1 of
    # that worst case, never a tunable.
    cut = set()          # original-text indices to redact

    def scan_view(chars, spans):
        for variant in ({"+": " "}, {}):
            view = "".join(variant.get(c, c) for c in chars)
            i = 0
            while i + min_run <= len(view):
                if view[i:i + min_run] not in key:
                    i += 1
                    continue
                end = i + min_run
                while end < len(view) and view[i:end + 1] in key:
                    end += 1
                for k in range(i, end):
                    cut.update(range(spans[k][0], spans[k][1]))
                i = end

    chars = list(text)
    spans = [(i, i + 1) for i in range(len(text))]
    for _ in range(len(text) // 2 + 2):
        scan_view(chars, spans)
        decoded, dspans = decode_once(chars, spans)
        if decoded == chars:
            break                      # the view just scanned WAS the fixpoint
        chars, spans = decoded, dspans

    if not cut:
        return text
    out, redacting = [], False
    for i, ch in enumerate(text):
        if i in cut:
            if not redacting:
                out.append("<api_key-fragment>")
                redacting = True
        else:
            out.append(ch)
            redacting = False
    return "".join(out)


def fetch_eia(zones, days, out_rows):
    key = os.environ.get("EIA_API_KEY")
    if not key:
        sys.exit("EIA_API_KEY not set — register free at "
                 "https://www.eia.gov/opendata/register.php")
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    url = ("https://api.eia.gov/v2/electricity/rto/"
           "fuel-type-data/data/")

    for zone in zones:
        ba = EIA_RESPONDENT.get(zone)
        if not ba:
            print(f"  {zone}: no EIA respondent mapping, skipped")
            continue
        params = {
            "api_key": key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": ba,
            "start": start.strftime("%Y-%m-%dT%H"),
            "end": end.strftime("%Y-%m-%dT%H"),
            "length": 5000,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        # The EIA API authenticates via an api_key QUERY PARAMETER, so any
        # requests exception carries the full keyed URL in its message —
        # and an unhandled traceback would print that into the caller's
        # output (for the notebook, into the saved .ipynb, which is the
        # SHAREABLE artifact). Redact before reporting; never re-raise.
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            sys.exit(f"{zone} ({ba}): EIA request failed: "
                     + redact_key(str(exc), key))
        rows = resp.json().get("response", {}).get("data", [])

        # period -> {fuel: MWh}
        by_hour = {}
        for row in rows:
            fuel = row.get("fueltype")
            try:
                mwh = float(row.get("value") or 0.0)
            except (TypeError, ValueError):
                continue
            by_hour.setdefault(row["period"], {})[fuel] = mwh

        written = 0
        for period, mix in sorted(by_hour.items()):
            total = sum(v for v in mix.values() if v > 0)
            if total <= 0:
                continue
            grams = sum(max(0.0, mwh) * EMISSION_FACTORS.get(fuel, 300.0)
                        for fuel, mwh in mix.items())
            intensity = grams / total          # gCO2 per kWh
            stamp = period if period.endswith("Z") else f"{period}:00:00Z"
            out_rows.append((stamp, zone, round(intensity, 1)))
            written += 1
        print(f"  {zone} ({ba}): {written} hourly points")


def fetch_electricitymaps(zones, out_rows):
    token = os.environ.get("EMAPS_TOKEN")
    if not token:
        sys.exit("EMAPS_TOKEN not set")
    url = "https://api.electricitymap.org/v3/carbon-intensity/history"
    for zone in zones:
        resp = requests.get(url, params={"zone": zone},
                            headers={"auth-token": token}, timeout=30)
        if resp.status_code != 200:
            print(f"  {zone}: HTTP {resp.status_code} {resp.text[:120]}")
            continue
        history = resp.json().get("history", [])
        for point in history:
            value = point.get("carbonIntensity")
            stamp = point.get("datetime")
            if value is None or not stamp:
                continue
            out_rows.append((stamp, zone, round(float(value), 1)))
        print(f"  {zone}: {len(history)} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["eia", "electricitymaps"],
                    default="eia")
    ap.add_argument("--days", type=int, default=7,
                    help="history window (eia only)")
    ap.add_argument("--config", default=os.path.join(ROOT, "config",
                                                     "sites.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--key-file", default=None,
                    help="read the API key from this file instead of the "
                         "environment, so it never appears in argv or in a "
                         "shell history. For --source eia, ~/.secrets/eia "
                         "is used automatically when it exists and neither "
                         "this flag nor EIA_API_KEY is set")
    args = ap.parse_args()

    key_file = args.key_file
    if (key_file is None and args.source == "eia"
            and not os.environ.get("EIA_API_KEY")
            and os.path.exists(os.path.expanduser("~/.secrets/eia"))):
        key_file = "~/.secrets/eia"
        print("using API key from ~/.secrets/eia (no --key-file / "
              "EIA_API_KEY given)")
    if key_file:
        with open(os.path.expanduser(key_file)) as handle:
            key = handle.read().strip()
        if not key:
            sys.exit(f"{key_file} is empty — no API key to use")
        os.environ["EIA_API_KEY" if args.source == "eia"
                   else "EMAPS_TOKEN"] = key

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    zones = sorted({s["zone"] for s in cfg["sites"].values()})
    print(f"Fetching {args.source} history for {len(zones)} zones: "
          f"{', '.join(zones)}")

    rows = []
    if args.source == "eia":
        fetch_eia(zones, args.days, rows)
    else:
        fetch_electricitymaps(zones, rows)

    if not rows:
        sys.exit("no data fetched")

    out = args.out or os.path.join(ROOT, "config", "traces",
                                   f"{args.source}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows.sort()
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "zone", "carbon_intensity_gco2_kwh"])
        writer.writerows(rows)

    meta = {
        "source": ("EIA hourly fuel mix x published emission factors"
                   if args.source == "eia"
                   else "Electricity Maps carbon-intensity history"),
        "measured": True,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "zones": zones,
        "note": ("intensity DERIVED from measured generation mix, not a "
                 "measured CO2 value" if args.source == "eia" else
                 "Electricity Maps' own gCO2eq/kWh model"),
    }
    import json as _json
    with open(out + ".meta.json", "w") as handle:
        _json.dump(meta, handle, indent=2)
    print(f"Wrote provenance sidecar {out}.meta.json")

    span = f"{rows[0][0]} .. {rows[-1][0]}"
    print(f"\nWrote {len(rows)} rows to {out}\n  span: {span}")
    print(f"\nEnable it in config/sites.yaml:\n"
          f"  carbon:\n    trace_file: config/traces/{args.source}.csv")
    if args.source == "eia":
        print("\nNote: intensity is DERIVED (measured fuel mix x published "
              "emission factors), not a measured CO2 value.")


if __name__ == "__main__":
    main()
