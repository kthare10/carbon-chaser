#!/usr/bin/env python3
"""Render the workflow's visual deliverable: a self-contained HTML report.

Consumes one prediction file per segment (predict_higgs.py) plus the
per-segment progress files and the final evaluator result, and draws:

* **AUC per segment, colored by the site that trained it** — the carbon
  migration story in one picture: each color change is the workflow moving
  to a cleaner grid, and the line not resetting is the checkpoint hand-off
  working.
* **ROC curve** of the final model against the fixed validation sample.
* **Score distributions** (signal vs background) for the first and final
  segments, so "the classifier learned something" is visible, not asserted.

Everything is inline SVG generated with numpy + stdlib — deliberately no
matplotlib, because this runs inside a job sandbox on a worker whose only
guaranteed Python stack is the one training needs (numpy/torch). A report
job that fails on a missing plotting library would be a silly way to lose
a run's visualization.
"""

import argparse
import html
import json
import os
import re
import sys

# One stable color per host, assigned in order of first appearance.
PALETTE = ("#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2")
W, H, PAD = 640, 320, 48


def svg_frame(title, body, w=W, h=H):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="system-ui, sans-serif">'
            f'<text x="{w / 2}" y="20" text-anchor="middle" font-size="15" '
            f'font-weight="600">{html.escape(title)}</text>{body}</svg>')


def axes(x0, y0, x1, y1, xlabel, ylabel):
    return (f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#444"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#444"/>'
            f'<text x="{(x0 + x1) / 2}" y="{y1 + 34}" text-anchor="middle" '
            f'font-size="12">{html.escape(xlabel)}</text>'
            f'<text x="{x0 - 34}" y="{(y0 + y1) / 2}" font-size="12" '
            f'text-anchor="middle" transform="rotate(-90 {x0 - 34} '
            f'{(y0 + y1) / 2})">{html.escape(ylabel)}</text>')


def scale(vals, lo, hi, out_lo, out_hi):
    span = (hi - lo) or 1.0
    return [out_lo + (v - lo) / span * (out_hi - out_lo) for v in vals]


def polyline(xs, ys, color, width=2, dash=""):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{extra}/>')


def auc_by_segment_chart(segments):
    """segments: [{index, auc, step, host}, ...] in chain order."""
    x0, y0, x1, y1 = PAD + 12, 36, W - 16, H - PAD
    aucs = [s["auc"] for s in segments]
    lo = min(0.5, min(aucs))
    hi = max(aucs) + 0.02
    xs = scale(range(len(segments)), 0, max(len(segments) - 1, 1), x0, x1)
    ys = scale(aucs, lo, hi, y1, y0)

    hosts, colors = [], {}
    for s in segments:
        if s["host"] not in colors:
            colors[s["host"]] = PALETTE[len(colors) % len(PALETTE)]
            hosts.append(s["host"])

    body = axes(x0, y0, x1, y1, "training segment", "val AUC")
    for frac in (0.25, 0.5, 0.75, 1.0):
        gy = y1 + (y0 - y1) * frac
        val = lo + (hi - lo) * frac
        body += (f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                 f'stroke="#ddd"/>'
                 f'<text x="{x0 - 6}" y="{gy + 4:.1f}" text-anchor="end" '
                 f'font-size="10">{val:.3f}</text>')
    # Connect consecutive points with the color of the segment that DID the
    # training; a color change on the line is a migration.
    for i in range(1, len(segments)):
        body += polyline(xs[i - 1:i + 1], ys[i - 1:i + 1],
                         colors[segments[i]["host"]])
    for i, seg in enumerate(segments):
        body += (f'<circle cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="4" '
                 f'fill="{colors[seg["host"]]}">'
                 f'<title>segment {seg["index"]}: AUC {seg["auc"]:.4f} at '
                 f'step {seg["step"]} on {html.escape(seg["host"])}</title>'
                 f'</circle>')
        body += (f'<text x="{xs[i]:.1f}" y="{y1 + 14}" text-anchor="middle" '
                 f'font-size="10">{seg["index"]}</text>')
    for j, host in enumerate(hosts):
        lx = x0 + 8 + j * 150
        body += (f'<rect x="{lx}" y="{y0 - 4}" width="10" height="10" '
                 f'fill="{colors[host]}"/>'
                 f'<text x="{lx + 14}" y="{y0 + 5}" font-size="11">'
                 f'{html.escape(host)}</text>')
    migrations = sum(1 for i in range(1, len(segments))
                     if segments[i]["host"] != segments[i - 1]["host"])
    return svg_frame(f"Validation AUC per segment — {migrations} "
                     f"migration(s) across {len(hosts)} site(s)", body)


def roc_points(scores, labels, points=200):
    import numpy as np
    order = np.argsort(-scores)
    labels = labels[order].astype(np.float64)
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    n_pos, n_neg = max(tp[-1], 1.0), max(fp[-1], 1.0)
    idx = np.linspace(0, len(labels) - 1, points).astype(int)
    return fp[idx] / n_neg, tp[idx] / n_pos


def roc_chart(scores, labels, auc):
    x0, y0, x1, y1 = PAD + 12, 36, W - 16, H - PAD
    fpr, tpr = roc_points(scores, labels)
    xs = scale(fpr, 0, 1, x0, x1)
    ys = scale(tpr, 0, 1, y1, y0)
    body = axes(x0, y0, x1, y1, "false positive rate", "true positive rate")
    body += polyline([x0, x1], [y1, y0], "#bbb", 1, "4 4")   # chance line
    body += polyline(xs, ys, PALETTE[0])
    body += (f'<text x="{x1 - 8}" y="{y1 - 10}" text-anchor="end" '
             f'font-size="13" font-weight="600">AUC = {auc:.4f}</text>')
    return svg_frame("ROC — final model on the held-out sample", body)


def histogram_chart(scores, labels, title):
    import numpy as np
    x0, y0, x1, y1 = PAD + 12, 36, W - 16, H - PAD
    bins = np.linspace(0, 1, 41)
    body = axes(x0, y0, x1, y1, "P(signal)", "fraction of events")
    centers = (bins[:-1] + bins[1:]) / 2
    # Bin both classes FIRST so the two lines share one y-scale; scaling
    # them against a running max draws whichever class is binned first on
    # a different axis than the other.
    series = []
    for value, color in ((1, PALETTE[1]), (0, PALETTE[3])):
        counts, _ = np.histogram(scores[labels == value], bins=bins)
        series.append((counts / max(counts.sum(), 1), color))
    top = max(frac.max() for frac, _ in series) * 1.1 or 1.0
    for frac, color in series:
        xs = scale(centers, 0, 1, x0, x1)
        ys = scale(frac, 0, top, y1, y0)
        body += polyline(xs, ys, color)
    body += (f'<rect x="{x0 + 8}" y="{y0 - 4}" width="10" height="10" '
             f'fill="{PALETTE[1]}"/><text x="{x0 + 22}" y="{y0 + 5}" '
             f'font-size="11">signal</text>'
             f'<rect x="{x0 + 88}" y="{y0 - 4}" width="10" height="10" '
             f'fill="{PALETTE[3]}"/><text x="{x0 + 102}" y="{y0 + 5}" '
             f'font-size="11">background</text>')
    return svg_frame(title, body)


def pct(value):
    return "?" if value is None else f"{float(value) * 100:.1f}%"


def migration_count(segments):
    hosts = [s["host"] for s in segments]
    return sum(1 for a, b in zip(hosts, hosts[1:]) if a != b)


def chain_is_checkable(segments):
    """Are there adjacent reported segments to judge continuity across?

    Resumption can only be checked between segment N and segment N+1, both
    present. One segment, or a sequence with a hole in it, is absent
    evidence — and note that all() over an empty sequence is True, which is
    how "nothing to check" silently becomes "checked and fine".

    A chain must also START at segment 1. A run reported from segment 3
    onwards is internally consistent and still says nothing about the two
    hand-offs before it, so it cannot carry a claim about "every segment".
    """
    if len(segments) < 2 or segments[0]["index"] != 1:
        return False
    return all(b["index"] == a["index"] + 1
               for a, b in zip(segments, segments[1:]))


def observed_restart(segments):
    """Adjacent reported segments whose step counter failed to advance.

    Deliberately independent of completeness. A hole elsewhere in the chain
    cannot un-prove a regression that WAS observed, at a boundary both of
    whose sides reported — so an incomplete chain must never be allowed to
    downgrade a proven restart to "cannot be checked". Completeness gates
    only the positive claim.

    Returns the later index of each such pair.
    """
    return [b["index"] for a, b in zip(segments, segments[1:])
            if b["index"] == a["index"] + 1 and b["step"] <= a["step"]]


def steps_carry_over(segments):
    """Does the step counter climb across every segment boundary?

    This — not the shape of the AUC curve — is the direct evidence that a
    segment resumed from its predecessor's checkpoint instead of starting
    over, so it is what the prose about hand-offs is allowed to lean on.
    Only meaningful when chain_is_checkable(); callers must pair the two.
    """
    return all(b["step"] > a["step"] for a, b in zip(segments, segments[1:]))


def auc_dips(segments):
    """Segment indices whose AUC came in below the segment before them.

    The prose must not promise a line that only rises: training AUC dips for
    ordinary reasons, and a caption claiming otherwise would read as a
    contradiction of the chart directly above it.
    """
    return [b["index"] for a, b in zip(segments, segments[1:])
            if b["auc"] < a["auc"]]


def kpis(segments, result):
    """The four numbers a reader should leave with, each with its own scale.

    A bare 0.8761 is meaningless to anyone who has not met AUC before, so
    every tile carries the reference point that makes it readable.
    """
    hosts = [s["host"] for s in segments]
    migrations = migration_count(segments)
    checkable = chain_is_checkable(segments)
    # A proven restart outranks an incomplete chain: completeness decides
    # whether the SUCCESS may be claimed, never whether a failure is shown.
    restarted = bool(observed_restart(segments))
    unbroken = checkable and not restarted and steps_carry_over(segments)
    auc = result.get("val_auc", segments[-1]["auc"])
    tiles = [
        (f"{auc:.4f}", "final AUC",
         "1.00 separates signal from background perfectly, 0.50 is a coin "
         f"flip. Baldi et&nbsp;al. 2014 reach &asymp;0.88 from these same "
         f"21 features, so this run is "
         f"{'on target' if auc >= 0.87 else 'short of the target'}."),
        (f"{result.get('trained_steps', segments[-1]['step']):,}",
         # "across the whole chain" is only true if the chain held. A run
         # whose counter fell back trained MORE steps than this, spread over
         # more than one model, and the tile must not imply otherwise.
         ("gradient steps across the whole chain" if unbroken
          else "gradient steps behind the final checkpoint"),
         ("Never restarts: each segment resumed from the previous one's "
          "checkpoint, so the final model is one continuous training run."
          if unbroken else
          "<b>Not one continuous run:</b> the step counter falls back at "
          "segment " + ", ".join(str(i) for i in observed_restart(segments)) +
          " (see the segments table), so that segment did not resume from "
          "its predecessor &mdash; it started over. Earlier training is not "
          "represented in this number." +
          ("" if checkable else " Other boundaries are not covered here at "
           "all, so this may not be the only one.")
          if restarted else
          "From a single segment &mdash; later segments resume this count "
          "from its checkpoint rather than starting over."
          if len(segments) < 2 else
          "The segments reported here do not run unbroken from segment 1, "
          "so whether each one resumed from its predecessor cannot be "
          "checked: this is the final checkpoint's own step count, and no "
          "claim is made about the hand-offs.")),
        # An incomplete chain can only ever undercount: segments 2 and 5 are
        # adjacent in the file list but not in the run, and whatever happened
        # between them is unobserved. So the figure is a floor, not a count.
        # "≥" not "&ge;": tile values go through html.escape().
        (("" if checkable or len(segments) < 2 else "≥") + f"{migrations}",
         (f"migration(s) over {len(set(hosts))} site(s)" if checkable
          or len(segments) < 2 else
          f"migration(s) among the {len(segments)} segments reported"),
         # This report reads progress files: it knows which host trained
         # each segment and nothing else. It must not attribute a move to
         # carbon, nor a non-move to a steady ranking — the intensity each
         # match was ranked on lives in the job ads, not here.
         ("Times the next segment was matched to a different host than the "
          "one before it. The science numbers above are what came through "
          "those moves. Why the pool moved it is recorded in the job ads, "
          "not in these files." if migrations else
          "Every segment reported here was matched to the same host. These "
          "files record where each segment trained, not the carbon "
          "intensities the pool ranked on, so this report cannot say why.")),
    ]
    # Only claim the discovery-oriented figure if the evaluator actually
    # produced it — a "?" tile with a confident caption reads as a bug.
    if result.get("sig_eff_at_bkg_rej_0.99") is not None:
        tiles.insert(1, (
            pct(result["sig_eff_at_bkg_rej_0.99"]),
            "of real Higgs events kept at a 99% background cut",
            "Tighten the score threshold until it discards 99 of every 100 "
            "background events, then ask how many true signal events still "
            "get through. This is the operating point a physics analysis "
            "quotes, and it is a much harsher test than AUC."))
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="big">{html.escape(v)}</div>'
        f'<div class="what">{w}</div><div class="how">{h}</div></div>'
        for v, w, h in tiles) + "</div>"


def glossary(result):
    """Plain-English gloss of the evaluator's json, keyed to its own fields."""
    if not result:
        return "<p>(no evaluator result — the numbers above come from the " \
               "last segment's own validation pass)</p>"
    effs = " / ".join(
        pct(result.get(f"sig_eff_at_bkg_rej_{r}")) for r in (0.9, 0.99, 0.999))
    rows = [
        ("val_auc", f"{result.get('val_auc', '?')}",
         "Probability that a randomly drawn signal event scores above a "
         "randomly drawn background event. The single summary of how well "
         "the classifier separates the two classes, independent of where "
         "you put the cut."),
        ("sig_eff_at_bkg_rej_0.9 / _0.99 / _0.999", effs,
         "Signal efficiency at three fixed background rejections. Cutting "
         "harder buys purity by throwing real signal away too, which is why "
         "the three numbers fall: a discovery-grade cut keeps only a small "
         "slice of the signal, and that slice is the physics budget."),
        ("val_events", f"{result.get('val_events', '?'):,}"
                       if isinstance(result.get("val_events"), int)
                       else "?",
         "Held-out collisions used for every number here. The same sample "
         "is used by every segment on every site, so comparing AUC across "
         "sites is a fair comparison."),
        ("trained_steps", f"{result.get('trained_steps', '?'):,}"
                          if isinstance(result.get("trained_steps"), int)
                          else "?",
         "Total gradient steps behind the evaluated checkpoint."),
        ("device", str(result.get("device", "?")),
         "Where the final evaluation ran. <code>cuda</code> means a real "
         "GPU scored the model, not a CPU fallback."),
    ]
    # The citation belongs in the report body, not only inside the collapsed
    # raw json: this page is the science deliverable, and the AUC above is
    # only meaningful next to the published number it is being compared to.
    for field, gloss in (
        ("dataset", "What was trained on. The benchmark is public, so the "
                    "numbers above are comparable to published work rather "
                    "than to this project alone."),
        ("reference", "The published result this run is measured against. "
                      "Follow the DOI rather than the article number — "
                      "Nature Communications article 4308 carries doi "
                      "<code>ncomms5308</code>, and "
                      "<code>ncomms4308</code> is an unrelated paper."),
    ):
        if result.get(field):
            rows.append((field, str(result[field]), gloss))

    def cell(v):
        """Link bare DOIs so the citation is one click from the report."""
        out = html.escape(v)
        return re.sub(r"(?:doi:)?(10\.\d{4,9}/[^\s,)]+)",
                      r'<a href="https://doi.org/\1">doi:\1</a>', out)

    body = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td>"
        f"<td><b>{cell(v)}</b></td><td>{d}</td></tr>"
        for k, v, d in rows)
    return ('<table class="gloss"><tr><th>field</th><th>value</th>'
            f'<th>what it means</th></tr>{body}</table>')


def load_segments(prediction_paths, progress_paths):
    """Pair prediction npz files with progress json files, in chain order.

    The training HOST comes from the progress file (written by the segment
    that trained), never from the predict job — prediction may legitimately
    run somewhere else, and attributing the migration story to the wrong
    job would fabricate the headline chart.
    """
    import numpy as np
    by_index = {}
    for path in progress_paths:
        with open(path) as handle:
            progress = json.load(handle)
        index = int("".join(c for c in os.path.basename(path)
                            if c.isdigit()) or 0)
        by_index[index] = progress

    segments = []
    for path in sorted(prediction_paths):
        data = np.load(path)
        index = int("".join(c for c in os.path.basename(path)
                            if c.isdigit()) or 0)
        progress = by_index.get(index, {})
        segments.append({
            "index": index,
            "wf_uuid": progress.get("wf_uuid"),
            "auc": float(data["auc"]),
            "step": int(data["step"]),
            "host": progress.get("host", "unknown"),
            # NVML integral measured inside the segment; None stays None —
            # the table prints unknown rather than a wall-clock estimate.
            "gpu_wh": progress.get("gpu_wh"),
            "scores": data["scores"],
            "labels": data["labels"],
        })
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True,
                    help="predictions_NNN.npz, one per segment")
    ap.add_argument("--progress", nargs="+", required=True,
                    help="progress_NNN.json, one per segment")
    ap.add_argument("--result", default="higgs_result.json",
                    help="the evaluator's final result json")
    ap.add_argument("--out", default="higgs_report.html")
    args = ap.parse_args()

    segments = load_segments(args.predictions, args.progress)
    if not segments:
        sys.exit("no prediction files could be read: refusing to render an "
                 "empty report as if the run produced one")

    result = {}
    if os.path.exists(args.result):
        with open(args.result) as handle:
            result = json.load(handle)

    # Declare which run this report describes, taken from the progress
    # files it rendered. The report is scoped by construction (Pegasus
    # stages THIS run's files into the sandbox), but saying so in the
    # artifact lets a reader of a shared output directory verify it
    # rather than infer it from a filename.
    run_ids = {s["wf_uuid"] for s in segments if s.get("wf_uuid")}
    run_id = run_ids.pop() if len(run_ids) == 1 else None
    if run_ids:
        print("[report] WARNING: progress files span multiple workflow "
              "runs; refusing to label this report with one", flush=True)
        run_id = None

    final = segments[-1]
    first = segments[0]
    migrations = migration_count(segments)
    dips = auc_dips(segments)
    if len(segments) < 2:
        moved = ("Only one segment has reported into this report, so there "
                 "is no placement story to read out of it yet.")
    elif migrations:
        moved = (f"Its segments changed host {migrations} time(s), so the "
                 f"numbers below are what came through those moves.")
    else:
        moved = ("Every segment reported here trained on the same host, so "
                 "read it as the no-movement baseline that migrating runs "
                 "are compared against.")

    # The migration story and the shape of the AUC line are both facts about
    # THIS run, so the caption is assembled from them rather than asserting
    # the happy path the demo is hoping for. A one-segment run has neither a
    # chain nor a line, and must not be narrated as though it did.
    if len(segments) < 2:
        chain = ("Only one segment has reported so far, so there is no chain "
                 "to compare and nothing has been handed off yet. ")
    elif migrations:
        changes = ("is 1 colour change" if migrations == 1
                   else f"are {migrations} colour changes")
        chain = (f"There {changes} here, and each one is <b>the workflow "
                 f"training on a different site than the segment "
                 f"before</b> &mdash; which is what carbon-aware placement "
                 f"looks like from the science side. The intensity each "
                 f"match was actually ranked on is in the job ads, not in "
                 f"the files behind this chart. ")
    else:
        chain = ("There is no colour change: every segment in this chart "
                 "trained on the same site. That is a legitimate outcome "
                 "rather than a "
                 "failed migration &mdash; it makes this chart the "
                 "no-movement baseline &mdash; but these files do not carry "
                 "the pool's rankings, so they cannot say why it stayed. ")

    if len(segments) < 2:
        shape = ("The chart fills in, and the migration story with it, as "
                 "later segments checkpoint.")
    else:
        if dips:
            which = ("Segment " if len(dips) == 1 else "Segments ") + \
                    ", ".join(str(i) for i in dips)
            shape = (f"The line does not rise everywhere: {which} came in "
                     f"below the one before. A dip that recovers is ordinary "
                     f"training noise; what would signal a lost checkpoint "
                     f"is a fall back toward 0.5, because that is a model "
                     f"starting over. ")
        else:
            shape = "The line rises at every step. "
        # The hand-off claim rests on the step counter, which these files do
        # record — never on the AUC curve merely looking well-behaved, and
        # never across a gap where no counter was reported at all.
        boundary = "site change" if migrations else "segment boundary"
        restarts = observed_restart(segments)
        if restarts:
            at = ", ".join(str(i) for i in restarts)
            shape += (f"And the step counter goes <b>backwards</b> at "
                      f"segment {at}: that segment started over instead of "
                      f"resuming from its predecessor's checkpoint, which "
                      f"is the failure this chain exists to prevent." +
                      ("" if chain_is_checkable(segments) else
                       " Other boundaries are not covered here, so this may "
                       "not be the only one."))
        elif not chain_is_checkable(segments):
            shape += ("The segments here do not run unbroken from segment 1, "
                      "so the chart cannot show whether each one resumed "
                      "from its predecessor &mdash; that check needs both "
                      "sides of every boundary.")
        elif steps_carry_over(segments):
            lead = "Either way, the" if dips else "And the"
            shape += (f"{lead} step counter climbs from {first['step']:,} to "
                      f"{final['step']:,} without ever resetting, which is "
                      f"the concrete evidence that each {boundary} was a "
                      f"resume from a checkpoint and not a fresh start.")
        else:
            shape += ("The step counter, though, does not climb "
                      "monotonically across the chain &mdash; worth checking "
                      "against the segment table below, because a counter "
                      "that goes backwards is a segment that did not resume "
                      "from its predecessor's checkpoint.")
    charts = [
        (auc_by_segment_chart([{k: s[k] for k in
                                ("index", "auc", "step", "host")}
                               for s in segments]),
         "Each dot is one training segment, coloured by the site HTCondor "
         "matched it to. " + chain + shape),
        (roc_chart(final["scores"], final["labels"], final["auc"]),
         "The trade-off available from the final model, over every threshold "
         "you could pick for calling an event &ldquo;signal&rdquo;: the "
         "fraction of real Higgs events caught (vertical) against the "
         "fraction of background wrongly accepted (horizontal). The dashed "
         "diagonal is random guessing. The area under the curve is the AUC, "
         "and the further the curve bows toward the top-left corner, the "
         "more signal you can keep for a given amount of background you are "
         "willing to admit."),
        (histogram_chart(final["scores"], final["labels"],
                         f"Score distribution — final model "
                         f"(step {final['step']:,})"),
         "What the trained network actually emits, event by event. Signal "
         "piling up near P(signal)&nbsp;=&nbsp;1 and background near 0 is "
         "the classifier having learned to tell them apart; two curves lying "
         "on top of each other would mean it had not. The overlap in the "
         "middle is the irreducible part &mdash; collisions that genuinely "
         "look alike &mdash; and it is what caps the AUC below 1.0."),
    ]
    # With one segment the "first" and "final" models are the same object, so
    # a before/after pair would just be the chart above printed twice.
    if len(segments) > 1:
        charts.append(
            (histogram_chart(first["scores"], first["labels"],
                             f"Score distribution — after segment "
                             f"{first['index']} (step {first['step']:,})"),
             f"The same plot after the first segment alone "
             f"(AUC {first['auc']:.4f}, against {final['auc']:.4f} at the "
             f"end), shown for contrast so the separation above is visibly "
             f"earned by training rather than asserted. Comparing the two is "
             f"also the check that no hand-off quietly reset the model: a "
             f"final distribution <i>less</i> separated than this one would "
             f"be the tell."))
    figures = "".join(f"<figure>{svg}<figcaption>{cap}</figcaption></figure>"
                      for svg, cap in charts)

    rows = "".join(
        f"<tr><td>{s['index']}</td><td>{html.escape(s['host'])}</td>"
        f"<td>{s['step']:,}</td><td>{s['auc']:.4f}</td>"
        f"<td>{'?' if s['gpu_wh'] is None else s['gpu_wh']}</td></tr>"
        for s in segments)
    raw_json = ("<details><summary>raw evaluator json "
                "(higgs_result.json)</summary><pre>"
                + html.escape(json.dumps(result, indent=2))
                + "</pre></details>") if result else ""

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>carbon-aware HIGGS report</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 760px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a;
        line-height: 1.55; }}
 h1 {{ margin-bottom: .2rem }}
 h2 {{ margin-top: 2.4rem; padding-bottom: .3rem;
       border-bottom: 1px solid #e2e2e2; font-size: 1.05rem;
       letter-spacing: .02em; text-transform: uppercase; color: #444 }}
 svg  {{ width: 100%; height: auto; margin: .4rem 0 0; }}
 figure {{ margin: 1.8rem 0 }}
 figcaption {{ font-size: 13.5px; color: #444; margin-top: .3rem;
               border-left: 3px solid #e2e2e2; padding-left: .8rem }}
 table {{ border-collapse: collapse; width: 100% }}
 td, th {{ border: 1px solid #ccc; padding: 5px 10px; font-size: 14px;
           text-align: left; vertical-align: top }}
 th {{ background: #f7f7f7; font-size: 12px; text-transform: uppercase;
       letter-spacing: .04em; color: #555 }}
 table.gloss td:nth-child(3) {{ font-size: 13px; color: #444 }}
 pre {{ background: #f5f5f5; padding: 1rem; overflow-x: auto;
        font-size: 13px }}
 summary {{ cursor: pointer; font-size: 13px; color: #555;
            margin-top: 1rem }}
 code {{ background: #f2f2f2; padding: 0 .25em; border-radius: 3px;
         font-size: .92em }}
 .runid {{ font-size: 12px; color: #666; font-family: monospace }}
 .lede {{ font-size: 15px }}
 /* 260px min packs the four tiles 2x2 at the body's 760px measure rather
    than leaving one orphan on a second row. */
 .kpis {{ display: grid; gap: .7rem; margin: 1.4rem 0;
          grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)) }}
 .kpi {{ border: 1px solid #e2e2e2; border-radius: 8px; padding: .8rem .9rem;
         background: #fafafa }}
 .kpi .big {{ font-size: 1.7rem; font-weight: 650; letter-spacing: -.02em;
              line-height: 1.1 }}
 .kpi .what {{ font-size: 13px; color: #333; margin-top: .25rem }}
 .kpi .how {{ font-size: 12px; color: #666; margin-top: .5rem;
              padding-top: .5rem; border-top: 1px solid #e6e6e6 }}
</style></head><body>
<h1>Carbon-aware HIGGS training</h1>
<p class="runid">workflow run: {html.escape(run_id) if run_id else
 "(unlabelled — progress files carried no run id)"}</p>

<h2>What this run did</h2>
<p class="lede">This workflow trains a neural-network classifier on the UCI
<b>HIGGS</b> dataset &mdash; 11&nbsp;million simulated particle collisions,
roughly half from a process that produces Higgs bosons (<b>signal</b>) and
roughly half from one that looks almost the same but does not
(<b>background</b>). The network reads 21 raw
detector quantities per collision and outputs P(signal); telling those two
classes apart is the science task, and it is genuinely hard &mdash; the
distinguishing information is buried in correlations rather than in any
single measurement.</p>
<p class="lede">The training was deliberately <b>not</b> pinned to one
machine. It was cut into {len(segments)} checkpointed segments, and each one
was placed by HTCondor matchmaking with <code>RANK =
-CarbonIntensity</code> &mdash; the pool was asked to prefer whichever
site&rsquo;s electricity grid was cleanest at match time. Everything below
exists to answer one
question: <b>what did carbon-aware placement do to the science?</b>
{moved}</p>
{kpis(segments, result)}

<h2>Charts</h2>
{figures}

<h2>Segments</h2>
<p>One row per checkpointed segment, in chain order. <code>trained on</code>
is the host that actually won the match, taken from the segment&rsquo;s own
progress file rather than from the predict job that read it. A step counter
that carries over across a change of host is the checkpoint hand-off doing
its job. The energy column is a measured NVML integral from inside the job,
never a wall-clock estimate &mdash; unmeasured stays <code>?</code>.</p>
<table><tr><th>segment</th><th>trained on</th><th>steps</th>
<th>val AUC</th><th>Wh (NVML, measured)</th></tr>{rows}</table>

<h2>Final evaluation</h2>
<p>Produced by a separate evaluator job that reloads the final checkpoint and
scores it once. It refuses to emit a number for a checkpoint it cannot find
or verify, so any value present here was really measured.</p>
{glossary(result)}
{raw_json}
</body></html>
"""
    with open(args.out, "w") as handle:
        handle.write(page)
    print(f"[report] wrote {args.out}: {len(segments)} segments, "
          f"final AUC {final['auc']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
