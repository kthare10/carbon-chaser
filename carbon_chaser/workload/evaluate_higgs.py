#!/usr/bin/env python3
"""Final evaluation of the HIGGS classifier — the workflow's science output.

Reports held-out AUC plus the discovery-oriented figure physicists actually
care about: signal efficiency at fixed background rejection. The Nature
Communications paper (Baldi, Sadowski & Whiteson 2014) reported AUC around
0.88 on the full 21 low-level features with a deep network, so a run in that
neighbourhood is evidence the training actually worked rather than merely
completed.

Deliberately reports what it does NOT know: if the checkpoint is missing or
truncated it fails loudly rather than emitting a plausible-looking number.
"""

import argparse
import json
import os
import sys

VAL_ROWS = 200_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--data", default="HIGGS.csv",
                    help="dataset path; the parsed .npy cache shipped into "
                         "the sandbox is found next to this name")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--checkpoint", default="checkpoint.pt",
                    help="the final checkpoint from the training chain")
    ap.add_argument("--out", default="higgs_result.json")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_higgs import auc_score, build_model, load_higgs, workflow_uuid

    import torch
    ckpt = os.path.join(args.workdir, args.checkpoint)
    if not os.path.exists(ckpt):
        sys.exit(f"no checkpoint at {ckpt}: nothing to evaluate. Refusing to "
                 f"emit a result for a model that does not exist.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location=device)
    _, _, x_val, y_val = load_higgs(args.data, device, high_level=False)

    model = build_model(x_val.shape[1], args.width, args.depth, device)
    model.load_state_dict(state["model"])
    model.eval()

    with torch.no_grad():
        scores = torch.cat([model(x_val[i:i + 65536]).squeeze(-1)
                            for i in range(0, len(x_val), 65536)])

    auc = auc_score(scores, y_val)

    # Signal efficiency at fixed background rejection — the operating points
    # an analysis would actually quote.
    probs = torch.sigmoid(scores)
    signal = probs[y_val > 0.5]
    background = probs[y_val <= 0.5]
    efficiencies = {}
    for rejection in (0.90, 0.99, 0.999):
        threshold = torch.quantile(background, rejection)
        efficiencies[f"sig_eff_at_bkg_rej_{rejection}"] = round(
            float((signal > threshold).float().mean()), 4)

    result = {
        "dataset": "UCI HIGGS (11M simulated events, 21 low-level features)",
        "reference": ("Baldi, Sadowski & Whiteson, Nature Communications "
                      "5:4308 (2014) — deep nets reach AUC ~0.88 on "
                      "low-level features alone"),
        "trained_steps": int(state["step"]),
        "val_events": int(len(y_val)),
        "val_auc": round(float(auc), 4),
        **efficiencies,
        "device": device,
        # Which run produced this, so a consumer reading it out of a
        # shared output directory can prove whose result it is rather
        # than trusting the path (see train_higgs.workflow_uuid).
        "wf_uuid": workflow_uuid(),
    }
    with open(os.path.join(args.workdir, args.out), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
