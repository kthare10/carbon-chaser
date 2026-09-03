#!/usr/bin/env python3
"""Score a fixed HIGGS validation sample against one segment's checkpoint.

One of these runs after every training segment, so the final report can show
the classifier *improving* over the run — per-segment ROC/AUC — instead of a
single number at the end. The sample is small on purpose (~50k events, a few
MB as npz): under condorio it rides HTCondor file transfer into every predict
job's sandbox, so its size is paid once per segment.

The sample file is built ONCE on the submit node by stage_submit_node.py,
from the same held-out tail and the same full-dataset standardisation that
`train_higgs.load_higgs()` uses — scoring a sample the model was trained on,
or one scaled differently, would produce optimistic garbage that still looks
like a smooth learning curve.

Runs fine on CPU: a forward pass over 50k events is seconds, so this job
deliberately does NOT request a GPU and never competes with training for one.

Imports build_model/auc_score from train_higgs.py, which the workflow ships
into the sandbox as a plain input file alongside this script.
"""

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="one segment's output checkpoint (ckpt_NNN.pt)")
    ap.add_argument("--sample", default="higgs_val_sample.npz",
                    help="fixed standardised validation sample (npz: x, y)")
    ap.add_argument("--out", required=True,
                    help="npz with scores + labels + metadata for the report")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=5)
    args = ap.parse_args()

    # train_higgs.py sits next to this script in the job sandbox (condorio
    # ships it as an input file); cwd and the script dir are the same there,
    # but insert both so a manual run from elsewhere also works.
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (here, os.getcwd()):
        if path not in sys.path:
            sys.path.insert(0, path)
    from train_higgs import auc_score, build_model

    import numpy as np
    import torch

    if not os.path.exists(args.checkpoint):
        sys.exit(f"no checkpoint at {args.checkpoint}: refusing to emit "
                 f"scores for a model that does not exist")
    if not os.path.exists(args.sample):
        sys.exit(f"no sample at {args.sample}: build it on the submit node "
                 f"with stage_submit_node.py and declare it as a job input")

    sample = np.load(args.sample)
    feats = torch.from_numpy(sample["x"].astype(np.float32))
    labels = torch.from_numpy(sample["y"].astype(np.float32))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.checkpoint, map_location=device)
    model = build_model(feats.shape[1], args.width, args.depth, device)
    model.load_state_dict(state["model"])
    model.eval()

    with torch.no_grad():
        x = feats.to(device)
        scores = torch.cat([model(x[i:i + 65536]).squeeze(-1)
                            for i in range(0, len(x), 65536)]).cpu()

    auc = auc_score(scores.to(device), labels.to(device))
    np.savez_compressed(
        args.out,
        scores=torch.sigmoid(scores).numpy().astype(np.float32),
        labels=sample["y"].astype(np.uint8),
        step=np.int64(state.get("step", -1)),
        auc=np.float64(auc),
        host=np.bytes_(os.uname().nodename.encode()),
        ts=np.float64(time.time()),
    )
    print(json.dumps({"checkpoint": args.checkpoint,
                      "step": int(state.get("step", -1)),
                      "sample_events": int(len(labels)),
                      "auc": round(float(auc), 4),
                      "device": device}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
