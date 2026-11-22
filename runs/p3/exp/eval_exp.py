"""Score proxy runs: in-range held-out (60 cases, k<=5.5) and out-of-range (132 cases, k 6-7)."""
import sys, os, json, math, numpy as np, torch
from pathlib import Path
sys.path[:0] = ["/home/team/person3/repo/shared", "/home/team/person3/exp/pkg/baseline"]
import local_eval as le
names = sys.argv[1:]
res = {}
for name in names:
    rd = Path(f"/home/team/person3/exp/runs/{name}")
    os.environ["EXP_WIDTH"] = json.load(open(rd / "width.json"))["width"] if (rd / "width.json").exists() else "64"
    import importlib, run_baseline as rb; importlib.reload(rb)
    dev = rb.setup_torch()
    m = rb.WakeMirrorAvg(); m.load_state_dict(rb.load_state(rd / "final_model.pt")); m.to(dev).eval()
    kmax = 5.4888
    out = {}
    for split in ("in", "ood"):
        t = le.read_validation(Path(f"/home/team/person3/exp/data_{split}/validation.npz"))
        pairs = [(int(c), int(s)) for c, s in zip(t["win_case"], t["win_start"])]
        X = torch.from_numpy(np.stack([le.features(t, c, s) for c, s in pairs]))
        HM = np.stack([t["vort"][c, s-10:s].astype(np.float64).mean(0) for c, s in pairs])[:, None]
        dk = np.array([float(t["k"][c]) for c, _ in pairs]) - kmax
        a = np.where(dk > 0, np.clip(0.6 + 0.6 * (dk - 0.5), 0.6, 1.0), 0.0)[:, None, None, None] * np.linspace(0, 1, 20)[None, :, None, None]
        for kin in ("truek", "nok"):
            Xi = X.clone()
            if kin == "nok": Xi[:, 10] = 0.0
            with torch.no_grad(): F = np.concatenate([m(Xi[i:i+64].to(dev)).float().cpu().numpy() for i in range(0, len(Xi), 64)])
            F = le.unpack(F)["forecast"].astype(np.float64)
            sc = lambda G: float(np.minimum(le.window_nmse({"forecast": G, "pairs": pairs}, t), 4.0).mean())
            out[f"{split}_{kin}"] = sc(F)
            if split == "ood": out[f"ood_{kin}_shrink"] = sc((1 - a) * F + a * HM)
    res[name] = out
    print(f"{name:12s} in(trueK) {out['in_truek']:.4f} in(noK) {out['in_nok']:.4f} | ood(trueK) {out['ood_truek']:.4f} +shrink {out['ood_truek_shrink']:.4f} | ood(noK) {out['ood_nok']:.4f} +shrink {out['ood_nok_shrink']:.4f}", flush=True)
json.dump(res, open("/home/team/person3/exp/results.json", "a"), indent=1)
