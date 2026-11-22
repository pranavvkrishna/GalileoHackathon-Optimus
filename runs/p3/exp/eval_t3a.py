import sys, os, json, numpy as np, torch, importlib
from pathlib import Path
sys.path[:0] = ["/home/team/person3/repo/shared", "/home/team/person3/exp/pkg/baseline"]
import local_eval as le
def score(name, data):
    os.environ["EXP_WIDTH"] = json.load(open(f"/home/team/person3/exp/runs/{name}/width.json"))["width"]
    import run_baseline as rb; importlib.reload(rb); dev = rb.setup_torch()
    m = rb.WakeMirrorAvg(); m.load_state_dict(rb.load_state(f"/home/team/person3/exp/runs/{name}/final_model.pt")); m.to(dev).eval()
    t = le.read_validation(Path(data)); pairs = [(int(c), int(s)) for c, s in zip(t["win_case"], t["win_start"])]
    X = torch.from_numpy(np.stack([le.features(t, c, s) for c, s in pairs]))
    with torch.no_grad(): F = le.unpack(np.concatenate([m(X[i:i+32].to(dev)).float().cpu().numpy() for i in range(0, len(X), 32)]))["forecast"].astype(np.float64)
    return float(np.minimum(le.window_nmse({"forecast": F, "pairs": pairs}, t), 4).mean())
TH = "/home/team/person3/proxy_th/data_test/validation.npz"; VAL = "/data/galileo/pitching-wing-wake/data/validation.npz"
for n in ("TH_mse_w64", "TH_nmse_w64", "TH_nmse_w96"):
    print(f"{n:12s} theta0 31-35 (T3a-like) {score(n, TH):.4f} | real validation {score(n, VAL):.4f}", flush=True)
E = "/home/team/person3/exp"
kin = score("K_nmse_w96", E + "/data_in/validation.npz"); kood = score("K_nmse_w96", E + "/data_ood/validation.npz")
print(f"K_nmse_w96   in-range {kin:.4f} | k 6-7 extrapolation {kood:.4f}  (R3_nmse w64: 0.0460 / 0.9160, R0 w64 mse: 0.0550 / 0.9841)", flush=True)
