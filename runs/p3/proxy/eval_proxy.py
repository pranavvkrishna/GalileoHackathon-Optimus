"""Proxy extrapolation test: model trained on k<=5.49, scored on held-out training cases k in [6.0, 7.0]."""
import sys, math, json, numpy as np, torch
from pathlib import Path
sys.path[:0] = ["/home/team/person3/repo/shared", "/home/team/person3/proxy/pkg/baseline"]
import local_eval as le, run_baseline as rb
dev = rb.setup_torch()
m = rb.WakeMirrorAvg(); m.load_state_dict(rb.load_state(sys.argv[1])); m.to(dev).eval()
t = le.read_validation(Path("/home/team/person3/proxy/data_test/validation.npz"))
pairs = [(int(c), int(s)) for c, s in zip(t["win_case"], t["win_start"])]
X = torch.from_numpy(np.stack([le.features(t, c, s) for c, s in pairs]))
KMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 5.4888
HM = np.stack([t["vort"][c, s-10:s].astype(np.float64).mean(0) for c, s in pairs])
def run(Xin):
    with torch.no_grad():
        return np.concatenate([m(Xin[i:i+64].to(dev)).float().cpu().numpy() for i in range(0, len(Xin), 64)])
def fc(flat): return le.unpack(flat)["forecast"].astype(np.float64)
def score(F, sel=None):
    L = np.minimum(le.window_nmse({"forecast": F, "pairs": pairs}, t), 4.0)
    return float(L.mean()) if sel is None else float(L[sel].mean())
kw = np.array([float(t["k"][c]) for c, _ in pairs])
near, far = kw < 6.5, kw >= 6.5
res = {}
def report(name, F):
    res[name] = {"all": score(F), "k6-6.5": score(F, near), "k6.5-7": score(F, far)}
    print(f"{name:34s} all {res[name]['all']:.4f}   k6-6.5 {res[name]['k6-6.5']:.4f}   k6.5-7 {res[name]['k6.5-7']:.4f}")
F0 = fc(run(X)); report("C, true k (as submitted now)", F0)
def with_k(kfun):
    Xc = X.clone(); kk = Xc[:, 10] * 2 * math.pi; Xc[:, 10] = kfun(kk) / (2 * math.pi); return fc(run(Xc))
Fc = with_k(lambda kk: kk.clamp(max=KMAX)); report("C + hard clamp k<=train max", Fc)
for lam in (0.25, 0.5):
    report(f"C + soft clamp (lambda {lam})", with_k(lambda kk: torch.where(kk > KMAX, KMAX + lam * (kk - KMAX), kk)))
ramp = np.linspace(0, 1, 20)[None, :, None, None]
for a in (0.1, 0.2, 0.3):
    report(f"hard clamp + ramp shrink {a}", (1 - a * ramp) * Fc + a * ramp * HM[:, None])
    report(f"true k + ramp shrink {a}", (1 - a * ramp) * F0 + a * ramp * HM[:, None])
json.dump(res, open("/home/team/person3/proxy/proxy_results.json", "w"), indent=1)
