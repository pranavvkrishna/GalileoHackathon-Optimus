import sys, torch
sys.path.insert(0, sys.argv[1])          # path to variants/B/pitching_airfoil_wake/baseline
import run_baseline as rb
torch.manual_seed(0)
x = torch.randn(3, 14, 48, 96); y = torch.randn(3, 20, 48, 96)
assert torch.equal(rb.mirror_x(rb.mirror_x(x)), x)
assert torch.equal(rb.mirror_field(rb.mirror_field(y)), y)
mx = rb.mirror_x(x)
assert torch.equal(mx[:, 10:12], x[:, 10:12])                       # k, theta0 unchanged
assert torch.equal(mx[:, 12:14], -x[:, 12:14])                      # phase + pi
assert torch.equal(mx[:, :10, 0], -x[:, :10, 47])                   # row 0 <-> row 47, sign flipped
m = rb.WakeMirrorAvg().eval()
with torch.no_grad():
    out = m(x); assert out.shape == (3, 96822)
    F  = out[:, :20*48*96].reshape(3, 20, 48, 96)
    Fg = m(rb.mirror_x(x))[:, :20*48*96].reshape(3, 20, 48, 96)
    err = (Fg - rb.mirror_field(F)).abs().max().item()
print("equivariance max abs err", err); assert err < 1e-4
sd = rb.WakeBaseline().state_dict(); m.load_state_dict(sd)          # same state_dict
print("ALL MIRROR CHECKS PASSED")
