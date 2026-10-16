#!/usr/bin/env python3
"""Offline correctness checks for frechet_selectors (CPU, no model).

Check 1: selector monotonicity  n(delta=0) >= n(0.25) >= n(0.5)  (stricter -> fewer).
Check 2: matched-factor mapping on FLAT confidence  frechet(delta) == factor(1-delta).
Check 3: default selector is an exact no-op vs v2's native threshold rule.
Check 4: -inf (non-candidate) positions are never committed.
Exit non-zero if any check fails.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import frechet_selectors as fs

torch.manual_seed(0)
NEG = float("-inf")
fails = []


def n_of(x1_p, mask, **kw):
    u, _ = fs.select(x1_p, mask, **kw)
    return u.sum(dim=1)


# ---- Check 1: monotonicity in delta (frechet) and in f (factor) ----
B, W = 64, 16
conf = torch.rand(B, W) * 0.5 + 0.5          # in [0.5, 1.0]
mask = torch.ones(B, W, dtype=torch.bool)
x1p = torch.where(mask, conf, torch.tensor(NEG))
n0 = n_of(x1p, mask, selector="frechet", frechet_delta=0.0)
n25 = n_of(x1p, mask, selector="frechet", frechet_delta=0.25)
n50 = n_of(x1p, mask, selector="frechet", frechet_delta=0.50)
if not (torch.all(n0 >= n25) and torch.all(n25 >= n50)):
    fails.append("Check1 frechet monotonicity violated")
# factor: larger f => commit more (looser); f=1.0 >= f=0.75 >= f=0.5
fn10 = n_of(x1p, mask, selector="factor", factor=1.0)
fn75 = n_of(x1p, mask, selector="factor", factor=0.75)
fn50 = n_of(x1p, mask, selector="factor", factor=0.50)
if not (torch.all(fn10 >= fn75) and torch.all(fn75 >= fn50)):
    fails.append("Check1 factor monotonicity violated")
print(f"Check1 monotonicity: frechet n(0/.25/.5) mean = "
      f"{n0.float().mean():.2f}/{n25.float().mean():.2f}/{n50.float().mean():.2f}  OK={('Check1 frechet monotonicity violated' not in fails)}")

# ---- Check 2: matched-factor mapping on FLAT confidence ----
for c in (0.80, 0.90, 0.95, 0.99):
    flat = torch.full((8, W), c)
    m = torch.ones(8, W, dtype=torch.bool)
    xp = torch.where(m, flat, torch.tensor(NEG))
    for d, f in [(0.00, 1.00), (0.25, 0.75), (0.50, 0.50), (0.10, 0.90)]:
        nf = n_of(xp, m, selector="frechet", frechet_delta=d)
        ng = n_of(xp, m, selector="matched_factor", frechet_delta=d)   # internal f=1-d
        nfac = n_of(xp, m, selector="factor", factor=f)
        if not (torch.equal(nf, ng) and torch.equal(nf, nfac)):
            fails.append(f"Check2 mapping fail c={c} d={d}: frechet={nf[0].item()} matched={ng[0].item()} factor={nfac[0].item()}")
print(f"Check2 matched-factor mapping (flat conf): OK={not any('Check2' in x for x in fails)}")

# ---- Check 3: default == v2 native threshold rule ----
thr = 0.9
conf2 = torch.rand(B, W)
# introduce some already-committed (non-candidate) positions
mask2 = torch.rand(B, W) > 0.3
x1p2 = torch.where(mask2, conf2, torch.tensor(NEG))
u_def, _ = fs.select(x1p2, mask2, selector="default", threshold=thr)
# native rule reproduction
native = (x1p2 > thr)
mx = x1p2.argmax(dim=-1)
native[torch.arange(B), mx] = True
native = native & mask2
if not torch.equal(u_def, native):
    fails.append("Check3 default no-op mismatch vs native threshold rule")
print(f"Check3 default no-op == native threshold: OK={'Check3 default no-op mismatch vs native threshold rule' not in fails}")

# ---- Check 4: -inf positions never committed; >=1 per row ----
for sel, kw in [("default", dict(threshold=0.9)), ("factor", dict(factor=0.75)),
                ("matched_factor", dict(frechet_delta=0.25)), ("frechet", dict(frechet_delta=0.25))]:
    u, _ = fs.select(x1p2, mask2, selector=sel, **kw)
    if torch.any(u & ~mask2):
        fails.append(f"Check4 {sel}: committed a non-candidate position")
    if torch.any(u.sum(dim=1) < 1):
        fails.append(f"Check4 {sel}: a row committed 0 tokens")
print(f"Check4 no non-candidate commits & >=1/row: OK={not any('Check4' in x for x in fails)}")

print("=" * 40)
if fails:
    print("FAILED CHECKS:")
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("ALL SELECTOR CHECKS PASSED")
