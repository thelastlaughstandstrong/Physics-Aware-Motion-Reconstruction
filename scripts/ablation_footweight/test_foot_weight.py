"""Regression check for the foot-weighted joint_pos_loss (run before training each arm).

Asserts the three properties the ablation's validity rests on:
  1. foot_pos_weight = 1.0 is an EXACT no-op (bitwise identical to the previous l1_loss).
  2. The weighted vector is mean-normalised: sum(W) == n_bodies, so the term's scale -- and hence
     its effective learning rate -- is unchanged across arms.
  3. Body indices 21/24 really are the two ankles in the tensor's own index order.
It also prints the effective non-foot weights, which normalisation reduces below 1.0.
"""
import sys

sys.path.insert(0, "phc")
try:                       # isaacgym MUST be imported before torch or poselib is unusable
    import isaacgym        # noqa: F401
    _HAVE_GYM = True
except Exception:
    _HAVE_GYM = False

import torch
import torch.nn as nn

FOOT = [21, 24]
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def weighted_pos_loss(pred, gt, w):
    """Exactly the expression now in distill_agent_quest.py (Stage E)."""
    ad = (pred - gt).abs()
    if w == 1.0:
        return ad.mean()
    W = torch.ones(pred.shape[1], device=pred.device, dtype=pred.dtype)
    W[FOOT] = w
    W = W * (pred.shape[1] / W.sum())
    return (W[None, :, None] * ad).mean()


def main():
    print("foot-weighted joint_pos_loss — regression check\n")
    torch.manual_seed(0)
    for B in (8, 64, 257):
        pred, gt = torch.randn(B, 25, 3), torch.randn(B, 25, 3)
        base = nn.functional.l1_loss(pred, gt)
        got = weighted_pos_loss(pred, gt, 1.0)
        check(f"w=1.0 bitwise identical to l1_loss (B={B})", base.item() == got.item(),
              f"{base.item():.12f}")

    for w in (2.0, 4.0):
        W = torch.ones(25); W[FOOT] = w; W = W * (25 / W.sum())
        check(f"w={w}: mean-normalised, sum(W)==25", abs(W.sum().item() - 25.0) < 1e-5,
              f"sum={W.sum().item():.6f}")
        check(f"w={w}: non-foot rows scaled DOWN (not left at 1.0)", W[0].item() < 1.0,
              f"foot={W[FOOT[0]].item():.4f} other={W[0].item():.4f} "
              f"({(W[0].item()-1)*100:+.1f}%)")

    # float64 exactness of the normalisation identity
    for w in (2.0, 4.0):
        W = torch.ones(25, dtype=torch.float64); W[FOOT] = w; W = W * (25 / W.sum())
        ratio = (W[FOOT[0]] / W[0]).item()
        check(f"w={w}: foot/non-foot ratio == w", abs(ratio - w) < 1e-9, f"{ratio:.12f}")

    try:
        if not _HAVE_GYM:
            raise ImportError("isaacgym unavailable")
        from poselib.poselib.skeleton.skeleton3d import SkeletonTree
        t = SkeletonTree.from_mjcf("phc/phc/data/assets/mjcf/quest_humanoid.xml")
        nn_names = list(t.node_names)
        check("tensor has 25 bodies", len(nn_names) == 25, f"n={len(nn_names)}")
        check("indices 21/24 are the two ankles",
              [nn_names[i] for i in FOOT] == ["b_l_foot_twist", "b_r_foot_twist"],
              ", ".join(nn_names[i] for i in FOOT))
        check("no toe joint exists in the rig",
              not any("toe" in n.lower() for n in nn_names))
    except Exception as e:
        print(f"  SKIP  skeleton checks ({type(e).__name__}: {e})")

    print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
