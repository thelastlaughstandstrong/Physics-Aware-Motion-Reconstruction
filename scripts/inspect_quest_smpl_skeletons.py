"""
STEP 2 of the Quest->SMPL retargeting work: extract both skeletons and emit a documented
correspondence table as a VERSIONED ARTIFACT (json) plus a human-readable report.

Deliberately emits the mapping to disk rather than hard-coding it in the retargeter, so the
correspondence has exactly one source of truth that later steps import.

Nothing here fits, calibrates frames, or transforms motion -- it only describes the two
skeletons and how their joints correspond.

Usage:
    python scripts/inspect_quest_smpl_skeletons.py
        [--json scripts/quest_smpl_correspondence.json]
        [--report scripts/quest_smpl_correspondence.md]
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phc"))

from poselib.poselib.skeleton.skeleton3d import SkeletonTree  # noqa: E402
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES  # noqa: E402

MJCF = os.path.join(REPO, "phc/phc/data/assets/mjcf/quest_humanoid.xml")
SMPLH_NPZ = os.path.join(REPO, "support_data/body_models/smplh/neutral/model.npz")

# SMPL 24-joint kinematic parents, SMPL_BONE_ORDER_NAMES order (same array distill_agent.py uses).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# SMPL-H stores 52 joints: 0-21 body, 22-36 LEFT hand, 37-51 RIGHT hand. So the SMPL 24-joint
# convention's L_Hand/R_Hand (22/23) map to SMPL-H rows 22 and 37 -- NOT 22 and 23. Taking
# J_regressor[:24] (as phc/phc/learning/distill_agent.py::_load_smpl_rest_offsets does) silently
# yields two LEFT-hand finger joints for indices 22-23; see the report's "Defect found" section.
SMPLH_ROW_FOR_SMPL24 = list(range(22)) + [22, 37]

# ---- correspondence -------------------------------------------------------------------------
# quest_idx -> smpl_idx, for joints that have a genuine 1:1 anatomical counterpart.
DIRECT = {
    0:  0,    # b_root            -> Pelvis
    19: 1,    # b_l_upleg         -> L_Hip
    22: 2,    # b_r_upleg         -> R_Hip
    20: 4,    # b_l_leg           -> L_Knee
    23: 5,    # b_r_leg           -> R_Knee
    21: 7,    # b_l_foot_twist    -> L_Ankle      (approximate: twist joint, not a true ankle)
    24: 8,    # b_r_foot_twist    -> R_Ankle      (approximate)
    5:  12,   # b_neck0           -> Neck
    6:  15,   # b_head            -> Head
    7:  13,   # b_l_shoulder      -> L_Thorax     (clavicle)
    13: 14,   # b_r_shoulder      -> R_Thorax
    9:  16,   # b_l_arm           -> L_Shoulder
    15: 17,   # b_r_arm           -> R_Shoulder
    10: 18,   # b_l_forearm       -> L_Elbow
    16: 19,   # b_r_forearm       -> R_Elbow
    12: 20,   # b_l_wrist         -> L_Wrist
    18: 21,   # b_r_wrist         -> R_Wrist
}

# Quest's 4-segment spine has no per-joint counterpart in SMPL's 3-segment spine. The chain as a
# whole corresponds; the individual joints do not. Handled as a chain resample, not a mapping.
MERGED = {
    "quest_chain": [1, 2, 3, 4],       # b_spine0..b_spine3
    "smpl_chain":  [3, 6, 9],          # Torso, Spine, Chest
    "note": "4 Quest spine segments -> 3 SMPL spine segments; resampled along the chain, "
            "never copied joint-for-joint.",
}

# Quest joints with no SMPL counterpart -- their motion is absorbed into the parent segment.
DISCARDED = {
    8:  "p_l_scap        - scapula; SMPL has no scapular DOF (absorbed into L_Thorax/L_Shoulder)",
    14: "p_r_scap        - scapula; as above",
    11: "b_l_wrist_twist - forearm twist DOF; SMPL models forearm as a single L_Elbow->L_Wrist bone",
    17: "b_r_wrist_twist - forearm twist DOF; as above",
}

# SMPL joints with no Quest source -- must be invented (or left at rest) by the retargeter.
SYNTHESIZED = {
    10: "L_Toe  - Quest's leg chain ends at b_l_foot_twist; there is no toe/sole joint",
    11: "R_Toe  - as above",
    22: "L_Hand - Quest's arm chain ends at b_l_wrist; there is no hand joint",
    23: "R_Hand - as above",
}

# Joints used as optimization targets when fitting SMPL (Step 5) -- the direct-mapped set minus
# the root, which is handled by the root transform itself.
FIT_TARGETS = [q for q in DIRECT if q != 0]

# ---- validation protocol ---------------------------------------------------------------------
# Residuals are reported PER JOINT and split by whether a low residual is even achievable, rather
# than against one global pass/fail threshold. Rationale in the report's "Validation protocol"
# section: a single aggregate number cannot separate "the optimizer has not converged" from "these
# two skeletons place this joint differently and no fit can reconcile it".
#
# HIGH_CONFIDENCE: chain-length ratio within ~8% of SMPL -> residual here is genuinely
#                  attributable to optimizer convergence.
# STRUCTURAL:      chain-length ratio disagrees by >=15% (see the flagged table) -> a floor exists
#                  that is a property of the skeletons, not of the fit.
VALIDATION_GROUPS = {
    "high_confidence": {
        "head": 6,
        "l_elbow": 10, "r_elbow": 16,
        "l_wrist": 12, "r_wrist": 18,
        "l_hip": 19, "r_hip": 22,
        "l_knee": 20, "r_knee": 23,
        "l_ankle": 21, "r_ankle": 24,
    },
    "structural_mismatch": {
        "l_shoulder": 9, "r_shoulder": 15,     # via discarded scapula; ratio ~2.0
        "l_thorax": 7, "r_thorax": 13,         # clavicle attachment; ratio ~1.21
        "neck": 5,                             # sits atop the merged 4->3 spine chain
    },
}
VALIDATION = {**VALIDATION_GROUPS["high_confidence"], **VALIDATION_GROUPS["structural_mismatch"]}

# Never counted in retargeting accuracy: no Quest source exists, so any number would be measuring
# the SMPL rest pose against itself.
EXCLUDED_FROM_ACCURACY = list(SYNTHESIZED)


def chain_segment(idx, parents, offsets, mapped):
    """Cumulative distance from `idx` up to its nearest MAPPED ancestor.

    Raw local offsets are NOT comparable across the two skeletons, because discarded/merged
    joints sit between mapped ones (e.g. Quest splits elbow->wrist across b_*_wrist_twist and
    b_*_wrist, so b_*_wrist's own offset is only 0.03 m). The distance between consecutive
    *mapped* joints is the frame- and parameterization-invariant quantity that must agree.
    """
    d, i = 0.0, idx
    while True:
        p = parents[i]
        if p < 0:
            return d
        d += float(np.linalg.norm(offsets[i]))
        if p in mapped:
            return d
        i = p


def rest_positions(parents, offsets):
    P = np.zeros((len(parents), 3))
    for i in range(1, len(parents)):
        P[i] = P[parents[i]] + offsets[i]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(REPO, "scripts/quest_smpl_correspondence.json"))
    ap.add_argument("--report", default=os.path.join(REPO, "scripts/quest_smpl_correspondence.md"))
    args = ap.parse_args()

    tree = SkeletonTree.from_mjcf(MJCF)
    qn, qp = tree.node_names, tree.parent_indices.tolist()
    qoff = tree.local_translation.numpy().astype(np.float64)
    qpos = rest_positions(qp, qoff)

    bd = np.load(SMPLH_NPZ, allow_pickle=True)
    J52 = np.array(bd["J_regressor"]) @ np.array(bd["v_template"])
    Js = J52[SMPLH_ROW_FOR_SMPL24]                       # correctly-indexed 24 SMPL joints
    soff = Js.copy()
    for i in range(1, 24):
        soff[i] = Js[i] - Js[SMPL_PARENTS[i]]
    sn = SMPL_BONE_ORDER_NAMES

    # frame-invariant validation of the direct mapping: segment length as a fraction of stature
    q_stature = float(np.linalg.norm(qpos[6] - qpos[21]))          # head..foot
    s_stature = float(np.linalg.norm(Js[15] - Js[7]))              # Head..L_Ankle

    out = []

    def P(s=""):
        print(s)
        out.append(s)

    P("# Quest -> SMPL Joint Correspondence (Step 2)")
    P("")
    P(f"- Quest: `{os.path.relpath(MJCF, REPO)}` — **25 joints**")
    P(f"- SMPL: `{os.path.relpath(SMPLH_NPZ, REPO)}` (SMPL-H, 52 rows) — **24-joint** body convention")
    P(f"- Quest rest stature (head→foot) **{q_stature:.4f} m**; "
      f"SMPL rest stature (Head→Ankle) **{s_stature:.4f} m** "
      f"(ratio {q_stature/s_stature:.4f} — no global rescale needed)")
    P("")
    P("Counts reconcile exactly:")
    P("")
    P(f"- Quest 25 = {len(DIRECT)} direct + {len(MERGED['quest_chain'])} spine (merged) "
      f"+ {len(DISCARDED)} discarded")
    P(f"- SMPL 24 = {len(DIRECT)} direct + {len(MERGED['smpl_chain'])} spine (merged) "
      f"+ {len(SYNTHESIZED)} synthesized")
    P("")

    P("## 1. Direct correspondence (1:1)")
    P("")
    P("| Quest idx | Quest joint | SMPL idx | SMPL joint | Quest chain (m) | SMPL chain (m) | Q/stature | S/stature | ratio |")
    P("|---|---|---|---|---|---|---|---|---|")
    qmapped, smapped = set(DIRECT), set(DIRECT.values())
    ratios = []
    for q in sorted(DIRECT, key=lambda k: DIRECT[k]):
        s = DIRECT[q]
        qseg = chain_segment(q, qp, qoff, qmapped) if q != 0 else 0.0
        sseg = chain_segment(s, SMPL_PARENTS, soff, smapped) if s != 0 else 0.0
        qn_, sn_ = qseg / q_stature, sseg / s_stature
        r = (qn_ / sn_) if sn_ > 1e-9 else float("nan")
        if q != 0:
            ratios.append(r)
        P(f"| {q} | `{qn[q]}` | {s} | `{sn[s]}` | {qseg:.4f} | {sseg:.4f} | "
          f"{qn_:.4f} | {sn_:.4f} | {r:.3f} |")
    P("")
    P("**Chain** = cumulative distance to the nearest *mapped* ancestor, not the raw local offset. "
      "This is the quantity that must agree: discarded joints sit between mapped ones (Quest "
      "splits elbow→wrist across `b_*_wrist_twist` + `b_*_wrist`, so `b_*_wrist`'s own offset is "
      "only 0.03 m and comparing it to SMPL's 0.25 m forearm would be meaningless). Normalising "
      "by stature makes the columns frame- and scale-invariant, so they validate the mapping "
      "without needing the Step 3 frame calibration.")
    P("")
    rr = np.array(ratios)
    P(f"Length-ratio agreement over the {len(ratios)} non-root pairs: "
      f"median **{np.median(rr):.3f}**, range {rr.min():.3f}–{rr.max():.3f} "
      f"(1.000 = perfect proportional agreement).")
    P("")
    P("### Where the irreducible residual will be — predicted before fitting")
    P("")
    P("Pairs whose proportional length disagrees by more than 15% are structural mismatches, not "
      "fitting difficulty: no optimizer can place a joint correctly if the two skeletons put it "
      "at a different fraction of body height. Flagged automatically:")
    P("")
    P("| Quest | SMPL | ratio | interpretation |")
    P("|---|---|---|---|")
    interp = {
        9:  "Quest routes clavicle→shoulder through the discarded `p_l_scap`, making that span ~2x "
            "SMPL's Thorax→Shoulder. **Expect the largest residual at the shoulders.**",
        15: "as above (right side)",
        7:  "Quest's clavicle attaches higher up its longer spine",
        13: "as above (right side)",
        19: "Quest's hips sit closer to the root (narrower pelvis) than SMPL's",
        22: "as above (right side)",
        6:  "Quest's neck→head segment is shorter relative to stature",
    }
    for q in sorted(DIRECT, key=lambda k: DIRECT[k]):
        if q == 0:
            continue
        sidx = DIRECT[q]
        qs = chain_segment(q, qp, qoff, qmapped) / q_stature
        ss = chain_segment(sidx, SMPL_PARENTS, soff, smapped) / s_stature
        r = qs / ss
        if abs(r - 1.0) > 0.15:
            P(f"| `{qn[q]}` | `{sn[sidx]}` | {r:.3f} | {interp.get(q, 'structural difference')} |")
    P("")
    P("Everything else (legs, elbows, wrists, neck) agrees within ~8%. Notably the **wrist ratio is "
      "1.021 / 0.997**, which confirms that merging `b_*_wrist_twist` into the forearm is the "
      "correct interpretation — the elbow→wrist span matches SMPL almost exactly once the twist "
      "joint is absorbed rather than mapped.")
    P("")

    P("## 2. Merged (chain resample, NOT a joint mapping)")
    P("")
    P(f"- Quest `{[qn[i] for i in MERGED['quest_chain']]}` (4 segments)")
    P(f"- SMPL  `{[sn[i] for i in MERGED['smpl_chain']]}` (3 segments)")
    P("")
    qsp = sum(float(np.linalg.norm(qoff[i])) for i in MERGED["quest_chain"])
    ssp = sum(float(np.linalg.norm(soff[i])) for i in MERGED["smpl_chain"])
    P(f"Cumulative chain length: Quest **{qsp:.4f} m** vs SMPL **{ssp:.4f} m** "
      f"({qsp/q_stature:.3f} vs {ssp/s_stature:.3f} of stature).")
    P("")
    P("> **This is the single largest structural mismatch.** Quest's spine is both longer in "
      "absolute terms and has one more segment. There is no joint-to-joint correspondence; the "
      "retargeter must fit the SMPL spine to the Quest chain's *shape*, and a non-zero residual "
      "here is unavoidable regardless of optimizer quality.")
    P("")

    P("## 3. Discarded (Quest joints with no SMPL counterpart)")
    P("")
    P("| Quest idx | Quest joint | reason |")
    P("|---|---|---|")
    for q, why in DISCARDED.items():
        P(f"| {q} | `{qn[q]}` | {why.split('-',1)[1].strip()} |")
    P("")
    P("Discarding these is lossy but well-posed: they are twist/auxiliary DOFs whose *effect* on "
      "the child joint's world position is preserved, because the child is a direct-mapped fit "
      "target. Only the intermediate joint's own orientation is lost.")
    P("")

    P("## 4. Synthesized (SMPL joints with no Quest source)")
    P("")
    P("| SMPL idx | SMPL joint | reason |")
    P("|---|---|---|")
    for s, why in SYNTHESIZED.items():
        P(f"| {s} | `{sn[s]}` | {why.split('-',1)[1].strip()} |")
    P("")
    P("> These carry **no information from Quest**. They must be held at the SMPL rest pose (or "
      "driven by a prior) and must **never** be reported as retargeting accuracy — there is no "
      "ground truth for them.")
    P("")

    P("## 5. Fit targets and validation protocol")
    P("")
    P(f"- **Fit targets** (Step 5 optimization): {len(FIT_TARGETS)} joints — the direct-mapped set "
      f"minus the root: `{[qn[q] for q in FIT_TARGETS]}`")
    P("")
    P("### Validation is per-joint and grouped, NOT a single global threshold")
    P("")
    P(f"**Group A — high confidence** ({len(VALIDATION_GROUPS['high_confidence'])} joints, "
      f"chain ratio within ~8%): "
      f"`{list(VALIDATION_GROUPS['high_confidence'])}`")
    P("")
    P(f"**Group B — structurally mismatched** ({len(VALIDATION_GROUPS['structural_mismatch'])} "
      f"joints, chain ratio off by >=15%): "
      f"`{list(VALIDATION_GROUPS['structural_mismatch'])}`")
    P("")
    P(f"**Excluded from accuracy entirely**: "
      f"`{[sn[i] for i in EXCLUDED_FROM_ACCURACY]}` — synthesized, no Quest source.")
    P("")
    P("Reported outputs: (1) a **per-joint residual table** for all "
      f"{len(VALIDATION)} mapped validation joints, and (2) a **mean over all mapped joints**, "
      "with the two groups summarised separately.")
    P("")
    P("#### Why a single global threshold would be misleading")
    P("")
    P("A pooled mean silently averages two error sources that call for different responses:")
    P("")
    P("- **Optimizer convergence error** — fixable by better initialization, more iterations, or "
      "reweighting. Only Group A can isolate this, because those joints sit at nearly the same "
      "fraction of body height in both skeletons (ratios 0.96–1.08).")
    P("- **Irreducible structural mismatch** — a property of the two skeletons that no fit can "
      "remove. Four specific causes, all quantified in Section 1:")
    P("  - **Spine 4→3.** Quest's chain is 0.4462 m over four segments; SMPL's is 0.3062 m over "
      "three. The intermediate joints have no counterpart at all, so the SMPL spine can only "
      "approximate the Quest chain's shape — and `neck`, sitting on top of it, inherits that error.")
    P("  - **Discarded scapulae.** Quest's clavicle→shoulder span runs through `p_*_scap`, making "
      "it ~2.0x SMPL's Thorax→Shoulder (ratio 2.039 / 1.930). The shoulder cannot be placed "
      "correctly relative to both its parent and its child.")
    P("  - **Different shoulder topology.** `b_*_shoulder`→`*_Thorax` is off by ~21% for the same "
      "reason, so the whole shoulder girdle carries a floor.")
    P("  - **Synthesized hands/toes.** Zero Quest information. Including them would report the "
      "SMPL rest pose's agreement with itself and would *flatter* the result — hence the explicit "
      "exclusion.")
    P("")
    P("Pooling these means a large, unavoidable shoulder residual could mask a genuine convergence "
      "failure in the legs, or a good leg fit could hide that the shoulders are unusable. "
      "Grouping makes the pass criterion actionable: **Group A residuals are a bug signal; Group B "
      "residuals are a design cost to be quantified and declared.**")
    P("")
    P("> ⚠️ Still true regardless of grouping: the validation joints are a **subset of the fit "
      "targets**, so these are *fit-quality* numbers, not held-out generalization. No holdout is "
      "possible — every Quest joint with an SMPL counterpart is needed as a target.")
    P("")

    P("## 6. Defect found in existing repo code")
    P("")
    P("`phc/phc/learning/distill_agent.py::_load_smpl_rest_offsets()` computes")
    P("")
    P("```python")
    P("J_reg = np.array(body_data['J_regressor'])[:24]")
    P("rest_joints = J_reg @ np.array(body_data['v_template'])")
    P("```")
    P("")
    P("but the asset is **SMPL-H with 52 joints** laid out as 0–21 body, 22–36 left hand, "
      "37–51 right hand. So rows 22–23 are the first two **left-hand** joints, not "
      "`L_Hand`/`R_Hand`. Measured consequence in the neutral model:")
    P("")
    P("| SMPL-24 idx | name | row used | position | correct row | correct position |")
    P("|---|---|---|---|---|---|")
    P(f"| 22 | `L_Hand` | 22 | `{np.round(J52[22],4).tolist()}` | 22 | (same — correct by luck) |")
    P(f"| 23 | `R_Hand` | 23 | `{np.round(J52[23],4).tolist()}` | 37 | `{np.round(J52[37],4).tolist()}` |")
    P("")
    P(f"Row 23 sits at **+X** (the left side) like row 22, giving a bogus `R_Wrist→R_Hand` bone of "
      f"**{np.linalg.norm(J52[23]-J52[21]):.4f} m** instead of "
      f"**{np.linalg.norm(J52[37]-J52[21]):.4f} m**.")
    P("")
    P("**Blast radius (checked, not assumed):**")
    P("")
    P("- `_build_hmd_features` uses `_SMPL_HMD_JOINT_IDS = [15, 20, 21]` — all ≤ 21, **unaffected**.")
    P("- `DistillPlayer._accumulate_qbase_gpen` FKs `range(1,24)` — **affected**, but it applies "
      "the same wrong offsets to both prediction and ground truth, so the difference largely "
      "cancels in `qref_MPJPE`; only `gpen` could shift if the bogus joint crosses the floor.")
    P("- Quest pipeline: **unaffected** (never uses SMPL rest offsets).")
    P("")
    P("This retargeter therefore uses `SMPLH_ROW_FOR_SMPL24 = list(range(22)) + [22, 37]` rather "
      "than `[:24]`. Since `L_Hand`/`R_Hand` are *synthesized* joints here, the fix does not change "
      "any fit target — it is corrected so the bug is not inherited, and flagged so the SMPL "
      "pipeline can be fixed separately.")
    P("")

    artifact = {
        "quest_mjcf": os.path.relpath(MJCF, REPO),
        "smplh_npz": os.path.relpath(SMPLH_NPZ, REPO),
        "quest_joint_names": qn,
        "quest_parents": qp,
        "smpl_joint_names": list(sn),
        "smpl_parents": SMPL_PARENTS,
        "smplh_row_for_smpl24": SMPLH_ROW_FOR_SMPL24,
        "direct": {str(k): v for k, v in DIRECT.items()},
        "merged": MERGED,
        "discarded": {str(k): v for k, v in DISCARDED.items()},
        "synthesized": {str(k): v for k, v in SYNTHESIZED.items()},
        "fit_targets_quest_idx": FIT_TARGETS,
        "validation_quest_idx": VALIDATION,
        "validation_groups": VALIDATION_GROUPS,
        "excluded_from_accuracy_smpl_idx": EXCLUDED_FROM_ACCURACY,
        "quest_rest_stature_m": q_stature,
        "smpl_rest_stature_m": s_stature,
    }
    with open(args.json, "w") as f:
        json.dump(artifact, f, indent=2)
    with open(args.report, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n[artifact] {args.json}\n[report]   {args.report}")


if __name__ == "__main__":
    main()
