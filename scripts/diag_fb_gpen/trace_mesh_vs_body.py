"""
Targeted diagnostic: per-frame relationship between the SIMULATED rigid-body foot height and the
RECONSTRUCTED SMPL-H mesh sole, on selected lower-body clips.

Answers steps 3-6 of the diagnosis: is the Finetuned Backbone's extra mesh GPen explained by a
lower root/foot (B), a different foot pose (C), a constant mesh-vs-collision offset exposed more
often (D), or contact/stance behaviour (E)?

Reuses the validated batched IK and the unchanged GPen definition. Writes only into
scripts/diag_fb_gpen/ -- it does not touch the full-dataset results.
"""
import json, os, sys
import numpy as np, torch, joblib

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ["scripts", REPO, "phc"]:
    sys.path.insert(0, os.path.join(REPO, p))
# Capture the stream BEFORE clearing argv: mesh_gpen_dataset parses argv at import, but clearing
# it first silently made every invocation fall back to the default stream (all three runs of the
# original loop processed finetuned_backbone).
_STREAM = sys.argv[1] if len(sys.argv) > 1 else "finetuned_backbone"
sys.argv = ["x"]
import mesh_gpen_dataset as MG
import retarget_quest_to_smpl as M
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from human_body_prior.body_model.body_model import BodyModel
from mesh_gpen_quest import BM_NEUTRAL, DMPL_NEUTRAL

FOOT_Q = [21, 24]          # quest FOOT_JOINTS, the repo's own definition
CONTACT_THRESH = 0.0871    # motion_lib_quest's own foot-height constant -- not a new threshold
OUT = os.path.join(REPO, "scripts/diag_fb_gpen")


def main():
    stream = _STREAM
    sel = json.load(open(f"{OUT}/selected_clips.json"))
    clips = sel["worst"][:12] + sel["control"][:5]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corr = json.load(open(os.path.join(REPO, "scripts/quest_smpl_correspondence.json")))
    soff = M.load_smpl_rest_offsets(corr)
    tree = SkeletonTree.from_mjcf(M.MJCF)
    bm = BodyModel(bm_fname=BM_NEUTRAL, num_betas=16, num_dmpls=8,
                   dmpl_fname=DMPL_NEUTRAL).to(dev)

    pos = MG.load_stream(stream, clips, tree)
    clips = [c for c in clips if c in pos]
    print(f"{stream}: {len(clips)} clips", flush=True)
    fits = MG.batched_ik([pos[c] for c in clips], soff, corr, dev, log=True)

    res = {}
    for c, p in zip(clips, fits):
        T = len(p["root"])
        root = torch.tensor(p["root"], dtype=torch.float32)
        body = torch.tensor(p["body"], dtype=torch.float32)
        tr = torch.tensor(p["trans"], dtype=torch.float32)
        pb = body[:, :21].reshape(T, 63)
        sole, pen = [], []
        with torch.no_grad():
            for s in range(0, T, 256):
                e = min(s + 256, T)
                v = bm(pose_body=pb[s:e].to(dev), root_orient=root[s:e].to(dev),
                       trans=tr[s:e].to(dev)).v
                sole.append(v[..., 2].min(-1).values.cpu().numpy())
                pen.append(torch.clamp(-v[..., 2], min=0).amax(-1).cpu().numpy())
        sole = np.concatenate(sole); pen = np.concatenate(pen)
        P = pos[c][:T]
        body_foot = P[:, FOOT_Q, 2].min(1)          # simulated rigid-body foot joint height
        pelvis = P[:, 0, 2]
        contact = body_foot < CONTACT_THRESH
        res[c] = dict(
            frames=int(T),
            pelvis_med=float(np.median(pelvis)),
            body_foot_med=float(np.median(body_foot)),
            sole_med=float(np.median(sole)),
            offset_med=float(np.median(body_foot - sole)),   # joint-to-sole vertical gap
            offset_p95=float(np.percentile(body_foot - sole, 95)),
            pen_rate=float((pen > 1e-4).mean()),
            pen_mean_pen_frames=float(pen[pen > 1e-4].mean() * 1000) if (pen > 1e-4).any() else 0.0,
            frac_contact=float(contact.mean()),
            pen_rate_contact=float((pen[contact] > 1e-4).mean()) if contact.any() else 0.0,
            pen_rate_flight=float((pen[~contact] > 1e-4).mean()) if (~contact).any() else 0.0,
            frac_pen_in_contact=float((contact & (pen > 1e-4)).sum() / max((pen > 1e-4).sum(), 1)),
        )
        print(f"  {c[:36]:36s} pelvis {res[c]['pelvis_med']:.3f} bodyfoot {res[c]['body_foot_med']:.3f} "
              f"sole {res[c]['sole_med']:+.3f} offset {res[c]['offset_med']*1000:6.1f}mm "
              f"penrate {res[c]['pen_rate']:.3f} (contact {res[c]['pen_rate_contact']:.3f} / "
              f"flight {res[c]['pen_rate_flight']:.3f})", flush=True)
    json.dump(res, open(f"{OUT}/trace_{stream}.json", "w"), indent=1)
    print(f"wrote {OUT}/trace_{stream}.json")


if __name__ == "__main__":
    main()
