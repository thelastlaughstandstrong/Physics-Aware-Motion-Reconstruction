"""
Physical plausibility metrics for AvatarPoser evaluation.
"""

import torch


def compute_ground_penetration(vertices, ground_z=0.0):
    """
    Depth of SMPL mesh vertices below the floor plane.

    Args:
        vertices  : [T, V, 3]  SMPL mesh vertices in world coords (metres, Z-up).
                    Obtain from body_model_output.v  (V=6890 for SMPL-H).
        ground_z  : Height of the floor plane (default 0.0 m).
                    AMASS places the floor at Z ≈ 0; the lowest GT vertex is
                    typically +0.01–0.02 m above zero, so 0.0 is a safe threshold.

    Returns:
        mean_depth_mm : mean depth over all (vertex, frame) pairs where depth > 0.
                        Returns 0 if no penetration occurs.
        max_depth_mm  : worst-case penetration depth across the sequence.
        pen_rate      : fraction of frames in which at least one vertex is below ground.

    Assumptions:
        - Flat horizontal floor only; does not handle sloped terrain.
        - Sequences are assumed pre-normalised so Z=0 is the floor (AMASS standard).
          Verify with the [DEBUG seq 0] block in the test script: GT pen rate should
          be near 0. If it is not, the sequence has incorrect global-translation
          calibration and the metric will be unreliable.
    """
    pen = torch.clamp(ground_z - vertices[..., 2], min=0.0)   # [T, V], depth below floor
    frame_max = pen.amax(dim=-1)                                # [T]

    pen_rate      = (frame_max > 1e-4).float().mean()
    max_depth_mm  = frame_max.max() * 1000.0

    penetrating   = pen[pen > 1e-4]
    mean_depth_mm = (penetrating.mean() * 1000.0
                     if penetrating.numel() > 0
                     else torch.tensor(0.0, device=vertices.device))

    return mean_depth_mm, max_depth_mm, pen_rate
