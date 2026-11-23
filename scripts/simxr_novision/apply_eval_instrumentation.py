#!/usr/bin/env python3
"""Apply this project's evaluation instrumentation to an official SimXR checkout.

The SimXR no-vision baseline in results_comparison.md was evaluated with two small modifications
to SimXR (https://github.com/ZhengyiLuo/SimXR) at revision
39081b831aa1c1e52a4a974fb78a1ed93f7580c2:

  phc/env/tasks/humanoid_im.py       expose simulated and reference body rotations in `extras`
  phc/learning/im_amp_players.py     collect them, remove two interactive debugger breakpoints,
                                     write phc_positions_for_eval.pkl, and exit after one pass

SimXR declares no license, so this repository does not include SimXR source code. The
instrumentation is stored in simxr_eval_instrumentation.json as an edit script: line-range
references into the official SimXR file plus this project's own added lines. No SimXR text is
stored. Applying it to an unmodified checkout reproduces the exact files used in the experiments,
which is verified by SHA-256 before and after.

Usage:
    git clone https://github.com/ZhengyiLuo/SimXR.git /path/to/SimXR
    git -C /path/to/SimXR checkout 39081b831aa1c1e52a4a974fb78a1ed93f7580c2
    python scripts/simxr_novision/apply_eval_instrumentation.py --simxr /path/to/SimXR --check
    python scripts/simxr_novision/apply_eval_instrumentation.py --simxr /path/to/SimXR

The script refuses to write unless the SimXR file is the expected unmodified version. Re-running
it on an already instrumented checkout reports "already applied" and changes nothing. Revert with
`git -C /path/to/SimXR checkout -- phc/`.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "simxr_eval_instrumentation.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _shift(line: str, delta: int) -> str:
    return line[delta:] if delta >= 0 else " " * (-delta) + line


def apply_segments(base: bytes, segments) -> bytes:
    a = base.decode("utf-8").splitlines(keepends=True)
    out = []
    for seg in segments:
        if seg and isinstance(seg[0], str):
            out.extend(seg)  # this project's own lines
        else:
            start, end, delta = seg
            if not (0 <= start <= end <= len(a)):
                raise ValueError("edit script does not match the SimXR file")
            out.extend(_shift(line, delta) for line in a[start:end])
    return "".join(out).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simxr", required=True, help="path to an official SimXR checkout")
    ap.add_argument("--check", action="store_true", help="report status only; do not modify files")
    args = ap.parse_args()
    spec = json.load(open(SPEC))
    plan, ok = [], True
    for f in spec["files"]:
        path = os.path.join(args.simxr, f["path"])
        if not os.path.exists(path):
            print(f"MISSING        {f['path']}")
            ok = False
            continue
        current = open(path, "rb").read()
        if sha256(current) == f["target_sha256"]:
            print(f"already applied {f['path']}")
            continue
        if sha256(current) != f["simxr_sha256"]:
            print(f"UNEXPECTED     {f['path']}: not the unmodified file at {spec['simxr_revision']}")
            ok = False
            continue
        result = apply_segments(current, f["segments"])
        if sha256(result) != f["target_sha256"]:
            print(f"FAILED         {f['path']}: reconstruction hash mismatch")
            ok = False
            continue
        plan.append((path, result, f["path"]))
        print(f"{'would apply' if args.check else 'ready'}    {f['path']}")
    if not ok:
        print("\nNothing was written. Check out the pinned SimXR revision and retry.")
        return 1
    if not args.check:
        for path, result, rel in plan:
            with open(path, "wb") as out:
                out.write(result)
            print(f"applied        {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
