"""Videos of a --terrain policy on its clips, in the look of tools/sim2sim_video.py: the robot (opaque), the reference
(translucent ghost) at their global poses and the clip's inferred boxes (terrain.json). Trajectories come from
``partymimic.evaluate --record`` on the training simulator:
    PYTHONPATH=src uv run python -m partymimic.evaluate --robot g1mj --run_dir <run> --ckpt <ckpt> \
        --motion_dir <dir with the clips and their terrain.json> --record traj.npz
    MUJOCO_GL=egl PYTHONPATH=src uv run python tools/terrain_video.py traj.npz <dir>/terrain.json g1mj <out dir> \
        --clips chair:Chair:<clip> box:"50 cm box":<clip> ladder:Ladder:<clip>:90 ... [--seconds 12] [--xml <MJCF of the page videos>]
Writes <out dir>/<robot>_<key>.mp4 for every clip that played to its end.
"""

import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim_video import FONT, FPS, GHOST, H, W, label, scene_model  # noqa: E402

NB = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("terrain")
    ap.add_argument("robot", help="robot package, e.g. g1mj, rp1")
    ap.add_argument("out_dir")
    ap.add_argument("--clips", nargs="+", required=True, metavar="KEY:TITLE:CLIP[:AZIMUTH]",
                    help="AZIMUTH: camera azimuth relative to the clip's initial heading (default 120)")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--name", default=None, help="robot name in the title (default: the package name)")
    ap.add_argument("--xml", default=None, help="MJCF to draw (same joint order; default: the robot package's)")
    a = ap.parse_args()
    if a.xml is None:
        from partymimic.robot import get_robot

        a.xml = get_robot(a.robot).mjcf_path
    name = (a.name or a.robot).upper()
    os.makedirs(a.out_dir, exist_ok=True)
    m = scene_model(a.xml, boxes=NB)
    gid = [m.geom(f"terrain{i}").id for i in range(NB)]
    mid = [int(m.body_mocapid[m.body(f"terrain{i}").id]) for i in range(NB)]
    nq = m.nq
    d = dict(np.load(a.traj))
    names = [str(n) for n in d["names"]]
    terr = json.load(open(a.terrain))
    font = ImageFont.truetype(FONT, 20)
    ren = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    for spec in a.clips:
        key, title, clip, *az = spec.split(":")
        c = names.index(clip)
        alive = d["alive"][:, c]
        L = int(d["clip_len"][c]) - 1
        if int(alive.sum()) < L:
            print(f"skip {clip}: terminated at {int(alive.sum())} of {L} steps", flush=True)
            continue
        boxes = np.array(terr.get(clip, []), np.float64).reshape(-1, 7)[:NB]
        d_rob, d_ref = mujoco.MjData(m), mujoco.MjData(m)
        for dd in (d_rob, d_ref):
            mujoco.mj_forward(m, dd)  # lights stay where they are at the initial pose, as in sim2sim_video
            dd.mocap_pos[:] = (0, 0, -10)
        for i, b in enumerate(boxes):
            m.geom_size[gid[i]] = b[3:6]
            d_rob.mocap_pos[mid[i]] = b[:3]
            d_rob.mocap_quat[mid[i]] = (np.cos(b[6] / 2), 0, 0, np.sin(b[6] / 2))
        ztop = float((boxes[:, 2] + boxes[:, 5]).max()) if len(boxes) else 0.0
        n = min(L, int(a.seconds * FPS))
        s0 = int(d["clip_start"][c])
        R0 = d["ref_root_R"][s0]
        yaw0 = np.degrees(np.arctan2(R0[1, 0], R0[0, 0]))
        cam.azimuth, cam.elevation, cam.distance = yaw0 + float(az[0] if az else 120), -12, 3.4 if ztop < 1 else 4.2
        look = np.array(d["root_p"][0, c], np.float64)
        out = os.path.join(a.out_dir, f"{a.robot}_{key}.mp4")
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                               "-preset", "slow", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
        gaps = []
        for t in range(n):
            f = int(d["f"][t, c])
            d_rob.qpos[:3], d_rob.qpos[3:7], d_rob.qpos[7:nq] = d["root_p"][t, c], d["root_quat"][t, c], d["q"][t, c]
            d_ref.qpos[:3], d_ref.qpos[7:nq] = d["ref_root_p"][f], d["ref_q"][f]
            mujoco.mju_mat2Quat(d_ref.qpos[3:7], np.ascontiguousarray(d["ref_root_R"][f], np.float64).reshape(-1))
            mujoco.mj_kinematics(m, d_rob), mujoco.mj_kinematics(m, d_ref)
            body = m.body_mocapid < 0
            body[0] = False
            gap = 1000 * np.linalg.norm(d_rob.xpos[body] - d_ref.xpos[body], axis=-1).mean()
            gaps.append(gap)
            root = 0.5 * (d_rob.qpos[:3] + d_ref.qpos[:3])
            look = 0.95 * look + 0.05 * root
            cam.lookat[:] = [look[0], look[1], max(0.55, look[2] - 0.2)]
            ren.update_scene(d_rob, camera=cam, scene_option=opt)
            scn = ren.scene
            n0 = scn.ngeom
            mujoco.mjv_addGeoms(m, d_ref, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
            for i in range(n0, scn.ngeom):
                scn.geoms[i].rgba[:] = GHOST
            img = label(ren.render(), [(f"PartyMimic on {name}  |  Terrain: {title} (training clip, --terrain)", (20, 20, 20)),
                                       (clip, (60, 60, 60)),
                                       ("robot (solid)   reference (blue ghost, global pose)   boxes inferred from the clip",
                                        (40, 90, 170)),
                                       (f"t = {t / FPS:5.2f} s   global body position gap {gap:4.0f} mm", (60, 60, 60))], font)
            ff.stdin.write(img.tobytes())
        ff.stdin.close()
        ff.wait()
        print({"key": key, "clip": clip, "boxes": len(boxes), "seconds": round(n / FPS, 1),
               "gap_mm": round(float(np.mean(gaps)), 1)}, flush=True)
    ren.close()


if __name__ == "__main__":
    main()
