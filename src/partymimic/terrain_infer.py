"""Static box terrain inferred from the reference motions, with a physical self-check (the --terrain pipeline).

From each clip's kinematics (10 Hz, the robot's MJCF with a floor plane at z 0; collision distances up to 1 m):
  step:   a still foot (< 0.15 m/s) that is the body's lowest point, more than 4 cm up for >= 0.3 s (boxes, stairs), and
          frames with the whole body off the floor without vertical motion for >= 0.4 s;
  seat:   >= 1 s with the pelvis still and low (< 0.89 standing height) and the center of mass > 6 cm outside the
          floor support: the pelvis' lowest point is the seat height;
  lean:   the same with the pelvis high: still geoms off the floor on the side of the offset carry it; the back or
          shoulders on it, or hands at shoulder height or above: a wall (thin vertical box); hands lower: a table top;
  ladder: >= 1 s with a foot and a hand still and raised and the body off the floor: rungs at the heights the feet
          stand and the hands rest on, plus two rails;
  stairs: more than two step heights: solid steps along a common axis, as wide as all footprints.
Every support height needs points from >= 0.3 s of the clip (brief floats of retargeted motions add no box); supports
lower than MIN_TOP are left out (the check runs on the boxes that are trained).
Boxes are shrunk (walls pushed back) away from body geoms inside them. A clip passes the self-check when no box holds
a body geom (ladders: the hands hold the rungs; other limbs for at most 0.3 s) and every stretch of more than 0.8 s
with the lowest point off the floor rests on a box top. Clips that fail (also those with no inferred box and an
unsupported stretch: hanging, climbing something not modeled) are moved out of the library, so they are not trained.

``build(motion_dir, robot)`` writes ``<motion_dir>/terrain.json`` (clip -> boxes, partymimic.terrain's layout),
``<motion_dir>/terrain_report.json`` (every clip with an inferred support or a failed check: kinds, ok, boxes) and
moves the failed clips to ``<motion_dir>_dropped/``. train.py --terrain runs it when terrain.json is missing.

    python -m partymimic.terrain_infer --robot g1mj --motion_dir <npz clips> [--workers 24] [--dry_run]
"""

import argparse
import glob
import json
import os
import shutil
import time
from multiprocessing import Pool

import mujoco
import numpy as np

from . import terrain
from .motion import load_npz
from .robot import get_robot

HZ = 10
MIN_TOP = 0.12  # lower supports are left out (floor)
_MODEL = None
_LOAD = None


def _files(motion_dir: str, robot_name: str):
    """The library's clips and loader, as MotionLib.from_dir lists them."""
    spec = get_robot(robot_name)
    files = sorted(glob.glob(os.path.join(motion_dir, spec.motion_glob)))
    if not files:
        return sorted(glob.glob(os.path.join(motion_dir, "*.npz"))), load_npz
    return files, spec.load_motion


def _init(robot_name: str, motion_dir: str):
    """Per worker: the scan model (robot plus a floor with a 1 m contact margin) and the body roles of the robot."""
    global _MODEL, _LOAD
    spec = get_robot(robot_name)
    ms = mujoco.MjSpec.from_file(spec.mjcf_path)
    ms.worldbody.add_geom(name="scan_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[100, 100, 0.1], margin=1.0)
    M = ms.compile()
    FLOOR = M.geom("scan_floor").id
    BOT = [g for g in range(M.ngeom) if g != FLOOR and (M.geom_contype[g] or M.geom_conaffinity[g])]
    BODY = {g: M.body(int(M.geom_bodyid[g])).name for g in BOT}
    root, torso = spec.tracked_bodies[0], spec.tracked_bodies[7]
    FOOT = [g for g in BOT if "ankle_roll" in BODY[g]]
    HAND = [g for g in BOT if "wrist" in BODY[g]]
    PELVIS = [g for g in BOT if BODY[g] == root]
    BACK = [g for g in BOT if BODY[g] in (root, torso) or "shoulder" in BODY[g]]
    assert FOOT and HAND and PELVIS, (robot_name, sorted(set(BODY.values())))
    RAD = {g: float(M.geom_size[g, 0]) if M.geom_type[g] in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                                mujoco.mjtGeom.mjGEOM_CYLINDER) else float(M.geom_rbound[g]) * 0.5
           for g in BOT}
    d = mujoco.MjData(M)  # standing pelvis height: the default pose with the lowest geom on the floor
    d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    d.qpos[7:] = spec.default_joint_pos_list()
    mujoco.mj_kinematics(M, d)
    H = float(-min(d.geom_xpos[g, 2] - RAD[g] for g in BOT))
    _MODEL = (M, FLOOR, BOT, BODY, FOOT, HAND, PELVIS, BACK, RAD, H)
    _LOAD = _files(motion_dir, robot_name)[1]


def _frames(path: str):
    rp, rq, q, fps = _LOAD(path)
    step = max(1, int(round(float(fps) / HZ)))
    return rp[::step], rq[::step], q[::step]


def runs(mask, n):
    out, s = [], None
    for i, x in enumerate(list(mask) + [False]):
        if x and s is None:
            s = i
        if not x and s is not None:
            if i - s >= n:
                out.append((s, i))
            s = None
    return out


def fit_box(p, margin, min_half):
    """p: [n, 2] support points -> cx, cy, hx, hy, yaw (box around them)."""
    c = p.mean(0)
    ax = np.array([1.0, 0.0])
    if len(p) > 2 and np.ptp(p, 0).max() > 0.02:
        ax = np.linalg.eigh(np.cov((p - c).T))[1][:, -1]
    yaw = float(np.arctan2(ax[1], ax[0]))
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    loc = (p - c) @ R
    lo, hi = loc.min(0) - margin, loc.max(0) + margin
    half = np.maximum((hi - lo) / 2, min_half)
    cw = c + R @ ((lo + hi) / 2)
    return float(cw[0]), float(cw[1]), float(half[0]), float(half[1]), yaw


def inside(box, xyz, rad, tol=0.02):
    """Body points (geom centers with radii) whose sphere reaches more than tol into the box."""
    cx, cy, cz, hx, hy, hz, yaw = box
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    loc = (xyz[:, :2] - [cx, cy]) @ R.T
    r = rad - tol
    return (np.abs(loc[:, 0]) < hx + r) & (np.abs(loc[:, 1]) < hy + r) & (xyz[:, 2] - r < cz + hz) & (xyz[:, 2] + r > cz - hz)



def hull_dist(pts, c):
    """Distance of point c outside the convex hull of 2-D pts (0 inside); single points and segments handled."""
    if len(pts) == 0:
        return np.inf
    if len(pts) < 3:
        return float(np.min(np.linalg.norm(pts - c, axis=1)))
    o = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[o]
    cross = lambda a, b, q: (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])  # noqa: E731
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in p[::-1]:
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    h = np.array(lo[:-1] + up[:-1])
    if len(h) < 3:
        return float(np.min(np.linalg.norm(pts - c, axis=1)))
    inside = all(cross(h[i], h[(i + 1) % len(h)], c) >= 0 for i in range(len(h)))
    if inside:
        return 0.0
    best = np.inf
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        t = np.clip(np.dot(c - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(a + t * (b - a) - c)))
    return best



def one(path: str):
    name = os.path.basename(path)[:-4]
    rp, rq, q = _frames(path)
    M, FLOOR, BOT, BODY, FOOT, HAND, PELVIS, BACK, RAD, H = _MODEL
    d = mujoco.MjData(M)
    T = len(q)
    low = np.ones(T)
    gpos = np.zeros((T, len(BOT), 3))  # geom centers
    glow = np.zeros((T, len(BOT)))  # lowest point height per geom (floor distance)
    com_off = np.zeros(T)
    com = np.zeros((T, 2))
    hullc = np.zeros((T, 2))
    gi = {g: i for i, g in enumerate(BOT)}
    for t in range(T):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = rp[t], rq[t], q[t]
        mujoco.mj_kinematics(M, d)
        mujoco.mj_comPos(M, d)
        mujoco.mj_collision(M, d)
        gpos[t] = d.geom_xpos[BOT]
        glow[t] = gpos[t, :, 2] - [RAD[g] for g in BOT]
        fl = []
        for c in d.contact[: d.ncon]:
            if FLOOR in (c.geom1, c.geom2):
                g = c.geom2 if c.geom1 == FLOOR else c.geom1
                if g in gi:
                    glow[t, gi[g]] = min(glow[t, gi[g]], c.dist) if c.dist < glow[t, gi[g]] else c.dist
                    if c.dist < 0.03:
                        fl.append(c.pos[:2].copy())
        low[t] = glow[t].min()
        fl = np.array(fl).reshape(-1, 2)
        com[t] = d.subtree_com[0, :2]
        com_off[t] = hull_dist(fl, com[t])
        hullc[t] = fl.mean(0) if len(fl) else rp[t, :2]
    gv = np.linalg.norm(np.gradient(gpos, axis=0), axis=-1) * HZ
    vz = np.abs(np.gradient(rp[:, 2]) * HZ)
    vh = np.linalg.norm(np.gradient(rp[:, :2], axis=0), axis=1) * HZ
    pel_still = (vh < 0.1) & (vz < 0.1)

    horiz, walls = [], []  # (x, y, top, t) support points; (x, y, z, nx, ny) wall contacts
    # step: a still foot that is the lowest point and off the floor; whole body lifted without vertical motion
    fi = [gi[g] for g in FOOT]
    stand = (gv[:, fi] < 0.15) & (glow[:, fi] > 0.04) & (glow[:, fi] <= low[:, None] + 0.01)
    for side in {BODY[g] for g in FOOT}:  # stood on for >= 0.3 s in a row (not the brief floats of retargeted gaits)
        cols = [j for j, g in enumerate(FOOT) if BODY[g] == side]
        for s, e in runs(stand[:, cols].any(1), 3):
            for t in range(s, e):
                for j in cols:
                    if stand[t, j]:
                        k = fi[j]
                        horiz.append((gpos[t, k, 0], gpos[t, k, 1], glow[t, k], t))
    for s, e in runs((low > 0.03) & (vz < 0.15), int(0.4 * HZ)):
        for t in range(s, e):
            for k in np.nonzero(glow[t] < low[t] + 0.03)[0]:
                horiz.append((gpos[t, k, 0], gpos[t, k, 1], glow[t, k], t))
    # seat
    seat = (low <= 0.03) & pel_still & (com_off > 0.06) & (rp[:, 2] < 0.89 * H)
    pi_ = [gi[g] for g in PELVIS]
    for s, e in runs(seat, HZ):
        for t in range(s, e):
            horiz.append((rp[t, 0], rp[t, 1], glow[t, pi_].min(), t))
    # hand / back support
    lean = (low <= 0.03) & pel_still & (com_off > 0.06) & (rp[:, 2] >= 0.89 * H)
    shoulder_z = None
    for s, e in runs(lean, HZ):
        for t in range(s, e):
            off = com[t] - hullc[t]
            off /= max(np.linalg.norm(off), 1e-6)
            if shoulder_z is None:
                shoulder_z = rp[t, 2] + 0.44 * H
            cand = [g for g in BOT if g not in FOOT and glow[t, gi[g]] >= 0.05 and gv[t, gi[g]] <= 0.05
                    and np.dot(gpos[t, gi[g], :2] - hullc[t], off) >= 0.05]
            wall_lean = any(g in BACK for g in cand)
            for g in cand:
                k = gi[g]
                if g in HAND and not wall_lean and gpos[t, k, 2] < rp[t, 2] + 0.32 * H:
                    horiz.append((gpos[t, k, 0], gpos[t, k, 1], glow[t, k], t))
                else:
                    walls.append((gpos[t, k, 0] + off[0] * RAD[g], gpos[t, k, 1] + off[1] * RAD[g], gpos[t, k, 2], off[0], off[1]))
    boxes = []
    hk = [gi[g] for g in HAND]
    climb = np.zeros(T, bool)
    for t in range(T):
        feet_up = any(gv[t, k] < 0.15 and glow[t, k] > 0.1 for k in fi)
        hand_up = any(gv[t, k] < 0.15 and gpos[t, k, 2] > rp[t, 2] + 0.32 * H for k in hk)
        climb[t] = feet_up and hand_up and low[t] > 0.03
    ladder = climb.sum() >= HZ
    if ladder:
        rungs, heading = [], []
        for t in np.nonzero(climb)[0]:
            w_, x_, y_, z_ = rq[t]
            heading.append(np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_)))
            for k in fi:
                if gv[t, k] < 0.15 and glow[t, k] > 0.1:
                    rungs.append((gpos[t, k, 0], gpos[t, k, 1], glow[t, k] - 0.015))
            for k in hk:
                if gv[t, k] < 0.15 and gpos[t, k, 2] > rp[t, 2] + 0.32 * H:  # the hand rests on the rung (no fingers)
                    rungs.append((gpos[t, k, 0], gpos[t, k, 1], gpos[t, k, 2] - RAD[HAND[hk.index(k)]] - 0.015))
        rungs = np.array(rungs)
        h = float(np.arctan2(np.mean(np.sin(heading)), np.mean(np.cos(heading))))
        n = np.array([np.cos(h), np.sin(h)])  # the robot faces the ladder: rungs ahead of it, across its heading
        tang = np.array([-n[1], n[0]])
        mid = rungs[:, :2].mean(0)
        along = float(np.median((rungs[:, :2] - mid) @ n))
        order = np.argsort(rungs[:, 2])
        levels = [rungs[idx] for idx in np.split(order, np.nonzero(np.diff(rungs[order, 2]) > 0.06)[0] + 1)]
        levels = [lv for lv in levels if len(lv) >= 2][: terrain.BOXES - 2]
        for lv in levels:
            c = lv[:, :2].mean(0)
            boxes.append([float(c[0]), float(c[1]), float(np.median(lv[:, 2])), 0.02, 0.25, 0.015, h])
        top = max(b[2] for b in boxes) + 0.3 if boxes else 2.0
        base = mid + n * along
        for sgn in (-1, 1):
            c = base + tang * sgn * 0.28
            boxes.append([float(c[0]), float(c[1]), top / 2, 0.02, 0.02, top / 2, h])
        near = np.convolve(climb, np.ones(2 * HZ + 1), "same") > 0  # platforms stood on before / after the climb
        horiz = [p for p in horiz if not near[p[3]]]
    if horiz:
        pts = np.array(horiz)
        order = np.argsort(pts[:, 2])
        clusters = [pts[idx] for idx in np.split(order, np.nonzero(np.diff(pts[order, 2]) > 0.06)[0] + 1)]
        # held for >= 0.3 s: a one-frame float of the whole body (a retargeting glitch) gives every foot geom a point
        clusters = [c for c in clusters if len(np.unique(c[:, 3])) >= 3 and np.median(c[:, 2]) >= 0.04]
        if len(clusters) > 2:  # a staircase: steps across its direction, as wide as all footprints (>= 0.8 m)
            allxy = np.concatenate([c[:, :2] for c in clusters])
            c0 = allxy.mean(0)
            ax = np.linalg.eigh(np.cov((allxy - c0).T))[1][:, -1]
            yaw = float(np.arctan2(ax[1], ax[0]))
            lat = np.array([-ax[1], ax[0]])
            half_w = max(float(np.ptp((allxy - c0) @ lat)) / 2 + 0.15, 0.4)
            tops = [float(np.median(cl[:, 2])) for cl in clusters]
            us = [(cl[:, :2] - c0) @ ax for cl in clusters]
            up = 1.0 if np.corrcoef([u.mean() for u in us], tops)[0, 1] >= 0 else -1.0
            far = max(float((u * up).max()) for u in us) + 0.06  # solid: every step runs on under the higher ones
            for top, u in zip(tops, us):
                lo, hi = float((u * up).min()) - 0.06, far
                c = c0 + ax * up * (lo + hi) / 2
                boxes.append([float(c[0]), float(c[1]), top / 2, max((hi - lo) / 2, 0.1), half_w, top / 2, yaw])
        else:
            for cl in clusters:
                top = float(np.median(cl[:, 2]))
                cx, cy, hx, hy, yaw = fit_box(cl[:, :2], 0.12, 0.25)
                boxes.append([cx, cy, top / 2, hx, hy, top / 2, yaw])
    if len(walls) >= 4:
        w = np.array(walls)
        n = w[:, 3:5].mean(0)
        n /= max(np.linalg.norm(n), 1e-6)
        yaw = float(np.arctan2(n[1], n[0]))
        dist = float(np.median(w[:, :2] @ n))
        tang = np.array([-n[1], n[0]])
        tt = w[:, :2] @ tang
        c = n * (dist + 0.06) + tang * (tt.min() + tt.max()) / 2
        boxes.append([float(c[0]), float(c[1]), 1.0, 0.05, max(float(np.ptp(tt)) / 2 + 0.4, 0.6), 1.0, yaw])
    boxes = [b for b in boxes if b[2] + b[5] >= MIN_TOP]  # the check runs on the terrain that is trained
    boxes = sorted(boxes, key=lambda b: -b[5])
    boxes = boxes[: terrain.BOXES]
    allp = gpos.reshape(-1, 3)
    rad = np.tile([RAD[g] for g in BOT], T)
    tol = 0.035 if ladder else 0.02  # feet and hands rest on the thin rungs
    for b in boxes:  # shrink away from body geoms inside (e.g. the feet on the floor before stepping up)
        wall = b[3] == 0.05 and b[5] == 1.0
        for _ in range(12):
            if not inside(b, allp, rad, tol).any():
                break
            if wall:  # walls move back along their normal
                b[0] += 0.01 * np.cos(b[6])
                b[1] += 0.01 * np.sin(b[6])
            elif b[3] != 0.02:  # not a rung or rail
                b[3], b[4] = max(b[3] * 0.85, 0.05), max(b[4] * 0.85, 0.12)
    if ladder:  # the hands hold the rungs; other limbs may brush them for up to 0.3 s
        arm = np.tile([g in HAND for g in BOT], T)
        hit = np.zeros(T, bool)
        for b in boxes:
            hit |= (inside(b, allp, rad, tol) & ~arm).reshape(T, len(BOT)).any(1)
        ok = int(hit.sum()) <= 3
    else:
        ok = all(not inside(b, allp, rad, tol).any() for b in boxes)
    if ok:  # every lowest point off the floor for more than 0.8 s rests on a box top (not a wall)
        k = glow.argmin(1)
        lp = gpos[np.arange(T), k]
        held = low <= 0.05
        for b in boxes:
            if b[3] == 0.05 and b[5] == 1.0 or b[5] > 0.3 and b[3] == 0.02:
                continue
            c, si = np.cos(b[6]), np.sin(b[6])
            dx, dy = lp[:, 0] - b[0], lp[:, 1] - b[1]
            on_xy = (np.abs(c * dx + si * dy) <= b[3] + 0.1) & (np.abs(-si * dx + c * dy) <= b[4] + 0.1)
            held |= on_xy & (np.abs(low - (b[2] + b[5])) <= 0.08)
        ok = not runs(~held, int(0.8 * HZ) + 1)
    kinds = (["ladder"] if ladder else []) + (["step/seat"] if horiz else []) + (["wall"] if len(walls) >= 4 else [])
    if not boxes and ok:
        return name, None
    if not boxes:
        kinds = ["unsupported"]
    return name, {"boxes": [[round(x, 4) for x in b] for b in boxes], "ok": bool(ok), "kinds": kinds,
                  "seat_s": float(seat.sum() / HZ), "lean_s": float(lean.sum() / HZ)}


def build(motion_dir: str, robot: str, workers: int = 24, dry_run: bool = False) -> dict:
    """Infers and checks every clip of motion_dir; returns the summary (see the module docstring)."""
    t0 = time.time()
    files = _files(motion_dir, robot)[0]
    with Pool(workers, initializer=_init, initargs=(robot, motion_dir)) as pool:
        res = {n: v for n, v in pool.imap_unordered(one, files, chunksize=16) if v}
    out = {n: v["boxes"] for n, v in res.items() if v["ok"]}
    failed = sorted(n for n, v in res.items() if not v["ok"])
    kinds = {}
    for n, v in res.items():
        for k in v["kinds"]:
            kinds.setdefault(k, [0, 0])[0 if v["ok"] else 1] += 1
    summary = {"robot": robot, "clips": len(files), "terrain": len(out), "dropped": len(failed),
               "kinds_ok_failed": kinds, "seconds": round(time.time() - t0, 1)}
    if not dry_run:
        drop = motion_dir.rstrip("/") + "_dropped"
        os.makedirs(drop, exist_ok=True)
        ext = os.path.splitext(files[0])[1] if files else ".npz"
        for n in failed:
            shutil.move(os.path.join(motion_dir, n + ext), os.path.join(drop, n + ext))
        json.dump(res, open(os.path.join(motion_dir, "terrain_report.json"), "w"))
        json.dump(summary, open(os.path.join(motion_dir, "terrain_summary.json"), "w"), indent=1)
        tmp = os.path.join(motion_dir, "terrain.json.tmp")
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, os.path.join(motion_dir, "terrain.json"))  # last: its presence marks the library as built
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--motion_dir", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dry_run", action="store_true", help="summary only: nothing written or moved")
    a = ap.parse_args()
    print(json.dumps(build(a.motion_dir, a.robot, a.workers, a.dry_run), indent=1))
