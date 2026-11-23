"""--terrain: static box terrain of the reference clips (inferred and checked by partymimic.terrain_infer).

``<motion_dir>/terrain.json`` maps a clip name to up to ``BOXES`` upright boxes [cx, cy, cz, hx, hy, hz, yaw] (world,
ground z 0); clips without an entry have flat ground. The simulator places a clip's boxes (mocap bodies with per-world
sizes) at every reset. The sim context xi carries them raw (center 0, scale 1, last in the layout); the world model and
critic see them as a height scan around the root of the current state (``scan``), so imagination follows the terrain.
The policy does not see the terrain: its reference (on terrain clips, raised to the terrain) is the only cue.
"""

from __future__ import annotations

import json
import os

import mujoco
import torch

BOXES, BOX = 16, 7  # 16: one solid box per stair step, a ladder is its rungs and two rails
XI_DIM = BOXES * BOX
PARK = (0.0, 0.0, -10.0, 0.05, 0.05, 0.05, 0.0)
# height scan points (m) in the root's heading frame: 11 along the heading (0.45 m behind to 1.05 m ahead), 5 across
SCAN_X, SCAN_Y = (-0.45, 1.05, 11), (-0.3, 0.3, 5)
SCAN_DIM = SCAN_X[2] * SCAN_Y[2]


def scan_points(device) -> torch.Tensor:
    """[SCAN_DIM, 2], made on the device (a CPU constant would be copied in every call, also in a captured graph)."""
    gx, gy = torch.linspace(*SCAN_X, device=device), torch.linspace(*SCAN_Y, device=device)
    return torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)

enabled = False
SOFT = 0.0  # --terrain_soft: edge width (m) of the scanned boxes; 0: hard edges (no gradient in x, y, yaw)


def load(motion_dir: str, names: list[str], device: str) -> tuple[torch.Tensor, int]:
    """Boxes [num_clips, BOXES, BOX] of the library's clips (unused slots parked) and the number of terrain clips."""
    path = os.path.join(motion_dir, "terrain.json")
    tj = json.load(open(path)) if os.path.exists(path) else {}
    rows = [(list(tj.get(n, [])) + [list(PARK)] * BOXES)[:BOXES] for n in names]
    assert all(len(tj.get(n, [])) <= BOXES for n in names)
    return torch.tensor(rows, dtype=torch.float32, device=device), sum(bool(tj.get(n)) for n in names)


def top(boxes: torch.Tensor, x: torch.Tensor, y: torch.Tensor, soft: float = 0.0) -> torch.Tensor:
    """Terrain height [..., P] (ground: 0) under the points (x, y) [..., P] for boxes [..., BOXES, BOX]; soft > 0:
    sigmoid edges of that width, so the height (and the scan) is differentiable in x, y and the heading."""
    b = boxes.unsqueeze(-3)                                                          # [..., 1, M, 7]
    dx, dy = x.unsqueeze(-1) - b[..., 0], y.unsqueeze(-1) - b[..., 1]
    c, s = b[..., 6].cos(), b[..., 6].sin()
    u, v = (c * dx + s * dy).abs(), (-s * dx + c * dy).abs()
    h = b[..., 2] + b[..., 5]
    if soft > 0:
        m = torch.sigmoid(((b[..., 3] - u) / soft).clamp(-30, 30)) * torch.sigmoid(((b[..., 4] - v) / soft).clamp(-30, 30))
        hm = m * h  # the highest box by gather: the compiled amax backward (ties by recomputed equality) gave NaN
        return hm.gather(-1, hm.argmax(-1, keepdim=True)).squeeze(-1).clamp(min=0)
    return torch.where((u < b[..., 3]) & (v < b[..., 4]), h, torch.zeros_like(dx)).amax(-1).clamp(min=0)


def scan(S, boxes_flat: torch.Tensor) -> torch.Tensor:
    """Terrain height at the SCAN points around the root of state S minus the root height [..., SCAN_DIM]."""
    p, R = S[0], S[1]
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    c, s = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)
    g = scan_points(p.device).to(p.dtype)
    x = p[..., 0:1] + c * g[:, 0] - s * g[:, 1]
    y = p[..., 1:2] + s * g[:, 0] + c * g[:, 1]
    return top(boxes_flat.unflatten(-1, (BOXES, BOX)), x, y, SOFT) - p[..., 2:3]


def with_scan(S, xi_n: torch.Tensor) -> torch.Tensor:
    """xi_n with its terrain columns (the last XI_DIM) replaced by the height scan of S."""
    if not enabled:
        return xi_n
    return torch.cat((xi_n[..., :-XI_DIM], scan(S, xi_n[..., -XI_DIM:])), -1)


def mirror(boxes_flat: torch.Tensor) -> torch.Tensor:
    """Reflection of the world y axis (mirror.Mirror): center y and yaw change sign."""
    b = boxes_flat.unflatten(-1, (BOXES, BOX)).clone()
    b[..., 1], b[..., 6] = -b[..., 1], -b[..., 6]
    return b.flatten(-2)


def add_to_spec(spec: mujoco.MjSpec) -> None:
    for i in range(BOXES):
        body = spec.worldbody.add_body(name=f"terrain{i}", mocap=True, pos=PARK[:3])
        body.add_geom(name=f"terrain{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.5, 0.5, 0.5), contype=1,
                      conaffinity=1, condim=3, rgba=(0.45, 0.6, 0.75, 1.0))
