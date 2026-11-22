"""partymimic.evaluate's protocol on libraries too large for one world per clip: the clips are loaded in groups of
--group files and, longest first, played in chunks of --worlds worlds (each once from its first frame, no randomization
or noise; a clip fails if the training termination triggers). Writes {norm_return, n_clips, n_failed, fail_rate,
failed: progress of each failed clip} to --out.

    python tools/eval_chunks.py --ckpt <ckpt.pt> --motion_dir <npz clips> --out <json> [--worlds 8192 --group 20000]
"""

import argparse
import glob
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--motion_dir", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--robot", type=str, default=None, help="default: the checkpoint's")
parser.add_argument("--worlds", type=int, default=8192)
parser.add_argument("--group", type=int, default=20000, help="clips loaded at a time")
args = parser.parse_args()
os.environ.setdefault("PM_LAZY_FK", "1")

import torch

import partymimic.networks as networks
from partymimic.dr import XiLayout
from partymimic.env import TrackingSim, make_sim
from partymimic.kinematics import Kinematics
from partymimic.mdp import MAX_STEP_REWARD, MDP
from partymimic.motion import MotionLib, load_npz
from partymimic.networks import Policy, policy_kwargs
from partymimic.robot import get_robot
import partymimic.terrain as terrain

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
a_ = ck["args"]
terrain.enabled = bool(a_.get("terrain"))  # the clips' <motion_dir>/terrain.json, as in training
robot = get_robot(args.robot or a_.get("robot", "g1"))
kin = Kinematics(robot, dev)
files = sorted(glob.glob(os.path.join(args.motion_dir, "*.npz")))
K, Hh = a_["goal_frames"], a_["hist"]
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
widths = lambda key, default: tuple(int(x) for x in a_.get(key, default).split(","))  # noqa: E731
sim_cfg = make_sim(robot)
t0 = time.time()
pi = None
ret_sum, n_clips, failed = 0.0, 0, {}
for g0 in range(0, len(files), args.group):
    motion = MotionLib(files[g0 : g0 + args.group], kin, dev, load_npz)
    motion.set_interface(a_.get("interface", "joint"), robot.tracked_bodies, a_.get("obs_err") or "none")
    if terrain.enabled:
        motion.set_terrain(args.motion_dir)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
    if pi is None:
        pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=Hh, enc_hidden=widths("enc_hidden", "2048,1024,512,256"),
                    dec_hidden=widths("dec_hidden", "2048,1024,512,256"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
        pi.load_state_dict(ck["pi"])
        pi.eval()
    C = len(motion.names)
    order = torch.argsort(motion.clip_len, descending=True)
    W = min(args.worlds, C)
    env = TrackingSim(sim_cfg, W, robot, motion, mdp, dev, eval_mode=True, randomize=False)
    with torch.no_grad():
        for c0 in range(0, C, W):
            clips = order[c0 : c0 + W]
            n = len(clips)
            clip_of = torch.cat((clips, clips[torch.arange(W - n, device=dev) % n]))  # the last chunk is padded
            env.done_eval[:] = False
            env.failed_eval[:] = False
            env.reset(torch.arange(W, device=dev), motion.clip_start[clip_of].clone())
            clip_end = motion.clip_end[clip_of]
            ret = torch.zeros(W, device=dev)
            steps = torch.zeros(W, device=dev)
            hist = None
            while not bool(env.done_eval.all()):
                o = env.observe()
                hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
                a = pi(hist, motion.goal(env.f, K))
                alive = ~env.done_eval.clone()
                out = env.step(a)
                ret += torch.where(alive, out["r"], 0.0)
                steps += alive.float()
                env.f = torch.minimum(env.f, clip_end - 1)
            L = motion.clip_len[clips]
            ret_sum += (ret[:n] / ((L - 1).float() * MAX_STEP_REWARD)).sum().item()
            prog = (steps[:n] / (L - 1).float()).clamp(max=1)
            for i in env.failed_eval[:n].nonzero().flatten().tolist():
                failed[motion.names[int(clips[i])]] = round(prog[i].item(), 3)
            n_clips += n
            print(f"    chunk {n_clips} ({time.time() - t0:.0f} s)", flush=True)
    print(f"  {n_clips}/{len(files)} clips, failed {len(failed)} ({time.time() - t0:.0f} s)", flush=True)
    del env, mdp, motion
    torch.cuda.empty_cache()

res = {"iteration": ck.get("iteration"), "wall_time_h": ck.get("wall_time", 0) / 3600, "n_clips": n_clips,
       "norm_return": ret_sum / n_clips, "n_failed": len(failed), "fail_rate": len(failed) / n_clips,
       "failed": failed, "eval_s": time.time() - t0}
json.dump(res, open(args.out + ".tmp", "w"), indent=1)
os.replace(args.out + ".tmp", args.out)
print(f"it {res['iteration']}: {n_clips - len(failed)}/{n_clips} clips completed ({100 * (1 - res['fail_rate']):.2f}%), "
      f"norm return {res['norm_return']:.4f} ({res['eval_s']:.0f} s)", flush=True)
