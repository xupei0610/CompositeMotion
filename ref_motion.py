from typing import Sequence
import os
import torch
import numpy as np
import yaml
import json
import pickle
from collections import namedtuple
import scipy.ndimage.filters as filters

from utils import quat2expmap,  quatconj, quatmultiply, slerp, quat2expmap, rotatepoint


Skeleton = namedtuple("Skeleton",
    "nodes parents trans rot"
)
Motion = namedtuple("Motion",
    "fps pos orient ang_vel lin_vel local_q local_p local_vel"
)

import xml.etree.ElementTree as XMLParser
def load_mjcf(filename: str):
    tree = XMLParser.parse(filename)
    doc = tree.getroot()
    world = doc.find("worldbody")
    if world is None:
        raise ValueError("Failed to find worldbody definition from MJCF file", filename)
    root = world.find("body")
    if root is None:
        raise ValueError("Failed to find any body definition from MJCF file", filename)
    
    nodes = []
    parents = []
    t, r = [], []
    def parse(node, pid):
        n = node.attrib.get("name")
        p = np.array(list(map(float, node.attrib.get("pos").split())))
        # NOTE for body rotation offset, only the quat attribute defined dierctly in the body element is supported
        q = node.attrib.get("quat")
        if q is None:
            q = [0., 0., 0., 1.]
        else:
            q = list(map(float, q.split()))
            q = np.array([q[1], q[2], q[3], q[0]])
        nodes.append(n)
        parents.append(pid)
        t.append(p)
        r.append(q)
        nid = len(nodes)-1
        for child in node.findall("body"):
            parse(child, nid)

    parse(root, -1)
    return Skeleton(
        nodes = nodes,
        parents = parents,
        trans = torch.from_numpy(np.array(t, dtype=float)),
        rot = torch.from_numpy(np.array(r, dtype=float))
    )

def compute_motion(fps:int, skeleton: Skeleton, local_q, local_p):
    orient = []
    pos = []
    for nid in range(len(skeleton.nodes)):
        pid = skeleton.parents[nid]
        if pid == -1:
            orient.append(quatmultiply(skeleton.rot[nid], local_q[:, nid]))
            pos.append(rotatepoint(skeleton.rot[nid].unsqueeze(0), local_p[:, nid]))
            root = nid
        else:
            q = quatmultiply(orient[pid], skeleton.rot[nid])
            orient.append(quatmultiply(q, local_q[:, nid]))
            pos.append(pos[pid] + rotatepoint(orient[pid], local_p[:, nid]+skeleton.trans[nid].unsqueeze(0)))

    orient = torch.stack(orient, 1) # N_frames x N_links x 4
    pos = torch.stack(pos, 1)       # N_frames x N_links x 3

    dq = quatmultiply(orient[1:], quatconj(orient[:-1]))
    ang_vel = quat2expmap(dq).mul_(fps)
    ang_vel = torch.cat((ang_vel, torch.zeros_like(ang_vel[-1:])), 0)
    ang_vel = filters.gaussian_filter1d(
        ang_vel.numpy(), 2, axis=0, mode="nearest"
    )
    ang_vel = torch.from_numpy(ang_vel) # N_frames x N_links x 3

    lin_vel = filters.gaussian_filter1d(
        np.gradient(pos.numpy(), axis=0), 2, axis=0, mode="nearest"
    )
    lin_vel = torch.from_numpy(lin_vel).mul_(fps) # N_frames x N_links x 3

    dq = quatmultiply(quatconj(local_q[:-1]), local_q[1:])
    local_ang_vel = quat2expmap(dq).mul_(fps)

    # remove root translation
    # NOTE We need the rotation joints put before translation joints 
    # if the root joint is controllable.
    local_p = local_p.clone()
    local_p[:, root] = 0

    local_lin_vel = local_p[1:] - local_p[:-1]
    local_lin_vel = local_lin_vel.mul_(fps)
    
    local_vel = local_ang_vel + local_lin_vel
    local_vel = torch.cat((local_vel, local_vel[-1:]))

    return Motion(
        fps=fps,
        pos=pos.to(torch.float),
        orient=orient.to(torch.float),
        ang_vel=ang_vel.to(torch.float),
        lin_vel=lin_vel.to(torch.float),
        local_q=local_q.to(torch.float),
        local_p=local_p.to(torch.float),
        local_vel=local_vel.to(torch.float)
    )


class ReferenceMotion():
    def __init__(self, motion_file: str or Sequence[str],
        character_model: str, 
        key_links: Sequence[int], 
        controllable_links: Sequence[int], 
        dofs: Sequence[int],
        device: torch.device
    ):
        self.character_model = character_model
        
        assert(len(controllable_links) == len(dofs))
        self.key_links = np.array(key_links)
        self.controllable_links = controllable_links
        self.dofs = []
        for i, dof in enumerate(dofs):
            assert all(_ in "XYZ0123" for _ in set(str(dof)))
            for axis in str(dof):
                if axis == "X" or axis == "0":
                    self.dofs.append(i*3)
                elif axis == "Y" or axis == "1":
                    self.dofs.append(i*3+1)
                elif axis == "Z" or axis == "2":
                    self.dofs.append(i*3+2)
                else:
                    self.dofs.extend((i*3, i*3+1, i*3+2))
        self.device = device

        self.motion = []
        self.motion_length = []
        self.motion_weight = []
        self.motion_dt = []
        self.motion_n_frames = []

        if type(motion_file) == str:
            self.load_motions(motion_file)
        else:
            for m in motion_file:
                self.load_motions(m)
            self.motion_weight = np.array(self.motion_weight)

    def load_motions(self, motion_file):
        self.motion_length = list(self.motion_length)
        self.motion_dt = list(self.motion_dt)
        self.motion_n_frames = list(self.motion_n_frames)

        if os.path.splitext(motion_file)[1] == ".yaml":
            with open(motion_file, 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            dirname = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []
            for item in motion_config['motions']:
                motion_weights.append(item['weight'])
                motion_files.append(os.path.join(dirname, item['file']))
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        n_motion_files = len(motion_files)
        tot_weights, tot_length = 0, 0
        tot_length_with_weights = 0
        for f, motion_file in enumerate(motion_files):
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, n_motion_files, motion_file))

            skeleton = load_mjcf(self.character_model)
            
            if os.path.splitext(motion_file)[1] == ".json":
                with open(motion_file, "r") as _:
                    motion_data = json.load(_)

                assert("fps" in motion_data.keys() or "sampling_rate" in motion_data.keys())
                if "fps" in motion_data.keys():
                    fps = motion_data["fps"]
                else:
                    fps = 1 / motion_data["sampling_rate"]
                n_frames = len(motion_data["frames"])

                r, t = [], []
                for frame in motion_data["frames"]:
                    r.append([])
                    t.append([])
                    for joint in skeleton.nodes:
                        if joint in frame:
                            q = frame[joint]
                            if len(q) == 2:
                                p, q = q[0], q[1]
                                assert (len(p) == 3 and len(q) == 4) or (len(p) == 4 and len(q) == 3)
                                if len(p) == 4 and len(q) == 3:
                                    p, q = q, p
                            elif len(q) == 3:
                                # translation
                                p, q = q, [0.,0.,0.,1.]
                            elif len(q) == 4:
                                p = [0.,0.,0.]
                            else:
                                assert len(frame[joint]) in [2,3,4]
                        else:
                            q = [0.,0.,0.,1.]
                            p = [0.,0.,0.]
                        r[-1].append(q)
                        t[-1].append(p)
                r = torch.from_numpy(np.array(r))
                t = torch.from_numpy(np.array(t))
                motion = compute_motion(fps, skeleton, r, t)
            else:
                # FIXME
                with open(motion_file, "rb") as _:
                    motion = pickle.load(_)
                n_frames = len(motion.pos)
                fps = motion.fps
            
            dt = 1.0 / fps
            motion_len = dt * (n_frames - 1)

            print("\t{:.4f}s, {:d} Hz, {:d} frames".format(motion_len, fps, n_frames))


            self.motion.append(motion)
            self.motion_dt.append(dt)
            self.motion_n_frames.append(n_frames)
            self.motion_length.append(motion_len)
            curr_weight = motion_weights[f]
            if curr_weight is None or curr_weight < 0:
                tot_length += motion_len
            elif curr_weight > 0:
                tot_weights += curr_weight
                tot_length_with_weights += motion_len
                tot_length += motion_len
            
        for f in range(n_motion_files):
            curr_weight = motion_weights[f]
            if tot_length != tot_length_with_weights:
                curr_length = self.motion_length[f-n_motion_files]
                if curr_weight is None or curr_weight < 0:
                    if tot_length_with_weights == 0:
                        curr_weight = curr_length/tot_length
                    else:
                        curr_weight = curr_length*tot_weights/tot_length_with_weights
                elif curr_weight > 0:
                    curr_weight *= tot_length_with_weights/tot_length
            self.motion_weight.append(curr_weight)

        self.motion_length = np.array(self.motion_length)
        self.motion_dt = np.array(self.motion_dt)
        self.motion_n_frames = np.array(self.motion_n_frames)

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(len(self.motion), sum(self.motion_length)))

    def sample(self, n, truncate_time=None):
        p = np.divide(self.motion_weight, sum(self.motion_weight))
        motion_ids = np.random.choice(len(self.motion), size=n, p=p, replace=True)
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)
        motion_len = self.motion_length[motion_ids]
        if (truncate_time is not None): motion_len -= truncate_time
        motion_time = phase * motion_len
        return motion_ids, motion_time

    def state(self, motion_ids, motion_times):
        n = len(motion_ids)
        n_key_links = len(self.key_links)
        n_controllable_links = len(self.controllable_links)
        device = self.device

        root_pos0 = torch.empty((n, 3), dtype=torch.float, requires_grad=False)
        root_pos1 = torch.empty((n, 3), dtype=torch.float, requires_grad=False)
        root_orient = torch.empty((n, 4), dtype=torch.float, requires_grad=False)
        root_orient0 = torch.empty((n, 4), dtype=torch.float, requires_grad=False)
        root_orient1 = torch.empty((n, 4), dtype=torch.float, requires_grad=False)
        root_lin_vel = torch.empty((n, 3), dtype=torch.float, requires_grad=False)
        root_ang_vel = torch.empty((n, 3), dtype=torch.float, requires_grad=False)
        joint_q0 = torch.empty((n, n_controllable_links, 4), dtype=torch.float, requires_grad=False)
        joint_q1 = torch.empty((n, n_controllable_links, 4), dtype=torch.float, requires_grad=False)
        joint_p0 = torch.empty((n, n_controllable_links, 3), dtype=torch.float, requires_grad=False)
        joint_p1 = torch.empty((n, n_controllable_links, 3), dtype=torch.float, requires_grad=False)
        joint_vel = torch.empty((n, n_controllable_links, 3), dtype=torch.float, requires_grad=False)
        link_pos0 = torch.empty((n, n_key_links, 3), dtype=torch.float, requires_grad=False)
        link_pos1 = torch.empty((n, n_key_links, 3), dtype=torch.float, requires_grad=False)
        link_orient0 = torch.empty((n, n_key_links, 4), dtype=torch.float, requires_grad=False)
        link_orient1 = torch.empty((n, n_key_links, 4), dtype=torch.float, requires_grad=False)
        link_lin_vel = torch.empty((n, n_key_links, 3), dtype=torch.float, requires_grad=False)
        link_ang_vel = torch.empty((n, n_key_links, 3), dtype=torch.float, requires_grad=False)

        motion_len = self.motion_length[motion_ids]
        num_frames = self.motion_n_frames[motion_ids]
        dt = self.motion_dt[motion_ids]

        phase = np.clip(motion_times / motion_len, 0.0, 1.0)
        fid0 = (phase * (num_frames - 1)).astype(int)
        fid1 = np.minimum(fid0 + 1, num_frames - 1)
        frac = np.minimum((motion_times - fid0 * dt) / dt, 1.)

        unique_ids = np.unique(motion_ids)
        motion_ids = torch.from_numpy(motion_ids)
        fid0 = torch.from_numpy(fid0)
        fid1 = torch.from_numpy(fid1)
        for uid in unique_ids:
            ids, = torch.where(motion_ids == uid)
            motion = self.motion[uid]

            fid0_ = fid0[ids]
            fid1_ = fid1[ids]

            root_pos0[ids]  = motion.pos[fid0_, 0]
            root_pos1[ids]  = motion.pos[fid1_, 0]
            root_orient0[ids] = motion.orient[fid0_, 0]
            root_orient1[ids]  = motion.orient[fid1_, 0]
            root_lin_vel[ids] = motion.lin_vel[fid0_, 0]
            root_ang_vel[ids] = motion.ang_vel[fid0_, 0]
            
            fid0_ = fid0_[:, None]
            fid1_ = fid1_[:, None]

            joint_q0[ids] = motion.local_q[fid0_, self.controllable_links]
            joint_q1[ids] = motion.local_q[fid1_, self.controllable_links]
            joint_p0[ids] = motion.local_p[fid0_, self.controllable_links]
            joint_p1[ids] = motion.local_p[fid1_, self.controllable_links]
            joint_vel[ids] = motion.local_vel[fid0_, self.controllable_links]

            link_pos0[ids] = motion.pos[fid0_, self.key_links]
            link_pos1[ids] = motion.pos[fid1_, self.key_links]
            link_orient0[ids] = motion.orient[fid0_, self.key_links]
            link_orient1[ids] = motion.orient[fid1_, self.key_links]
            link_lin_vel[ids] = motion.lin_vel[fid0_, self.key_links]
            link_ang_vel[ids] = motion.ang_vel[fid0_, self.key_links]

        frac = torch.tensor(frac, device=device,  dtype=root_pos0.dtype).unsqueeze_(-1)
        frac_ = frac[..., None]

        root_pos0 = root_pos0.to(device)
        root_pos1 = root_pos1.to(device)
        root_orient0 = root_orient0.to(device)
        root_orient1 = root_orient1.to(device)
        root_lin_vel = root_lin_vel.to(device)
        root_ang_vel = root_ang_vel.to(device)
        joint_q0 = joint_q0.to(device)
        joint_q1 = joint_q1.to(device)
        joint_p0 = joint_p0.to(device)
        joint_p1 = joint_p1.to(device)
        joint_vel = joint_vel.to(device)
        link_pos0 = link_pos0.to(device)
        link_pos1 = link_pos1.to(device)
        link_orient0 = link_orient0.to(device)
        link_orient1 = link_orient1.to(device)
        link_lin_vel = link_lin_vel.to(device)
        link_ang_vel = link_ang_vel.to(device)

        root_pos = ((1.0-frac)*root_pos0).add_(frac*root_pos1)
        root_orient = slerp(root_orient0, root_orient1, frac)
        link_pos = ((1.0-frac_)*link_pos0).add_(frac_*link_pos1)
        link_orient = slerp(link_orient0, link_orient1, frac_)
        joint_p = ((1.0-frac_)*joint_p0).add_(frac_*joint_p1)
        joint_q = slerp(joint_q0, joint_q1, frac_)
        joint_pos = (quat2expmap(joint_q) + joint_p).view(joint_q.size(0), -1)[:, self.dofs]
        joint_vel = joint_vel.view(joint_q.size(0), -1)[:, self.dofs]

        root_tensor = torch.cat((root_pos, root_orient, root_lin_vel, root_ang_vel), -1)
        link_tensor = torch.cat((link_pos, link_orient, link_lin_vel, link_ang_vel), -1)
        joint_tensor = torch.stack((joint_pos, joint_vel), -1)
        return root_tensor, link_tensor, joint_tensor
