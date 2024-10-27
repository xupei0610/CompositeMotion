from typing import Sequence, Union
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
    "nodes parents trans rot free_root dofs"
)
Motion = namedtuple("Motion",
    "fps pos orient ang_vel lin_vel local_q local_p local_vel"
)

import xml.etree.ElementTree as XMLParser
def load_mjcf(filename: Union[str, Sequence[str]]):
    if type(filename) == str:
        filename = [filename]
    
    nodes = []
    parents = []
    t, r = [], []
    dofs = []
    dof_offset = [0]
    def parse(node, pid):
        n = node.attrib.get("name")
        p = np.array(list(map(float, node.attrib.get("pos").split())))
        # NOTE for body rotation offset, only the quat attribute defined directly in the body element is supported
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
        has_dof = False
        for joint in node.findall("joint"):
            has_dof = True
            axis = list(map(int, joint.attrib.get("axis").split()))
            assert sum(axis)==1 and 1 in axis, joint.attrib.get("name")
            dofs.append((joint.attrib.get("name"), nid, dof_offset[0]*3+axis.index(1)))
        dof_offset[0] += int(has_dof)
        for child in node.findall("body"):
            parse(child, nid)

    free_root = []
    for f in filename:
        tree = XMLParser.parse(f)
        doc = tree.getroot()
        world = doc.find("worldbody")
        if world is None:
            raise ValueError("Failed to find worldbody definition from MJCF file", f)
        root = world.find("body")
        if root is None:
            raise ValueError("Failed to find any body definition from MJCF file", f)
        freejoint = root.find("freejoint")
        free_root.append(freejoint is not None)
    
        parse(root, -1)
    return Skeleton(
        nodes = nodes,
        parents = parents,
        trans = torch.from_numpy(np.array(t, dtype=float)),
        rot = torch.from_numpy(np.array(r, dtype=float)),
        free_root = free_root,
        dofs = dofs
    )

def compute_motion(fps:int, skeleton: Skeleton, local_q, local_p):
    orient = []
    pos = []
    i = 0
    for nid in range(len(skeleton.nodes)):
        pid = skeleton.parents[nid]
        if pid == -1:
            orient.append(quatmultiply(skeleton.rot[nid], local_q[:, nid]))
            pos.append(rotatepoint(skeleton.rot[nid].unsqueeze(0), local_p[:, nid]))
            if not skeleton.free_root[i]:
                pos[-1] += skeleton.trans[nid].unsqueeze(0)
            i += 1
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
    def __init__(self, motion_file: Union[str, Sequence[str]],
        character_model: Union[str, Sequence[str]], 
        key_links: Sequence[int],
        device: torch.device
    ):
        self.device = device

        self.motion = []
        self.motion_length = []
        self.motion_weight = []
        self.motion_dt = []
        self.motion_n_frames = []

        self.skeleton = load_mjcf(character_model)
        self.dofs = [idx for _, __, idx in self.skeleton.dofs]
        if key_links is None:
            key_links = list(np.arange(len(self.skeleton.nodes)))
        controllable_links = sorted(list(set([idx for _, idx, __  in self.skeleton.dofs])))
        self.n_key_links = len(key_links)
        self.n_controllable_links = len(controllable_links)
        if type(motion_file) == str:
            self.load_motions(motion_file, self.skeleton, controllable_links, key_links)
        else:
            for m in motion_file:
                self.load_motions(m, self.skeleton, controllable_links, key_links)
            self.motion_weight = np.array(self.motion_weight)

    def load_motions(self, motion_file, skeleton, controllable_links, key_links):
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
                with open(motion_file, "rb") as _:
                    motion = pickle.load(_)
                n_frames = len(motion.pos)
                fps = motion.fps
            
            dt = 1.0 / fps
            motion_len = dt * (n_frames - 1)
            motion_ = (
                motion.pos[:,key_links].clone().detach().requires_grad_(False),
                motion.orient[:,key_links].clone().detach().requires_grad_(False),
                torch.cat((motion.lin_vel[:,key_links], motion.ang_vel[:,key_links]), -1).detach().requires_grad_(False),
                motion.local_q[:,controllable_links].clone().detach().requires_grad_(False),
                motion.local_p[:,controllable_links].clone().detach().requires_grad_(False),
                motion.local_vel[:,controllable_links].clone().detach().requires_grad_(False)
            )

            print("\t{:.4f}s, {:d} Hz, {:d} frames".format(motion_len, fps, n_frames))


            del motion
            self.motion.append(motion_)
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
        self.motion_weight = np.divide(self.motion_weight, sum(self.motion_weight))

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(len(self.motion), sum(self.motion_length)))

    def sample(self, n, truncate_time=None):
        motion_ids = np.random.choice(len(self.motion), size=n, p=self.motion_weight, replace=True)
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)
        motion_len = self.motion_length[motion_ids]
        if truncate_time is not None:
            motion_len = np.maximum(motion_len - truncate_time, 0)
        motion_time = phase * motion_len
        return motion_ids, motion_time

    @torch.no_grad
    def state(self, motion_ids, motion_times, with_joint_tensor=True):
        n = len(motion_ids)
        device = self.device

        n_key_links = self.n_key_links
        link_pos0 = torch.empty((n, n_key_links, 3), dtype=torch.float)
        link_pos1 = torch.empty((n, n_key_links, 3), dtype=torch.float)
        link_orient0 = torch.empty((n, n_key_links, 4), dtype=torch.float)
        link_orient1 = torch.empty((n, n_key_links, 4), dtype=torch.float)
        link_vel = torch.empty((n, n_key_links, 6), dtype=torch.float)
        if with_joint_tensor:
            n_controllable_links = self.n_controllable_links
            joint_q0 = torch.empty((n, n_controllable_links, 4), dtype=torch.float)
            joint_q1 = torch.empty((n, n_controllable_links, 4), dtype=torch.float)
            joint_p0 = torch.empty((n, n_controllable_links, 3), dtype=torch.float)
            joint_p1 = torch.empty((n, n_controllable_links, 3), dtype=torch.float)
            joint_vel = torch.empty((n, n_controllable_links, 3), dtype=torch.float)
            joint_tensor = torch.empty((n, len(self.dofs), 2), dtype=torch.float, device=device)
        link_tensor = torch.empty((n, n_key_links, 13), dtype=torch.float, device=device)

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
            ids = motion_ids == uid
            motion = self.motion[uid]

            fid0_ = fid0[ids]
            fid1_ = fid1[ids]

            link_pos0[ids] = motion[0][fid0_]
            link_pos1[ids] = motion[0][fid1_]
            link_orient0[ids] = motion[1][fid0_]
            link_orient1[ids] = motion[1][fid1_]
            link_vel[ids] = motion[2][fid0_]

            if with_joint_tensor:
                joint_q0[ids] = motion[3][fid0_]
                joint_q1[ids] = motion[3][fid1_]
                joint_p0[ids] = motion[4][fid0_]
                joint_p1[ids] = motion[4][fid1_]
                joint_vel[ids] = motion[5][fid0_]

        link_pos0 = link_pos0.to(device)
        link_pos1 = link_pos1.to(device)
        link_orient0 = link_orient0.to(device)
        link_orient1 = link_orient1.to(device)
        link_tensor[..., 7:] = link_vel.to(device)
        frac = torch.tensor(frac, device=link_pos0.device, dtype=link_pos0.dtype).unsqueeze_(-1)
        one_frac = 1 - frac
        frac_ = frac[..., None]
        one_frac_ = one_frac[..., None]

        link_tensor[..., :3] = (one_frac_*link_pos0).add_(frac_*link_pos1)
        link_tensor[..., 3:7] = slerp(link_orient0, link_orient1, frac_)
        if not with_joint_tensor:
            return link_tensor

        joint_q0 = joint_q0.to(device)
        joint_q1 = joint_q1.to(device)
        joint_p0 = joint_p0.to(device)
        joint_p1 = joint_p1.to(device)
        joint_vel = joint_vel.to(device)
        joint_p = joint_p0.mul_(one_frac_).add_(frac_*joint_p1)
        joint_q = slerp(joint_q0, joint_q1, frac_)
        joint_tensor[..., 0] = quat2expmap(joint_q).add_(joint_p).view(joint_q.size(0), -1)[:, self.dofs]
        joint_tensor[..., 1] = joint_vel.view(joint_q.size(0), -1)[:, self.dofs]
        return link_tensor, joint_tensor
