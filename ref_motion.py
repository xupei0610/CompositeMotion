from typing import Optional, Sequence, Union
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
        if "pos" in node.attrib:
            p = np.array(list(map(float, node.attrib.get("pos").split())))
        else:
            p = np.array([0, 0, 0])
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
            if joint.attrib.get("type") == "free":
                # root joint
                continue
            has_dof = True
            assert "axis" in joint.attrib, joint.attrib.get("name")
            try:
                axis = list(map(int, joint.attrib.get("axis").split()))
            except:
                print("Failed to parse joint", joint.attrib.get("name"))
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
        if freejoint is None:
            basejoint = root.findall("joint")
            if basejoint and basejoint[0].attrib.get("type") == "free":
                freejoint = basejoint
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
        key_links: Optional[Sequence[int]]=None, 
        device: Optional[torch.device]=None
    ):
        self.device = device

        self.motion = []
        self.motion_length = []
        self.motion_weight = []
        self.motion_dt = []
        self.motion_n_frames = []

        self.skeleton = load_mjcf(character_model)
        self.dofs = [d[2] for d in self.skeleton.dofs]
        if key_links is None:
            key_links = list(np.arange(len(self.skeleton.nodes)))
        controllable_links = sorted(list(set([d[1] for d in self.skeleton.dofs])))
        if type(motion_file) == str:
            self.load_motions(motion_file, self.skeleton, controllable_links, key_links)
        else:
            for m in motion_file:
                self.load_motions(m, self.skeleton, controllable_links, key_links)
        self.prepare_data()

    def prepare_data(self):
        self.motion_link_pos_tensor = torch.cat([m[0] for m in self.motion]).to(self.device)
        self.motion_link_orient_tensor = torch.cat([m[1] for m in self.motion]).to(self.device)
        self.motion_link_vel_tensor = torch.cat((torch.cat([m[2] for m in self.motion]), torch.cat([m[3] for m in self.motion])),-1).to(self.device)
        self.motion_joint_q_tensor = torch.cat([m[4] for m in self.motion]).to(self.device)
        self.motion_joint_p_tensor = torch.cat([m[5] for m in self.motion]).to(self.device)
        self.motion_joint_vel_tensor = torch.cat([m[6] for m in self.motion]).to(self.device)
        self.motion_dt_tensor = torch.tensor([m[7] for m in self.motion], dtype=torch.float, device=self.device)
        self.motion_n_frames_tensor = torch.tensor([m[0].size(0)-1 for m in self.motion], dtype=torch.int, device=self.device)
        self.motion_length = np.array([m[7]*(m[0].size(0)-1) for m in self.motion])
        self.motion_length_tensor = torch.from_numpy(self.motion_length).to(device=self.device, dtype=torch.float)

        self.motion_tensor_offset = torch.cumsum(torch.tensor([0]+[m[0].size(0) for m in self.motion[:-1]]), 0).to(self.device)
        self.has_translation_joint = torch.any(self.motion_joint_p_tensor[4] > 1e-6).item()

        tot_weights, tot_length = 0, 0
        tot_length_with_weights = 0
        for m in self.motion:
            w, t = m[8], m[0].size(0)-1
            if w is None or w < 0:
                tot_length += t
            elif w > 0:
                tot_weights += w
                tot_length_with_weights += t
                tot_length += t
        motion_weight = []
        for m in self.motion:
            w, t = m[8], m[0].size(0)-1
            if tot_length != tot_length_with_weights and (w is None or w < 0):
                if tot_length_with_weights == 0:
                    w = t/tot_length
                else:
                    w = t*tot_weights/tot_length_with_weights
            motion_weight.append(w)
        self.motion_weight = np.array(motion_weight)
        self.motion_weight /= np.sum(self.motion_weight)

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(len(self.motion), sum(self.motion_length)))

    def load_motions(self, motion_file, skeleton, controllable_links, key_links):
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
            motion_weights = [None]

        n_motion_files = len(motion_files)
        for f, (w, motion_file) in enumerate(zip(motion_weights, motion_files)):
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

            self.motion.append((
                motion.pos[:,key_links],
                motion.orient[:,key_links],
                motion.lin_vel[:,key_links],
                motion.ang_vel[:,key_links],
                motion.local_q[:,controllable_links],
                motion.local_p[:,controllable_links],
                motion.local_vel[:,controllable_links],
                dt, w
            ))

            print("\t{:.4f}s, {:d} Hz, {:d} frames".format(motion_len, fps, n_frames))


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
        motion_ids = torch.from_numpy(motion_ids)
        motion_times = torch.from_numpy(motion_times).to(device=self.device, dtype=torch.float)
        n_frames = self.motion_n_frames_tensor[motion_ids]
        motion_len = self.motion_length_tensor[motion_ids]
        dt = self.motion_dt_tensor[motion_ids]
        motion_id_offset = self.motion_tensor_offset[motion_ids]

        fid0 = (motion_times / motion_len).clip_(min=0, max=1).mul_(n_frames).to(torch.int)
        fid1 = (fid0+1).clip_(max=n_frames)
        frac = (motion_times - fid0*dt).div_(dt).clip_(max=1).view(-1, 1, 1)
        
        fid0.add_(motion_id_offset)
        fid1.add_(motion_id_offset)

        link_pos0 = self.motion_link_pos_tensor[fid0]
        link_pos1 = self.motion_link_pos_tensor[fid1]
        link_orient0 = self.motion_link_orient_tensor[fid0]
        link_orient1 = self.motion_link_orient_tensor[fid1]
        link_vel = self.motion_link_vel_tensor[fid0]
        if with_joint_tensor:
            joint_q0 = self.motion_joint_q_tensor[fid0]
            joint_q1 = self.motion_joint_q_tensor[fid1]
            if self.has_translation_joint:
                joint_p0 = self.motion_joint_p_tensor[fid0]
                joint_p1 = self.motion_joint_p_tensor[fid1]
            joint_vel = self.motion_joint_vel_tensor[fid1]

        one_frac = 1.0-frac
        link_pos = link_pos0.mul_(one_frac).add_(frac*link_pos1)
        link_orient = slerp(link_orient0, link_orient1, frac)
        link_tensor = torch.cat((link_pos, link_orient, link_vel), -1)
        if with_joint_tensor:
            joint_q = slerp(joint_q0, joint_q1, frac)
            if self.has_translation_joint:
                joint_p = joint_p0.mul_(one_frac).add_(frac*joint_p1)
                joint_pos = (quat2expmap(joint_q) + joint_p).view(joint_q.size(0), -1)[:, self.dofs]
            else:
                joint_pos = (quat2expmap(joint_q)).view(joint_q.size(0), -1)[:, self.dofs]
            joint_vel = joint_vel.view(joint_q.size(0), -1)[:, self.dofs]
            joint_tensor = torch.stack((joint_pos, joint_vel), -1)
            return link_tensor, joint_tensor
        else:
            return link_tensor
