from typing import Sequence
import os
import yaml
import json
import torch
import numpy as np

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

from utils import quat2axang, quat2expmap, wrap2pi, quatconj, quatmultiply, slerp

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
        self.dofs = dofs
        self.n_dofs = sum(self.dofs)
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

            skeleton = SkeletonTree.from_mjcf(self.character_model)
            if os.path.splitext(motion_file)[1] == ".json":
                skeleton = SkeletonTree.from_mjcf(self.character_model)

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
                    t.append(frame["base_position"])
                    for k in skeleton.node_names:
                        if k not in frame: continue
                        q = frame[k]
                        if len(q) == 1:
                            half_theta = q[0]/2
                            s, c = np.sin(half_theta), np.cos(half_theta)
                            q = [0, s, 0, c] # assume joint is always along y axis
                        r[-1].append(q)
                r = torch.from_numpy(np.array(r))
                t = torch.from_numpy(np.array(t))
                state = SkeletonState.from_rotation_and_root_translation(skeleton, r, t, is_local=True)
                motion = SkeletonMotion.from_skeleton_state(state, fps)
            else:
                motion = SkeletonMotion.from_file(motion_file)
                fps = motion.fps
                n_frames = motion.tensor.shape[0]
            # pre-load transformation
            motion.global_translation
            motion.global_rotation
            motion.local_rotation
            motion.global_root_velocity
            motion.global_root_angular_velocity

            dt = 1.0 /fps
            motion_len = dt * (n_frames - 1)

            print("\t{:.4f}s, {:d} Hz, {:d} frames".format(motion_len, fps, n_frames))

            joint_vels = []
            for frame in range(n_frames - 1):
                joint_q0 = motion.local_rotation[frame]
                joint_q1 = motion.local_rotation[frame + 1]
                dq = quatmultiply(quatconj(joint_q0), joint_q1)
                vel = quat2expmap(dq).div_(dt)

                j = 0
                v = torch.zeros((self.n_dofs, ), dtype=torch.float, device=self.device)
                for i, dof in zip(self.controllable_links, self.dofs):
                    if dof == 3:
                        v[j:j+dof] = vel[i]
                    elif dof == 1:
                        v[j] = vel[i][1] # assume joint is always along y axis
                    else:
                        assert(False), "Unsupported joint type with DoF not equal to 1 or 3."
                    j += dof

                joint_vels.append(v)
            joint_vels.append(joint_vels[-1])
            motion.joint_vels = torch.stack(joint_vels).cpu()

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

    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float, device=self.device, requires_grad=False)

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
        n_links = self.motion[0].num_joints
        n_key_links = len(self.key_links)

        root_pos0 = np.empty((n, 3))
        root_pos1 = np.empty((n, 3))
        root_orient = np.empty((n, 4))
        root_orient0 = np.empty((n, 4))
        root_orient1 = np.empty((n, 4))
        root_lin_vel = np.empty((n, 3))
        root_ang_vel = np.empty((n, 3))
        joint_q0 = np.empty((n, n_links, 4))
        joint_q1 = np.empty((n, n_links, 4))
        joint_vel = np.empty((n, self.n_dofs))
        link_pos0 = np.empty((n, n_key_links, 3))
        link_pos1 = np.empty((n, n_key_links, 3))
        link_orient0 = np.empty((n, n_key_links, 4))
        link_orient1 = np.empty((n, n_key_links, 4))
        link_lin_vel = np.empty((n, n_key_links, 3))
        link_ang_vel = np.empty((n, n_key_links, 3))

        motion_len = self.motion_length[motion_ids]
        num_frames = self.motion_n_frames[motion_ids]
        dt = self.motion_dt[motion_ids]

        phase = np.clip(motion_times / motion_len, 0.0, 1.0)
        fid0 = (phase * (num_frames - 1)).astype(np.int)
        fid1 = np.minimum(fid0 + 1, num_frames - 1)
        frac = np.minimum((motion_times - fid0 * dt) / dt, 1.)

        key_links = self.key_links[np.newaxis, :]

        unique_ids = np.unique(motion_ids)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            motion = self.motion[uid]

            fid0_ = fid0[ids]
            fid1_ = fid1[ids]

            root_pos0[ids, :]  = motion.global_translation[fid0_, 0].numpy()
            root_pos1[ids, :]  = motion.global_translation[fid1_, 0].numpy()

            root_orient0[ids, :] = motion.global_rotation[fid0_, 0].numpy()
            root_orient1[ids, :]  = motion.global_rotation[fid1_, 0].numpy()

            joint_q0[ids, :, :]= motion.local_rotation[fid0_].numpy()
            joint_q1[ids, :, :] = motion.local_rotation[fid1_].numpy()

            root_lin_vel[ids, :] = motion.global_root_velocity[fid0_].numpy()
            root_ang_vel[ids, :] = motion.global_root_angular_velocity[fid0_].numpy()
            joint_vel[ids, :] = motion.joint_vels[fid0_].numpy()
            
            fid0_ = fid0_[:, np.newaxis]
            fid1_ = fid1_[:, np.newaxis]

            link_pos0[ids, :, :] = motion.global_translation[fid0_, key_links].numpy()
            link_pos1[ids, :, :] = motion.global_translation[fid1_, key_links].numpy()
            
            link_orient0[ids, :, :] = motion.global_rotation[fid0_, key_links].numpy()
            link_orient1[ids, :, :] = motion.global_rotation[fid1_, key_links].numpy()
            link_lin_vel[ids, :] = motion.global_velocity[fid0_, key_links].numpy()
            link_ang_vel[ids, :] = motion.global_angular_velocity[fid0_, key_links].numpy()


        frac = self.to_tensor(frac).unsqueeze_(-1)

        root_pos0 = self.to_tensor(root_pos0)
        root_pos1 = self.to_tensor(root_pos1)
        root_orient0 = self.to_tensor(root_orient0)
        root_orient1 = self.to_tensor(root_orient1)
        root_lin_vel = self.to_tensor(root_lin_vel)
        root_ang_vel = self.to_tensor(root_ang_vel)
        joint_q0 = self.to_tensor(joint_q0)
        joint_q1 = self.to_tensor(joint_q1)
        link_pos0 = self.to_tensor(link_pos0)
        link_pos1 = self.to_tensor(link_pos1)
        joint_vel = self.to_tensor(joint_vel)

        link_orient0 = self.to_tensor(link_orient0)
        link_orient1 = self.to_tensor(link_orient1)
        link_lin_vel = self.to_tensor(link_lin_vel)
        link_ang_vel = self.to_tensor(link_ang_vel)

        frac_exp = frac.unsqueeze(-1)
        
        link_orient = slerp(link_orient0, link_orient1, frac_exp)
        root_pos = (1.0 - frac) * root_pos0 + frac * root_pos1
        root_orient = slerp(root_orient0, root_orient1, frac)
        link_pos = (1.0 - frac_exp) * link_pos0 + frac_exp * link_pos1
        
        joint_q = slerp(joint_q0, joint_q1, frac_exp)
        joint_pos = torch.zeros((n, self.n_dofs), dtype=torch.float, device=self.device)
        j = 0
        for i, dof in zip(self.controllable_links, self.dofs):
            if (dof == 3):
                joint_pos[:, j:j+dof] = quat2expmap(joint_q[:, i])
            elif (dof == 1):
                ax, ang = quat2axang(joint_q[:, i])
                ax = ax[..., 1] # assume joint is always along y axis
                joint_pos[:, j] = wrap2pi(ang*ax)
            j += dof

        root_tensor = torch.cat((root_pos, root_orient, root_lin_vel, root_ang_vel), -1)
        link_tensor = torch.cat((link_pos, link_orient, link_lin_vel, link_ang_vel), -1)
        joint_tensor = torch.stack((joint_pos, joint_vel), -1)

        return root_tensor, link_tensor, joint_tensor
