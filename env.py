from typing import Callable, Optional, List, Dict, Any
from collections import namedtuple
import os
from isaacgym import gymapi, gymtorch
import torch

from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quatdiff_normalized

def parse_kwarg(kwargs: dict, key: str, default_val: Any):
    return kwargs[key] if key in kwargs else default_val

class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[List[str]]=None, ob_horizon: Optional[int]=None, 
        parent_link: Optional[str]=None,
        replay_speed: Optional[Callable]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.parent_link = parent_link
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight

DiscriminatorProperty = namedtuple("DiscriminatorProperty",
    "name key_links parent_link replay_speed ob_horizon id"
)


class Env(object):
    UP_AXIS = 2
    CHARACTER_MODEL = None
    CAMERA_POS= 0, -4.5, 2.0
    CAMERA_FOLLOWING = True

    def __init__(self,
        n_envs: int, fps: int=30, frameskip: int=2,
        episode_length: Optional[Callable or int] = 300,
        control_mode: str = "position",
        substeps: int = 2,
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,
        **kwargs
    ):
        self.viewer = None
        assert(control_mode in ["position", "torque", "free"])
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1./self.fps
        self.substeps = substeps
        self.control_mode = control_mode
        self.episode_length = episode_length
        self.device = torch.device(compute_device)
        self.camera_pos = self.CAMERA_POS
        self.camera_following = self.CAMERA_FOLLOWING
        if graphics_device is None:
            graphics_device = compute_device
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model

        sim_params = self.setup_sim_params()
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
        self.add_ground()
        self.envs, self.actors = self.create_envs(n_envs)
        self.setup_action_normalizer()
        self.create_tensors()

        self.gym.prepare_sim(self.sim)

        n_actors_per_env = self.gym.get_actor_count(self.envs[0])
        self.actor_ids = torch.arange(n_actors_per_env * len(self.envs), dtype=torch.int32, device=self.device).view(len(self.envs), -1)
        controllable_actors = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            dof = self.gym.get_actor_dof_count(self.envs[0], i)
            if dof > 0: controllable_actors.append(i)
        self.actor_ids_having_dofs = \
            n_actors_per_env * torch.arange(len(self.envs), dtype=torch.int32, device=self.device).unsqueeze(-1) + \
            torch.tensor(controllable_actors, dtype=torch.int32, device=self.device).unsqueeze(-2)
        
        self.refresh_tensors()
        self.viewer_pause = False
        self.viewer_advance = False
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        base_pos = self.root_pos[tar_env].cpu()
        self.cam_target = gymapi.Vec3(*self.vector_up(1.0, [base_pos[0], base_pos[1], base_pos[2]]))

        self.simulation_step = 0
        self.lifetime = torch.zeros(len(self.envs), dtype=torch.int64, device=self.device)
        self.done = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)
        self.info = dict(lifetime=self.lifetime)

        self.act_dim = self.action_scale.size(-1)
        self.ob_dim = self.observe().size(-1)
        self.rew_dim = self.reward().size(-1)

        for i in range(self.gym.get_actor_count(self.envs[0])):
            rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], i)
            print("Links", sorted(rigid_body.items(), key=lambda x:x[1]), len(rigid_body))
            dof = self.gym.get_actor_dof_dict(self.envs[0], i)
            print("Joints", sorted(dof.items(), key=lambda x:x[1]), len(dof))

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)

    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[self.UP_AXIS] = val
        return base_vector
    
    def setup_sim_params(self, physx_params=dict()):
        p = gymapi.SimParams()
        p.dt = self.step_time/self.frameskip
        p.substeps = self.substeps
        p.up_axis = gymapi.UP_AXIS_Z if self.UP_AXIS == 2 else gymapi.UP_AXIS_Y
        p.gravity = gymapi.Vec3(*self.vector_up(-9.81))
        p.num_client_threads = 0
        p.physx.num_threads = 4
        p.physx.solver_type = 1
        p.physx.num_subscenes = 4  # works only for CPU 
        p.physx.num_position_iterations = 4
        p.physx.num_velocity_iterations = 0
        p.physx.contact_offset = 0.01
        p.physx.rest_offset = 0.0
        p.physx.bounce_threshold_velocity = 0.2
        p.physx.max_depenetration_velocity = 10.0
        p.physx.default_buffer_size_multiplier = 5.0
        p.physx.max_gpu_contact_pairs = 8*1024*1024
        # FIXME IsaacGym Pr4 will provide unreliable results when collecting from all substeps
        p.physx.contact_collection = \
            gymapi.ContactCollection(gymapi.ContactCollection.CC_LAST_SUBSTEP) 
        #gymapi.ContactCollection(gymapi.ContactCollection.CC_ALL_SUBSTEPS)
        for k, v in physx_params.items():
            setattr(p.physx, k, v)
        p.use_gpu_pipeline = True # force to enable GPU
        p.physx.use_gpu = True
        return p

    def add_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(*self.vector_up(1.0))
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def add_actor(self, env, i):
        pass

    def create_envs(self, n: int, actuate_all_dofs=True, asset_options=dict()):
        if self.control_mode == "position":
            control_mode = gymapi.DOF_MODE_POS
        elif self.control_mode == "torque":
            control_mode = gymapi.DOF_MODE_EFFORT
        else:
            control_mode = gymapi.DOF_MODE_NONE

        envs, actors = [], []
        env_spacing = 3
        asset_opt = gymapi.AssetOptions()
        
        asset_opt.angular_damping = 0.01
        asset_opt.max_angular_velocity = 100.0
        asset_opt.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        for k, v in asset_options.items():
            setattr(asset_opt, k, v)
        actor_asset = self.gym.load_asset(self.sim,
            os.path.abspath(os.path.dirname(self.character_model)),
            os.path.basename(self.character_model),
            asset_opt)

        spacing_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        spacing_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        n_envs_per_row = int(n**0.5)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(0.89))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        if actuate_all_dofs:
            self.actuated_dof = [i for i in range(self.gym.get_asset_dof_count(actor_asset))]
        else:
            actuators = []
            for i in range(self.gym.get_asset_actuator_count(actor_asset)):
                name = self.gym.get_asset_actuator_joint_name(actor_asset, i)
                actuators.append(self.gym.find_asset_dof_index(actor_asset, name))
                if actuators[-1] == -1:
                    raise ValueError("Failed to find joint with name {}".format(name))
            self.actuated_dof = sorted(actuators) if len(actuators) else None

        for i in range(n):
            env = self.gym.create_env(self.sim, spacing_lower, spacing_upper, n_envs_per_row)
            actor = self.gym.create_actor(env, actor_asset, start_pose, "actor", i, -1, 0)
            self.add_actor(env, i)
            envs.append(env)
            actors.append(actor)
            
            dof_prop = self.gym.get_asset_dof_properties(actor_asset)
            for k in range(len(dof_prop)):
                if k in self.actuated_dof:
                    dof_prop[k]["driveMode"] = control_mode
                else:
                    dof_prop[k]["stiffness"] = 0
                    dof_prop[k]["damping"] = 0
            self.gym.set_actor_dof_properties(env, actor, dof_prop)
        return envs, actors

    def render(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(*self.vector_up(self.camera_pos[2], 
            [base_pos[0]+self.camera_pos[0], base_pos[1]+self.camera_pos[1], base_pos[2]+self.camera_pos[1]]))
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "TOGGLE_CAMERA_FOLLOWING")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "TOGGLE_PAUSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "SINGLE_STEP_ADVANCE")
    
    def update_viewer(self):
        self.gym.poll_viewer_events(self.viewer)
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "QUIT" and event.value > 0:
                exit()
            if event.action == "TOGGLE_CAMERA_FOLLOWING" and event.value > 0:
                self.camera_following = not self.camera_following
            if event.action == "TOGGLE_PAUSE" and event.value > 0:
                self.viewer_pause = not self.viewer_pause
            if event.action == "SINGLE_STEP_ADVANCE" and event.value > 0:
                self.viewer_advance = not self.viewer_advance
        if self.camera_following: self.update_camera()
        self.gym.step_graphics(self.sim)

    def update_camera(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, self.envs[tar_env])
        dx, dy = cam_trans.p.x - self.cam_target.x, cam_trans.p.y - self.cam_target.y
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(base_pos[0]+dx, base_pos[1]+dy, cam_trans.p.z)
        self.cam_target = gymapi.Vec3(base_pos[0], base_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)

    def refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def create_tensors(self):
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(root_tensor)
        self.root_tensor = root_tensor.view(len(self.envs), -1, 13)

        num_links = self.gym.get_env_rigid_body_count(self.envs[0])
        link_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        link_tensor = gymtorch.wrap_tensor(link_tensor)
        self.link_tensor = link_tensor.view(len(self.envs), num_links, -1)

        num_dof = self.gym.get_env_dof_count(self.envs[0])
        joint_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        joint_tensor = gymtorch.wrap_tensor(joint_tensor)
        # IsaacGym Preview 3 supports fix, revolute and prismatic (1d) joints only
        self.joint_tensor = joint_tensor.view(len(self.envs), num_dof, -1)  # n_envs x n_dof x 2

        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.contact_force_tensor = contact_force_tensor.view(len(self.envs), -1, 3)

        if len(self.actuated_dof) == self.joint_tensor.size(1):
            self.action_tensor = None
        else:
            self.action_tensor = torch.zeros_like(self.joint_tensor[..., 0])

    def setup_action_normalizer(self):
        action_lower, action_upper = [], []
        action_scale = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            actor = self.gym.get_actor_handle(self.envs[0], i)
            dof_prop = self.gym.get_actor_dof_properties(self.envs[0], actor)
            # FIXME actuated dof setting only supports the primary actor
            if len(dof_prop) < 1: continue
            if self.control_mode == "torque":
                action_lower.extend([-dof_prop["effort"][j] for j in self.actuated_dof])
                action_upper.extend([dof_prop["effort"][j] for j in self.actuated_dof])
                action_scale.append(1)
            else: # self.control_mode == "position":
                action_lower.extend([min(dof_prop["lower"][j], dof_prop["upper"][j]) for j in self.actuated_dof])
                action_upper.extend([max(dof_prop["lower"][j], dof_prop["upper"][j]) for j in self.actuated_dof])
                action_scale.append(2)
        action_offset = 0.5 * np.add(action_upper, action_lower)
        action_scale *= 0.5 * np.subtract(action_upper, action_lower)
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=self.device)

    def process_actions(self, actions):
        a = actions*self.action_scale + self.action_offset
        if self.action_tensor is None:
            return a
        self.action_tensor[:, self.actuated_dof] = a
        return self.action_tensor

    def reset(self):
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime)
        self.request_quit = False
        self.obs = None

    def reset_done(self):
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)
                if len(env_ids) == len(self.envs) or self.obs is None:
                    self.obs = self.observe()
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info
    
    def reset_envs(self, env_ids):
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)

        self.root_tensor[env_ids] = ref_root_tensor
        self.link_tensor[env_ids] = ref_link_tensor
        self.joint_tensor[env_ids] = ref_joint_tensor

        actor_ids = self.actor_ids[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            actor_ids, n_actor_ids
        )
        actor_ids = self.actor_ids_having_dofs[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor),
            actor_ids, n_actor_ids
        )

        self.lifetime[env_ids] = 0

    def do_simulation(self):
        for _ in range(self.frameskip):
            self.gym.simulate(self.sim)
        self.simulation_step += 1

    def step(self, actions):
        if not self.viewer_pause or self.viewer_advance:
            self.apply_actions(actions)
            self.do_simulation()
            self.refresh_tensors()
            self.lifetime += 1
            if self.viewer is not None:
                self.gym.fetch_results(self.sim, True)
                self.viewer_advance = False

        if self.viewer is not None:
            self.update_viewer()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)    # sync to simulation dt

        rewards = self.reward()
        terminate = self.termination_check()                    # N
        if self.viewer_pause:
            overtime = None
        else:
            overtime = self.overtime_check()
        if torch.is_tensor(overtime):
            self.done = torch.logical_or(overtime, terminate)
        else:
            self.done = terminate
        self.info["terminate"] = terminate
        self.obs = self.observe()
        self.request_quit = False if self.viewer is None else self.gym.query_viewer_has_closed(self.viewer)
        return self.obs, rewards, self.done, self.info

    def apply_actions(self, actions):
        actions = self.process_actions(actions)
        if self.control_mode == "position":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_position_target_tensor(self.sim, actions)
        elif self.control_mode == "torque":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, actions)
        else:
            actions = torch.stack((actions, torch.zeros_like(actions)), -1)
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_state_tensor(self.sim, actions)

    def init_state(self, env_ids):
        pass
    
    def observe(self, env_ids=None):
        pass
    
    def overtime_check(self):
        if self.episode_length:
            if callable(self.episode_length):
                return self.lifetime >= self.episode_length(self.simulation_step)
            return self.lifetime >= self.episode_length
        return None

    def termination_check(self):
        return torch.zeros(len(self.envs), dtype=torch.bool, device=self.device)

    def reward(self):
        return torch.ones((len(self.envs), 0), dtype=torch.float32, device=self.device)


from ref_motion import ReferenceMotion
import numpy as np


class ICCGANHumanoid(Env):

    CHARACTER_MODEL = "assets/humanoid.xml"
    CONTROLLABLE_LINKS = ["torso", "head", 
        "right_upper_arm", "right_lower_arm",
        "left_upper_arm", "left_lower_arm", 
        "right_thigh", "right_shin", "right_foot",
        "left_thigh", "left_shin", "left_foot"]
    DOFS =  [3, 3, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3]
    CONTACTABLE_LINKS = ["right_foot", "left_foot"]
    UP_AXIS = 2

    GOAL_DIM = 0
    GOAL_REWARD_WEIGHT = None
    ENABLE_GOAL_TIMER = False
    GOAL_TENSOR_DIM = None

    OB_HORIZON = 4
    KEY_LINKS = None    # All links
    PARENT_LINK = None  # root link


    def __init__(self, *args,
        motion_file: str,
        discriminators: Dict[str, DiscriminatorConfig],
    **kwargs):
        contactable_links = parse_kwarg(kwargs, "contactable_links", self.CONTACTABLE_LINKS)
        controllable_links = parse_kwarg(kwargs, "controllable_links", self.CONTROLLABLE_LINKS)
        dofs = parse_kwarg(kwargs, "dofs", self.DOFS)
        goal_reward_weight = parse_kwarg(kwargs, "goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.enable_goal_timer = parse_kwarg(kwargs, "enable_goal_timer", self.ENABLE_GOAL_TIMER)
        self.goal_tensor_dim = parse_kwarg(kwargs, "goal_tensor_dim", self.GOAL_TENSOR_DIM)
        self.ob_horizon = parse_kwarg(kwargs, "ob_horizon", self.OB_HORIZON)
        self.key_links = parse_kwarg(kwargs, "key_links", self.KEY_LINKS)
        self.parent_link = parse_kwarg(kwargs, "parent_link", self.PARENT_LINK)
        super().__init__(*args, **kwargs)

        n_envs = len(self.envs)
        n_links = self.char_link_tensor.size(1)
        n_dofs = self.char_joint_tensor.size(1)
        
        controllable_links = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link)
            for link in controllable_links]

        if contactable_links is None:
            self.contactable_links = None
        elif contactable_links:
            contact = np.zeros((n_envs, n_links), dtype=bool)
            for link in contactable_links:
                lid = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link)
                assert(lid >= 0), "Unrecognized contactable link {}".format(link)
                contact[:, lid] = True
            self.contactable_links = torch.tensor(contact).to(self.contact_force_tensor.device)
        else:
            self.contactable_links = False

        init_pose = motion_file
        self.ref_motion = ReferenceMotion(motion_file=init_pose, character_model=self.character_model,
            key_links=np.arange(n_links), controllable_links=controllable_links, dofs=dofs,
            device=self.device
        )

        ref_motion = {init_pose: self.ref_motion}
        disc_ref_motion = dict()
        for id, config in discriminators.items():
            m = init_pose if config.motion_file is None else config.motion_file
            if m not in ref_motion:
                ref_motion[m] = ReferenceMotion(motion_file=m, character_model=self.character_model,
                    key_links=np.arange(n_links), controllable_links=controllable_links, dofs=dofs,
                    device=self.device
                )
            key = (ref_motion[m], config.replay_speed)
            if config.ob_horizon is None:
                config.ob_horizon = self.ob_horizon+1
            if key not in disc_ref_motion: disc_ref_motion[key] = [0, []]
            disc_ref_motion[key][0] = max(disc_ref_motion[key][0], config.ob_horizon)
            disc_ref_motion[key][1].append(id)

        if goal_reward_weight is not None:
            reward_weights = torch.empty((len(self.envs), self.rew_dim), dtype=torch.float32, device=self.device)
            if not hasattr(goal_reward_weight, "__len__"):
                goal_reward_weight = [goal_reward_weight]
            assert(self.rew_dim == len(goal_reward_weight))
            for i, w in zip(range(self.rew_dim), goal_reward_weight):
                reward_weights[:, i] = w
        elif self.rew_dim:
            goal_reward_weight = []
            assert(self.rew_dim == len(goal_reward_weight))

        n_comp = len(discriminators) + self.rew_dim
        if n_comp > 1:
            self.reward_weights = torch.zeros((n_envs, n_comp), dtype=torch.float32, device=self.device)
            weights = [disc.weight for _, disc in discriminators.items() if disc.weight is not None]
            total_weights = sum(weights) if weights else 0
            assert(total_weights <= 1), "Discriminator weights must not be greater than 1."
            n_unassigned = len(discriminators) - len(weights)
            rem = 1 - total_weights
            for disc in discriminators.values():
                if disc.weight is None:
                    disc.weight = rem / n_unassigned
                elif n_unassigned == 0:
                    disc.weight /= total_weights
        else:
            self.reward_weights = None

        self.discriminators = dict()
        max_ob_horizon = self.ob_horizon+1
        for i, (id, config) in enumerate(discriminators.items()):
            key_links = None if config.key_links is None else sorted([
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link) for link in config.key_links
            ])
            parent_link = None if config.parent_link is None else \
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], config.parent_link)

            assert(key_links is None or all(lid >= 0 for lid in key_links))
            assert(parent_link is None or parent_link >= 0)
            
            self.discriminators[id] = DiscriminatorProperty(
                name = id,
                key_links = key_links,
                parent_link = parent_link,
                replay_speed = config.replay_speed,
                ob_horizon = config.ob_horizon,
                id=i
            )
            if self.reward_weights is not None:
                self.reward_weights[:, i] = config.weight
            max_ob_horizon = max(max_ob_horizon, config.ob_horizon)

        if max_ob_horizon != self.state_hist.size(0):
            self.state_hist = torch.empty((max_ob_horizon, *self.state_hist.shape[1:]),
                dtype=self.root_tensor.dtype, device=self.device)
        self.disc_ref_motion = [
            (ref_motion, replay_speed, max_ob_horizon, [self.discriminators[id] for id in disc_ids])
            for (ref_motion, replay_speed), (max_ob_horizon, disc_ids) in disc_ref_motion.items()
        ]

        if self.rew_dim > 0:
            if self.rew_dim > 1:
                self.reward_weights *= (1-reward_weights.sum(dim=-1, keepdim=True))
            else:
                self.reward_weights *= (1-reward_weights)
            self.reward_weights[:, -self.rew_dim:] = reward_weights
            
        self.info["ob_seq_lens"] = torch.zeros_like(self.lifetime)  # dummy result
        self.info["disc_obs"] = self.observe_disc(self.state_hist)  # dummy result
        self.info["disc_obs_expert"] = self.info["disc_obs"]        # dummy result
        self.goal_dim = self.GOAL_DIM
        self.state_dim = (self.ob_dim-self.goal_dim)//self.ob_horizon
        self.disc_dim = {
            name: ob.size(-1)
            for name, ob in self.info["disc_obs"].items()
        }
    
    def reset_done(self):
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        info["reward_weights"] = self.reward_weights
        return obs, info
    
    def reset(self):
        if self.goal_tensor is not None:
            self.goal_tensor.zero_()
            if self.goal_timer is not None: self.goal_timer.zero_()
        super().reset()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_goal(env_ids)
        
    def reset_goal(self, env_ids):
        pass
    
    def step(self, actions):
        self.state_hist[:-1] = self.state_hist[1:].clone()
        obs, rews, dones, info = super().step(actions)
        info["disc_obs"] = self.observe_disc(self.state_hist)
        info["disc_obs_expert"], info["disc_seq_len"] = self.fetch_real_samples()
        return obs, rews, dones, info

    def overtime_check(self):
        if self.goal_timer is not None:
            self.goal_timer -= 1
            env_ids = torch.nonzero(self.goal_timer <= 0).view(-1)
            if len(env_ids) > 0: self.reset_goal(env_ids)
        return super().overtime_check()

    def termination_check(self):
        if self.contactable_links is None:
            return torch.zeros_like(self.done)
        masked_contact = self.char_contact_force_tensor.clone()
        if self.contactable_links is not False:
            masked_contact[self.contactable_links] = 0          # N x n_links x 3

        contacted = torch.any(masked_contact > 1., dim=-1)  # N x n_links
        too_low = self.link_pos[..., self.UP_AXIS] < 0.15    # N x n_links

        terminate = torch.any(torch.logical_and(contacted, too_low), -1)    # N x
        terminate *= (self.lifetime > 1)
        return terminate

    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        return self.ref_motion.state(motion_ids, motion_times)
    
    def create_tensors(self):
        super().create_tensors()
        n_dofs = self.gym.get_actor_dof_count(self.envs[0], 0)
        n_links = self.gym.get_actor_rigid_body_count(self.envs[0], 0)
        self.root_pos, self.root_orient = self.root_tensor[:, 0, :3], self.root_tensor[:, 0, 3:7]
        self.root_lin_vel, self.root_ang_vel = self.root_tensor[:, 0, 7:10], self.root_tensor[:, 0, 10:13]
        self.char_root_tensor = self.root_tensor[:, 0]
        if self.link_tensor.size(1) > n_links:
            self.link_pos, self.link_orient = self.link_tensor[:, :n_links, :3], self.link_tensor[:, :n_links, 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[:, :n_links, 7:10], self.link_tensor[:, :n_links, 10:13]
            self.char_link_tensor = self.link_tensor[:, :n_links]
        else:
            self.link_pos, self.link_orient = self.link_tensor[..., :3], self.link_tensor[..., 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[..., 7:10], self.link_tensor[..., 10:13]
            self.char_link_tensor = self.link_tensor
        if self.joint_tensor.size(1) > n_dofs:
            self.joint_pos, self.joint_vel = self.joint_tensor[:, :n_dofs, 0], self.joint_tensor[:, :n_dofs, 1]
            self.char_joint_tensor = self.joint_tensor[:, :n_dofs]
        else:
            self.joint_pos, self.joint_vel = self.joint_tensor[..., 0], self.joint_tensor[..., 1]
            self.char_joint_tensor = self.joint_tensor
        
        self.char_contact_force_tensor = self.contact_force_tensor[:, :n_links]
    
        self.state_hist = torch.empty((self.ob_horizon+1, len(self.envs), 13 + n_links*13),
            dtype=self.root_tensor.dtype, device=self.device)

        self.key_links = None if self.key_links is None else sorted([
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link) for link in self.key_links
        ])
        
        if self.goal_tensor_dim:
            try:
                self.goal_tensor = [
                    torch.zeros((len(self.envs), dim), dtype=self.root_tensor.dtype, device=self.device)
                    for dim in self.goal_tensor_dim
                ]
            except TypeError:
                self.goal_tensor = torch.zeros((len(self.envs), self.goal_tensor_dim), dtype=self.root_tensor.dtype, device=self.device)
        else:
            self.goal_tensor = None
        self.goal_timer = torch.zeros((len(self.envs), ), dtype=torch.int32, device=self.device) if self.enable_goal_timer else None

    def observe(self, env_ids=None):
        self.ob_seq_lens = self.lifetime+1 #(self.lifetime+1).clip(max=self.state_hist.size(0)-1)
        n_envs = len(self.envs)
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[-1] = torch.cat((
                self.char_root_tensor, self.char_link_tensor.view(n_envs, -1)
            ), -1)
            env_ids = None
        else:
            n_envs = len(env_ids)
            self.state_hist[-1, env_ids] = torch.cat((
                self.char_root_tensor[env_ids], self.char_link_tensor[env_ids].view(n_envs, -1)
            ), -1)
        return self._observe(env_ids)
    
    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens, self.key_links, self.parent_link
            ).flatten(start_dim=1)
        else:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids], self.key_links, self.parent_link
            ).flatten(start_dim=1)
    
    def observe_disc(self, state):
        seq_len = self.info["ob_seq_lens"]+1
        res = dict()
        if torch.is_tensor(state):
            # fake
            for id, disc in self.discriminators.items():
                res[id] = observe_iccgan(state[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link, include_velocity=False)
            return res
        else:
            # real
            seq_len_ = dict()
            for disc_name, s in state.items():
                disc = self.discriminators[disc_name]
                res[disc_name] = observe_iccgan(s[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link, include_velocity=False)
                seq_len_[disc_name] = seq_len
            return res, seq_len_

    def fetch_real_samples(self):
        n_inst = len(self.envs)

        samples = dict()
        for ref_motion, replay_speed, ob_horizon, discs in self.disc_ref_motion:
            dt = self.step_time
            if replay_speed is not None:
                dt /= replay_speed(n_inst)
            motion_ids, motion_times0 = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
            motion_ids = np.tile(motion_ids, ob_horizon)
            motion_times = np.concatenate((motion_times0, *[motion_times0+dt*i for i in range(1, ob_horizon)]))
            root_tensor, link_tensor, joint_tensor = ref_motion.state(motion_ids, motion_times)
            real = torch.cat((
                root_tensor, link_tensor.view(root_tensor.size(0), -1)
            ), -1).view(ob_horizon, n_inst, -1)

            for d in discs: samples[d.name] = real
        return self.observe_disc(samples)


@torch.jit.script
def observe_iccgan(state_hist: torch.Tensor, seq_len: torch.Tensor,
    key_links: Optional[List[int]]=None, parent_link: Optional[int]=None,
    include_velocity: bool=True
):
    # state_hist: L x N x (1+N_links) x 13

    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    root_tensor = state_hist[..., :13]
    link_tensor = state_hist[...,13:].view(n_hist, n_inst, -1, 13)
    if key_links is None:
        link_pos, link_orient = link_tensor[...,:3], link_tensor[...,3:7]
    else:
        link_pos, link_orient = link_tensor[:,:,key_links,:3], link_tensor[:,:,key_links,3:7]

    if parent_link is None:
        origin = root_tensor[-1,:, :3]              # N x 3
        orient = root_tensor[-1,:,3:7]              # N x 4

        heading = heading_zup(orient)               # N
        up_dir = torch.zeros_like(origin)
        up_dir[..., UP_AXIS] = 1                    # N x 3
        orient_inv = axang2quat(up_dir, -heading)   # N x 4
        orient_inv = orient_inv.view(1, -1, 1, 4)   # 1 x N x 1 x 4

        origin = origin.clone()
        origin[..., UP_AXIS] = 0                    # N x 3
        origin.unsqueeze_(-2)                       # N x 1 x 3
    else:
        origin = link_tensor[:,:, parent_link, :3]  # L x N x 3
        orient = link_tensor[:,:, parent_link,3:7]  # L x N x 4
        orient_inv = quatconj(orient)               # L x N x 4
        orient_inv.unsqueeze_(-2)                   # L x N x 1 x 4
        origin = origin.unsqueeze(-2)               # L x N x 1 x 3

    ob_link_pos = link_pos - origin                                     # L x N x n_links x 3 
    ob_link_pos = rotatepoint(orient_inv, ob_link_pos)
    ob_link_orient = quatmultiply(orient_inv, link_orient)              # L x N x n_links x 4

    if include_velocity:
        if key_links is None:
            link_lin_vel, link_ang_vel = link_tensor[...,7:10], link_tensor[...,10:13]
        else:
            link_lin_vel, link_ang_vel = link_tensor[:,:,key_links,7:10], link_tensor[:,:,key_links,10:13]
        ob_link_lin_vel = rotatepoint(orient_inv, link_lin_vel)         # L x N x n_links x 3
        ob_link_ang_vel = rotatepoint(orient_inv, link_ang_vel)         # L x N x n_links x 3
        ob = torch.cat((ob_link_pos, ob_link_orient,
            ob_link_lin_vel, ob_link_ang_vel), -1)                      # L x N x n_links x 13
    else:
        ob = torch.cat((ob_link_pos, ob_link_orient), -1)               # L x N x n_links x 7
    ob = ob.view(n_hist, n_inst, -1)                                    # L x N x (n_links x 7 or 13)

    ob1 = ob.permute(1, 0, 2)                                           # N x L x (n_links x 7 or 13)
    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    return ob2




class ICCGANHumanoidTarget(ICCGANHumanoid):

    GOAL_REWARD_WEIGHT = 0.5
    GOAL_DIM = 4
    GOAL_TENSOR_DIM = 3
    ENABLE_GOAL_TIMER = True

    GOAL_RADIUS = 0.5
    SP_LOWER_BOUND = 1.2
    SP_UPPER_BOUND = 1.5
    GOAL_TIMER_RANGE = 90, 150
    GOAL_SP_MEAN = 1
    GOAL_SP_STD = 0.25
    GOAL_SP_MIN = 0
    GOAL_SP_MAX = 1.25

    SHARP_TURN_RATE = 1

    def __init__(self, *args, **kwargs):
        self.goal_radius = parse_kwarg(kwargs, "goal_radius", self.GOAL_RADIUS)
        self.sharp_turn_rate = parse_kwarg(kwargs, "sharp_turn_rate", self.SHARP_TURN_RATE)
        self.sp_lower_bound = parse_kwarg(kwargs, "sp_lower_bound", self.SP_LOWER_BOUND)
        self.sp_upper_bound = parse_kwarg(kwargs, "sp_upper_bound", self.SP_UPPER_BOUND)
        self.goal_timer_range = parse_kwarg(kwargs, "goal_timer_range", self.GOAL_TIMER_RANGE)
        self.goal_sp_mean = parse_kwarg(kwargs, "goal_sp_mean", self.GOAL_SP_MEAN)
        self.goal_sp_std = parse_kwarg(kwargs, "goal_sp_std", self.GOAL_SP_STD)
        self.goal_sp_min = parse_kwarg(kwargs, "goal_sp_min", self.GOAL_SP_MIN)
        self.goal_sp_max = parse_kwarg(kwargs, "goal_sp_max", self.GOAL_SP_MAX)
        super().__init__(*args, **kwargs)

    def update_viewer(self):
        super().update_viewer()
        self.gym.clear_lines(self.viewer)
        n_lines = 10
        tar_x = self.goal_tensor[:, 0].cpu().numpy()

        p = self.root_pos.cpu().numpy()
        zero = np.zeros_like(tar_x)+0.05
        tar_y = self.goal_tensor[:, 1].cpu().numpy()
        lines = np.stack([
            np.stack((p[:,0], p[:,1], zero+0.01*i, tar_x, tar_y, zero), -1)
        for i in range(n_lines)], -2)
        for e, l in zip(self.envs, lines):
            self.gym.add_lines(self.viewer, e, n_lines, l, [[1., 0., 0.] for _ in range(n_lines)])
        n_lines = 10
        target_pos = self.goal_tensor.cpu().numpy()
        lines = np.stack([
            np.stack((
                target_pos[:, 0], target_pos[:, 1], zero,
                target_pos[:, 0]+self.goal_radius*np.cos(2*np.pi/n_lines*i), 
                target_pos[:, 1]+self.goal_radius*np.sin(2*np.pi/n_lines*i),
                zero
            ), -1)
        for i in range(n_lines)], -2)
        for e, l in zip(self.envs, lines):
            self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 0., 1.] for _ in range(n_lines)])
    
    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan_target(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens,
                self.goal_tensor, self.goal_timer, sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )
        else:
            return observe_iccgan_target(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.goal_tensor[env_ids], self.goal_timer[env_ids], sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )

    def reset_goal(self, env_ids, goal_tensor=None, goal_timer=None):
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer
        
        n_envs = len(env_ids)
        all_envs = n_envs == len(self.envs)
        root_orient = self.root_orient if all_envs else self.root_orient[env_ids]

        small_turn = torch.rand(n_envs, device=self.device) > self.sharp_turn_rate
        large_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(2*np.pi)
        small_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).sub_(0.5).mul_(2*(np.pi/3))

        heading = heading_zup(root_orient)
        small_angle += heading
        theta = torch.where(small_turn, small_angle, large_angle)

        timer = torch.randint(self.goal_timer_range[0], self.goal_timer_range[1], (n_envs,), dtype=self.goal_timer.dtype, device=self.device)
        if self.goal_sp_min == self.goal_sp_max:
            vel = self.goal_sp_min
        elif self.goal_sp_std == 0:
            vel = self.goal_sp_mean
        else:
            vel = torch.nn.init.trunc_normal_(torch.empty(n_envs, dtype=torch.float32, device=self.device), mean=self.goal_sp_mean, std=self.goal_sp_std, a=self.goal_sp_min, b=self.goal_sp_max)
        
        dist = vel*timer*self.step_time
        dx = dist*torch.cos(theta)
        dy = dist*torch.sin(theta)

        if all_envs:
            self.init_dist = dist
            goal_timer.copy_(timer)
            goal_tensor[:,0] = self.root_pos[:,0] + dx
            goal_tensor[:,1] = self.root_pos[:,1] + dy
        else:
            self.init_dist[env_ids] = dist
            goal_timer[env_ids] = timer
            goal_tensor[env_ids,0] = self.root_pos[env_ids,0] + dx
            goal_tensor[env_ids,1] = self.root_pos[env_ids,1] + dy

    def reward(self, goal_tensor=None, goal_timer=None):
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer

        p = self.root_pos
        p_ = self.state_hist[-1][:, :3]

        dp_ = goal_tensor - p_
        dp_[:, self.UP_AXIS] = 0
        dist_ = torch.linalg.norm(dp_, ord=2, dim=-1)
        v_ = dp_.div_(goal_timer.unsqueeze(-1)*self.step_time)

        v_mag = torch.linalg.norm(v_, ord=2, dim=-1)
        sp_ = (dist_/self.step_time).clip_(max=v_mag.clip(min=self.sp_lower_bound, max=self.sp_upper_bound))
        v_ *= (sp_/v_mag).unsqueeze_(-1)

        dp = p - p_
        dp[:, self.UP_AXIS] = 0
        v = dp / self.step_time
        r = (v - v_).square_().sum(1).mul_(-3/(sp_*sp_)).exp_()

        dp = goal_tensor - p
        dp[:, self.UP_AXIS] = 0
        dist = torch.linalg.norm(dp, ord=2, dim=-1)
        self.near = dist < self.goal_radius

        r[self.near] = 1
        
        if self.viewer is not None:
           self.goal_timer[self.near] = self.goal_timer[self.near].clip(max=20)
        
        return r.unsqueeze_(-1)

    def termination_check(self, goal_tensor=None):
        if goal_tensor is None: goal_tensor = self.goal_tensor
        fall = super().termination_check()
        dp = goal_tensor - self.root_pos
        dp[:, self.UP_AXIS] = 0
        dist = dp.square_().sum(-1).sqrt_()
        too_far = dist-self.init_dist > 3
        return torch.logical_or(fall, too_far)


@torch.jit.script
def observe_iccgan_target(state_hist: torch.Tensor, seq_len: torch.Tensor,
    target_tensor: torch.Tensor, timer: torch.Tensor,
    sp_upper_bound: float, fps: int
):
    ob = observe_iccgan(state_hist, seq_len)

    root_pos = state_hist[-1, :, :3]
    root_orient = state_hist[-1, :, 3:7]

    dp = target_tensor - root_pos
    x = dp[:, 0]
    y = dp[:, 1]
    heading_inv = -heading_zup(root_orient)
    c = torch.cos(heading_inv)
    s = torch.sin(heading_inv)
    x, y = c*x-s*y, s*x+c*y

    dist = (x*x + y*y).sqrt_()
    sp = dist.mul(fps/timer)

    too_close = dist < 1e-5
    x = torch.where(too_close, x, x/dist)
    y = torch.where(too_close, y, y/dist)
    sp.clip_(max=sp_upper_bound)
    dist.div_(3).clip_(max=1.5)

    return torch.cat((ob.flatten(start_dim=1), x.unsqueeze_(-1), y.unsqueeze_(-1), sp.unsqueeze_(-1), dist.unsqueeze_(-1)), -1)



class ICCGANHumanoidTargetAiming(ICCGANHumanoidTarget):
    
    GOAL_REWARD_WEIGHT = 0.25, 0.25
    GOAL_DIM = 4+3
    GOAL_TENSOR_DIM = 3+4

    def create_tensors(self):
        super().create_tensors()
        self.hand_link = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "right_hand")
        self.lower_arm_link = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "right_lower_arm")

        self.aiming_start_link = self.lower_arm_link
        self.aiming_end_link = self.hand_link

        self.x_dir = torch.zeros_like(self.root_pos)
        self.x_dir[..., 0] = 1
        self.reverse_rotation = torch.zeros_like(self.root_orient)
        self.reverse_rotation[..., self.UP_AXIS] = 1

    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan_target_aiming(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens,
                self.goal_tensor, self.goal_timer,
                sp_upper_bound=self.sp_upper_bound, goal_radius=self.goal_radius, fps=self.fps
            )
        else:
            return observe_iccgan_target_aiming(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.goal_tensor[env_ids], self.goal_timer[env_ids],
                sp_upper_bound=self.sp_upper_bound, goal_radius=self.goal_radius, fps=self.fps
            )

    def update_viewer(self):
        super().update_viewer()

        target_tensor = self.goal_tensor[:, :3]
        aiming_tensor = self.goal_tensor[:, 3:]

        target_dir = target_tensor - self.root_pos
        target_dir[..., self.UP_AXIS] = 0
        dist = torch.linalg.norm(target_dir, ord=2, dim=-1, keepdim=True)
        not_near = (dist > self.goal_radius).squeeze_(-1)
        dist = dist[not_near]

        if dist.nelement() < 1: return

        target_dir = target_dir[not_near]
        target_dir.div_(dist)
        link_pos = self.link_pos[not_near]

        x_dir = self.x_dir[:target_dir.size(0)]
        q = quatdiff_normalized(x_dir, target_dir)
        # ensure 180 degree rotation is around the up axis
        q = torch.where(target_dir[:, :1] < -0.99999,
            self.reverse_rotation, q)

        aiming_dir = rotatepoint(quatmultiply(q, aiming_tensor), x_dir)

        start = link_pos[:, self.aiming_start_link]
        end = start + aiming_dir

        start = start.cpu().numpy()
        end = end.cpu().numpy()
        not_near = torch.nonzero(not_near).view(-1).cpu().numpy()
        n_lines = 10
        lines = np.stack([
            np.stack((start[:,0], start[:,1], start[:,2]+0.005*i, end[:, 0], end[:, 1], end[:,2]+0.005*i), -1)
        for i in range(-n_lines//2, n_lines//2)], -2)
        for i, l in zip(not_near, lines):
            e = self.envs[i]
            self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 1., 0.] for _ in range(n_lines)])
            
    def reset_goal(self, env_ids):
        super().reset_goal(env_ids, self.goal_tensor[:, :3])
        self.reset_aiming_goal(env_ids)
    
    def reset_aiming_goal(self, env_ids):
        n_envs = len(env_ids)
        elev = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(-np.pi/6)
        azim = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(np.pi/4)
        if self.viewer is not None: azim.add_(0.3)

        elev /= 2
        azim /= 2
        cp = torch.cos(elev) # y
        sp = torch.sin(elev)
        cy = torch.cos(azim) # z
        sy = torch.sin(azim)

        w = cp*cy  # cr*cp*cy + sr*sp*sy
        x = -sp*sy # sr*cp*cy - cr*sp*sy
        y = sp*cy  # cr*sp*cy + sr*cp*sy
        z = cp*sy  # cr*cp*sy - sr*sp*cy
        
        if n_envs == len(self.envs):
            self.goal_tensor[:, 3] = x
            self.goal_tensor[:, 4] = y
            self.goal_tensor[:, 5] = z 
            self.goal_tensor[:, 6] = w
        else:
            self.goal_tensor[env_ids, 3] = x
            self.goal_tensor[env_ids, 4] = y
            self.goal_tensor[env_ids, 5] = z
            self.goal_tensor[env_ids, 6] = w

    def reward(self):
        target_tensor = self.goal_tensor[:, :3]
        aiming_tensor = self.goal_tensor[:, 3:]
        
        target_rew = super().reward(target_tensor)

        dp = target_tensor - self.root_pos
        dp[..., self.UP_AXIS] = 0
        dist = torch.linalg.norm(dp, ord=2, dim=-1, keepdim=True)
        
        target_dir = dp / dist
        q0 = quatdiff_normalized(self.x_dir, target_dir)
        q = torch.where(target_dir[:, :1] < -0.99999,
            self.reverse_rotation, q0)

        aiming_dir = rotatepoint(quatmultiply(q, aiming_tensor), self.x_dir)

        hand_pos = self.link_pos[:, self.aiming_end_link]
        fore_arm_pos = self.link_pos[:, self.aiming_start_link]

        fore_arm_dir = hand_pos - fore_arm_pos
        arm_len = torch.linalg.norm(fore_arm_dir, ord=2, dim=-1, keepdim=True)
        fore_arm_dir.div_(arm_len)

        target_hand_pos = fore_arm_pos + arm_len * aiming_dir
        e = torch.linalg.norm(target_hand_pos.sub_(hand_pos), ord=2, dim=-1).div_(arm_len.squeeze_(-1))
        aiming_rew = e.mul_(-2).exp_()
         
        rest_rew = fore_arm_dir[..., self.UP_AXIS].div(0.8).clip_(min=0, max=1)
        
        aiming_rew = torch.where(self.near, rest_rew, aiming_rew).unsqueeze_(-1)

        r = torch.cat((target_rew, aiming_rew), -1)
        return r

    def termination_check(self):
        return super().termination_check(self.goal_tensor[:, :3])


@torch.jit.script
def observe_iccgan_target_aiming(state_hist: torch.Tensor, seq_len: torch.Tensor, 
    goal_tensor: torch.Tensor, timer: torch.Tensor,
    sp_upper_bound: float, goal_radius: float, fps: int
):
    UP_AXIS = 2

    target_tensor = goal_tensor[..., :3]
    aiming_tensor = goal_tensor[..., 3:]

    target_ob = observe_iccgan_target(state_hist, seq_len, target_tensor, timer, sp_upper_bound=sp_upper_bound, fps=fps)
    
    root_pos = state_hist[-1, :, :3]
    root_orient = state_hist[-1, :, 3:7]
    heading = heading_zup(root_orient)
    up_dir = torch.zeros_like(root_pos)
    up_dir[..., UP_AXIS] = 1
    orient_inv = axang2quat(up_dir, -heading)

    dp = target_tensor - root_pos
    dp[..., UP_AXIS] = 0
    dist = torch.linalg.norm(dp, ord=2, dim=-1, keepdim=True)
    
    x_dir = torch.zeros_like(dp)
    x_dir[..., 0] = 1
    target_dir = dp / dist
    q = quatdiff_normalized(x_dir, target_dir)

    # ensure 180 degree rotation is around the up axis
    reverse = torch.zeros_like(q)
    reverse[..., UP_AXIS] = 1
    q = torch.where(target_dir[:, :1] < -0.99999,
        reverse, q)

    aiming_dir = quatmultiply(q, aiming_tensor)
    aiming_dir = quatmultiply(q, aiming_tensor)
    aiming_dir = quatmultiply(orient_inv, aiming_dir)
    aiming_dir = rotatepoint(aiming_dir, x_dir)

    near = dist.squeeze_(-1) < goal_radius
    # aiming_dir[near] = 0 # not supported by script
    aiming_dir[near, 0] = 0
    aiming_dir[near, 1] = 0
    aiming_dir[near, 2] = 0
    
    return torch.cat((target_ob, aiming_dir), -1)



class ICCGANHumanoidJugglingTarget(ICCGANHumanoidTarget):
    
    GOAL_DIM = 4+19

    N_BALLS = 3
    BALL_RADIUS = 0.05
    BALL_MASS = 0.057
    BALL_PROPERTIES = dict(
        compliance=0.0,
        friction=0.0,
        restitution=0.0,
        rolling_friction=0.0,
        torsion_friction=0.0
    )
    THROW_INTERVAL = 20
    DWELL_TIME = 12

    def __init__(self, *args, **kwargs):
        self.n_balls = parse_kwarg(kwargs, "n_balls", self.N_BALLS)
        assert(self.n_balls > 1)
        self.throw_interval =  parse_kwarg(kwargs, "throw_interval", self.THROW_INTERVAL)
        self.dwell_time = parse_kwarg(kwargs, "dwell_time", self.DWELL_TIME)
        self.ball_radius = parse_kwarg(kwargs, "ball_radius", self.BALL_RADIUS)
        self.ball_mass = parse_kwarg(kwargs, "ball_mass", self.BALL_MASS)
        self.ball_properties = self.BALL_PROPERTIES.copy()
        if "ball_properties" in kwargs:
            self.ball_properties.update(kwargs["ball_properties"])
        super().__init__(*args, **kwargs)
    
    def create_envs(self, n: int):
        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.density = self.ball_mass / (4*np.pi*(self.ball_radius**3)/3)
        ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_asset_options)
        props = self.gym.get_asset_rigid_shape_properties(ball_asset)
        for k, v in self.BALL_PROPERTIES.items():
            setattr(props[0], k, v)
        self.gym.set_asset_rigid_shape_properties(ball_asset, props)
        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(0.0, 5.0, 0.0)
        ball_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.ball_asset = ball_asset
        self.ball_init_pose = ball_pose
        return super().create_envs(n)

    def add_actor(self, env, i):
        self.left_hand_link = self.gym.find_actor_rigid_body_handle(env, 0, "left_hand")
        self.right_hand_link = self.gym.find_actor_rigid_body_handle(env, 0, "right_hand")
        left_lower_arm_link = self.gym.find_actor_rigid_body_handle(env, 0, "left_lower_arm")
        right_lower_arm_link = self.gym.find_actor_rigid_body_handle(env, 0, "right_lower_arm")
        collison_mask = 2**self.n_balls - 1
        rb_shape = self.gym.get_actor_rigid_body_shape_indices(env, 0)
        rb_shape_props = self.gym.get_actor_rigid_shape_properties(env, 0)
        rb_shape_props[rb_shape[self.left_hand_link].start].filter = collison_mask
        rb_shape_props[rb_shape[self.right_hand_link].start].filter = collison_mask
        rb_shape_props[rb_shape[left_lower_arm_link].start].filter = collison_mask
        rb_shape_props[rb_shape[right_lower_arm_link].start].filter = collison_mask
        self.gym.set_actor_rigid_shape_properties(env, 0, rb_shape_props)
        for k in range(1, self.n_balls+1):
            self.gym.create_actor(env, self.ball_asset, self.ball_init_pose, "ball", i, -1, 0)
            r, g, b = (k//4)%2, (k//2)%2, k%2
            self.gym.set_rigid_body_color(env, k, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(r, g, b))
            rb_shape_props = self.gym.get_actor_rigid_shape_properties(env, k)
            rb_shape_props[0].filter = 1
            self.gym.set_actor_rigid_shape_properties(env, k, rb_shape_props)
        
        self.first_catch_frame = self.throw_interval-1
        self.last_catch_frame = 1

    def create_tensors(self):
        super().create_tensors()
        self.ball_root_tensor = self.root_tensor[:, 1:]
        self.ball_contact_force_tensor = self.contact_force_tensor[:, -2:]

        self.env_range = torch.arange(len(self.envs),
            dtype=torch.int64, device=self.ball_root_tensor.device)
        self.env_range_ = self.env_range.unsqueeze(-1)
        self.ball_index = torch.arange(self.ball_root_tensor.size(1),
            dtype=torch.int64, device=self.ball_root_tensor.device
            ).add_(self.link_tensor.size(1)-self.ball_root_tensor.size(1)
            ).unsqueeze_(0).repeat(len(self.envs), 1)
        self.ball_attach = torch.zeros((len(self.envs), self.ball_root_tensor.size(1)),
            dtype=torch.int64, device=self.ball_root_tensor.device)
        
        self.target_ball_release = torch.zeros((len(self.envs), ),
            dtype=torch.int64, device=self.ball_root_tensor.device)
        self.target_ball_left = torch.zeros_like(self.target_ball_release)
        self.target_ball_right = torch.ones_like(self.target_ball_release)

        self.timer_left = torch.empty_like(self.target_ball_release)
        self.timer_right = torch.empty_like(self.timer_left)

        self.v_out_left = torch.zeros_like(self.ball_root_tensor[:, 0, 0])
        self.v_out_right = torch.zeros_like(self.v_out_left)
        self.ball_h_out = torch.zeros_like(self.ball_root_tensor[:,:,self.UP_AXIS])
        self.dist2target_left = torch.zeros_like(self.v_out_left)
        self.dist2target_right = torch.zeros_like(self.dist2target_left)
        self.catachable_left = torch.zeros_like(self.dist2target_left, dtype=torch.bool)
        self.catachable_right = torch.zeros_like(self.dist2target_right, dtype=torch.bool)
        self.miss_left = torch.zeros_like(self.dist2target_left, dtype=torch.bool)
        self.miss_right = torch.zeros_like(self.dist2target_right, dtype=torch.bool)

        n_balls = self.ball_root_tensor.size(1)
        self.v_out_target = 0.5*9.81* ((self.throw_interval//2)*n_balls-self.dwell_time)*self.step_time


    def init_state(self, env_ids):
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = super().init_state(env_ids)

        n_balls = self.ball_attach.size(1)
        for i in range(n_balls):
            self.ball_attach[env_ids, i] = self.right_hand_link if i % 2 else self.left_hand_link
        ball_root_tensor = ref_link_tensor[(self.env_range_[:len(env_ids)], self.ball_attach[env_ids])]

        self.timer_left[env_ids] = self.throw_interval//2
        self.timer_right[env_ids] = self.throw_interval
        self.target_ball_release[env_ids] = 0
        self.target_ball_left[env_ids] = 0
        self.target_ball_right[env_ids] = 1

        self.dist2target_left[env_ids] = 0
        self.dist2target_right[env_ids] = 0
        self.v_out_left[env_ids] = self.v_out_target
        self.v_out_right[env_ids] = self.v_out_target
        self.catachable_left[env_ids] = True
        self.catachable_right[env_ids] = True
        self.miss_left[env_ids] = False
        self.miss_right[env_ids] = False

        ref_root_tensor = torch.cat((ref_root_tensor.unsqueeze_(1), ball_root_tensor), 1)
        ref_link_tensor = torch.cat((ref_link_tensor, ball_root_tensor), 1)
        return ref_root_tensor, ref_link_tensor, ref_joint_tensor

    def refresh_tensors(self):
        super().refresh_tensors()
        catch_range = 2
        
        self.timer_left -= 1
        self.timer_right -= 1

        n_balls = self.ball_root_tensor.size(1)
        target_ball_release = self.target_ball_release % n_balls
        throw_left = self.timer_left == 0
        throw_right = self.timer_right == 0
        release = torch.logical_or(throw_left, throw_right)
        target_ball = (self.env_range, target_ball_release)
        self.ball_attach[target_ball] = torch.where(release,
            self.ball_index[target_ball], self.ball_attach[target_ball]
        )
        
        self.timer_left[throw_left] = self.throw_interval
        self.miss_left[throw_left] = False
        self.target_ball_left[throw_left] = target_ball_release[throw_left]+(2-n_balls)
        self.timer_right[throw_right] = self.throw_interval
        self.miss_right[throw_right] = False
        self.target_ball_right[throw_right] = target_ball_release[throw_right]+(2-n_balls)

        v_out = self.ball_root_tensor[target_ball][:, self.UP_AXIS+7]
        self.v_out_left[throw_left] = v_out[throw_left]
        self.v_out_right[throw_right] = v_out[throw_right]
        self.ball_h_out[target_ball] = torch.where(release,
            self.ball_root_tensor[:,:,self.UP_AXIS][target_ball], self.ball_h_out[target_ball]
        )
        
        timer = self.timer_left
        target_ball = (self.env_range, self.target_ball_left)
        catched_already = self.ball_attach[target_ball] == self.left_hand_link
        hand_link = torch.full_like(self.env_range, self.left_hand_link)
        ball_pos = self.ball_root_tensor[target_ball][:, :3]
        hand_pos = self.link_tensor[:, self.left_hand_link, :3]
        dist = torch.linalg.norm(hand_pos - ball_pos, ord=2, dim=-1)
        t_before_throw = (timer-self.last_catch_frame).mul(self.step_time)
        catch = torch.all(torch.stack((
            timer >= self.last_catch_frame,
            timer <= self.first_catch_frame,
            dist < self.ball_radius*catch_range
        ), -1), -1)
        self.miss_left = torch.where(timer == self.last_catch_frame,
            ~torch.logical_or(catched_already, catch),
            self.miss_left
        )
        self.ball_attach[target_ball] = torch.where(
            torch.logical_or(timer == self.last_catch_frame, catch),
            hand_link, self.ball_attach[target_ball]
        )
        ball_vel = self.ball_root_tensor[target_ball][:, 7:10]
        root_h = self.root_pos[:, self.UP_AXIS]
        h = torch.clip(hand_pos[:, self.UP_AXIS], min=root_h) - ball_pos[:, self.UP_AXIS]
        v = ball_vel[:, self.UP_AXIS]
        delta = v.square().sub_(19.62*h)
        t = torch.where(delta > 0, delta.sqrt().add_(v).div_(9.81), v/9.81)
        self.catachable_left = torch.any(torch.stack((
            catched_already, dist < self.ball_radius*catch_range,
            torch.all(torch.stack((
                delta>=0, t>0, t<t_before_throw
            ), -1), -1)
        ), -1), -1)
        t.clip_(max=t_before_throw)
        dp = hand_pos - ball_pos
        dp[:, self.UP_AXIS] += 4.905*t.square() - ball_vel[:, self.UP_AXIS]*t
        dist = torch.where(t < 0,
            dist, torch.linalg.norm(dp, ord=2, dim=-1)
        )
        self.dist2target_left = torch.where(catched_already,
            self.dist2target_left, dist
        )

        timer = self.timer_right
        target_ball = (self.env_range, self.target_ball_right)
        catched_already = self.ball_attach[target_ball] == self.right_hand_link
        hand_link = torch.full_like(self.env_range, self.right_hand_link)
        ball_pos = self.ball_root_tensor[target_ball][:, :3]
        hand_pos = self.link_tensor[:, self.right_hand_link, :3]
        dist = torch.linalg.norm(hand_pos - ball_pos, ord=2, dim=-1)
        t_before_throw = (timer-self.last_catch_frame).mul(self.step_time)
        catch = torch.all(torch.stack((
            timer >= self.last_catch_frame,
            timer <= self.first_catch_frame,
            dist < self.ball_radius*catch_range
        ), -1), -1)
        self.miss_right = torch.where(timer == self.last_catch_frame,
            ~torch.logical_or(catched_already, catch),
            self.miss_right
        )
        self.ball_attach[target_ball] = torch.where(
            torch.logical_or(timer == self.last_catch_frame, catch),
            hand_link, self.ball_attach[target_ball]
        )
        ball_vel = self.ball_root_tensor[target_ball][:, 7:10]
        h = torch.clip(hand_pos[:, self.UP_AXIS], min=root_h) - ball_pos[:, self.UP_AXIS]
        v = ball_vel[:, self.UP_AXIS]
        delta = v.square().sub_(19.62*h)
        t = torch.where(delta > 0, delta.sqrt().add_(v).div_(9.81), v/9.81)
        self.catachable_right = torch.any(torch.stack((
            catched_already, dist < self.ball_radius*catch_range,
            torch.all(torch.stack((
                delta>=0, t>0, t<t_before_throw
            ), -1), -1)
        ), -1), -1)
        t.clip_(max=t_before_throw)
        dp = hand_pos - ball_pos
        dp[:, self.UP_AXIS] += 4.905*t.square() - ball_vel[:, self.UP_AXIS]*t
        dist = torch.where(t < 0,
            dist, torch.linalg.norm(dp, ord=2, dim=-1)
        )
        self.dist2target_right = torch.where(catched_already,
            self.dist2target_right, dist
        )

        self.ball_root_tensor.copy_(self.link_tensor[(self.env_range_, self.ball_attach)])
        actor_ids = self.actor_ids[:, 1:].flatten().contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )
        self.target_ball_release[release] += 1


    def reward(self):
        n_balls = self.ball_root_tensor.size(1)
        if n_balls % 2:
            started_left = self.target_ball_release >= n_balls
            started_right = self.target_ball_release >= n_balls-1
        else:
            started_left = self.target_ball_release >= n_balls-1
            started_right = self.target_ball_release >= n_balls

        catched = self.ball_attach[(self.env_range, self.target_ball_left)] == self.left_hand_link
        dist2 = self.dist2target_left.square()
        rew_left_catch = 0.9*dist2.mul(-20).exp_()+0.2*dist2.mul(-1).exp_()
        dh = self.ball_h_out[(self.env_range, self.target_ball_left)] - self.link_tensor[:, self.left_hand_link, self.UP_AXIS]
        dh[catched] = 0
        rew_left_catch = dh.square_().mul_(-20).exp_()*0.1 + 0.9*(~self.miss_left)*rew_left_catch

        rew_left_throw = (self.v_out_left/self.v_out_target - 1).square_().mul_(-5).exp_()
        rew_left = torch.where(self.timer_left == self.throw_interval,
            rew_left_throw, started_left*rew_left_catch)

        catched = self.ball_attach[(self.env_range, self.target_ball_right)] == self.right_hand_link
        dist2 = self.dist2target_right.square()
        rew_right_catch = 0.9*dist2.mul(-20).exp_()+0.2*dist2.mul(-1).exp_()
        dh = self.ball_h_out[(self.env_range, self.target_ball_right)] - self.link_tensor[:, self.right_hand_link, self.UP_AXIS]
        dh[catched] = 0
        rew_right_catch = dh.square_().mul_(-20).exp_()*0.1 + 0.9*(~self.miss_right)*rew_right_catch

        rew_right_throw = (self.v_out_right/self.v_out_target - 1).square_().mul_(-5).exp_()
        rew_right = torch.where(self.timer_right == self.throw_interval,
            rew_right_throw, started_right*rew_right_catch)
        
        rew = rew_left.mul_(0.5) + rew_right.mul_(0.5)
        rew = rew.unsqueeze_(-1)

        tar_rew = super().reward()
        return torch.cat((rew, tar_rew), -1)

    def _observe(self, env_ids):
        target_ball = (self.env_range, self.target_ball_left)
        catched_left = self.ball_attach[target_ball] == self.left_hand_link
        catched_left.unsqueeze_(-1)
        target_ball_left = self.ball_root_tensor[target_ball]

        target_ball = (self.env_range, self.target_ball_right)
        catched_right = self.ball_attach[target_ball] == self.right_hand_link
        catched_right.unsqueeze_(-1)
        target_ball_right = self.ball_root_tensor[target_ball]

        n_balls = self.ball_attach.size(1)
        next_release = self.target_ball_release % n_balls
        next_catch = next_release.add_(2-n_balls)
        target_ball = (self.env_range, next_catch)
        target_ball_next = self.ball_root_tensor[target_ball]

        if env_ids is None:
            res = observe_iccgan_juggling_target(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens,
                self.goal_tensor, self.goal_timer,
                self.sp_upper_bound, self.fps,

                target_ball_left, catched_left,
                target_ball_right, catched_right,
                target_ball_next, 
                self.timer_left/self.throw_interval 
            )
        else:
            res = observe_iccgan_juggling_target(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.goal_tensor[env_ids], self.goal_timer[env_ids],
                self.sp_upper_bound, self.fps,

                target_ball_left[env_ids], catched_left[env_ids],
                target_ball_right[env_ids], catched_right[env_ids],
                target_ball_next[env_ids],  
                self.timer_left[env_ids]/self.throw_interval
            )
        return res

    def termination_check(self):
        return ICCGANHumanoid.termination_check(self)

@torch.jit.script
def observe_iccgan_juggling_target(state_hist: torch.Tensor, seq_len: torch.Tensor,
    goal_tensor: torch.Tensor, goal_timer: torch.Tensor, sp_upper_bound: float, fps: int,
    target_ball_left: torch.Tensor, catched_left: torch.Tensor,
    target_ball_right: torch.Tensor, catched_right: torch.Tensor, 
    target_ball_next: torch.Tensor,
    timer: torch.Tensor):
    UP_AXIS = 2

    ob = observe_iccgan_target(state_hist, seq_len, goal_tensor, goal_timer, sp_upper_bound=sp_upper_bound, fps=fps)

    root_pos = state_hist[-1, :, :3]
    root_orient = state_hist[-1, :, 3:7]
    
    origin = root_pos.clone()
    origin[..., UP_AXIS] = 0                                            # N x 3
    heading = heading_zup(root_orient)
    up_dir = torch.zeros_like(origin)
    up_dir[..., UP_AXIS] = 1
    heading_orient_inv = axang2quat(up_dir, -heading)                   # N x 4

    ball_pos = target_ball_left[:, :3] - origin
    ball_lin_vel = target_ball_left[:, 7:10]
    ball_pos = rotatepoint(heading_orient_inv, ball_pos)
    ball_lin_vel = rotatepoint(heading_orient_inv, ball_lin_vel)
    ball_state_l = torch.cat((ball_pos, ball_lin_vel), -1)
    ball_state_l = torch.where(
        catched_left, torch.zeros_like(ball_state_l), ball_state_l
    )

    ball_pos = target_ball_right[:, :3] - origin
    ball_lin_vel = target_ball_right[:, 7:10]
    ball_pos = rotatepoint(heading_orient_inv, ball_pos)
    ball_lin_vel = rotatepoint(heading_orient_inv, ball_lin_vel)
    ball_state_r = torch.cat((ball_pos, ball_lin_vel), -1)
    ball_state_r = torch.where(
        catched_right, torch.zeros_like(ball_state_r), ball_state_r
    )

    ball_pos = target_ball_next[:, :3] - origin
    ball_lin_vel = target_ball_next[:, 7:10]
    ball_pos = rotatepoint(heading_orient_inv, ball_pos)
    ball_lin_vel = rotatepoint(heading_orient_inv, ball_lin_vel)
    ball_state_n = torch.cat((ball_pos, ball_lin_vel), -1)

    return torch.cat((ob, ball_state_l, ball_state_r, ball_state_n, timer.unsqueeze_(-1)), -1)
