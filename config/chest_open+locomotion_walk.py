import numpy as np

env_cls = "ICCGANHumanoidTarget"
env_params = dict(
    episode_length = 500,
    motion_file = "assets/motions/clips_walk.yaml",
    goal_reward_weight = [0.5],

    goal_radius = 0.5,
    sp_lower_bound = 1.2,
    sp_upper_bound = 1.5,
    goal_timer_range = (90, 150),
    goal_sp_mean = 1,
    goal_sp_std = 0.25,
    goal_sp_min = 0,
    goal_sp_max = 1.25
)

training_params = dict(
    max_epochs = 100000,
    save_interval = 10000,
    terminate_reward = -25
)

discriminators = {
    "chest_open/upper": dict(
        motion_file = "assets/motions/gym/chest_open+walk_in_place.json",
        key_links = ["torso", "head", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand"],
        parent_link = "pelvis",
        local_pos = True,
        replay_speed = lambda n: np.random.uniform(0.8, 1.2, size=(n,))
    ),
    "walk/lower": dict(
        key_links = ["pelvis", "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin", "left_foot"],
        parent_link = None
    )
}
