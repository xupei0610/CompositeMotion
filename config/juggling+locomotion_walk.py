import numpy as np

env_cls = "ICCGANHumanoidJugglingTarget"
env_params = dict(
    episode_length = 500,
    motion_file = "assets/motions/clips_walk.yaml",
    goal_reward_weight = [0.6, 0.1],

    throw_interval = 20,
    dwell_time = 12,

    sharp_turn_rate = 0,
    goal_radius = 0.75,
    sp_lower_bound = 1,
    sp_upper_bound = 1.2,
    goal_timer_range = (90, 150),
    goal_sp_mean = 1.,
    goal_sp_std = 0.,
    goal_sp_min = 1,
    goal_sp_max = 1
)

training_params = dict(
    max_epochs = 140000,
    save_interval = 30000,
    terminate_reward = -25
)

discriminators = {
    "juggling/hand": dict(
        motion_file = "assets/motions/gym/juggling.json",
        key_links = ["right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand"],
        parent_link = "torso",
        local_pos = True,
        replay_speed = lambda n: np.random.uniform(0.8, 1.2, size=(n,)),
        weight = 0.1
    ),
    "walk/body": dict(
        key_links = ["torso", "head", "pelvis", "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin", "left_foot"],
        parent_link = None,
        weight=0.4
    )
}
