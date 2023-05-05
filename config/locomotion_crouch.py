env_cls = "ICCGANHumanoidTarget"
env_params = dict(
    episode_length = 500,
    motion_file = "assets/motions/clips_crouch.yaml",
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
    max_epochs = 50000,
    save_interval = 10000,
    terminate_reward = -25
)

discriminators = {
    "crouch/full": dict(
        parent_link = None,
    )
}

