env_cls = "ICCGANHumanoidTarget"
env_params = dict(
    episode_length = 500,
    motion_file = "assets/motions/clips_run.yaml",
    goal_reward_weight = [0.5],

    goal_radius = 0.5,
    sp_lower_bound = 2,
    sp_upper_bound = 4,
    goal_timer_range = (60, 90),
    goal_sp_mean = 1.5,
    goal_sp_std = 0.5,
    goal_sp_min = 1,
    goal_sp_max = 3
)

training_params = dict(
    max_epochs = 100000,
    save_interval = 10000,
    terminate_reward = -25
)

discriminators = {
    "run/full": dict(
        parent_link = None,
    )
}
