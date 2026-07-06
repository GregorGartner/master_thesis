# Project Notes

## Python Environment

Use the Conda environment wrapper in this repo for running project commands:

```bash
bash ./in_env python cartpole_ppo_sb3_training.py
bash ./in_env python lqr_theta_sweep_eval.py
bash ./in_env python episode_traj_and_regression_plot.py
bash ./in_env python -m py_compile unified_context_ppo.py
```

The wrapper runs commands inside the `gym_env2` Conda environment.

Direct execution via `./in_env ...` may fail on some mounted filesystems with
`permission denied`; if that happens, use `bash ./in_env ...` instead.
