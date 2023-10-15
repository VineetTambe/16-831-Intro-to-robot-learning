The script to run the q1 experiment can be found in /rob831/scripts/experiment1_runner.bash

Command to run experiment for q2:

```
python rob831/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q2_10_10 -ntu 10 -ngsptu 10 --no_gpu
```

Command to run experiment for q2:
```
python rob831/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q3_10_10 -ntu 10 -ngsptu 10 --no_gpu
```

The script used to plot the graphs for q1 is in rob831/scripts/q1_plotter.ipynb
To plot it download the csv directly from the tensorboard dashboard