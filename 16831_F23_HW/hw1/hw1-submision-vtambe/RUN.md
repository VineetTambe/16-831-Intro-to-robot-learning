To generate the numbers for Section 1 Question 2 and Question 3 (for Ant and Humanoid): 
run the following commands:

1. `Ant-v2`: 

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1 --eval_batch_size 5000 
```

2. `Hopper-v2`: 

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Hopper.pkl \
--env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
--video_log_freq -1 --eval_batch_size 5000 
```

3. `HalfCheetah-v2`: 

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_HalfCheetah --n_iter 1 \
--expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl \
--video_log_freq -1 --eval_batch_size 5000 
```

4. `Walker2d-v2`: 

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Walker2d.pkl \
--env_name Walker2d-v2 --exp_name bc_Walker2d --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
--video_log_freq -1 --eval_batch_size 5000 
```

5. `Humanoid-v2`: 

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 --eval_batch_size 5000 
```

To generate the numbers for Section 1 Question 4 run the following commands from `--n_layers` going from [2, 4, 8, 16, 32, 64, 128, 256, 512]:

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 --eval_batch_size 5000 --n_layers 2
```
