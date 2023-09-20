import matplotlib.pyplot as plt

eval_avg_return = [
    368.77,
    333.01,
    276.50,
    115.57,
    235.31,
    394.44,
    382.49,
    398.62,
    382.49,
]
eval_std_return = [116.82, 78.98, 67.63, 3.56, 15.15, 23.40, 18.00, 26.34, 18.00]
initial_avg_return = 10344.51

n_layers = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# plot eval_avg_return , eval_std_retur and intial_avg_return vs num_agent_train_steps_per_iter
plt.figure()
plt.plot(n_layers, eval_avg_return, label="eval_avg_return")
plt.plot(n_layers, eval_std_return, label="eval_std_return")
plt.scatter(n_layers, eval_avg_return)
plt.scatter(n_layers, eval_std_return)
# plt.plot(num_agent_train_steps_per_iter, initial_avg_return, label="initial_avg_return")
plt.xlabel("n_layers")
plt.legend()
plt.savefig("Humanoidv2_n_layers_eval_avg_return.png")
plt.show()


# eval_avg_return = [
#     2866.81,
#     5444.47,
#     5344.69,
#     4585.64,
#     5513.61,
# ]

# eval_std_return = [2437.65, 28.15, 277.42, 1948.91, 47.70]

# initial_avg_return = 5566.84

# num_agent_train_steps_per_iter = [1000, 1500, 2000, 2500, 3000]


# # print(len(eval_avg_return))
# # print(len(eval_std_return))
# # print(len(initial_avg_return))
# # print(len(num_agent_train_steps_per_iter))

# # plot eval_avg_return , eval_std_retur and intial_avg_return vs num_agent_train_steps_per_iter
# plt.figure()
# plt.plot(num_agent_train_steps_per_iter, eval_avg_return, label="eval_avg_return")
# plt.plot(num_agent_train_steps_per_iter, eval_std_return, label="eval_std_return")
# plt.plot(
#     num_agent_train_steps_per_iter,
#     [initial_avg_return for i in range(len(eval_avg_return))],
#     label="initial_avg_return",
# )
# plt.scatter(num_agent_train_steps_per_iter, eval_avg_return)
# plt.scatter(num_agent_train_steps_per_iter, eval_std_return)
# # plt.plot(num_agent_train_steps_per_iter, initial_avg_return, label="initial_avg_return")
# plt.xlabel("num_agent_train_steps_per_iter")
# plt.legend()
# plt.savefig("eval_avg_return.png")
# plt.show()
