from collections import OrderedDict

from rob831.critics.bootstrapped_continuous_critic import BootstrappedContinuousCritic
from rob831.infrastructure.replay_buffer import ReplayBuffer
from rob831.infrastructure.utils import *
from rob831.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
from rob831.infrastructure import pytorch_util as ptu


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params["gamma"]
        self.standardize_advantages = self.agent_params["standardize_advantages"]

        self.actor = MLPPolicyAC(
            self.agent_params["ac_dim"],
            self.agent_params["ob_dim"],
            self.agent_params["n_layers"],
            self.agent_params["size"],
            self.agent_params["discrete"],
            self.agent_params["learning_rate"],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        for _ in range(self.agent_params["num_critic_updates_per_agent_update"]):
            critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            # if itrs % 100 == 0:
            #     print("critic loss: ", critic_loss)

        # advantage = estimate_advantage(...)
        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        for _ in range(self.agent_params["num_actor_updates_per_agent_update"]):
            actor_loss = self.actor.update(ob_no, ac_na, advantage)
            # if itrs % 100 == 0:
            #     print("actor loss: ", actor_loss)

        loss = OrderedDict()
        loss["Loss_Critic"] = critic_loss
        loss["Loss_Actor"] = actor_loss

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:

        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # 1) query the critic with ob_no, to get V(s)
        V_s = self.critic.forward(ob_no)

        # 2) query the critic with next_ob_no, to get V(s')
        V_s_prime = self.critic.forward(next_ob_no) * (1.0 - terminal_n)

        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        Q_s_a = re_n + self.gamma * V_s_prime

        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)

        adv_n = Q_s_a - V_s
        adv_n = ptu.to_numpy(adv_n)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)

        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
