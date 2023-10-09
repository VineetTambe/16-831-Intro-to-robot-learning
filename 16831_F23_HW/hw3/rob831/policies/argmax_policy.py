import numpy as np
from rob831.infrastructure import pytorch_util as ptu


class ArgMaxPolicy(object):
    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        ## TODO return the action that maxinmizes the Q-value
        # at the current observation as the output
        # TODO verify the axis = 1 bit
        action = np.argmax(ptu.to_numpy(self.critic.q_net(observation)), axis=1)

        return action.squeeze()
