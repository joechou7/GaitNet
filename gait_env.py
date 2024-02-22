import gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.env_checker import check_env


class GaitEnv(gym.Env):
    """
    Custom gait environment that follows gym interface.
    """
    def __init__(self, n_channels, n_frames, epi_len, data, model, att_mmt, reward, num_phases, phase_frame_idx):
        self.epi_len = epi_len
        self.data = data
        self.label = data.y
        self.model = model
        self.att_mmt = att_mmt
        self.reward = reward
        self.phase_frame_idx = phase_frame_idx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize the agent's current time step.
        self.cur_step = 0
        # Define action space and observation space.
        self.action_space = spaces.MultiBinary(num_phases)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, n_channels, n_frames), dtype=np.float32)


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent's current time step.
        self.cur_step = 0
        # Get observation.
        observation = self.data.x.unsqueeze(0).cpu().numpy()

        return observation


    def step(self, action):
        # Take action, i.e. adjust the phase attention.
        phase_att = torch.tensor(action).to(self.device)
        if phase_att.sum() == 0:
            phase_att[0] = 1
            phase_att[5] = 1
        phase_att = phase_att / phase_att.sum()
        pre_phase_att = self.data.phase_att
        phase_att = self.att_mmt * pre_phase_att + phase_att
        phase_att = phase_att / phase_att.sum()

        frame_att = torch.zeros_like(self.data.frame_att)
        for k, (i, j) in enumerate(self.phase_frame_idx):
            frame_att[i:j] = phase_att[k] / (j-i)

        # Update current time step.
        self.cur_step += 1

        # Get observation.
        self.data.key_frames = self.data.x * frame_att
        observation = self.data.x.unsqueeze(0).cpu().numpy()

        # Reward function.
        with torch.no_grad():
            cur_pred = self.model(self.data)

        if cur_pred.argmax() == self.label:
            reward = self.reward
        else:
            reward = -self.reward

        # Do we reach the episode length?
        done = bool(self.cur_step == self.epi_len)

        # Optionally we can pass additional info, we are not using that for now.
        info = {}

        return observation, reward, done, info


    def render(self, mode="human"):
        pass


    def close(self):
        pass



# # It will check your custom environment and output additional warnings if needed.
# env = GaitEnv()
# check_env(env)

