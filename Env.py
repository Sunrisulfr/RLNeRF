# import gym
# from gym import spaces
import numpy as np


class PlanGymEnv():

    #observation ==> DT  : img 3 * 800 * 800
    #index ==> PlanData ==> reward

    def __init__(self, NeRFModel, PlanData, maxPlanLength):
        self.action_space = 2
        self.observation_space = (3,800,800)
        self.view_num = 0
        self.n_views = maxPlanLength
        self.env_model = NeRFModel
        self.PlanData = PlanData
        # self.PlanSpace = PlanSpace
        self.observations = []
        self.indexs = []
    
    def step(self, action):
        self.view_num += 1
        self.observation, index = self.PlanData.get_view(action)
        self.observations.append(self.observation)
        self.indexs.append(index)
        reward= self.env_model.cal_reward(self.indexs, self.PlanData) - self.reward_baseline
        # reward= self.env_model.cal_uncert_reward(self.indexs, self.PlanData)/100
        # self.state = state
        if self.view_num == self.n_views:
            done = 1
        else:
            done = 0 
        return self.observation[0], reward, done, {}
    
    def reset(self, init_action):
        self.view_num = 1
        self.observation, index = self.PlanData.get_view(init_action)
        self.indexs = [index]
        self.observations = [self.observation]
        self.reward_baseline = self.env_model.cal_reward(self.indexs, self.PlanData)
        # self.state = self.env_model.get_state(self.observations)
        return self.observation
        
    def render(self):
        return None
        
    def close(self):
        return None




class PlanEnv(object):

    def __init__(self, NeRFModel):
        self.action_dim = 2
        self.env_model = NeRFModel
        

    def step(self, actions):
        """Returns reward, terminated, info."""
        raise NotImplementedError

    def get_obs(self):
        """Returns all agent observations in a list."""
        raise NotImplementedError

    def get_obs_size(self):
        """Returns the size of the observation."""
        raise NotImplementedError

    def get_state(self):
        """Returns the global state."""
        raise NotImplementedError

    def get_state_size(self):
        """Returns the size of the global state."""
        raise NotImplementedError

    def reset(self):
        """Returns initial observations and states."""
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "episode_limit": self.episode_limit}
        return env_info