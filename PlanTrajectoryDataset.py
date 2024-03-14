import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

def discount_cumsum(x, gamma):
    x = np.array(x)
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


def create_offline_dataset(max_t_length, t_number, eval_model, metadata):
    
    dataset = []
    # t_observations = []
    # # t_rays = []
    # t_states = []
    # t_actions = []
    # t_rewards = []
    random_index = metadata.get_random_index(max_t_length, t_number)
    for i in range(t_number):
        traj = {}
        observations = []
        # rays = []
        states = []
        actions = []
        rewards = []
        # poses = []
        index0 = random_index[i][0]
        # observation, pose = metadata.get_index_data(index0)
        # print(random_index[i])
        reward0 = eval_model.cal_reward(np.array([index0]*(max_t_length+1)),metadata)
        observation = index0
        observations.append(observation)
        # rays.append(ray)
        # poses.append(pose)
        for j in range(1,max_t_length+1):
            index = random_index[i][j]
            # observation, pose = metadata.get_index_data(index)
            observation = index  #REM Question
            observations.append(observation)
            # rays.append(ray)
            # poses.append(pose)
            if j < max_t_length:
                pair_index = np.concatenate((random_index[i][:j+1], np.array([index0]*(max_t_length-j))), axis=0)
            else:
                pair_index = random_index[i]
            # print(pair_index)
            
            # reward, state = eval_model.cal_reward(random_index[i][:j])
            reward = eval_model.cal_reward(pair_index,metadata)
            # reward = 0
            # state = 0
            action = metadata.sps[index]
            actions.append(action)
            rewards.append(reward - reward0)
            # states.append(state)

        traj["observations"] = observations[:-1]
        # traj["states"] = states[:-1]
        traj["actions"] = actions
        traj["rewards"] = rewards

        dataset.append(traj)

def create_offline_uncert_dataset(max_t_length, t_number, eval_model, metadata):
    
    dataset = []
    # t_observations = []
    # # t_rays = []
    # t_states = []
    # t_actions = []
    # t_rewards = []
    random_index = metadata.get_random_index(max_t_length, t_number)
    for i in range(t_number):
        traj = {}
        observations = []
        # rays = []
        states = []
        actions = []
        rewards = []
        # poses = []
        index0 = random_index[i][0]
        # observation, pose = metadata.get_index_data(index0)
        # print(random_index[i])
        # reward0 = eval_model.cal_reward(np.array([index0]*(max_t_length+1)),metadata)
        observation = index0
        observations.append(observation)
        # rays.append(ray)
        # poses.append(pose)
        for j in range(1,max_t_length+1):
            index = random_index[i][j]
            # observation, pose = metadata.get_index_data(index)
            observation = index  #REM Question
            observations.append(observation)
            # rays.append(ray)
            # poses.append(pose)
            if j < max_t_length:
                pair_index = np.concatenate((random_index[i][:j+1], np.array([index0]*(max_t_length-j))), axis=0)
            else:
                pair_index = random_index[i]
            # print(pair_index)
            
            # reward, state = eval_model.cal_reward(random_index[i][:j])
            reward = eval_model.cal_uncert_reward(pair_index,metadata)
            # reward = 0
            # state = 0
            action = metadata.sps[index]
            actions.append(action)
            rewards.append(reward)
            # states.append(state)

        traj["observations"] = observations[:-1]
        # traj["states"] = states[:-1]
        traj["actions"] = actions
        traj["rewards"] = rewards

        dataset.append(traj)

    with open("./data/offline_uncert_data.pickle", 'wb') as f:
        pickle.dump(dataset, f)




class PlanTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale, metadata):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        # min_len =3
        states = []
        self.trajectories = self.trajectories[:80]
        for traj in self.trajectories:
            traj_len = len(traj['observations'])
            # min_len = min(min_len, traj_len)
            observations = []
            for index in traj['observations']:
                img, _, _ = metadata.get_index_data(index)
               
                observations.append(img.numpy())
            traj['observations'] = np.array(observations)
            
            traj['actions'] = np.array(traj['actions'], dtype='float32')
            traj['actions'][:,0] = (traj['actions'][:,0] - 180.0) / 180.0
            traj['actions'][:,1] = (traj['actions'][:,1] - 45.0) / 45.0
            traj['rewards'] = np.array(traj['rewards'], dtype='float32')/100
        
            
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 0.9) / rtg_scale

        print(self.trajectories[0]["rewards"].shape)
        print(self.trajectories[0]["actions"].shape)
        print(self.trajectories[0]["observations"].shape)

        print(len(self.trajectories), "~~")
        #actions: tuple, imgs: tensor, [640000, 3]
        
        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        # for traj in self.trajectories:
        #     traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]
   
        padding_len = self.context_len - traj_len
        

        # padding with zeros
        states = torch.from_numpy(traj['observations'])
        states = torch.cat([states,
                            torch.zeros(([padding_len] + list(states.shape[1:])),
                            dtype=states.dtype)],
                            dim=0)

        actions = torch.from_numpy(traj['actions'])
        actions = torch.cat([actions,
                            torch.zeros(([padding_len] + list(actions.shape[1:])),
                            dtype=actions.dtype)],
                            dim=0)

        returns_to_go = torch.from_numpy(traj['returns_to_go'])
        returns_to_go = torch.cat([returns_to_go,
                            torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                            dtype=returns_to_go.dtype)],
                            dim=0)

        timesteps = torch.arange(start=0, end=self.context_len, step=1)

        traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                torch.zeros(padding_len, dtype=torch.long)],
                                dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask