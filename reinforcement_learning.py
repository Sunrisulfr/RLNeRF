import torch
import argparse
import time
import numpy as np


# state: imgs+vol
# action: delta_pos
# reward: PSNR

#      delta_pos
#      |
#-----------------------------------------------------------#
#                       Transformer
#-----------------------------------------------------------#
# |    |    |     |     
# imgs vol  pos  rew 

from PlanData import PlanData
from Env import PlanGymEnv
from PlanTrajectoryDataset import create_offline_dataset, PlanTrajectoryDataset, create_offline_uncert_dataset
from reward import Reward
from DTtrain import train


#uncert reward: change Model, Render, ENV-cal_reward, dataset, create_offline_uncert_dataset, cal_uncert_reward, testun1

def main(args):
    metadata = PlanData(args)
    reward_model = Reward()
    #[698,681,808,672,736]np.random.choice(self.img_number, t_length+1, replace=False)
    # r = 0
    # for i in range(10):
    #     random_index = np.random.choice(metadata.img_number, 5, replace=False)
    #     baseline = reward_model.cal_reward(random_index[:1], metadata)
    #     for i in range(2,6):
    #         r += (reward_model.cal_reward(random_index[:i], metadata) - baseline)
    # print(r/10)
    # exit()
    n_views = 5
    max_t_length = n_views - 1
    t_number = 160
    # create_offline_uncert_dataset(max_t_length, t_number, reward_model, metadata)
    # exit()

    dataset = PlanTrajectoryDataset(args.datasetdir, 4, 0.5, metadata)

    env = PlanGymEnv(reward_model, metadata, n_views)

    
    train(args, dataset, env)

    #val

 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, default='/home/disk/smyData/NeRF_Data-20230814T023608Z-001/NeRF_Data/blend_files/blend_files/results_test_100_20/',
                        help='input data directory')
    parser.add_argument("--datasetdir", type=str, default='./data/offline_uncert_data.pickle',
                        help='input data directory')
    parser.add_argument('--imgScale_train', type=float, default=1.0)
    parser.add_argument('--imgScale_test', type=float, default=1.0)



    # parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=10)
    parser.add_argument('--num_eval_ep', type=int, default=5)

    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='./RLLog')

    parser.add_argument('--context_len', type=int, default=10)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wt_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=100)

    parser.add_argument('--max_train_iters', type=int, default=50)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(args)
