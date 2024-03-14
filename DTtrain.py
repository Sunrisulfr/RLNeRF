import argparse
import os
import sys
import random
import csv
import time
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from decision_tranformer.PlanTrajectoryDataset import PlanTrajectoryDataset
from RLAlgorithm.utils import D4RLTrajectoryDataset, evaluate_on_env
from RLAlgorithm.DTmodels import DecisionTransformer

def train(args, traj_dataset, env):

    # dataset = args.dataset
    # rtg_sacle = args.rtg_scale


    

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes

    batch_size = args.batch_size            # training batch size
    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability


    device = torch.device(args.device)

    # traj_dataset = PlanTrajectoryDataset(dataset_path, context_len, rtg_scale)
    
    # for data in traj_dataset:
    #     print(data)
    # exit()


    traj_data_loader = DataLoader(
                            traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            # pin_memory=True,
                            drop_last=True
                        )


   
    # print(traj_data_loader)
    # exit()
    data_iter = iter(traj_data_loader)

    #?
    state_mean, state_std = traj_dataset.get_state_stats()

    # env = gym.make(env_name)
    

    # state_dim = env.observation_space[0]
    state_dim = 512
    act_dim = env.action_space

    # state_feature_dim = 

    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
            ).to(device)


   
    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lambda steps: min((steps+1)/warmup_steps, 1)
                        )

    total_updates = 0


    # Log Info

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    log_dir = args.log_dir
    save_model_name =  "DT" + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = "DT" + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)
    print("~~~~~~~~~~~~~statrt train~~~~~~~~~~")
    for i_train_iter in range(max_train_iters):

        log_action_losses = []

        model.train()
        
        for i in range(num_updates_per_iter):
            print("~~~round" + str(i) + "~~~") 
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
               
            except StopIteration:
                data_iter = iter(traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
          
           
            timesteps = timesteps.to(device)    # B x T
           
            states = states.to(device)          # B x T x state_dim
            actions = actions.to(device)        # B x T x act_dim
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) # B x T x 1
            traj_mask = traj_mask.to(device)    # B x T
            action_target = torch.clone(actions).detach().to(device)

            state_preds, action_preds, return_preds = model.forward(
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                            returns_to_go=returns_to_go
                                                        )
            # only consider non padded elements
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()
          
            # time.sleep(10)

        
        # evaluate action accuracy
        print("~~~~~~~~~~~~~~evaluation~~~~~~~~~~~~~~~~~~~")
        rtg_target = 2
        torch.cuda.empty_cache()
        time.sleep(10)
        
        results = evaluate_on_env(model, device, context_len, env, rtg_target, args.rtg_scale,
                                num_eval_ep, max_eval_ep_len, state_mean, state_std)
        
        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']


        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + time_elapsed  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "action loss: " +  format(mean_action_loss, ".5f") + '\n' +
                "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' 
                # + "eval d4rl score: " + format(eval_d4rl_score, ".5f")
            )
    
        print(log_str)

        log_data = [time_elapsed, total_updates, mean_action_loss,
                    eval_avg_reward, eval_avg_ep_len]

        csv_writer.writerow(log_data)

        torch.save(model.state_dict(), save_model_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)

    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train(args)


