

import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import pickle


from data.ray_utils import *


class PlanData():
    def __init__(self, args,  split='train'):
        self.args = args
        self.root_dir = args.datadir
        self.split = split
        # self.intervals = intervals
        downsample = args.imgScale_train if split=='train' else args.imgScale_test
        assert int(800*downsample)%32 == 0, \
            f'image width is {int(800*downsample)}, it should be divisible by 32, you may need to modify the imgScale'
        self.img_wh = (int(800*downsample),int(800*downsample))
        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.index_dict = {}
        self.read_meta()

        self.white_back = True

    def read_meta(self):
        
        with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
            
            self.meta = json.load(f)

        # sub select training views from pairing file
        # if os.path.exists('configs/pairs.th'):
        #     name = os.path.basename(self.root_dir)
        #     self.img_idx = torch.load('configs/pairs.th')[f'{name}_{self.split}']
        #     self.meta['frames'] = [self.meta['frames'][idx] for idx in self.img_idx]
        #     print(f'===> {self.split}ing index: {self.img_idx}')
        self.img_number = len(self.meta['frames'])
        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh
        
        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)


        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.sps = []
        for frame in self.meta['frames']:
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)

            image_path = frame['file_path'] +".png"
            
            index_str = image_path[len(self.root_dir + "r_"):-4]
            index_strs = index_str.split('_')
            index, alpha, beta = int(index_strs[0]), int(float(index_strs[1])), int(float(index_strs[2]))

            self.index_dict[(alpha, beta)] = index
            self.sps.append((alpha, beta))

            self.image_paths += [image_path]


        
            # img = Image.open(image_path)
            # img = img.resize(self.img_wh, Image.LANCZOS)
            # img = self.transform(img)  # (4, h, w)
            # img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            # self.all_masks += [img[:,-1:]>0]
            # img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
        
            # self.all_rgbs += [img]

        
            # # rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

            # # self.all_rays += [torch.cat([rays_o, rays_d,
            # #                              self.near * torch.ones_like(rays_o[:, :1]),
            # #                              self.far * torch.ones_like(rays_o[:, :1])],
            # #                             1)]  # (h*w, 8)
            # self.all_masks += []
            
        # exit()
        # self.poses = np.stack(self.poses)
        # if 'train' == self.split:
            # self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
        
            # self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
           
        # else:
        #     self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
        #     self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        #     self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

        

    def read_source_views(self, file=f"transforms.json", pair_idx=[0], device=torch.device("cpu")):
            with open(os.path.join(self.root_dir, file), 'r') as f:
                meta = json.load(f)

            w, h = self.img_wh
            focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
            focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

            src_transform = T.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

            # if do not specify source views, load index from pairing file
            # if pair_idx is None:
            #     name = os.path.basename(self.root_dir)
            #     pair_idx = torch.load('configs/pairs.th')[f'{name}_train'][:3]
            #     print(f'====> ref idx: {pair_idx}')

            imgs, proj_mats = [], []
            intrinsics, c2ws, w2cs = [],[],[]
            for i,idx in enumerate(pair_idx):
                frame = meta['frames'][idx]
                c2w = np.array(frame['transform_matrix']) @ self.blender2opencv
                w2c = np.linalg.inv(c2w)
                c2ws.append(c2w)
                w2cs.append(w2c)

                # build proj mat from source views to ref view
                proj_mat_l = np.eye(4)
                intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
                intrinsics.append(intrinsic.copy())
                intrinsic[:2] = intrinsic[:2] / 4  # 4 times downscale in the feature space
                proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
                if i == 0:  # reference view
                    ref_proj_inv = np.linalg.inv(proj_mat_l)
                    proj_mats += [np.eye(4)]
                else:
                    proj_mats += [proj_mat_l @ ref_proj_inv]

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB
                imgs.append(src_transform(img))

            pose_source = {}
            pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
            pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
            pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

            near_far_source = [2.0,6.0]
            imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
            proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
            return imgs, proj_mats, near_far_source, pose_source



    def read_views(self, file=f"transforms.json", device=torch.device("cpu")):
        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        w, h = self.img_wh
        focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
        focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # if do not specify source views, load index from pairing file
        # if pair_idx is None:
        #     name = os.path.basename(self.root_dir)
        #     pair_idx = torch.load('configs/pairs.th')[f'{name}_train'][:3]
        #     print(f'====> ref idx: {pair_idx}')

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        for frame in self.meta['frames']:
            c2w = np.array(frame['transform_matrix']) @ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy())
            # intrinsic[:2] = intrinsic[:2] / 4  # 4 times downscale in the feature space
            # proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            # if i == 0:  # reference view
            #     ref_proj_inv = np.linalg.inv(proj_mat_l)
            #     proj_mats += [np.eye(4)]
            # else:
            #     proj_mats += [proj_mat_l @ ref_proj_inv]

            # image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            # img = Image.open(image_path)
            # img = img.resize(self.img_wh, Image.LANCZOS)
            # img = self.transform(img)  # (4, h, w)
            # img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB
            # imgs.append(src_transform(img))

        self.pose_source = {}
        self.pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        self.pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        self.pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

        # near_far_source = [2.0,6.0]
        # imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        # proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        # return imgs, proj_mats, near_far_source, pose_source

    def load_poses_all(self, file=f"transforms.json"):
        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        c2ws = []
        for i,frame in enumerate(meta['frames']):
            c2ws.append(np.array(frame['transform_matrix']) @ self.blender2opencv)
        return np.stack(c2ws)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def query(self, key:tuple):
        alpha, beta = key
        min_distance = 361.0
        index = 0

        beta = round(beta / (90 / 20)) * (90 / 20)
        for i in range(self.img_number):
            a1, b1 = self.sps[i]
            if(abs(b1 - beta) < 0.00001):
                distance = abs(a1 - alpha)
                if(distance < min_distance):
                    min_distance = distance
                    index = i
            
        return index
        # return self.index_dict[key]

    def get_view(self, sp):

        alpha, beta = sp
        alpha  = alpha * 180 + 180
        beta = beta * 45 + 45
        index = self.query((alpha, beta))
        img, ray, pose_info = self.get_index_data(index)

        return img, index

    def get_index_data(self, index):
        image_path = self.image_paths[index]
        img = Image.open(image_path)
       
        
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)  # (4, h, w)
        
        img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
        self.all_masks += [img[:,-1:]>0]
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
    
        img = img.reshape(self.img_wh[0], self.img_wh[1], 3).permute(2,0,1)
        
        # img = self.all_rgbs[index]
        # ray = self.all_rays[index]
        # pose = self.poses[index]
        # c2w = self.pose_source['c2ws'][index]
        # w2c = self.pose_source["w2cs"][index]
        # intrinsic = self.pose_source["intrinsics"][index]

        # return img, (pose, c2w, w2c, intrinsic)
        return img, None, None


    def get_index_rays(self, index):
        pose = self.poses[index]
        c2w = torch.FloatTensor(pose)
        rays_o, rays_d = get_rays(self.directions, c2w) 
        rays = torch.cat([rays_o, rays_d,
                                         self.near * torch.ones_like(rays_o[:, :1]),
                                         self.far * torch.ones_like(rays_o[:, :1])],
                                        1)
        return rays

    def get_random_index(self, t_length, t_number):
        random_index = []
        atjs = np.array([
            [20, 53, 87, 1005, 1015],
        [0, 33, 67, 1005, 1015], #0, 72
        [0, 20, 40, 60, 80],
        [100, 119, 138, 157, 179],
        [195, 218, 240, 262, 1046], #4.5, 90
        [195, 213, 231, 249, 267], #9
        [285, 302, 319, 336, 353], #13.5
        [370, 386, 402, 418, 434], #18.0
        [450, 465, 480, 495, 510], #22.5 
        [525, 539, 553, 567, 581], #27.0
        [595, 608, 621, 634, 647], #31.5
        [660, 672, 684, 696, 708], #36.0
        [720, 731, 742, 753, 764], #40.5
        [775, 785, 795, 805, 815], #45
        [825, 834, 843, 852, 861],#49.5
        [1000, 1004, 1008, 1012, 1016], #72
        ]) #45  
        al = atjs.shape[0]
        for i in range(al):
            for t in range(4):
                random_index.append(atjs[i,:] + t + 1) 
        trajs = np.array([[698,681,808,672,736],
        [698,681,808,672,2],
        [698,681,2,672,808],
        [2,681,808,672,698],
        [698,2,808,672,681],
        [2,698,681,672,808],
        [2,698,681,736,808],
        [808,698,681,672,698]] )   
        al2 = trajs.shape[0]
        for i in range(al2):
            random_index.append(atjs[i,:])
            for t in range(4):
                random_index.append(trajs[i,:] + np.random.randint(4, size=5) - 2)         


        for _ in range(t_number - 4 * (al+al2)):
            random_index.append(np.random.choice(self.img_number, t_length+1, replace=False))

        return random_index


    # def create_offline_dataset(self, max_t_length, t_number, eval_model):
    #     t_observations = []
    #     t_rays = []
    #     t_states = []
    #     t_actions = []
    #     t_rewards = []
    #     random_index = self.get_random_index(max_t_length, t_number)
    #     for i in range(t_number):
    #         observations = []
    #         rays = []
    #         states = []
    #         actions = []
    #         rewards = []
    #         poses = []
    #         for j in range(max_t_length+1):
    #             index = random_index[i][j]
    #             observation, ray, pose = self.get_index_data(index)
    #             observations.append(observation)
    #             rays.append(ray)
    #             poses.append(pose)
    #             if(j > 0):
    #                 reward, state = eval_model.cal_reward(observations, rays, pose)
    #                 action = self.sps[index]
    #                 actions.append(action)
    #                 rewards.append(reward)
    #                 states.append(state)
    #         t_observations.append(observations[:-1])
    #         t_rays.append(rays[:-1])
    #         t_states.append(states[:-1])
    #         t_rewards.append(rewards)
    #         t_actions.append(actions)


    #     t_observations = np.array(t_observations)
    #     t_actions = np.array(t_actions)
    #     t_rewards = np.array(t_rewards)
    #     t_actions = np.array(t_actions)
           
    #     dataset = {}
    #     dataset["observations"] = t_observations
    #     dataset["states"] = t_states
    #     dataset["actions"] = t_actions
    #     dataset["rewards"] = t_rewards

    #     with open("./rl_data/offline_data.pickle", 'wb') as f:
    #         pickle.dump(dataset, f)






    # def __len__(self):
    #     if self.split == 'train':
    #         return len(self.all_rays)
    #     return len(self.all_rgbs)

    # def __getitem__(self, idx):

    #     if self.split == 'train':  # use data in the buffers
    #         # view, ray_idx = torch.randint(0,len(self.all_rays),(1,)), torch.randperm(self.all_rays.shape[1])[:self.args.batch_size]
    #         # sample = {'rays': self.all_rays[view,ray_idx],
    #         #           'rgbs': self.all_rgbs[view,ray_idx]}
    #         sample = {'rays': self.all_rays[idx],
    #                   'rgbs': self.all_rgbs[idx]}

    #     else:  # create data for each image separately
    #         # frame = self.meta['frames'][idx]
    #         # c2w = torch.FloatTensor(frame['transform_matrix']) @ self.blender2opencv

    #         img = self.all_rgbs[idx]
    #         rays = self.all_rays[idx]
    #         mask = self.all_masks[idx] # for quantity evaluation

    #         sample = {'rays': rays,
    #                   'rgbs': img,
    #                   'mask': mask}
    #     return sample












