import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed


class VOSDataset(Dataset):
    """
    Works for JIGSAWS training
    For each sequence:
    - Pick three frames
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, root, max_jump, task, is_bl,subset=None):
        self.root = root
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.classes = ["needle","rightgrasper","leftgrasper","ring", "thread"]#[] ,"rightgrasper","needle","leftgrasper","ring" "thread"
        if task =='Suturing':
        
            self.dir_half = np.array(['Suturing_S07_T01', 'Suturing_S07_T02', 'Suturing_S07_T03','Suturing_S07_T04', 'Suturing_S07_T05', 'Suturing_S08_T01'])
            self.dir_test = np.array(['Suturing_03_T01', 'Suturing_S02_T04','Suturing_S02_T01', 'Suturing_S03_T04','Suturing_S03_T05','Suturing_S05_T03'])
        elif task=='Needle_Passing':
            self.dir_half = np.array(['Needle_Passing_S06_T01','Needle_Passing_S06_T03','Needle_Passing_S06_T04','Needle_Passing_S08_T02','Needle_Passing_S08_T04','Needle_Passing_S08_T05','Needle_Passing_S09_T03'])
            self.dir_test = np.array(['Needle_Passing_S08_T02','Needle_Passing_S04_T01','Needle_Passing_S05_T03','Needle_Passing_S05_T05'])
        else:
            self.dir_half = np.array(['Knot_Tying_S06_T04','Knot_Tying_S06_T05','Knot_Tying_S07_T01','Knot_Tying_S07_T02','Knot_Tying_S07_T03','Knot_Tying_S07_T04','Knot_Tying_S07_T05'])
            self.dir_test = np.array(['Knot_Tying_S09_T05','Knot_Tying_S05_T05','Knot_Tying_S03_T05','Knot_Tying_S05_T03','Knot_Tying_S03_T02'])

        self.videos = []
        self.videos_gt = []
        self.frames = {}

        vid_list = sorted(os.listdir(os.path.join(self.root,"images")))
        vid_list = set(vid_list) - set(self.dir_half) - set(self.dir_test)
        #print(vid_list)
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            
            frames = sorted(os.listdir(os.path.join(self.root, "images",vid)))
            self.frames[vid] = frames

            
            for clas in self.classes:
                path_gt =os.path.join(self.root,'mask_output_'+clas,vid)
                if not os.path.exists(path_gt):
                    continue
                frames = sorted(os.listdir(path_gt))
                
                if len(frames) < 3:
                    continue
                
                self.videos_gt.append(path_gt ) # the gt directory
                self.videos.append(vid)

        #print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        # self.pair_im_lone_transform = transforms.Compose([
        #     transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        # ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        # self.all_im_lone_transform = transforms.Compose([
        #     transforms.ColorJitter(0.1, 0.03, 0.03, 0),
        #     transforms.RandomGrayscale(0.05),
        # ])

        # if self.is_bl:
        #     # Use a different cropping scheme for the blender dataset because the image size is different
        #     self.all_im_dual_transform = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BICUBIC)
        #     ])

        #     self.all_gt_dual_transform = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
        #     ])
        # else:
        #     # self.all_im_dual_transform = transforms.Compose([
        #     #     transforms.RandomHorizontalFlip(),
        #     #     transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)
        #     # ])

        #     self.all_gt_dual_transform = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
            #])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        video_gt = self.videos_gt[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.root, "images",video)
        vid_gt_path = video_gt
        frames = self.frames[video]

        trial = False
        trials = 0
        idx =0
        
        while (trials==0 or trial ) and trials <5 : #and trials <5
            info['frames'] = [] # Appended with actual frames

            # Don't want to bias towards beginning/end
            this_max_jump = min(len(frames), self.max_jump)
            start_idx = np.random.randint(len(frames)-this_max_jump+1)
            f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
            f1_idx = min(f1_idx, len(frames)-this_max_jump, len(frames)-1)

            f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
            f2_idx = min(f2_idx, len(frames)-this_max_jump//2, len(frames)-1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            for f_idx in frames_idx:
                #print('idx')
                #print(f_idx)
                if f_idx==len(frames):
                    f_idx = len(frames)-1
                jpg_name = frames[f_idx][:-4] + '.png'
                png_name = frames[f_idx][:-4] + '.png'
                if (not os.path.exists(path.join(vid_im_path, jpg_name))) or (not os.path.exists(path.join(vid_gt_path, png_name))):
                    trial = True
                    break
                    
                else:
                    trial = False
            if trial == True:
                continue
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            
            images = []
            masks = []
            target_object = None
            
            for f_idx in frames_idx:
                #print('fidx'+str(f_idx),'frame'+str(len(frames)))
                if f_idx==len(frames):
                    f_idx = len(frames)-1
                jpg_name = frames[f_idx][:-4] + '.png'
                png_name = frames[f_idx][:-4] + '.png'
                #point()
                

                reseed(sequence_seed)
                

                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                #info['frames'].append(frames_idx)
                # info['frames'].append(path.join(vid_im_path, jpg_name))
                #this_im = self.all_im_dual_transform(this_im)
                #this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('L')
                info['frames'].append(path.join(vid_gt_path, png_name))
                #this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                #this_im = self.pair_im_lone_transform(this_im) # Removed image color change
                reseed(pairwise_seed) 
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)
            
            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0)
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:]
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, 480, 640), dtype=np.int)
        
        cls_gt[tar_masks[:,0] > 0.05] = 1 # was  0.001 0.001
        cls_gt[sec_masks[:,0] >0.05 ] = 2
        #breakpoint()

        data = {
            'rgb': images,
            'gt': tar_masks,
            'cls_gt': cls_gt,
            'sec_gt': sec_masks,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)