"""
Generic evaluation script 
The segmentation mask for each object when they first appear is required

Optimized for compatibility, not speed.
We will resize the input video to 480p -- check generic_test_dataset.py if you want to change this behavior
AMP default on.

Usage: python eval_generic.py  --output <some output path>
python eval_generic.py  --output /Documents/video_object_segmentation/output

"""


import os
from os import path
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from model.eval_network import STCN
from dataset.generic_test_deeplabdataset import GenericTestDataset
from util.tensor_util import unpad
from inference_core_yv import InferenceCore
from progressbar import progressbar
from util.hyper_para import HyperParameters

para_dir = HyperParameters()
para_dir.parse()
jigsaws_gt_dir =para_dir['jigsaws']

"""
Arguments loading
"""
parser = ArgumentParser()

parser.add_argument('--model', default='saves/Jan13_12.37.44_KT_all_JIGSAWS500100/Jan13_12.37.44_KT_all_JIGSAWS500100_checkpoint.pth')

# S, NP, KT model directories
model_dirs = {"Suturing":"saves/Jan13_10.10.40_S_all_JIGSAWS500100/Jan13_10.10.40_S_all_JIGSAWS500100_checkpoint.pth", 
"Needle_Passing":"saves/Jan12_22.39.27_NP_all_JIGSAWS500100/Jan12_22.39.27_NP_all_JIGSAWS500100_checkpoint.pth",
"Knot_Tying":"saves/Jan13_12.37.44_KT_all_JIGSAWS500100/Jan13_12.37.44_KT_all_JIGSAWS500100_checkpoint.pth"}

parser.add_argument('--output')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp_off', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()


out_path = args.output
args.amp = not args.amp_off

# Simple setup
os.makedirs(out_path, exist_ok=True)
torch.autograd.set_grad_enabled(False)
#"needle","thread","leftgrasper","ring" ,"rightgrasper"
btypes = ["needle","thread","leftgrasper","ring" ,"rightgrasper"] #,"rightgrasper","needle","leftgrasper","ring" ,"rightgrasper","needle","leftgrasper","ring" #"leftgrasper","thread" ,"rightgrasper","ring"
tasks = ["Suturing","Needle_Passing","Knot_Tying"] #"Suturing", #,"Knot_Tying" Suturing "Needle_Passing","Knot_Tying",


for task in tasks:
    imagedir=os.path.join(jigsaws_gt_dir,f'{task}/images')
    model_dir = model_dirs[task]
    for btype in btypes:
        if btype =='needle' and task=='Knot_Tying':continue
        if btype =='ring' and (task=='Knot_Tying' or task=="Suturing"):continue

        maskdir = # mask directory
        
        # Setup Dataset
        if not os.path.exists(maskdir):
            continue
        test_dataset = GenericTestDataset(imagedir=imagedir,maskdir=maskdir,task=task)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)



        # Load our checkpoint
        top_k = args.top
        prop_model = STCN().cuda().eval()

        # Performs input mapping such that stage 0 model can be loaded
        prop_saved = torch.load(model_dir)
        prop_saved = prop_saved['network']
        #breakpoint()
        for k in list(prop_saved.keys()):
            if k == 'value_encoder.conv1.weight':
                if prop_saved[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
                    prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
        prop_model.load_state_dict(prop_saved)

        # Start eval
        import time
            
        for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):
            #breakpoint()
            

            with torch.cuda.amp.autocast(enabled=args.amp):
                rgb = data['rgb']
                msk = data['gt'][0]
                info = data['info']
                name = info['name'][0]
                num_objects = len(info['labels'][0])
                print(info['label_backward'])
                gt_obj = info['gt_obj']

                size = info['size']
                #palette = data['palette'][0]
                print(sorted(list(gt_obj.keys())))

                print('Processing', name, '...')

                # Frames with labels, but they are not exhaustively labeled
                frames_with_gt = sorted(list(gt_obj.keys()))
                
                processor = InferenceCore(prop_model, rgb, num_objects=num_objects, top_k=top_k, 
                                            mem_every=args.mem_every, include_last=args.include_last)

                # min_idx tells us the starting point of propagation
                # Propagating before there are labels is not useful
                min_idx = 99999
                start_time = time.time()
                
                for i, frame_idx in enumerate(frames_with_gt):
                    min_idx = min(frame_idx, min_idx)
                    # Note that there might be more than one label per frame
                    obj_idx = gt_obj[frame_idx][0].tolist()
                    
                    print(frame_idx)
                    # Map the possibly non-continuous labels into a continuous scheme
                    obj_idx = [info['label_convert'][o].item() for o in obj_idx]
                    

                    # Append the background label
                    with_bg_msk = torch.cat([
                        1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
                        msk[:,frame_idx],
                    ], 0).cuda()

                    # We perform propagation from the current frame to the next frame with label
                    if i == len(frames_with_gt) - 1:
                        processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
                    else:
                        processor.interact(with_bg_msk, frame_idx, frames_with_gt[i+1]+1, obj_idx)

                # Do unpad -> upsample to original size (we made it 480p)
                out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')

                for ti in range(processor.t):
                    prob = unpad(processor.prob[:,ti], processor.pad)
                    prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
                    out_masks[ti] = torch.argmax(prob, dim=0)

                out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

                # Remap the indices to the original domain
                idx_masks = np.zeros_like(out_masks)
                for i in range(1, num_objects+1): #num_objects
                    backward_idx = info['label_backward'][i].item()
                    idx_masks[out_masks==i] = backward_idx
                
                print("--- %s seconds ---" % (time.time() - start_time))
                # Save the resultsis

                this_out_path = path.join(task,out_path+f'_{btype}', name)
                os.makedirs(this_out_path, exist_ok=True)
                for f in range(idx_masks.shape[0]):
                    if f >= min_idx:
                        img_E = Image.fromarray(idx_masks[f])
                        #img_E.putpalette(palette)
                        #print(info['frames'][f][0], info['frames'][f])
                        img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))#info['frames'][f][0]

                print(f'finish {this_out_path}')
                del rgb
                del msk
                del processor
