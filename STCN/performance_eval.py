# with prestored mask
import os
from eval_metrics import eval_iou, f_measure
import numpy as np
from PIL import Image
from matplotlib.pylab import plt
from util.hyper_para import HyperParameters

para_dir = HyperParameters()
para_dir.parse()
jigsaws_gt_dir =para_dir['jigsaws']


videos = ['Needle_Passing_S08_T02','Needle_Passing_S04_T01','Needle_Passing_S05_T03','Needle_Passing_S05_T05']
btypes = ["thread","rightgrasper","needle","leftgrasper","ring"]  #"thread","rightgrasper","needle","leftgrasper","ring"
tasks = ["Suturing","Needle_Passing", "Knot_Tying"] #,"Knot_Tying","Suturing"
for task in tasks:
    if task =='Suturing':
        vids = np.array([ 'Suturing_S02_T04','Suturing_S02_T01', 'Suturing_S03_T04','Suturing_S03_T05','Suturing_S05_T03'])
    elif task=='Needle_Passing':
        vids = np.array(['Needle_Passing_S04_T01','Needle_Passing_S05_T03','Needle_Passing_S05_T05'])
    else:
        vids = np.array(['Knot_Tying_S09_T05','Knot_Tying_S05_T05','Knot_Tying_S03_T05','Knot_Tying_S05_T03','Knot_Tying_S03_T02'])
    for btype in btypes:
        ious = []
        fvals = []
        avgs = []
        for vid in vids:
            #vid = "Needle_Passing_S04_T01"
            #task = "Needle_Passing"
            #btype = "thread"
            gtmask_dir =  os.path.join(jigsaws_gt_dir,f'{task}/mask_output_{btype}')
            mask_dir =f'/Documents/video_object_segmentation/data/Prestore_{btype}'
            existdir = f'/Documents/video_object_segmentation/data/Prestore_{btype}/{vid}'
            if os.path.exists(existdir)==False:
                continue
            frames = sorted(os.listdir(os.path.join(mask_dir, vid)))
            
            for idx,frame in enumerate(frames):
                if idx ==0:continue
                gt_dir = os.path.join(gtmask_dir,vid,frame)
                seg_dir = os.path.join(mask_dir,vid,frame)
                loc_mask = f'data/Prestore_{btype}/{vid}/{frame}'
                mask = Image.open(loc_mask).convert('L')
                mask = np.asarray(mask)
                loc_gt =os.path.join(jigsaws_gt_dir, f'{task}/mask_output_{btype}/{vid}/{frame}')
                mask_gt =  Image.open(loc_gt).convert('L')
                mask_gt = np.asarray(mask_gt)
                if len(np.unique(mask_gt))==1:
                    continue
                iouval,fval = eval_iou(mask,mask_gt), f_measure(mask,mask_gt)
                avg = round(1/2*(iouval+fval),3)
                ious.append(iouval)
                fvals.append(fval)
                avgs.append(avg)
        if len(ious)==0: continue
        mean_iou = round(np.mean(ious),3)
        mean_f = round(np.mean(fvals),3)
        mean_mean = round(np.mean(avgs),3)
        print(f'{task} {btype} mean iou: { mean_iou} mean f: {mean_f} mean mean: {mean_mean}')