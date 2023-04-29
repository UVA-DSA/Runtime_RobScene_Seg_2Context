import cv2 as cv
import numpy as np
import glob
import os
import pathlib

def main():
    TASK = "Suturing"
    TRIAL = "Suturing_S02_T01"
    MASK_SET_A = "2023_ICRA"
    MASK_SET_B = "2023_DL"
    CWD=os.getcwd()
    I = VideoInterface(CWD,TASK,TRIAL,MASK_SET_A,MASK_SET_B)
    I.makevideos(TRIAL, MASK_SET_A)

class VideoInterface:
    def __init__(self, CWD, TASK,TRIAL,MASK_SET_A,MASK_SET_B):
        self.CWD = CWD
        self.task = TASK
        self.trial = TRIAL
        self.imagesDir = os.path.join(self.CWD, "data", "images", TRIAL)   
        self.labeledDir_A= os.path.join(self.CWD,"eval","labeled_images",MASK_SET_A,TRIAL)
        self.labeledDir_B= os.path.join(self.CWD,"eval","labeled_images",MASK_SET_B,TRIAL)
        self.OS = "windows" 

    def makevideos(self, TRIAL, MASK_SET_A):
        TrialNum = 0
        #TrialRoot = os.path.join(self.CWD,self.task,"deeplab_labeled_images",Trial)
        #         
        img_array_A = []
        size = (0,0)
        for filename in glob.glob(self.labeledDir_A+'/*.png'):
            img = cv.imread(filename)            
            height, width, layers = img.shape
            #size = (height,width*2) if size == (0,0) else size
            size = (height,width) if size == (0,0) else size
            img_array_A.append(img)
        

        img_array_B = []
        
        for filename in glob.glob(self.labeledDir_B+'/*.png'):
            img = cv.imread(filename)
            height, width, layers = img.shape
            img_array_B.append(img)

        out = cv.VideoWriter(self.CWD+'/'+ MASK_SET_A + "-" + TRIAL+'.avi',cv.VideoWriter_fourcc(*'DIVX'), 15, (size[1],size[0]))
        
        #minRange = min(len(img_array_A),len(img_array_B))

        for i in range(0,len(img_array_A)):
            for j in range(3):
                try:
                    '''
                    h1, w1 = img_array_A[i].shape[:2]
                    h2, w2 = img_array_B[i].shape[:2]
                    #create empty matrix
                    vis = np.zeros((size[0],size[1],3), np.uint8)
                    vis[:h1, :w1,:3] = img_array_A[i]
                    vis[:h2, w1:w1+w2,:3] = img_array_B[i]
                    #vis = cv.cvtColor(vis, cv.COLOR_)
                    vos = cv.imread(vis, cv.IMREAD_GRAYSCALE)
                    out.write(vos)
                    '''                  

                    
                    #vis = np.concatenate((img_array_A[i], img_array_B[i]), axis=1)
                    vis = img_array_A[i]
                    out.write(vis)
                    
                    

                    #out.write(img_array_A[i])
                except Exception as e:
                    print("Err!",e)
        
        out.release()
        print("Saved Video",TRIAL)
        TrialNum+=1

        print("Processed ",TrialNum,"trials")

main();