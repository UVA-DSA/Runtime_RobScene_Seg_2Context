import numpy as np
import cv2 as cv
import os,sys
import utils
import pathlib

colors =["#E36D6D","#5E81B5","#D47BC9","#7CEB8E","#C9602A","#77B9E0","#A278F0","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#D66F6D","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5"]

class Contour_Iterator:
    def __init__(self, MASK_SET, TRIAL, CDW):
        self.CWD = CDW
        self.MASK_SET = MASK_SET 
        self.TRIAL = TRIAL 
        self.MASK_ROOT = os.path.join(self.CWD, "data", "masks", MASK_SET)
        self.imagesDir = os.path.join(self.CWD, "data", "images", TRIAL)
        self.tissueFname =  os.path.join(self.CWD,"tissue_keypoints", TRIAL+".json")
        self.grasperJawFname = os.path.join(self.CWD,"grasper_jaw_keypoints",TRIAL+".json")
        self.OS = "windows"

    def dilation(self, img):
        kernel = np.ones ((5, 5), np.uint8)
        dilated = cv.dilate(img, kernel, iterations=1)
        return dilated 

    def getLabelClassnames(self,TRIAL):        
        if "Knot" in TRIAL:
            return ["leftgrasper","rightgrasper","thread"]
        elif "Needle" in TRIAL:
            return ["leftgrasper","rightgrasper","thread","needle"]
        elif "Suturing" in TRIAL:
            return ["leftgrasper","rightgrasper","thread","needle"]

    def ExtractContoursTrial(self, TRIAL, FRAME_NUMS):
        classNameIndex=0
        ContourFiles = []
        label_classes = self.getLabelClassnames(TRIAL)
        RingFile = ""
        if "Knot_Tying" in TRIAL:
            for label_class in label_classes:
                ContourFname = self.findMaskContours(label_class,TRIAL,FRAME_NUMS,SAVE_TEST_IMAGE=True,SAVE_DATA=True)
                ContourFiles.append(ContourFname)
                classNameIndex+=1
        # TODO: add the contourfiles for the other tasks
        elif "Needle_Passing" in TRIAL:
            for label_class in label_classes:
                #ContourFname = self.findAllContoursTask(label_class,label_classNames[classNameIndex],EPOCH,TRIAL,FRAME_NUMS,SAVE_TEST_IMAGE=True,SAVE_DATA=True)
                ContourFname = self.findMaskContours(label_class,TRIAL,FRAME_NUMS,SAVE_TEST_IMAGE=True,SAVE_DATA=True)
                ContourFiles.append(ContourFname)
                classNameIndex+=1
            RingFile = self.findRingContoursTimed("ring",TRIAL,FRAME_NUMS,SAVE_DATA=True,SAVE_TEST_IMAGE=True)
        elif "Suturing" in TRIAL:
            for label_class in label_classes:
                #ContourFname = self.findAllContoursTask(label_class,label_classNames[classNameIndex],EPOCH,TRIAL,FRAME_NUMS,SAVE_TEST_IMAGE=True,SAVE_DATA=True)
                ContourFname = self.findMaskContours(label_class,TRIAL,FRAME_NUMS,SAVE_TEST_IMAGE=True,SAVE_DATA=True)
                ContourFiles.append(ContourFname)
                classNameIndex+=1
        return label_classes, ContourFiles, RingFile

    def ExtractContours(self, BATCH_SIZE, EPOCH,TRIAL, FRAME_NUMS):
        #label_classes, label_classNames = self.getLabelClassnames()
        classNameIndex=0
        task = self.task
        label_classes, label_classNames, ContourFiles = [],[],[]
        RingFile = ""
        if "Knot" in task:
            label_classes, label_classNames = ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks"], ["2023_grasper_L","2023_grasper_R","2023_thread" ]
            for label_class in label_classes:
                ContourFname = self.findAllContoursTimed(label_class,label_classNames[classNameIndex],EPOCH,TRIAL,FRAME_NUMS,SAVE_TEST_IMAGE=True,SAVE_DATA=True)
                ContourFiles.append(ContourFname)
                classNameIndex+=1
        # TODO: add the contourfiles for the other tasks
        elif "Needle" in task:
            label_classes, label_classNames = ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks","2023_needle_masks"], ["2023_grasper_L","2023_grasper_R","2023_thread","2023_needle"] # add Needle,
            for label_class in label_classes:
                ContourFname = self.findAllContoursTimed(label_class,label_classNames[classNameIndex],EPOCH,TRIAL,FRAME_NUMS,SAVE_TEST_IMAGE=True,SAVE_DATA=True)
                ContourFiles.append(ContourFname)
                classNameIndex+=1
            RingFile = self.findRingContoursTimed("2023_ring_masks","",EPOCH,TRIAL,FRAME_NUMS,SAVE_DATA=True)
        elif "Suturing" in task:
            label_classes, label_classNames =  ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks","2023_needle_masks"], ["2023_grasper_L","2023_grasper_R","2023_thread","2023_needle"] # add Needle
            for label_class in label_classes:
                ContourFname = self.findAllContoursTimed(label_class,label_classNames[classNameIndex],EPOCH,TRIAL,FRAME_NUMS,SAVE_TEST_IMAGE=True,SAVE_DATA=True)
                ContourFiles.append(ContourFname)
                classNameIndex+=1
        return label_classes, label_classNames, ContourFiles, RingFile

    def findMaskContours(self, label_class,TRIAL,FRAME_NUMS, SAVE_TEST_IMAGE=False, SAVE_DATA=False , DEBUG=False):
        # to find the contours for a single mask class,
        # we need the directory of all the .pngs we need to process
        # and where to output them
        TrialRoot = os.path.join(self.CWD,"data","masks",self.MASK_SET,label_class,TRIAL)
        #OutRoot = TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\2023_contour_images")
        OutRoot = os.path.join(self.CWD,"eval","contour_images",self.MASK_SET,label_class,TRIAL)
        PointsRoot = os.path.join(self.CWD,"data","contours",self.MASK_SET,label_class)

        VIATemplate =  os.path.join(self.CWD,"contour_template.json")
        VIAOutput =  os.path.join(PointsRoot,TRIAL+".json") # This is where we try to separate contour points by epoch
        # load json points for trial
        VIA = utils.ViaJSONTemplate(VIATemplate)

        if(not os.path.isdir(OutRoot)):
                path = pathlib.Path(OutRoot)
                path.mkdir(parents=True, exist_ok=True)

        if(not os.path.isdir(PointsRoot)):
                path = pathlib.Path(PointsRoot)
                path.mkdir(parents=True, exist_ok=True)

        for file in FRAME_NUMS:            
            file = "frame_"+file+".png"

            if ".png" not in file:
                continue
            
            imageFname = os.path.join(TrialRoot,file)
            if not os.path.isfile(imageFname): continue
            
            #img_3 = np.zeros([500,700,3],dtype=np.uint8)
            #img_3.fill(255)

            videoFrame = os.path.join(self.imagesDir,file)
            testFname = os.path.join(OutRoot,file)
            #frameNumber = int(file.replace(".png","").split("_")[1])
            im = cv.imread(imageFname)
            fileSizeInBytes = os.path.getsize(videoFrame)
            imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
            ret, thresh = cv.threshold(imgray, 1, 255, 0)

            contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
            #colors = 
            if(len(contours) ==0):continue
            areas = []
            largestArea = 0
            
            for k in range(len(contours)):                    
                cnt = contours[k]
                area = cv.contourArea(cnt)
                areas.append(area)
                if area>largestArea:
                    largestArea=area

            Regions = []
            RegionAttributes = []
            areasInOrderSaved = []
            
            rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
            for area in sorted(areas,reverse=True):
                origIndex = areas.index(area)
                # smoothing and drop out turned off
                #if len(Regions) <= 2:
                #    if area > 15 or len(Regions) == 0:
                if area < 20:
                    continue
                areasInOrderSaved.append(area)
                cnt = contours[origIndex]                        
                X = []
                Y = []
                epsilon = 0.001*cv.arcLength(cnt,True) #0.01 smaller number for less smoothing
                approx = cv.approxPolyDP(cnt,epsilon,True)
                pts = []
                for points in approx:
                    x =int(points[0][0])
                    y = int(points[0][1])
                    X.append(x)
                    Y.append(y)
                    pts.append([x,y])
                newShape = np.array([pts], np.int32)

                #cv.drawContours(im,[approx],0,rbg,1)
                cv.polylines(im, [newShape], True, (0,0,255), thickness=1)
                cv.putText(im,label_class,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                RegionAttributes.append(label_class)
                Regions.append([X,Y])
                #else: 
                #    break
            #if DEBUG:
            #    print(areasInOrderSaved,"------------",areas, end=" ")
            #print(areasInOrderSaved,end=" ")
                    
            VIA.addFrameMultiRegion(file, fileSizeInBytes, Regions, RegionAttributes)
            
            if SAVE_TEST_IMAGE:
                cv.imwrite(testFname,im)
            if DEBUG:
                if len(contours) > 2:
                    print("=======================================================================================================================================================================================================================>",LabelClassName)
                    print("len contours:",len(contours),hierarchy)
                elif len(contours) > 1:
                    print("===================================================================================>",label_class)
                    print("len contours:",len(contours),hierarchy)
                else:
                    print("len contours:",len(contours),hierarchy)
        
        if SAVE_DATA:
            VIA.save(VIAOutput)    
        return VIAOutput

    def findAllContoursTimed(self, LabelClass,LabelClassName,EPOCH,TRIAL,FRAME_NUMS, SAVE_TEST_IMAGE=False, SAVE_DATA=False , DEBUG=False):
        
        #print("Finding contours for object:",LabelClass, "for trial:",TRIAL)               
        TrialRoot = os.path.join(self.CWD,self.task,LabelClass,TRIAL)
        OutRoot = TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\2023_contour_images")
        PointsRoot = os.path.join(self.CWD,self.task,"2023_contour_points",LabelClass)
        #TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contour_points")
        VIATemplate =  os.path.join(self.CWD,"contour_template.json")
        VIAOutput =  os.path.join(PointsRoot,TRIAL+"_"+str(EPOCH)+".json") # This is where we try to separate contour points by epoch
        # load json points for trial
        VIA = utils.ViaJSONTemplate(VIATemplate)


        for file in FRAME_NUMS:            
            file = "frame_"+file+".png"

            if ".png" not in file:
                continue
            
            imageFname = os.path.join(TrialRoot,file)
            if not os.path.isfile(imageFname): continue
            if(not os.path.isdir(OutRoot)):
                path = pathlib.Path(OutRoot)
                path.mkdir(parents=True, exist_ok=True)
            if(not os.path.isdir(PointsRoot)):
                path = pathlib.Path(PointsRoot)
                path.mkdir(parents=True, exist_ok=True)
            
            #img_3 = np.zeros([500,700,3],dtype=np.uint8)
            #img_3.fill(255)

            #outFname =  os.path.join(OutRoot,file.replace(".png",".npy"))
            non_pred_name = file.replace("_pred","")
            videoFrame = os.path.join(self.imagesDir,TRIAL,non_pred_name)
            testFname = os.path.join(OutRoot,file)
            #frameNumber = int(file.replace(".png","").split("_")[1])
            im = cv.imread(imageFname)
            fileSizeInBytes = os.path.getsize(videoFrame)
            imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
            ret, thresh = cv.threshold(imgray, 1, 255, 0)

            contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
            #colors = 
            if(len(contours) ==0):continue
            areas = []
            largestIndex = -1
            largestArea = 0
            
            
            for k in range(len(contours)):                    
                cnt = contours[k]
                area = cv.contourArea(cnt)
                areas.append(area)
                if area>largestArea:
                    largestIndex=k
                    largestArea=area

            Regions = []
            RegionAttributes = []
            areasInOrderSaved = []
            
            rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
            for area in sorted(areas,reverse=True):
                origIndex = areas.index(area)
                # smoothing and drop out turned off
                #if len(Regions) <= 2:
                #    if area > 15 or len(Regions) == 0:
                areasInOrderSaved.append(area)
                cnt = contours[origIndex]                        
                X = []
                Y = []
                epsilon = 0.001*cv.arcLength(cnt,True) #0.01 smaller number for less smoothing
                approx = cv.approxPolyDP(cnt,epsilon,True)
                pts = []
                for points in approx:
                    x =int(points[0][0])
                    y = int(points[0][1])
                    X.append(x)
                    Y.append(y)
                    pts.append([x,y])
                newShape = np.array([pts], np.int32)

                #cv.drawContours(im,[approx],0,rbg,1)
                cv.polylines(im, [newShape], True, (0,0,255), thickness=1)
                cv.putText(im,LabelClass,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                RegionAttributes.append(LabelClassName)
                Regions.append([X,Y])
                #else: 
                #    break
            #if DEBUG:
            #    print(areasInOrderSaved,"------------",areas, end=" ")
            #print(areasInOrderSaved,end=" ")
                    
            VIA.addFrameMultiRegion(non_pred_name, fileSizeInBytes, Regions, RegionAttributes)
            
            if SAVE_TEST_IMAGE:
                cv.imwrite(testFname,im)
            if DEBUG:
                if len(contours) > 2:
                    print("=======================================================================================================================================================================================================================>",LabelClassName)
                    print("len contours:",len(contours),hierarchy)
                elif len(contours) > 1:
                    print("===================================================================================>",LabelClassName)
                    print("len contours:",len(contours),hierarchy)
                else:
                    print("len contours:",len(contours),hierarchy)
        
        if SAVE_DATA:
            VIA.save(VIAOutput)    
        return VIAOutput

    def findRingContoursTimed(self,label_class,TRIAL,FRAME_NUMS, SAVE_TEST_IMAGE=False, SAVE_DATA=False, DEBUG=False):
        TrialRoot = os.path.join(self.CWD,"data","masks",self.MASK_SET,label_class,TRIAL)

        OutRoot = os.path.join(self.CWD,"eval","contour_images",self.MASK_SET,label_class,TRIAL)
        PointsRoot = os.path.join(self.CWD,"data","contours",self.MASK_SET,label_class)
        #TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contour_points")
        VIATemplate =  os.path.join(self.CWD,"contour_template.json")
        VIAOutput =  os.path.join(PointsRoot,TRIAL+".json")
        # load json points for trial
        VIA = utils.ViaJSONTemplate(VIATemplate)

        if(not os.path.isdir(OutRoot)):
            path = pathlib.Path(OutRoot)
            path.mkdir(parents=True, exist_ok=True)

        if(not os.path.isdir(PointsRoot)):
            path = pathlib.Path(PointsRoot)
            path.mkdir(parents=True, exist_ok=True)

        
        for file in FRAME_NUMS:
            file = "frame_"+file+".png"
            if ".png" not in file:
                continue
            
            imageFname = os.path.join(TrialRoot,file)
            if not os.path.isfile(imageFname): continue

            #outFname =  os.path.join(OutRoot,file.replace(".png",".npy"))
            non_pred_name = file.replace("_pred","")
            videoFrame = os.path.join(self.imagesDir,non_pred_name)
            testFname = os.path.join(OutRoot,file)
            frameNumber = int(file.replace(".png","").split("_")[1])
            im = cv.imread(imageFname)
            fileSizeInBytes = os.path.getsize(videoFrame)
            imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
            ret, thresh = cv.threshold(imgray, 1, 255, 0)

            contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
            #colors = 

            #RegionAttributes = ["Ring_4","Ring_5","Ring_6","Ring_7"]
            RegionAttributes = []
            Regions = []
            if(len(contours) ==0):continue
            areas = []
            largestArea = 0
            for k in range(len(contours)):                    
                cnt = contours[k]
                area = cv.contourArea(cnt)
                if area < 15: 
                    continue
                areas.append(area)
                
                M = cv.moments(cnt)
                #print( M )
                try:
                    cx = int(M['m10']/(M['m00']+ 1e-5))
                    cy = int(M['m01']/(M['m00']+ 1e-5))
                except Exception as e:
                    print(e,"weird moment error")
                    continue
                ringID,closestIndex = self.idRing(cx,cy)                        
                #Rcontours[closestIndex].append(cnt)
                if area>largestArea:
                    largestIndex=k
                    largestArea=area
                X = []
                Y = []
                epsilon = 0.001*cv.arcLength(cnt,True)
                approx = cv.approxPolyDP(cnt,epsilon,True)
                for points in approx:
                    x =int(points[0][0])
                    y = int(points[0][1])
                    X.append(x)
                    Y.append(y)
                RegionAttributes.append("Ring_"+ringID)
                Regions.append([X,Y])
                rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                cv.putText(im,"Ring_"+ringID,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                cv.drawContours(im,[approx],0,rbg,1)

            ##### DRAW
            
            
            #rbg = tuple(int(colors[i].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
            #cv.drawContours(im,approx,0,rbg,thickness)
            
            #Regions = [[X,Y],[X,Y]]
            VIA.addFrameMultiRegion(non_pred_name, fileSizeInBytes, Regions, RegionAttributes)
            #VIA.addRings(file, LabelClassName, PolyPointsY)
            '''
            contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            thickness-=1
            for cnt in contours:
                #cnt = contours[4]
                rbg = tuple(int(colors[i].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                cv.drawContours(im, 
                                [cnt],
                                0, 
                                rbg, 
                                thickness)
                cv.putText(im,"mask_"+str(i),(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX, thickness,rbg)
                i=i+1
            
            '''
            if SAVE_TEST_IMAGE:                        
                cv.imwrite(testFname,im)
            if DEBUG:
                print("\tlen contours:",len(contours),'\n\t' + str(hierarchy).replace('\n', '\n\t'))
            #return
            #print(len(contours), end =" ")
        if SAVE_DATA:
            VIA.save(VIAOutput)
        return VIAOutput

    def idRing(self, cx, cy):
        points = [ [185,207],[290,213],[394,206],[497,236]]
        closestIndex = -1
        closestDist = 10000
        for i in range(len(points)):
            p = points[i]
            d =utils.distTwoPoints([cx,cy],p)
            if d < closestDist:
                closestDist = d
                closestIndex = i
        return str(closestIndex+4),closestIndex

    def findRingContours(self,LabelClass,LabelClassName, SAVE_TEST_IMAGE=False, SAVE_DATA=False, DEBUG=False):
        Dirs = []
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            break
        print("find Contours for Rings in trials:",Dirs)
        TrialNum = 0
        for Trial in Dirs:
            TrialRoot = os.path.join(self.CWD,self.task,LabelClass,Trial)
            OutRoot = TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contours")
            PointsRoot = os.path.join(self.CWD,self.task,"contour_points",LabelClass)
            #TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contour_points")
            VIATemplate =  os.path.join(self.CWD,"contour_template.json")
            VIAOutput =  os.path.join(PointsRoot,Trial+".json")
            # load json points for trial
            VIA = utils.ViaJSONTemplate(VIATemplate)
            print("\n\tTrial:",Trial)

            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if ".png" not in file:
                        continue
                    
                    imageFname = os.path.join(TrialRoot,file)
                    if(not os.path.isdir(OutRoot)):
                        path = pathlib.Path(OutRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    if(not os.path.isdir(PointsRoot)):
                        path = pathlib.Path(PointsRoot)
                        path.mkdir(parents=True, exist_ok=True)

                    #outFname =  os.path.join(OutRoot,file.replace(".png",".npy"))
                    non_pred_name = file.replace("_pred","")
                    videoFrame = os.path.join(self.imagesDir,Trial,non_pred_name)
                    testFname = os.path.join(OutRoot,file)
                    frameNumber = int(file.replace(".png","").split("_")[1])
                    im = cv.imread(imageFname)
                    fileSizeInBytes = os.path.getsize(videoFrame)
                    imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
                    ret, thresh = cv.threshold(imgray, 1, 255, 0)

                    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
                    #colors = 

                    #RegionAttributes = ["Ring_4","Ring_5","Ring_6","Ring_7"]
                    RegionAttributes = []
                    Regions = []
                    if(len(contours) ==0):continue
                    areas = []
                    largestIndex = -1
                    largestArea = 0
                    for k in range(len(contours)):                    
                        cnt = contours[k]
                        area = cv.contourArea(cnt)
                        areas.append(area)
                        
                        M = cv.moments(cnt)
                        #print( M )
                        try:
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                        except Exception as e:
                            print(e,"weird moment error")
                            continue
                        ringID,closestIndex = self.idRing(cx,cy)                        
                        #Rcontours[closestIndex].append(cnt)
                        if area>largestArea:
                            largestIndex=k
                            largestArea=area
                        X = []
                        Y = []
                        epsilon = 0.01*cv.arcLength(cnt,True)
                        approx = cv.approxPolyDP(cnt,epsilon,True)
                        for points in approx:
                            x =int(points[0][0])
                            y = int(points[0][1])
                            X.append(x)
                            Y.append(y)
                        RegionAttributes.append("Ring_"+ringID)
                        Regions.append([X,Y])

                    #ringIDs = ["Ring_4","Ring_5","Ring_6","Ring_7"]

                    rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))        
                    #cnt = contours[largestIndex]
                    
                    ##### DRAW
                    #cv.drawContours(im,[approx],0,rbg,1)
                    #cv.putText(im,LabelClassName,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                    
                    #rbg = tuple(int(colors[i].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                    #cv.drawContours(im,approx,0,rbg,thickness)
                    
                    

                    #Regions = [[X,Y],[X,Y]]
                    VIA.addFrameMultiRegion(non_pred_name, fileSizeInBytes, Regions, RegionAttributes)
                    #VIA.addRings(file, LabelClassName, PolyPointsY)
                    '''
                    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    thickness-=1
                    for cnt in contours:
                        #cnt = contours[4]
                        rbg = tuple(int(colors[i].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                        cv.drawContours(im, 
                                        [cnt],
                                        0, 
                                        rbg, 
                                        thickness)
                        cv.putText(im,"mask_"+str(i),(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX, thickness,rbg)
                        i=i+1
                    
                    '''
                    if SAVE_TEST_IMAGE:                        
                        cv.imwrite(testFname,im)
                    if DEBUG:
                        print("\tlen contours:",len(contours),'\n\t' + str(hierarchy).replace('\n', '\n\t'))
                    #return
                    print(len(contours), end =" ")
            if SAVE_DATA:
                VIA.save(VIAOutput)
            TrialNum+=1
        print("Processed ",TrialNum,"trials")

    def findAllContoursUnion(self, LabelClass,LabelClassName, trialName, filename):
        Dirs = []
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            break
        print("Trials:",Dirs)
        TrialNum = 0
        for Trial in Dirs:
            TrialRoot = os.path.join(self.CWD,self.task,LabelClass,Trial)
            OutRoot = TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contours")
            PointsRoot = os.path.join(self.CWD,self.task,"contour_points",LabelClass)
            #TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contour_points")
            VIATemplate =  os.path.join(self.CWD,"contour_template.json")
            VIAOutput =  os.path.join(PointsRoot,Trial+".json")
            # load json points for trial
            VIA = utils.ViaJSONTemplate(VIATemplate)

            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if ".png" not in file:
                        continue
                    
                    imageFname = os.path.join(TrialRoot,file)
                    if(not os.path.isdir(OutRoot)):
                        path = pathlib.Path(OutRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    if(not os.path.isdir(PointsRoot)):
                        path = pathlib.Path(PointsRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    
                    img_3 = np.zeros([1512,1512,3],dtype=np.uint8)
                    img_3.fill(255)

                    #outFname =  os.path.join(OutRoot,file.replace(".png",".npy"))
                    non_pred_name = file.replace("_pred","")
                    videoFrame = os.path.join(self.imagesDir,Trial,non_pred_name)
                    testFname = os.path.join(OutRoot,file)
                    frameNumber = int(file.replace(".png","").split("_")[1])
                    im = cv.imread(imageFname)
                    fileSizeInBytes = os.path.getsize(videoFrame)
                    imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
                    ret, thresh = cv.threshold(imgray, 1, 255, 0)

                    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
                    #colors = 
                    if(len(contours) ==0):continue
                    areas = []
                    largestIndex = -1
                    largestArea = 0
                    
                    for k in range(len(contours)):                    
                        cnt = contours[k]
                        area = cv.contourArea(cnt)
                        areas.append(area)
                        if area>largestArea:
                            largestIndex=k
                            largestArea=area
                    Regions = []
                    RegionAttributes = []
                    areasInOrderSaved = []
                    
                    rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                    for area in sorted(areas,reverse=True):
                        origIndex = areas.index(area)
                        if len(Regions) <= 2:
                            if area > 15 or len(Regions) == 0:
                                areasInOrderSaved.append(area)
                                cnt = contours[origIndex]                        
                                X = []
                                Y = []
                                epsilon = 0.01*cv.arcLength(cnt,True)
                                approx = cv.approxPolyDP(cnt,epsilon,True)
                                scalar = 10
                                pts = []
                                for points in approx:
                                    x =int(points[0][0])
                                    y = int(points[0][1])
                                    X.append(x)
                                    Y.append(y)
                                    pts.append([x,y])
                                newShape = np.array([pts], np.int32)

                                #cv.drawContours(im,[approx],0,rbg,1)
                                cv.polylines(img_3, [newShape], True, (0,0,255), thickness=8)
                                #cv.putText(im,LabelClassName,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                                RegionAttributes.append(LabelClassName)
                                Regions.append([X,Y])
                        else: 
                            break
                    print(areasInOrderSaved,"------------",areas)
                            
                    VIA.addFrameMultiRegion(non_pred_name, fileSizeInBytes, Regions, RegionAttributes)
                    #cv.imwrite(testFname,img_3)
                    if False:
                        if len(contours) > 2:
                            print("=======================================================================================================================================================================================================================>",LabelClassName)
                            print("len contours:",len(contours),hierarchy)
                        elif len(contours) > 1:
                            print("===================================================================================>",LabelClassName)
                            print("len contours:",len(contours),hierarchy)
                        else:
                            print("len contours:",len(contours),hierarchy)
                    #return
            VIA.save(VIAOutput)
            TrialNum+=1
        print("Processed ",TrialNum,"trials")

    def findAllContours(self, LabelClass,LabelClassName, SAVE_TEST_IMAGE=False, SAVE_DATA=False , DEBUG=False):
        
        Dirs = []
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            break
        print("Finding contours for object:",LabelClass, "for trials:",Dirs)
        TrialNum = 0
        for Trial in Dirs:
            TrialRoot = os.path.join(self.CWD,self.task,LabelClass,Trial)
            OutRoot = TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\2023_contour_images")
            PointsRoot = os.path.join(self.CWD,self.task,"2023_contour_points",LabelClass)
            #TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contour_points")
            VIATemplate =  os.path.join(self.CWD,"contour_template.json")
            VIAOutput =  os.path.join(PointsRoot,Trial+".json")
            # load json points for trial
            VIA = utils.ViaJSONTemplate(VIATemplate)
            print("\n\n\tTrial:",Trial,":",LabelClass)

            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if ".png" not in file:
                        continue
                    
                    imageFname = os.path.join(TrialRoot,file)
                    if(not os.path.isdir(OutRoot)):
                        path = pathlib.Path(OutRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    if(not os.path.isdir(PointsRoot)):
                        path = pathlib.Path(PointsRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    
                    img_3 = np.zeros([500,700,3],dtype=np.uint8)
                    img_3.fill(255)

                    #outFname =  os.path.join(OutRoot,file.replace(".png",".npy"))
                    non_pred_name = file.replace("_pred","")
                    videoFrame = os.path.join(self.imagesDir,Trial,non_pred_name)
                    testFname = os.path.join(OutRoot,file)
                    frameNumber = int(file.replace(".png","").split("_")[1])
                    im = cv.imread(imageFname)
                    fileSizeInBytes = os.path.getsize(videoFrame)
                    imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
                    ret, thresh = cv.threshold(imgray, 1, 255, 0)

                    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
                    #colors = 
                    if(len(contours) ==0):continue
                    areas = []
                    largestIndex = -1
                    largestArea = 0
                    
                    
                    for k in range(len(contours)):                    
                        cnt = contours[k]
                        area = cv.contourArea(cnt)
                        areas.append(area)
                        if area>largestArea:
                            largestIndex=k
                            largestArea=area

                    Regions = []
                    RegionAttributes = []
                    areasInOrderSaved = []
                    
                    rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                    for area in sorted(areas,reverse=True):
                        origIndex = areas.index(area)
                        # smoothing and drop out turned off
                        #if len(Regions) <= 2:
                        #    if area > 15 or len(Regions) == 0:
                        areasInOrderSaved.append(area)
                        cnt = contours[origIndex]                        
                        X = []
                        Y = []
                        epsilon = 0.001*cv.arcLength(cnt,True) #0.01 smaller number for less smoothing
                        approx = cv.approxPolyDP(cnt,epsilon,True)
                        pts = []
                        for points in approx:
                            x =int(points[0][0])
                            y = int(points[0][1])
                            X.append(x)
                            Y.append(y)
                            pts.append([x,y])
                        newShape = np.array([pts], np.int32)

                        #cv.drawContours(im,[approx],0,rbg,1)
                        cv.polylines(img_3, [newShape], True, (0,0,255), thickness=1)
                        #cv.putText(im,LabelClassName,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                        RegionAttributes.append(LabelClassName)
                        Regions.append([X,Y])
                        #else: 
                        #    break
                    if DEBUG:
                        print(areasInOrderSaved,"------------",areas, end=" ")
                    print(areasInOrderSaved,end=" ")
                            
                    VIA.addFrameMultiRegion(non_pred_name, fileSizeInBytes, Regions, RegionAttributes)
                    
                    if SAVE_TEST_IMAGE:
                        cv.imwrite(testFname,img_3)
                    if DEBUG:
                        if len(contours) > 2:
                            print("=======================================================================================================================================================================================================================>",LabelClassName)
                            print("len contours:",len(contours),hierarchy)
                        elif len(contours) > 1:
                            print("===================================================================================>",LabelClassName)
                            print("len contours:",len(contours),hierarchy)
                        else:
                            print("len contours:",len(contours),hierarchy)
                    #return
            if SAVE_DATA:
                VIA.save(VIAOutput)
            TrialNum+=1
        print("Processed ",TrialNum,"trials")

    def findContours(self, LabelClass,LabelClassName):
        Dirs = []
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            break
        print("Trials:",Dirs)
        TrialNum = 0
        for Trial in Dirs:
            TrialRoot = os.path.join(self.CWD,self.task,LabelClass,Trial)
            OutRoot = TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contours")
            PointsRoot = os.path.join(self.CWD,self.task,"contour_points",LabelClass)
            #TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contour_points")
            VIATemplate =  os.path.join(self.CWD,"contour_template.json")
            VIAOutput =  os.path.join(PointsRoot,Trial+".json")
            # load json points for trial
            VIA = utils.ViaJSONTemplate(VIATemplate)

            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if ".png" not in file:
                        continue
                    
                    imageFname = os.path.join(TrialRoot,file)
                    if(not os.path.isdir(OutRoot)):
                        path = pathlib.Path(OutRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    if(not os.path.isdir(PointsRoot)):
                        path = pathlib.Path(PointsRoot)
                        path.mkdir(parents=True, exist_ok=True)

                    #outFname =  os.path.join(OutRoot,file.replace(".png",".npy"))
                    non_pred_name = file.replace("_pred","")
                    videoFrame = os.path.join(self.imagesDir,Trial,non_pred_name)
                    testFname = os.path.join(OutRoot,file)
                    frameNumber = int(file.replace(".png","").split("_")[1])
                    im = cv.imread(imageFname)
                    fileSizeInBytes = os.path.getsize(videoFrame)
                    imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
                    ret, thresh = cv.threshold(imgray, 1, 255, 0)

                    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
                    #colors = 
                    if(len(contours) ==0):continue
                    areas = []
                    largestIndex = -1
                    largestArea = 0
                    
                    for k in range(len(contours)):                    
                        cnt = contours[k]
                        area = cv.contourArea(cnt)
                        areas.append(area)
                        if area>largestArea:
                            largestIndex=k
                            largestArea=area
                    rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))        
                    cnt = contours[largestIndex]
                    epsilon = 0.01*cv.arcLength(cnt,True)
                    approx = cv.approxPolyDP(cnt,epsilon,True)
                    cv.drawContours(im,[approx],0,rbg,1)
                    cv.putText(im,LabelClassName,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                    #rbg = tuple(int(colors[i].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                    #cv.drawContours(im,approx,0,rbg,thickness)
                    X = []
                    Y = []
                    for points in approx:
                        x =int(points[0][0])
                        y = int(points[0][1])
                        X.append(x)
                        Y.append(y)
                    VIA.addFrame(non_pred_name, fileSizeInBytes, X,Y)
                    #VIA.addRings(file, LabelClassName, PolyPointsY)
                    '''
                    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    thickness-=1
                    for cnt in contours:
                        #cnt = contours[4]
                        rbg = tuple(int(colors[i].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                        cv.drawContours(im, 
                                        [cnt],
                                        0, 
                                        rbg, 
                                        thickness)
                        cv.putText(im,"mask_"+str(i),(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX, thickness,rbg)
                        i=i+1
                    
                    '''
                    cv.imwrite(testFname,im)
                    print("len contours:",len(contours),hierarchy)
                    #return
            VIA.save(VIAOutput)
            TrialNum+=1
        print("Processed ",TrialNum,"trials")
