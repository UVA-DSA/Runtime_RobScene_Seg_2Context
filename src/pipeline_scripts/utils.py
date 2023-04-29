import os, sys
from re import template
import json
import pathlib
import math
from turtle import position
from xmlrpc.client import Boolean
from cv2 import KeyPoint, threshold
from matplotlib.colors import cnames
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont
from scipy import interpolate
import time
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
from shapely.geometry import LineString

# from dataclasses import dataclass
# from itertools import accumulate
from scipy import ndimage

# from imantics import Polygons, Mask
import utils


def distTwoPoints(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def isIntersecting(maskA, maskB):
    rows = len(maskA)
    cols = len(maskA[0])
    for i in range(rows):
        for j in range(cols):
            if maskA[i][j] == 1 and maskB[i][j] == 1:
                return True, (i, j)
    return False, (0, 0)


def imageToTXT(file):
    fileArr = file.split(".")
    return "".join(fileArr[:-1]) + ".txt"


def imageToJSON(file):
    fileArr = file.split(".")
    return "".join(fileArr[:-1]) + ".json"


def imageToNPY(file):
    fileArr = file.split(".")
    temp = "".join(fileArr[:-1]) + ".npy"
    return temp.replace(".npy", "_gt_pred.npy")


def CrossProduct(A):
    # Stores coefficient of X
    # direction of vector A[1]A[0]
    X1 = A[1][0] - A[0][0]

    # Stores coefficient of Y
    # direction of vector A[1]A[0]
    Y1 = A[1][1] - A[0][1]

    # Stores coefficient of X
    # direction of vector A[2]A[0]
    X2 = A[2][0] - A[0][0]

    # Stores coefficient of Y
    # direction of vector A[2]A[0]
    Y2 = A[2][1] - A[0][1]

    # Return cross product
    return X1 * Y2 - Y1 * X2


# Function to check if the polygon is
# convex polygon or not
def isConvex(points):

    # Stores count of
    # edges in polygon
    N = len(points)

    # Stores direction of cross product
    # of previous traversed edges
    prev = 0

    # Stores direction of cross product
    # of current traversed edges
    curr = 0

    # Traverse the array
    for i in range(N):

        # Stores three adjacent edges
        # of the polygon
        temp = [points[i], points[(i + 1) % N], points[(i + 2) % N]]

        # Update curr
        curr = CrossProduct(temp)

        # If curr is not equal to 0
        if curr != 0:

            # If direction of cross product of
            # all adjacent edges are not same
            if curr * prev < 0:
                return False
            else:

                # Update curr
                prev = curr

    return True


def dist(self, A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def save(file, lines):
    with open(file, "w+") as f:
        for item in lines:
            f.write("%s\n" % item)

def saveAppend(file, lines):
    with open(file,  "a") as f:
        for item in lines:
            f.write("%s\n" % item)


"""
JSONInterface deals with the particular JSON format of the annotations
It's set up now to read labels from Via
"""


class ViaJSONInterface:
    def __init__(self, jsonLoc,TestMode=False,MatchLines=0):
        self.json_location = jsonLoc
        self.frameNumbers = []
        self.dataArr = []
        with open(self.json_location) as f:
            data = json.load(f)
            # self.data = data
            # self.imageDict = data["_via_img_metadata"]
            imageDict = data["_via_img_metadata"]
            self.dataDict = {}
            for frame in imageDict:
                # now song is a dictionary
                frameName = frame[: len("frame_0080.png")]
                frameNumber = int(frameName.replace("frame_", "").replace(".png", ""))
                self.frameNumbers.append(frameNumber)
                # print("Saved Via Label:",frameName)
                regions = data["_via_img_metadata"][frame]["regions"]
                points = [
                    [region["shape_attributes"]["cx"], region["shape_attributes"]["cy"]]
                    for region in regions
                ]
                self.dataDict[str(frameNumber)] = points
                self.dataArr.append(points)
                if(len(regions) != MatchLines and TestMode):
                    print("\t\t Error: unexpected number of datapoints (",len(regions),MatchLines,") ",jsonLoc,"frame",frameNumber )
                # for attribute, value in song.items():
                #    print(attribute, value) # example usage

    def getDataDict(self):
        return self.dataDict

    def getFrameNumbers(self):
        return self.frameNumbers

    def getDataArr(self):
        return self.dataArr

class VIARingJSON:
    def __init__(self, jsonLoc):
        self.json_location = jsonLoc
        self.frameNumbers = []
        self.dataArr = []
        self.ringArr = []

        self.R4 = {}
        self.R5 = {}
        self.R6 = {}
        self.R7 = {}
        
        # print("VIAjsonLoc:",jsonLoc)
        with open(self.json_location) as f:
            data = json.load(f)
            self.dataDict = {}
            self.ringDict = {}
            for frame in data:
                # now song is a dictionary
                thisFrame = data[frame]
                frameName = thisFrame["filename"]
                frameNumber = int(frameName.replace("frame_", "").replace(".png", ""))
                self.frameNumbers.append(frameNumber)
                # print("Saved Via Label:",frameName)
                regions = thisFrame["regions"]
                r4p = []
                r5p = []
                r6p = []
                r7p = []

                for region in regions:
                    points = [(region["shape_attributes"]["all_points_x"][i], region["shape_attributes"]["all_points_y"][i]) for i in range(len(region["shape_attributes"]["all_points_y"]))]
                    ringID =  region["region_attributes"]["ringID"]                     
                    if("4" in ringID):
                        r4p.append(points)
                    if("5" in ringID):
                        r5p.append(points)
                    if("6" in ringID):
                        r6p.append(points)
                    if("7" in ringID):
                        r7p.append(points)

                self.R4[str(frameNumber)] = r4p
                self.R5[str(frameNumber)] = r5p
                self.R6[str(frameNumber)] = r6p
                self.R7[str(frameNumber)] = r7p
                
               # self.dataDict[str(frameNumber)] = points
                #self.dataArr.append(points)
                #self.ringDict[str(frameNumber)] = ringIDs
                # for attribute, value in song.items():
                #    print(attribute, value) # example usage
    def getRingsPoints(self):
        return self.R4,self.R5,self.R6,self.R7
    def getDataDict(self):
        return self.dataDict

    def getFrameNumbers(self):
        return self.frameNumbers
    
    def getRingIDs(self):
        return self.ringArr

    def getDataArr(self):
        return self.dataArr

class VIAPolyJSON:
    def __init__(self, jsonLoc):
        self.json_location = jsonLoc
        self.frameNumbers = []
        self.dataArr = []
        # print("VIAjsonLoc:",jsonLoc)
        with open(self.json_location) as f:
            data = json.load(f)
            self.dataDict = {}
            for frame in data:
                # now song is a dictionary
                thisFrame = data[frame]
                frameName = thisFrame["filename"]
                frameNumber = int(frameName.replace("frame_", "").replace(".png", ""))
                self.frameNumbers.append(frameNumber)
                # print("Saved Via Label:",frameName)
                regions = thisFrame["regions"]
                points = [[(region["shape_attributes"]["all_points_x"][i], region["shape_attributes"]["all_points_y"][i]) for i in range(len(region["shape_attributes"]["all_points_y"]))]for region in regions]
                self.dataDict[str(frameNumber)] = points
                self.dataArr.append(points)
                # for attribute, value in song.items():
                #    print(attribute, value) # example usage

    def getDataDict(self):
        return self.dataDict

    def getFrameNumbers(self):
        return self.frameNumbers

    def getDataArr(self):
        return self.dataArr

    def getClassKey(self, label_class):
        label_classes = [
            "2023_grasper_L_masks",
            "2023_grasper_R_masks",
            "2023_needle_masks",
            "2023_thread_masks",
        ]
        label_classNames = ["2023_grasper_L", "2023_grasper_R", "2023_needle", "2023_thread"]
        for i in range(len(label_class)):
            if label_classes[i] in label_class:
                return label_classNames[i]


class ViaJSONTemplate:
    def __init__(self, jsonLoc):
        self.json_location = jsonLoc
        with open(self.json_location) as f:
            self.data = json.load(f)
            self.data["_via_img_metadata"] = {}
            self._via_image_id_list = []
            f.close()

    def addFrameMultiRegion(self, file, fileSizeInBytes, Regions, RegionAttributes):
        # self.data["_via_img_metadata"][file+str(fileSizeInBytes)] = self.imageEntry
        thisFrame = {}
        thisFrame["filename"] = file
        thisFrame["size"] = fileSizeInBytes

        thisFrame["regions"] = []
        i=0
        for r in Regions:     
            RegionAttribute = RegionAttributes[i]
            [X,Y] = r
            region = {}
            region["shape_attributes"] = {}
            region["shape_attributes"]["name"] = "polygon"
            region["shape_attributes"]["all_points_x"] = X
            region["shape_attributes"]["all_points_y"] = Y
            region["region_attributes"] = {"ringID":RegionAttribute}

            thisFrame["regions"].append(region)
            i+=1

        thisFrame["file_attributes"] = {}

        try:
            self._via_image_id_list.append(file)
        except Exception as e:
            print(e, "list probably empty")
            self._via_image_id_list = ["" + file]

        self.data["_via_img_metadata"][file + str(fileSizeInBytes)] = thisFrame
    def addFrame(self, file, fileSizeInBytes, X, Y):
        # self.data["_via_img_metadata"][file+str(fileSizeInBytes)] = self.imageEntry
        thisFrame = {}
        thisFrame["filename"] = file
        thisFrame["size"] = fileSizeInBytes

        regions = {}

        regions["shape_attributes"] = {}
        regions["shape_attributes"]["name"] = "polygon"
        regions["shape_attributes"]["all_points_x"] = X
        regions["shape_attributes"]["all_points_y"] = Y
        regions["region_attributes"] = {}

        thisFrame["regions"] = []
        thisFrame["file_attributes"] = {}
        thisFrame["regions"].append(regions)

        try:
            self._via_image_id_list.append(file)
        except Exception as e:
            print(e, "list probably empty")
            self._via_image_id_list = ["" + file]

        self.data["_via_img_metadata"][file + str(fileSizeInBytes)] = thisFrame

    def save(self, outFName):
        self.data["_via_image_id_list"] = self._via_image_id_list
        with open(outFName, "w+") as f:
            json.dump(self.data["_via_img_metadata"], f)


"""
JSONInterface deals with the particular JSON format of the annotations
It's set up now to read labels as we received them from Cogito

If the JSON annotations are in a different format, you can edit the getPolygons and getKeyPoints methods
"""


class CogitoJSONInterface:
    def __init__(self, jsonLoc):
        self.json_location = jsonLoc
        with open(self.json_location) as f:
            data = json.load(f)
            self.data = data
            self.meta = data["metadata"]
            self.instances = data["instances"]

    """
    Returns a list of polygons
    each polygon is a list of points ordered as [x1, y1, x2, y2, ... , xn, yn]
    """

    def getPolygons(self):
        polygonSeries = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "polygon":
                polygonSeries.append(instance["points"])
                cn.append(instance["className"])
        return cn, polygonSeries

    """
    Returns a list of PolyLines
    each polyline is a list of points ordered as [x1, y1, x2, y2, ... , xn, yn]
    """

    def getKeyPoints(self):
        keyPoints = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "point":
                keyPoints.append([instance["x"], instance["y"]])
                cn.append(instance["className"])
        return cn, keyPoints

    """
    Returns a list of PolyLines
    each PolyLine is a list [x, y]
    """

    def getPolyLines(self):
        polylineSeries = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "polyline":
                polylineSeries.append(instance["points"])
                cn.append(instance["className"])
        return cn, polylineSeries


class NPYInterface3:
    def __init__(self):
        pass

    def loadArr(maskFName):
        [mask_gt, mask] = np.load(maskFName, allow_pickle=True)
        mask[mask > 0.95] = 1  #! instead of 0.97
        mask[mask < 0.95] = 0  #! instead of 0.97
        mask = np.squeeze(mask)
        return mask


class NPYInterface2:
    def __init__(self):
        pass

    def getIntersection_cached(self, grasper_gt, grasper, threadSource):
        [grasper_gt, grasper] = grasper_gt, grasper
        [thread_gt, thread] = np.load(threadSource, allow_pickle=True)
        grasper[grasper > 38] = 1  #! instead of 0.97
        grasper[grasper < 38] = 0  #! instead of 0.97
        thread[thread > 38] = 1
        thread[thread < 38] = 0
        grasper = np.squeeze(grasper)
        thread = np.squeeze(thread)
        (x_center, y_center) = ndimage.center_of_mass(grasper)
        # drawObject = plt.Circle((y_center,x_center),radius=10,color='red', fill=True)
        inter, (i, j) = self.isIntersecting(grasper, thread)
        # print(inter)
        return (y_center, x_center), (i, j), inter

    def getIntersectionFastest(self, s1, s2):
        (x_center, y_center) = (0, 0)
        # drawObject = plt.Circle((y_center,x_center),radius=10,color='red', fill=True)
        inter, (i, j) = self.isIntersecting(s1, s2)
        # print(inter)
        return (y_center, x_center), (i, j), inter

    def getIntersectionFast(self, grasperSource, threadSource):
        [grasper_gt, grasper] = np.load(grasperSource, allow_pickle=True)
        [thread_gt, thread] = np.load(threadSource, allow_pickle=True)
        grasper[grasper > 0.95] = 1  #! instead of 0.97
        grasper[grasper < 0.95] = 0  #! instead of 0.97
        thread[thread > 0.95] = 1
        thread[thread < 0.95] = 0
        grasper = np.squeeze(grasper)
        thread = np.squeeze(thread)
        (x_center, y_center) = (0, 0)
        # drawObject = plt.Circle((y_center,x_center),radius=10,color='red', fill=True)
        inter, (i, j) = self.isIntersecting(grasper, thread)
        # print(inter)
        return (y_center, x_center), (i, j), inter

    def getIntersection(self, grasperSource, threadSource):
        [grasper_gt, grasper] = np.load(grasperSource, allow_pickle=True)
        [thread_gt, thread] = np.load(threadSource, allow_pickle=True)
        grasper[grasper > 0.95] = 1  #! instead of 0.97
        grasper[grasper < 0.95] = 0  #! instead of 0.97
        thread[thread > 0.95] = 1
        thread[thread < 0.95] = 0
        grasper = np.squeeze(grasper)
        thread = np.squeeze(thread)
        (x_center, y_center) = ndimage.center_of_mass(grasper)
        # drawObject = plt.Circle((y_center,x_center),radius=10,color='red', fill=True)
        inter, (i, j) = self.isIntersecting(grasper, thread)
        # print(inter)
        return (y_center, x_center), (i, j), inter

    def isIntersecting(self, maskA, maskB):
        rows = len(maskA)
        cols = len(maskA[0])
        for i in range(rows):
            for j in range(cols):
                if maskA[i][j] == 1 and maskB[i][j] == 1:
                    return True, (i, j)
        return False, (0, 0)


class NPYInterface:
    def __init__(self, npyLoc):
        self.grasper_loc = npyLoc
        self.thread_loc = npyLoc.replace("deeplab_grasper_v1", "deeplab_thread_v1")
        self.needle_loc = npyLoc.replace("deeplab_grasper_v1", "deeplab_needle_v1")
        self.ring_loc = npyLoc.replace("deeplab_grasper_v1", "deeplab_ring_v1")
        # with open(self.npyarr_location) as f:
        self.grasperData = np.load(self.grasper_loc, allow_pickle=True)
        print("\t\t\t GrasperData", type(self.grasperData), self.grasperData.shape)
        self.g1 = self.grasperData[0]
        self.g2 = self.grasperData[1]
        print("\t\t\t l1", self.g1.shape, "l2", self.g2.shape)
        try:
            self.threadData = np.load(self.thread_loc, allow_pickle=True)
            print("\t\t\t ThreadData", self.threadData.shape)
            self.t1 = self.threadData[0]
            self.t2 = self.threadData[1]
            print("\t\t\t T1", self.t1.shape, "T2", self.t2.shape)
        except:
            pass
        try:
            self.needleData = np.load(self.needle_loc, allow_pickle=True)
        except:
            pass

        try:
            self.ringData = np.load(self.ring_loc, allow_pickle=True)
        except:
            pass
        self.instances = []

    def getPolygons(self):
        polygonSeries = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "polygon":
                polygonSeries.append(instance["points"])
                cn.append(instance["className"])
        return cn, polygonSeries

    def getKeyPoints(self):
        keyPoints = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "point":
                keyPoints.append([instance["x"], instance["y"]])
                cn.append(instance["className"])
        return cn, keyPoints

    def getPolyLines(self):
        polylineSeries = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "polyline":
                polylineSeries.append(instance["points"])
                cn.append(instance["className"])
        return cn, polylineSeries


class MPInterface:
    def __init__(self, MPLoc):
        self.mp_loc = MPLoc
        self.transcript = []
        with open(self.mp_loc) as file:
            for line in file:
                # print(line.rstrip())
                self.transcript.append(line.rstrip())

    def getMP(self, index):
        # print("GetMP Matching",index)
        for i in range(1, len(self.transcript)):
            l_s = self.transcript[i].split(" ")
            if int(l_s[1]) > index:
                return " ".join(l_s)


class ContextInterface:
    def __init__(self, ContextLoc, enable):
        self.empty = True
        if enable:
            return
        self.c_loc = ContextLoc
        self.transcript = []
        try:
            with open(self.c_loc) as file:
                for line in file:
                    self.transcript.append(line.rstrip())
            self.empty = len(self.transcript) == 0
        except Exception as e:
            # print(e)
            pass

    def getContext(self, index):
        if self.empty:
            return "Building Prediction"
        for i in range(1, len(self.transcript)):
            l_s = self.transcript[i].split(" ")
            if int(l_s[0]) > index:
                # return " ".join(min(0,i-1))
                return self.transcript[max(0, i - 1)]


class ContextInterface2:
    def __init__(self, ContextLoc):
        self.c_loc = ContextLoc
        self.transcript = []
        try:
            with open(self.c_loc) as file:
                for line in file:
                    self.transcript.append(line.rstrip())
            self.empty = len(self.transcript) == 0
        except Exception as e:
            print(e)
            pass

    def getContext(self, index):

        for i in range(1, len(self.transcript)):
            l_s = self.transcript[i].split(" ")
            if int(l_s[0]) > index:
                # return " ".join(min(0,i-1))
                return self.transcript[max(0, i - 1)]

        return ""


class JSONInterface:
    def __init__(self, jsonLoc):
        self.json_location = jsonLoc
        with open(self.json_location) as f:
            data = json.load(f)
            self.data = data
            self.meta = data["metadata"]
            self.instances = data["instances"]

    """
    Returns a list of polygons
    each polygon is a list of points ordered as [x1, y1, x2, y2, ... , xn, yn]
    """

    def getPolygons(self):
        polygonSeries = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "polygon":
                polygonSeries.append(instance["points"])
                cn.append(instance["className"])
        return cn, polygonSeries

    def getPolygonsDict(self):
        polyDict = {}
        polygonSeries = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "polygon":
                if instance["className"] not in polyDict.keys():
                    polyDict[instance["className"]] = []
                polyDict[instance["className"]].append(instance["points"])
        return polyDict

    """
    Returns a list of PolyLines
    each polyline is a list of points ordered as [x1, y1, x2, y2, ... , xn, yn]
    """

    def getKeyPoints(self):
        keyPoints = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "point":
                keyPoints.append([instance["x"], instance["y"]])
                cn.append(instance["className"])
        return cn, keyPoints

    def getKeyPointsDict(self):
        kpDict = {}
        for instance in self.instances:
            instance_type = instance["type"]
            if instance_type == "point":
                if instance["className"] not in kpDict.keys():
                    kpDict[instance["className"]] = []
                kpDict[instance["className"]].append([instance["x"], instance["y"]])
        return kpDict

    """
    Returns a list of PolyLines
    each PolyLine is a list [x, y]
    """

    def getPolyLines(self):
        polylineSeries = list()
        cn = list()
        for instance in self.instances:
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if instance_type == "polyline":
                polylineSeries.append(instance["points"])
                cn.append(instance["className"])
        return cn, polylineSeries

    def getPolyLinesDict(self):
        linesDict = {}
        for instance in self.instances:
            instance_type = instance["type"]
            if instance_type == "polyline":
                if instance["className"] not in linesDict.keys():
                    linesDict[instance["className"]] = []
                linesDict[instance["className"]].append(instance["points"])
        return linesDict
