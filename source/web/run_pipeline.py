#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import re
import json
import sys,os
import subprocess
import cv2

current_path=os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, current_path)

import face_recognition_adapter as adapter

path_to_dataset= '/home/anastasiia/Documents/project/dataset/'
name_dir=["All", "Asyok", "Ion", "Nastya", "daryafret", "Malinka"]
#name_dir=["Malinka"]
#mode=["train" , "test"]
mode="train"

for name in name_dir:
    path_images_in_dir=[]
    p=path_to_dataset+name+"/"+mode+"/";
    for r, d, f in os.walk(p):
        for file in f:
            if '.jpg' in file:
                path_images_in_dir.append(os.path.join(r, file))         

    for image_path in path_images_in_dir:
        image = cv2.imread(image_path)
        fname=os.path.basename(image_path)
        fnameWithoutExt=os.path.splitext(fname)[0]

        path_for_calculate_map=current_path+"/mAP-master/input/detection-results/"+fnameWithoutExt+".txt"
        path_for_result_detection_net=current_path+"/mAP-master/data/predicted/"+fnameWithoutExt+".txt"

        detects, recogns, aligns, time  = adapter.recognize_faces(image,path_for_calculate_map,path_for_result_detection_net);
