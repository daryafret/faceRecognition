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

def get_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def get_top_one_error(pred_bbox,pred_class,gt_bbox, gt_class, countTrue, countFalse):
    countFalse+=len(gt_class)
    for i in range(len(gt_bbox)):
        maxIOU=0
        t=False
        for j in range(len(pred_bbox)):
            IOU=get_iou(pred_bbox[j], gt_bbox[i])
            if (IOU>maxIOU):
                maxIOU=IOU
                if (j>=len(gt_bbox)):
                    t=False
                else:
                    t=pred_class[j]==gt_class[i]
        if maxIOU>=0.5 and t==True:
            countTrue+=1
        if maxIOU<0.5:
            countFalse-=1
        return countTrue, countFalse

countTrue=0
countFalse=0

path_to_dataset= '/home/anastasiia/Documents/project/dataset/'
name_dir=["All", "Asyok", "Ion", "Nastya", "daryafret", "Malinka"]
#name_dir=["Malinka"]
mode="train"

path_predicted = current_path+'/mAP-master/data/predicted/'
path_to_save_for_mAP = current_path+"/mAP-master/input/ground-truth/"
path_to_save_for_other = current_path+'/mAP-master/data/groundtruth/'

#path_predicted = current_path+'/data/predicted/'
#path_to_save_for_mAP = current_path+"/input/ground-truth/"
#path_to_save_for_other = current_path+'/data/groundtruth/'


#run pipeline
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

#write ground-truth files
for name in name_dir:
    path_to_json_file=path_to_dataset+name+'/'+mode+'/'+name+'_'+mode+'.json'
    if (name!="Ion" or (name=="Ion" and mode=="train")):
        with open(path_to_json_file) as json_file:  
            data = json.load(json_file)
        keys=list(data.keys())
        key = keys[0]
        for key in keys:
            fname = re.search(r'^([^.]+)', data[key]['filename']).group(0)
            final_fname_map=path_to_save_for_mAP+fname+'.txt'
            f_map = open(final_fname_map,"w")
            final_fname_other=path_to_save_for_other+fname+'.txt'
            f_other = open(final_fname_other,"w")
            regions = data[key]['regions']
            for region in regions:
                bbox = region['shape_attributes']
                f_map.write("detection "+ str(bbox['x'])+ " " + str(bbox['y']) +" " + str(bbox['x']+bbox['width']) + " " + str(bbox['y']+bbox['height'])+'\n')
                f_other.write(region['region_attributes']['class']+" "+ str(bbox['x'])+ " " + str(bbox['y']) +" " + str(bbox['x']+bbox['width']) + " " + str(bbox['y']+bbox['height'])+'\n')
            f_map.close()
            f_other.close()
    else:
        with open(path_to_json_file) as json_file:  
            data = json.load(json_file)
        keys=list(data.keys())
        key_for_images=keys[1]
        data_image=list(data[key_for_images].keys());
        for key in data_image:
            fname = re.search(r'^([^.]+)', data[key_for_images][key]['filename']).group(0)
            final_fname_map=path_to_save_for_mAP+fname+'.txt'
            f_map = open(final_fname_map,"w")
            final_fname_other=path_to_save_for_other+fname+'.txt'
            f_other = open(final_fname_other,"w")
            regions = data[key_for_images][key]['regions']
            for region in regions:
                bbox = region['shape_attributes']
                f_map.write("detection "+ str(bbox['x'])+ " " + str(bbox['y']) +" " + str(bbox['x']+bbox['width']) + " " + str(bbox['y']+bbox['height']) +'\n')
                f_other.write(region['region_attributes']['class']+" "+ str(bbox['x'])+ " " + str(bbox['y']) +" " + str(bbox['x']+bbox['width']) + " " + str(bbox['y']+bbox['height'])+'\n')    
            f_map.close()
            f_other.close()

#some calculation for top-1
predicred_files = []

for r, d, f in os.walk(path_predicted):
    for file in f:
        predicred_files.append(os.path.join(r, file))

for f in predicred_files:
    with open(f) as file:
        fname=os.path.basename(f)
        fnameWithoutExt=os.path.splitext(fname)[0]
        file_contents = file.read()

    #get predicted data from files
    pred_bbox = []
    allRect = re.findall(r'\d.*', file_contents)
    for bbox in allRect:
        rect = bbox.split()
        pred_bbox.append([int(rect[1]), int(rect[2]), int(rect[3]), int(rect[4])])

    pred_class=[]
    allClasses = re.findall(r'(Ion|Asyok|daryafret|Malinka|Nastya|Unknown)', file_contents)
    for c in allClasses:
        pred_class.append(c)
    # print(pred_class)

    #get groundTruth data from files
    path_gt = path_to_save_for_other+fnameWithoutExt+'.txt'
    with open(path_gt) as gt_file:
        gt_file_contents = gt_file.read()

    gt_bbox = []
    gt_class = []
    gt_allRect = re.findall(r'\d.*', gt_file_contents)
    for bbox in gt_allRect:
        gt_rect = bbox.split()
        gt_bbox.append([int(gt_rect[0]), int(gt_rect[1]), int(gt_rect[2]), int(gt_rect[3])])

    gt_allClasses = re.findall(r'(Ion|Asyok|daryafret|Malinka|Nastya|Unknown)', gt_file_contents)
    for c in gt_allClasses:
        gt_class.append(c)

    countTrue,countFalse =get_top_one_error(pred_bbox,pred_class,gt_bbox, gt_class, countTrue, countFalse) 


countTrue=countTrue+0.0
#print(countTrue)
#print(countFalse)
print("top-1 error = " + str(countTrue/countFalse))     
   
#subprocess.call("python3 " + os.path.dirname(os.path.abspath(sys.argv[0])) + "/main.py", shell=True)
subprocess.call("python3 " + os.path.dirname(os.path.abspath(sys.argv[0])) + "/mAP-master/main.py", shell=True)

