import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
from torchvision import models
import random
import math



yolov5_weight = 'yolov5s.pt' # yolov5 model



frame_can_be_missed = 50 # exceding this value will remove that object id from the list
min_dist_threshold_value = 100 # if an object appear this many px far from where it was lost, then that object will assigned a different id



################################################## object detection #####################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(yolov5_weight, map_location=device)
stride = int(model.stride.max()) 
cudnn.benchmark = True
names = model.module.names if hasattr(model, 'module') else model.names


def object_detection(frame):
	img = torch.from_numpy(frame)
	img = img.permute(2, 0, 1 ).float().to(device)
	img /= 255.0  
	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	pred = model(img, augment=False)[0]
	pred = non_max_suppression(pred, 0.21, 0.45) # prediction, conf, iou

	detection_result = []
	for i, det in enumerate(pred):
		if len(det): 
			for d in det: # d = (x1, y1, x2, y2, conf, cls)
				x1 = int(d[0].item())
				y1 = int(d[1].item())
				x2 = int(d[2].item())
				y2 = int(d[3].item())
				conf = round(d[4].item(), 2)
				c = int(d[5].item())
				
				detected_name = names[c]
				detection_result.append([x1, y1, x2, y2, conf, c])
				
				## Bounding box
				frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2) # box
				#frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
	return (frame, detection_result)




############################################# Object tracking ######################################################
list_of_center_dict = {}
previous_bbox_dic = {} # it store bbox of previous frame
not_found_id_list = list()
def object_tracker(results): # this results should be only riders and not any other classes
	results_ = results.copy()
	global previous_bbox_dic
	global not_found_id_list
	new_bbox_dic = {}

	# stores bbox center of this single frame
	bbox_center_list = [] 
	for result in results_: # Loop all the riders in an image
		x1, y1, x2, y2, conf, c = result # single bbox
		bbox_center = (int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)) # finding center of bbox
		bbox_center_list.append(bbox_center)

	# loop through, all the centers of riders from a single frame(current frame)
	for center in bbox_center_list:
		if len(previous_bbox_dic) == 0: 
			rand_key = random.randint(10, 800000)
			new_bbox_dic[rand_key] = center # # adding new center of a rider, bcz there wasn't any prev, frame.
			list_of_center_dict[rand_key] = [center]  # plot
		else:
			dist_list = []
			for key, center_prev in previous_bbox_dic.items():
				dist = math.dist(center, center_prev)
				# dist = math.sqrt((center[0]-center_prev[0])**2 + (center[1]-center_prev[1])**2)
				dist_list.append(dist)

			min_dist_value = min(dist_list)
			# print(min_dist_value)
			if min_dist_value < min_dist_threshold_value: # updating frame center
				min_dist_idx = dist_list.index(min_dist_value)
				min_dist_key = list(previous_bbox_dic.keys())[min_dist_idx] # from prev frame
				new_bbox_dic[min_dist_key] = center # updating position of existing bbox 
				previous_bbox_dic[min_dist_key] = center # updating prev dict
				
				old_list = list_of_center_dict[min_dist_key] # plot
				old_list.append(center)
				list_of_center_dict[min_dist_key] = old_list

			else: 
				random_id = random.randint(10, 8000)
				new_bbox_dic[random_id] = center  # adding new center of a rider, bcz itz newly discovered
				list_of_center_dict[random_id] = [center] # plot



	detection_count = len(list(new_bbox_dic.keys())) # count number of rider detected in this frame
	# print('detection count: ---------->>>>>>',len(new_bbox_dic))


	for key in list(previous_bbox_dic.keys()): # removing dic key bcz that object is no more visible
		if key not in list(new_bbox_dic.keys()):
			not_found_id_list.append(key)
			if not_found_id_list.count(key) > frame_can_be_missed:
				previous_bbox_dic.pop(key) # remove that id from prev list. 
				list_of_center_dict.pop(key) # plot


	for key, value in new_bbox_dic.items(): # adding newly discoverd riders into the prev_dict
		if key not in list(previous_bbox_dic.keys()):
			previous_bbox_dic[key] = value

	# return previous_bbox_dic
	return (list_of_center_dict, detection_count)

	
