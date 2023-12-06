'''
 Before running call:
export PATH=Packages/anaconda/bin\:$PATH
export PYTHONPATH=Caffe/caffe-dilation/build_master_release/python:$PYTHONPATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:Caffe/caffe-dilation/build_master_release/lib"
export PATH=$PATH\:Packages/anaconda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH\:/usr/local/cuda-7.5/lib64
export PATH=$PATH\:/usr/local/cuda-7.5/bin

Bodyparts in order: Standing    Biking
1. head             Upper 30%   Upper 50%
2. neck             Upper 40%   Upper 50%
3. R. shoulder      Upper 40%   Upper 60%
4. R. elbow         Upper 50%   Upper 75%
5. R. wrist         Middle 60%  Middle 60%
6. L. shoulder      Upper 40%
7. L. elbow         Upper 40%
8. L. wrist         Upper 50%
9. R. hip           Middle 50%  Middle 60%
10. R. knee         Lower 50%   Lower 75%
11. R. ankle        Lower 20%   Lower 50%
12. L. hip          Middle 50%
13. L. knee         Lower 50%
14. L. ankle        Lower 20%
'''

# The script evaluates the approximate position of the joints by assuming teh above

# from PyQt4.QtCore import QObject
# matplotlib.use("Qt4Agg")
# matplotlib.use('Agg')# " Uncomment when ssh"

# import matplotlib.pyplot as plt
# import pylab

import math
import numpy as np
from scipy import ndimage
from RL.settings import STATISTICS_INDX
from commonUtils.ReconstructionUtils import ROAD_LABELS,SIDEWALK_LABELS, OBSTACLE_LABELS_NEW, OBSTACLE_LABELS, MOVING_OBSTACLE_LABELS, CHANNELS,NUM_SEM_CLASSES
# Checks if two lists overlap. buf is precision
def overlap(cluster1, cluster2, buf):
    # Buffer length relative to the length of the square.
    # y_dist = math.ceil(max(cluster1[1] - cluster1[0], cluster2[1] - cluster2[0]) * buf)
    # x_dist = math.ceil(max(cluster1[3] - cluster1[2], cluster2[3] - cluster2[2]) * buf)
    #

    if len(cluster1)==6:
        return overlap_3D(cluster1, cluster2)
    vals = [False, False]
    for dim in range(2):
        i = dim * 2
        # print (str(cluster1[i:i + 2])+" "+str(cluster2[i:i + 2])+" dim "+str(i))
        # print ("Between "+str(between(cluster1[i:i + 2], cluster2[i:i + 2]))+" "+str(between(cluster2[i:i + 2], cluster1[i:i + 2])))
        # print ("Inside " + str( inside( cluster1[i:i + 2], cluster2[i:i + 2]))+" "+str(inside(cluster1[i:i + 2], cluster2[i:i + 2])))
        if between(cluster1[i:i + 2], cluster2[i:i + 2]) or between(cluster2[i:i + 2], cluster1[i:i + 2]) or inside(
                cluster1[i:i + 2], cluster2[i:i + 2]) or inside(cluster1[i:i + 2], cluster2[i:i + 2]):
            vals[dim] = True
    #         print (vals)
    # print (vals[0] and vals[1])
    return vals[0] and vals[1]
    #return len(intersection(cluster1, cluster2, x_dist, y_dist))

def overlap_3D(cluster1, cluster2):

    # Buffer length relative to the length of the square.
    vals=[False,False, False]
    for dim in range(3):
        i=dim*2
        #print str(cluster1[i:i + 2])+" "+str(cluster2[i:i + 2])
        if between(cluster1[i:i+2], cluster2[i:i+2]) or between(cluster2[i:i + 2], cluster1[i:i + 2]) or inside(cluster1[i:i+2], cluster2[i:i+2])or inside(cluster1[i:i+2], cluster2[i:i+2]):
            vals[dim]=True
            #print "overlap"
    #print vals[0] and vals[1] and vals[2]

    return vals[0] and vals[1] and vals[2]

def between (pair1, pair2):
    return round(pair1[0])<=round(pair2[0])<=round(pair1[1]) or round(pair1[0])<=round(pair2[1])<=round(pair1[1])

def inside (pair1, pair2):
    return round(pair1[0])<=round(pair2[0]) and round(pair2[1])<=round(pair1[1])

# Union of two rectangles
def union(a, b):
    y_min = min(a[0], b[0])
    y_max = max(a[1], b[1])
    x_min = min(a[2], b[2])
    x_max = max(a[3], b[3])
    return (y_min, y_max, x_min, x_max)


# Intersection of two rectangles
def intersection(a, b, x_dist, y_dist):
    x = max(a[0], b[0])
    y = max(a[2], b[2])
    w = min(a[1], b[1])
    h = min(a[3], b[3])
    if w - x < -y_dist or h - y < -x_dist: return ()
    return (x, y, w, h)

# creates float range.
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def save_obj(obj, name ):
    import pickle
    with open( name[:-4] + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    import pickle
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding="latin1", fix_imports=True)

def shortest_dist(positions_pers, pred_indx):
    min_dist = np.square(pred_indx[0] - positions_pers[0][0]) + np.square(pred_indx[1] - positions_pers[1][0])
    # print len(positions_pers[0])
    for k in range(0, len(positions_pers[0])):
        d = np.square(pred_indx[0] - positions_pers[0][k]) + np.square(pred_indx[1] - positions_pers[1][k])
        if (d < min_dist):
            min_dist = d
    return np.sqrt(min_dist)


def label_show(label):
    colors = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                       [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                       [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
    colors = np.uint8(colors)
    label_r = colors[label.ravel()]
    label_r = label_r.reshape((label.shape[0], label.shape[1], 3))
    return label_r

# Checks if point x, y is in square sq.

def in_square(sq, x, y, x_dist, y_dist):
    return (sq[0] <= y + y_dist and sq[1] + y_dist >= y) and (sq[2] <= x + x_dist and sq[3] + x_dist >= x)

def find_border(v_pos):
    test_positions=[]
    # for x in range(v_pos.shape[0]):
    #     for y in range(v_pos.shape[1]):
    #         if x > 0 and y > 0 and( v_pos[x - 1, y]!=v_pos[x, y] or v_pos[x , y-1]!=v_pos[x, y]):
    #             if v_pos[x , y] == 1:
    #                 test_positions.append([x, y])
    #             if v_pos[x - 1, y] == 1:
    #                 test_positions.append([x-1, y])
    #             if v_pos[x, y - 1] == 1:
    #                 test_positions.append([x, y - 1])
    difference=v_pos-ndimage.grey_erosion(v_pos, size=(3,3))
    test_positions=np.transpose(np.nonzero(difference))
    # print "Border "#+str(test_positions.shape)
    # print test_positions
    return test_positions

def mark_out_car_cv_trajectory( prior, car_bbox, car_goal, car_vel, agent_size, time=-1):
    # print (" Before marking out cv trajectory "+ str(np.sum(prior==0))+ "  time "+str(time)+" vel "+str(car_vel[1:]))
    if len(car_vel)==3:
        car_vel_two_dim = car_vel[1:]
    else:
        car_vel_two_dim= car_vel
    speed = np.linalg.norm(car_vel_two_dim)
    if speed>1e-4:

        car_vel_unit=(1/speed)*car_vel_two_dim

        limits = np.array([car_bbox[2]-agent_size[1]+car_vel_unit[0]-1, car_bbox[3]+agent_size[1]+car_vel_unit[0]+2, car_bbox[4]-agent_size[2]+car_vel_unit[1]-1,
                  car_bbox[5]+ agent_size[2]+car_vel_unit[1]+2])


        car_pos=np.array([np.mean(limits[0:1]), np.mean(limits[2:])])


        for dim in range(len(limits)):
            limits[dim] = int(min(max(limits[dim], 0), prior.shape[dim // 2] - 1))

        # print("Limits " + str(limits))
        while np.linalg.norm(car_pos-car_goal[1:])> np.sqrt(2) and limits[0]<limits[1] and limits[2]< limits[3]:
            #print (" In while loop ")
            limits_int=limits.astype(int)
            if time< 0:

                prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]]=np.zeros_like(prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]])
                # print (" Prior limits " + str(limits_int)+ (np.sum(np.abs(prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]]))))
            else:
                prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]] = time*np.ones_like(
                    prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]])
                time=time+1

            limits = np.array([limits[0]+car_vel_unit[0], limits[1]+car_vel_unit[0], limits[2]+car_vel_unit[1],limits[3] +car_vel_unit[1]])
            car_pos = [np.mean(limits[0:1]), np.mean(limits[2:])]

            for dim in range(len(limits)):
                limits[dim] = min(max(limits[dim], 0), prior.shape[dim // 2] - 1)
    #     print("After while loop time "+str(time)+" limits "+str(limits)+" car pos "+str(car_pos))
    # print (" After marking out cv trajectory " + str(np.sum(prior == 0)))

    return prior

def find_occlusions(occlusion_map, occupied_positions, car_pos, car_speed, car_vel, max_dim, dimensions,lidar_occlusion, max_angle=np.pi / 4.0):
    # print (" Car speed "+str(car_speed)+"car pos "+str(car_pos))
    if car_speed > 0.081 ** 2:  # Less than 1km/h
        alpha = np.arctan2(car_vel[1], car_vel[0])

        # print (" Alpha "+str(alpha))
        if lidar_occlusion:
            borders_img = getBordersImg(occupied_positions)
            occlusion_map = np.zeros_like(occlusion_map)
        else:
            for x in range(dimensions[0]):
                for y in range(dimensions[1]):
                    dir = np.array([x, y]) - car_pos  # car_pos_front
                    alpha_x = np.arctan2(dir[1], dir[0])
                    if alpha - max_angle < alpha_x and alpha_x < alpha + max_angle:  # (dir[1]<m1*dir[0]+(car_min_dim*.5) or dir[1]>m2*dir[0]-(car_min_dim*.5)): # Inside cone of possible initializations
                        occlusion_map[x, y] = 0
                        # print (str(x)+" "+str(y)+" "+str(occlusion_map[x,y])+" alpha_x "+str(alpha_x)+" alpha "+str(alpha)+" min "+str(alpha-np.pi/4.0)+" max "+str(alpha+np.pi/4.0))

            # print("After for loops  occlusion map: "+str(np.sum(occlusion_map)))
            # Find borders in the area seen by the car.
            borders_img = np.logical_and(getBordersImg(occupied_positions), np.logical_not(occlusion_map))
            # print("Borders image: " + str(np.sum(borders_img)))

        occlusion_map = getOcclusionImg(occlusion_map, borders_img, car_pos, max_dim, occupied_positions)
        # print("Get occlusion Image : " + str(np.sum(occlusion_map)))

        # Remove holes in occlusion mask
        binstruct = ndimage.generate_binary_structure(2, 2)
        occlusion_map = ndimage.binary_closing(occlusion_map, binstruct)
        # print("After closing  : " + str(np.sum(occlusion_map)))
    return occlusion_map

def getBordersImg( img):
    # Discover the borders of the objects in the image by XOR between an erosion with 2x2 structure AND original image
    binstruct = ndimage.generate_binary_structure(2, 2)
    eroded_img = ndimage.binary_erosion(img, binstruct)
    edges_img = (img > 0) ^ eroded_img
    return edges_img

# Given the pos of the car, unit velocity => occlusion img
# BorderImg shows some of the borders of validImg
# validImg- valid locations
def getOcclusionImg(occlusion_map, bordersImg, car_pos, max_dim, validImg):
    edge_points = np.nonzero(bordersImg)

    for pos in range(len(edge_points[0])):
        edge_point = np.array([edge_points[0][pos], edge_points[1][pos]], dtype=np.float)

        # Find the direction from the observer (i.e. car) to the current location. This is the direction of the light ray.
        dir = edge_point - car_pos
        dir_length = np.linalg.norm(dir)

        # If within the observers collision diameter, then this edge is an edge indicating the valid locations
        # on the border to the observer (i.e. car) and this is not an obstacle occluding anything.

        if dir_length > max_dim:
            dir = dir / dir_length  # unit length
            # Find if we move towards the observer (-dir) we are at an invalid
            # position
            pos_towards_observer = edge_point - dir
            roundedPoint_row = int(np.round(pos_towards_observer[0]))
            roundedPoint_col = int(np.round(pos_towards_observer[1]))

            occupied_before_edge = not validImg[int(edge_point[0]), int(edge_point[1])]

            if roundedPoint_row > 0 and roundedPoint_col > 0 and \
                    roundedPoint_row < occlusion_map.shape[0] and roundedPoint_col < occlusion_map.shape[1]:
                occupied_before_edge = occupied_before_edge or not validImg[roundedPoint_row, roundedPoint_col]
            past_occluding_object=False
            while True:
                edge_point += dir
                roundedPoint_row = int(np.round(edge_point[0]))
                roundedPoint_col = int(np.round(edge_point[1]))

                if roundedPoint_row > 0 and roundedPoint_col > 0 and \
                        roundedPoint_row < occlusion_map.shape[0] and roundedPoint_col < occlusion_map.shape[1]:
                    # Occluded areas are behind edges in which if we move towards the observer (-dir) we are at an invalid
                    # position, but if we move away from the observer (+dir) we are at a valid position.
                    if not past_occluding_object and validImg[roundedPoint_row, roundedPoint_col] == 1:
                        past_occluding_object=True
                    if past_occluding_object and occupied_before_edge:
                        occlusion_map[roundedPoint_row, roundedPoint_col] = 1
                else:
                    break  # Outside of img

    return occlusion_map

def get_heatmap( people, reconstruction):
    from sklearn.neighbors import KernelDensity
    people_centers = []
    for people_list in people:
        for pers in people_list:
            people_centers.append(np.mean(pers, axis=1)[1:])
    heatmap = np.zeros((reconstruction.shape[1], reconstruction.shape[2]))
    if len(people_centers) > 1:
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=0.0001, kernel='exponential')
        kde.fit(people_centers)

        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x, y] = kde.score_samples(np.array([x, y]).reshape(1, -1))
                if np.isnan(heatmap[x, y]):
                    heatmap[x, y] = 0
    # convolutional solution
    #import scipy.ndimage
    #heatmap = scipy.ndimage.gaussian_filter(np.sum(self.reconstruction[:, :, :, CHANNELS.pedestrian_trajectory], axis=0), 15)

    # Normalize heatmap
    max_value = np.max(heatmap[np.isfinite(heatmap)])
    min_value = np.min(heatmap[np.isfinite(heatmap)])
    if max_value - min_value > 0:
        heatmap = (heatmap - min_value) / (max_value - min_value)
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):

            if np.isinf(heatmap[x, y]):
                if heatmap[x, y] > 0:
                    heatmap[x, y] = 1
                else:
                    heatmap[x, y] = 0
                    # print "Done renormalizing"
    return heatmap


def is_car_on_road(car_bbox, road):
    car_in_dims=max(car_bbox[2:4]) < road.shape[1 - 1] and min(car_bbox[2:4]) > 0 and max(car_bbox[4:]) <road.shape[2 - 1] and min(car_bbox[4:]) > 0
    car_on_road=np.any(road[car_bbox[2]:car_bbox[3], car_bbox[4]:car_bbox[5]])
    return car_in_dims and car_on_road

def find_extreme_road_pos(car, car_pos, ortogonal_car_vel, car_dim, road):
    next_pos=car_pos+ortogonal_car_vel
    car_bbox=get_bbox_of_car(next_pos, car, car_dim)

    while is_car_on_road(car_bbox, road):
        next_pos = next_pos + ortogonal_car_vel
        car_bbox = get_bbox_of_car(next_pos, car, car_dim)
    return next_pos

def get_bbox_of_car( car_opposite, car, car_dims):
    car_bbox=[]
    car_bbox.append(int(round(car[0])))
    car_bbox.append(int(round(car[1])))
    max_dim=max(car_dims[1:])
    min_dim = max(car_dims[1:])
    if (car[3]-car[2])>(car[5]-car[4]):
        car_bbox.append(int(round(car_opposite[1]-max_dim)))
        car_bbox.append(int(round(car_opposite[1] + max_dim+1)))
        car_bbox.append(int(round(car_opposite[2]-min_dim)))
        car_bbox.append(int(round(car_opposite[2] +min_dim+1)))
    else:
        car_bbox.append(int(round(car_opposite[1] - min_dim)))
        car_bbox.append(int(round(car_opposite[1] + min_dim + 1)))
        car_bbox.append(int(round(car_opposite[2] - max_dim)))
        car_bbox.append(int(round(car_opposite[2] + max_dim + 1)))
    return car_bbox

def get_bbox_of_car_vel(car_opposite, car_vel,car_dims):
    car_bbox = []
    car_bbox.append(int(round(car_opposite[0]-car_dims[0])))
    car_bbox.append(int(round(car_opposite[0]+car_dims[0]+1)))
    max_dim = max(car_dims[1:])
    min_dim = max(car_dims[1:])
    if (car_vel[1]) > (car_vel[2]):
        car_bbox.append(int(round(car_opposite[1] - max_dim)))
        car_bbox.append(int(round(car_opposite[1] + max_dim + 1)))
        car_bbox.append(int(round(car_opposite[2] - min_dim)))
        car_bbox.append(int(round(car_opposite[2] + min_dim + 1)))
    else:
        car_bbox.append(int(round(car_opposite[1] - min_dim)))
        car_bbox.append(int(round(car_opposite[1] + min_dim + 1)))
        car_bbox.append(int(round(car_opposite[2] - max_dim)))
        car_bbox.append(int(round(car_opposite[2] + max_dim + 1)))
    return car_bbox

def get_road( reconstruction):
    reconstruction_semantic = (reconstruction[:, :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(np.int)
    road = np.zeros(reconstruction.shape[1:3])
    for label in ROAD_LABELS:
        road_label_3d = reconstruction_semantic == label
        road_label = np.any(road_label_3d, axis=0)
        # print("Label "+str(label)+" sum "+str(np.sum(road_label_3d))+" sum of only this label with any "+str(np.sum(road_label))+" size "+str(road_label.shape))
        # print(np.where(road_label))
        road = np.logical_or(road_label, road)
        # print("Road after addition "+str(np.sum(self.valid_init.road)))
    return road
def get_goal_frames( statistics, ep, ped_id=-1):
    if ped_id>=0:
        goal_frames = statistics[ep, ped_id, :, STATISTICS_INDX.frames_of_goal_change]
        goal_frames = goal_frames[goal_frames > 0]
        goal_frames[0] = 0
    else:
        goal_frames_all = statistics[ep, :, :, STATISTICS_INDX.frames_of_goal_change]
        goal_frames_list=[]
        for id in range(statistics.shape[1]):
            goal_frames=np.squeeze(goal_frames_all[id,:])
            goal_frames = goal_frames[goal_frames > 0]
            goal_frames[0] = 0
            goal_frames_list.append(goal_frames)
        return goal_frames_list
    return goal_frames