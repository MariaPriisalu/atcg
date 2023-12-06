import numpy as np
import copy
from commonUtils.ReconstructionUtils import ROAD_LABELS,SIDEWALK_LABELS, OBSTACLE_LABELS_NEW, OBSTACLE_LABELS, MOVING_OBSTACLE_LABELS, NUM_SEM_CLASSES

def is_object_in_field_of_view(point_of_view, object, movement_direction, field_of_view):
    if np.linalg.norm(movement_direction[1:])<1e-5:
        return True
    min_x_A, max_x_A, min_y_A, max_y_A = get_extremes(object)
    directions = get_directions(max_x_A, min_x_A, min_y_A, max_y_A, point_of_view)
    for direction in directions:
        if np.linalg.norm(direction)<1e-5:
            return True
        angle=np.arccos(np.dot(direction, movement_direction[1:])/(np.linalg.norm(direction)*np.linalg.norm(movement_direction[1:])))
        if angle <= field_of_view/2 :
            return True
    return False

def is_object_occluded(point_of_view, object, frame, agent_shape, episode, is_fake_episode):

    pedestrian_list = episode.people[frame]
    car_list = episode.cars[frame]

    if is_object_occluded_by_controllable_pedestrians(point_of_view, object, episode.pedestrian_data, frame, agent_shape):
        return True
    if is_object_occluded_by_controllable_car(point_of_view, object, episode.car_data, frame):
        return True
    if is_object_occluded_by_external_obj(point_of_view, object, car_list):
        return True
    if is_object_occluded_by_external_obj(point_of_view, object, pedestrian_list):
        return True
    if is_occluded_by_static_object(point_of_view, object,episode.reconstruction):
        return True
    return False


def is_object_occluded_by_controllable_pedestrians(point_of_view, object, pedestrian_data, frame, agent_shape):
    obj_center=get_obj_center(object)
    for extr_ped in pedestrian_data:
        extr_ped_center=extr_ped.agent[frame]
        person_B=np.array([[extr_ped_center[0]-agent_shape[0], extr_ped_center[0]+agent_shape[0]],[extr_ped_center[1]-agent_shape[1], extr_ped_center[1]+agent_shape[1]],[extr_ped_center[2]-agent_shape[2], extr_ped_center[2]+agent_shape[2]]])
        if np.linalg.norm(extr_ped_center[1:]-point_of_view[1:])>0.5 and np.linalg.norm(extr_ped_center[1:]-point_of_view[1:])<np.linalg.norm(obj_center[1:]-point_of_view[1:]):
            if is_B_occluding_A(object, person_B, point_of_view):
                return True
    return False

def is_object_occluded_by_external_obj(point_of_view, object, car_list):
    obj_center=get_obj_center(object)
    for extr_car in car_list:
        if len(extr_car) >= 6:
            extr_car_copy=[extr_car[0], extr_car[1]-1,extr_car[2], extr_car[3]-1,extr_car[4], extr_car[5]-1]
        else:
            extr_car_copy = extr_car
        extr_ped_center=get_obj_center(extr_car_copy)
        if  np.linalg.norm(extr_ped_center[1:]-point_of_view[1:])>0.5 and np.linalg.norm(extr_ped_center[1:]-point_of_view[1:])<np.linalg.norm(obj_center[1:]-point_of_view[1:]):
            if is_B_occluding_A(object, extr_car_copy, point_of_view):
                return True
    return False

def is_object_occluded_by_controllable_car(point_of_view, object, car_data, frame):
    obj_center=get_obj_center(object)
    for extr_car in car_data:
        if np.linalg.norm(extr_car.car[frame][1:]-point_of_view[1:])>0.5 and np.linalg.norm(extr_car.car[frame][1:]-point_of_view[1:])<np.linalg.norm(obj_center[1:]-point_of_view[1:]):
            if is_B_occluding_A(object, extr_car.car_bbox[frame], point_of_view):
                return True
    return False

def get_obj_center(object):
    if len(object)>=6:
        return np.array([np.mean(object[0:2]),np.mean(object[2:4]),np.mean(object[4:])])
    else:
        return np.array([np.mean(object[0,:]),np.mean(object[1,:]),np.mean(object[2,:])])

def is_B_occluding_A(object_A, object_B, point_of_view):

    min_x_A, max_x_A, min_y_A, max_y_A= get_extremes(object_A)
    min_x_B, max_x_B, min_y_B, max_y_B = get_extremes(object_B)

    directions = get_directions(max_x_A, min_x_A, min_y_A, max_y_A, point_of_view)
    for direction in directions:
        t_min_x, t_max_x = find_temporal_borders(min_x_B - point_of_view[1], max_x_B - point_of_view[1],direction[1 - 1])
        t_min_y, t_max_y = find_temporal_borders(min_y_B - point_of_view[2], max_y_B - point_of_view[2], direction[2 - 1])
        t_min = max(t_min_x, t_min_y)
        t_max = min(t_max_x, t_max_y)
        if t_min <= t_max and t_max >= 0 and t_min <= 1 and t_max >= 0:
            return True
    return False


def get_directions(max_x_A, min_x_A, min_y_A, max_y_A, point_of_view):
    directions = []
    directions.append(np.array([min_x_A, min_y_A]) - point_of_view[1:])
    directions.append(np.array([max_x_A, max_y_A]) - point_of_view[1:])
    directions.append(np.array([min_x_A, max_y_A]) - point_of_view[1:])
    directions.append(np.array([max_x_A, min_y_A]) - point_of_view[1:])
    return directions


def find_temporal_borders(min_value,max_value,divisor):
    if divisor>0:
        return min_value/divisor,max_value/divisor
    elif divisor <0:
        return max_value / divisor, min_value / divisor
    elif (min_value<=0 and 0<=max_value) or (min_value>=0 and 0>=max_value):
        return 0,1
    else:
        return 1, 0

def get_extremes(object):
    if len(object)>=6:
        return object[2], object[3],object[4],object[5]
    else:
        return object[1,0], object[1,1], object[2,0], object[2,1]

def is_occluded_by_static_object(point_of_view, object,reconstruction):
    min_x_A, max_x_A, min_y_A, max_y_A = get_extremes(object)
    obj_center = get_obj_center(object)
    directions = get_directions(max_x_A, min_x_A, min_y_A, max_y_A, point_of_view)
    directions.append(np.array([obj_center[1]-point_of_view[1],obj_center[2]-point_of_view[2]]))
    for direction in directions:
        final_pos=point_of_view[1:]+direction
        if np.linalg.norm(direction)>1e-5:
            unit_dir=direction*(1/np.linalg.norm(direction))*.5
            cur_pos=copy.copy(point_of_view[1:])
            cur_pos_x = int(round(cur_pos[0]))
            cur_pos_y = int(round(cur_pos[1]))
            while np.linalg.norm(final_pos-cur_pos)>0.5 and cur_pos_x<reconstruction.shape[1] and cur_pos_y<reconstruction.shape[2]  and  cur_pos_x>=0 and  cur_pos_y>=0 :
                if is_valid(reconstruction[:,cur_pos_x,cur_pos_y], new_carla=False)==False:
                    return True
                cur_pos=cur_pos+unit_dir
                cur_pos_x=int(round(cur_pos[0]))
                cur_pos_y=int(round(cur_pos[1]))
    return False


def is_valid(segmentation, new_carla=False):
    local_seg=segmentation*NUM_SEM_CLASSES
    if new_carla:
        obstacles = OBSTACLE_LABELS_NEW
    else:
        obstacles = OBSTACLE_LABELS

    for label in obstacles:
        if (local_seg.astype(int) == label).sum()>0:
            return False
    return True
