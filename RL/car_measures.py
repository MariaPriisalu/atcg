import numpy as np

from RL.settings import CAR_MEASURES_INDX
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES, OBSTACLE_LABELS_NEW, OBSTACLE_LABELS, \
    MOVING_OBSTACLE_LABELS, cityscapes_labels_dict, SIDEWALK_LABELS
from utils.utils_functions import overlap


from RL.occlusion_utils import is_object_occluded, is_object_in_field_of_view

def collide_with_people( frame_input, people_overlapped, x_range, y_range, z_range, people,pedestrian_data,agent_size ):
    # print ("Car measure:  People hit by car frame" + str(frame_input)+" bbox "+str([y_range[0], y_range[1], z_range[0], z_range[1]]))
    # print ("People in frame "+str(people[frame_input]))
    collide_with_agents=[]
    for person in pedestrian_data:
        bbox = [person.agent[frame_input][0] - agent_size[0],
                person.agent[frame_input][0] + agent_size[0] ,
                person.agent[frame_input][1]- agent_size[1],
                person.agent[frame_input][1] + agent_size[1] ,
                person.agent[frame_input][2] - agent_size[2],
                person.agent[frame_input][2] + agent_size[2] ]
        if overlap(bbox[2:], [y_range[0], y_range[1], z_range[0], z_range[1]], 1):
            people_overlapped.append(bbox)
            collide_with_agents.append(person)

    for person in people[frame_input]:
        x_pers = [min(person[1, :]), max(person[1, :]), min(person[1, :]), max(person[1, :]), min(person[2, :]),
                  max(person[2, :])]
        if overlap(x_pers[2:], [y_range[0], y_range[1], z_range[0], z_range[1]], 1):
            people_overlapped.append(person)

            return people_overlapped,collide_with_agents
    return people_overlapped,collide_with_agents


def intercept_hero_car( id, frame_in, cars,cars_data, bbox, all_frames=False, agent_frame=-1):
    # can be replaced by reurning the content of hero_car_measures
    # Return number of cars overlapped

    overlapped = 0
    frames = []
    if all_frames or frame_in >= len(cars):
        frames = list(range(len(cars)))
    else:
        frames.append(frame_in)


    for frame in frames:
        # print (" Cars "+str(cars[frame]))
        for car in cars[frame]:
            car_locally = np.array(car).copy()
            car_locally[1] = car_locally[1] - 1
            car_locally[3] = car_locally[3] - 1
            car_locally[5] = car_locally[5] - 1
            if overlap(car_locally[2:6], bbox[2:], 1) or overlap(bbox[2:], car_locally[2:6], 1):

                overlapped += 1
        for car_id,car in enumerate(cars_data):
            if car_id!=id:
                if overlap(car.car_bbox[frame][2:6], bbox[2:], 1) or overlap(bbox[2:], car.car_bbox[frame][2:6], 1):
                    overlapped += 1
    return overlapped


def end_of_episode_measures_car(id, frame, measures,episode, bbox, end_on_hit_by_pedestrians, goal, allow_car_to_live_through_collisions,agent_size, carla_new=False):
    people_overlapped=[]
    per_id = -1
    car=[]
    x_range = bbox[0:2]
    y_range = bbox[2:4]
    z_range = bbox[4:]
    car_pos=np.array([np.mean(x_range),np.mean(y_range), np.mean(z_range)])


    # print (" End of episode measures in car: frame "+str(frame )+" car pos "+str(bbox)+ " evaluate "+str(frame+1))

    # Hit by car

    measures[frame, CAR_MEASURES_INDX.hit_by_car] = max(measures[max(frame - 1, 0), CAR_MEASURES_INDX.hit_by_car], intercept_hero_car(id, frame + 1, all_frames=False, cars=episode.cars,cars_data=episode.car_data,  bbox=bbox))

    if not allow_car_to_live_through_collisions and frame>0:
        measures[frame, CAR_MEASURES_INDX.agent_dead] = max(measures[max(frame - 1, 0), CAR_MEASURES_INDX.hit_by_car],
                                                            measures[frame, CAR_MEASURES_INDX.agent_dead])
    collide_pedestrians_list, collide_with_agents_list = collide_with_people(frame+1 ,people_overlapped, x_range, y_range, z_range, episode.people,episode.pedestrian_data,  agent_size )
    collide_pedestrians=len(collide_pedestrians_list)>0
    collide_with_agents=len(collide_with_agents_list)>0
    if collide_pedestrians:
        measures[frame, CAR_MEASURES_INDX.hit_pedestrians] = 1
    if collide_with_agents:
        measures[frame, CAR_MEASURES_INDX.hit_by_agent] = 1
    if end_on_hit_by_pedestrians and frame>0:
        measures[frame, CAR_MEASURES_INDX.hit_pedestrians]=max(measures[max(frame - 1, 0), CAR_MEASURES_INDX.hit_pedestrians],measures[frame, CAR_MEASURES_INDX.hit_pedestrians])
        if not allow_car_to_live_through_collisions:
            measures[frame, CAR_MEASURES_INDX.agent_dead] = max(measures[max(frame - 1, 0), CAR_MEASURES_INDX.hit_pedestrians],measures[frame, CAR_MEASURES_INDX.agent_dead] )


    measures[frame, CAR_MEASURES_INDX.hit_obstacles] = max(intercept_objects_cars(bbox,episode.reconstruction, no_height=True,carla_new=carla_new),measures[frame, CAR_MEASURES_INDX.hit_obstacles])
    if not allow_car_to_live_through_collisions and frame>0:
        measures[frame, CAR_MEASURES_INDX.hit_obstacles] =max(measures[frame, CAR_MEASURES_INDX.hit_obstacles], measures[frame-1, CAR_MEASURES_INDX.hit_obstacles])
        measures[frame, CAR_MEASURES_INDX.agent_dead] = max(measures[max(frame - 1, 0), CAR_MEASURES_INDX.hit_obstacles]>0,measures[frame, CAR_MEASURES_INDX.agent_dead])
    if len(goal)>0:  # If using a general goal
        measures[frame, CAR_MEASURES_INDX.dist_to_goal] = np.linalg.norm(np.array(goal[1:]) - car_pos[1:])
        if y_range[0]<=goal[1] and y_range[1]>=goal[1] and z_range[0]<=goal[2] and z_range[1]>=goal[2]:
            measures[frame,CAR_MEASURES_INDX.goal_reached]=1
    # min_dist, id =find_closest_controllable_pedestrian(frame, car_pos,episode, agent_size, np.array([0,0,0]), 2*np.pi,is_fake_episode=False)
    # measures[frame, CAR_MEASURES_INDX.dist_to_agent]= min_dist
    return measures


def car_bounding_box(bbox, reconstruction, no_height=False, channel=3):
    if no_height:
        x_range = [0, reconstruction.shape[0]]
    else:
        x_range = [bbox[0], bbox[1]]
    segmentation = (
        reconstruction[int(x_range[0]):int(x_range[1]) + 1, int(bbox[2]):int(bbox[3]) + 1,
        int(bbox[4]):int(bbox[5]) + 1, channel] * int(NUM_SEM_CLASSES)).astype(
        np.int32)
    return segmentation


def intercept_objects_cars( bbox,reconstruction, no_height=False, cars=False,carla_new=False):  # Normalisera/ binar!
    segmentation = car_bounding_box(bbox, reconstruction,no_height)
    if carla_new:
        obstacles=OBSTACLE_LABELS_NEW
    else:
        obstacles = OBSTACLE_LABELS
    count=0
    for label in obstacles:
        count+=(segmentation==label).sum()
    if cars:
        for label in MOVING_OBSTACLE_LABELS:
            count += (segmentation == label).sum()
        count += (segmentation == cityscapes_labels_dict['person'] ).sum()
    if count > 0:
        if (bbox[1]-bbox[0])*(bbox[3]-bbox[2])>0:
            return count *1.0/((bbox[1]-bbox[0])*(bbox[3]-bbox[2]))#*(z_range[1]+2-z_range[0]))
        else:
            return 0
    return 0
def find_closest_car(frame, pos,episode, agent_shape, movement_direction, field_of_view, is_fake_episode):
    closest_car, min_dist=find_closest_car_in_list(frame, pos, episode, agent_shape, movement_direction, field_of_view, is_fake_episode)
    closest_car_agent, min_dist_agent=find_closest_controllable_car(frame, pos, episode, agent_shape, movement_direction, field_of_view, is_fake_episode)
    if min_dist_agent<0 or min_dist<min_dist_agent and min_dist>=0:
        return closest_car, min_dist
    return closest_car_agent, min_dist_agent

def find_closest_car_in_list(frame, pos,episode, agent_shape, movement_direction, field_of_view, is_fake_episode):
    closest_car = []
    min_dist = -1
    for car in episode.cars[min(frame, len(episode.cars) - 1)]:
        car_local=np.array(car).copy()
        car_local[1]=car_local[1]-1
        car_local[3] = car_local[3] - 1
        car_local[5] = car_local[5] - 1
        car_pos = np.array([np.mean(car_local[0:2]), np.mean(car_local[2:4]), np.mean(car_local[4:])])
        in_fov=is_object_in_field_of_view(pos, car_local, movement_direction, field_of_view)
        occluded=is_object_occluded(pos, car_local, frame, agent_shape, episode, is_fake_episode)
        if in_fov and not occluded and ( min_dist < 0 or min_dist > np.linalg.norm(np.array(pos[1:]) - car_pos[1:])):
            closest_car = car_pos
            min_dist = np.linalg.norm(np.array(pos[1:]) - car_pos[1:])
    # print (" Closest car "+str(closest_car )+" min dist "+str( min_dist))
    return closest_car, min_dist

def find_closest_controllable_car(frame, pos,episode, agent_shape, movement_direction, field_of_view,is_fake_episode):
    closest_car = []
    min_dist = -1
    for car in episode.car_data:
        car_pos = car.car[min(frame, len(car.car) - 1)]
        in_fov = is_object_in_field_of_view(pos, car.car_bbox[min(frame, len(car.car) - 1)], movement_direction, field_of_view)
        occluded = is_object_occluded(pos, car.car_bbox[min(frame, len(car.car) - 1)], frame, agent_shape, episode,is_fake_episode)

        if in_fov and not occluded and ( min_dist < 0 or min_dist > np.linalg.norm(np.array(pos[1:]) - car_pos[1:]) and np.linalg.norm(np.array(pos[1:]) - car_pos[1:])>0):
            closest_car = car_pos
            min_dist = np.linalg.norm(np.array(pos[1:]) - car_pos[1:])

    return closest_car, min_dist


def find_closest_controllable_pedestrian(frame, pos,episode, agent_shape, movement_direction, field_of_view,is_fake_episode):
    closest_pedestrian = []
    min_dist = -1
    id=-1
    for local_id, pedestrian in  enumerate(episode.pedestrian_data):

        pedestrian_pos = pedestrian.agent[min(frame, len(pedestrian.agent) - 1)]
        pedestrian_bbox =np.array([pedestrian_pos[0]-agent_shape[0],pedestrian_pos[0]+agent_shape[0],pedestrian_pos[1]-agent_shape[1],pedestrian_pos[1]+agent_shape[1],pedestrian_pos[2]-agent_shape[2],pedestrian_pos[2]+agent_shape[2]  ])
        in_fov = is_object_in_field_of_view(pos, pedestrian_bbox, movement_direction, field_of_view)
        occluded = is_object_occluded(pos, pedestrian_bbox, frame, agent_shape,  episode,is_fake_episode)
        if in_fov and not occluded and (min_dist < 0 or min_dist > np.linalg.norm(np.array(pos[1:]) - pedestrian_pos[1:])):
            closest_pedestrian = pedestrian
            min_dist = np.linalg.norm(np.array(pos[1:]) - pedestrian_pos[1:])
            id= local_id

    return min_dist, id

def find_closest_pedestrian_in_list(frame, pos,episode, agent_shape, movement_direction, field_of_view,is_fake_episode):
    closest_pedestrian = []
    min_dist = -1
    for person in episode.people[min(frame, len(episode.people) - 1)]:
        person_pos = np.mean(person, axis=1)
        in_fov = is_object_in_field_of_view(pos, person, movement_direction, field_of_view)
        occluded = is_object_occluded(pos, person, frame, agent_shape,  episode,is_fake_episode)
        if in_fov and not occluded and (min_dist < 0 or min_dist > np.linalg.norm(np.array(pos[1:]) - person_pos[1:])):
            closest_pedestrian = person_pos
            min_dist = np.linalg.norm(np.array(pos[1:]) - person_pos[1:])

    return closest_pedestrian, min_dist

def iou_sidewalk_car( bbox, reconstruction, no_height=True):
    segmentation = car_bounding_box(bbox, reconstruction,no_height)
    area =0
    for label in SIDEWALK_LABELS:
        area+=(segmentation == label).sum()
    divisor=(bbox[1]-bbox[0]+1)*(bbox[3]-bbox[2]+1)*(bbox[5]-bbox[4]+1)

    return area * 1.0 / divisor