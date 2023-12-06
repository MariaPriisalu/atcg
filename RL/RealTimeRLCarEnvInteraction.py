# This is a wrapper around agent_car object.
# It is better like this to keep this as a wrapper only for real time interaction either with online/offline data, and let the aggregated agent_car other solve low-level decision making mechanisms
# Interaction test class for one parameter car. Might include import to tensorflow so under RL to not mess with imports-

from settings import  CAR_MEASURES_INDX,CAR_REWARD_INDX
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES
from pedestrian_data_holder import PedestrianDataHolder
from car_data_holder import CarDataHolder


#import commonUtils.RealTimeEnv.CarlaWorldManagement as CarlaWorldManagement

import settings
import os

if settings.run_settings.realTimeEnvOnline:
    import sys
    sys.path.append("./commonUtils/RealTimeEnv")
    import CarlaWorldManagement as CarlaWorldManagement  # to be sure that it works from both data gathering scripts and RLAgent
    from commonUtils.RealTimeEnv.CarlaRealTimeUtils import carlaVector3DToNumpy
else:
    import copy
from dotmap import DotMap
import numpy as np
from car_measures import iou_sidewalk_car
from commonUtils.ReconstructionUtils import ROAD_LABELS,SIDEWALK_LABELS, OBSTACLE_LABELS_NEW, OBSTACLE_LABELS, MOVING_OBSTACLE_LABELS, CHANNELS
from sklearn.preprocessing import normalize
import time
import pickle
import joblib
import scipy

from memory_profiler import profile as profilemem
from settings import run_settings
from initializer_car_data_holder import InitializerCarDataHolder

memoryLogFP_decisions = None
if run_settings.memoryProfile:
    memoryLogFP_decisions = open("memoryLogFP_RlEnvdecisions.log", "w+")
from utils.utils_functions import overlap, find_border, mark_out_car_cv_trajectory, find_occlusions, is_car_on_road, find_extreme_road_pos, get_bbox_of_car, get_bbox_of_car_vel,get_road,get_heatmap
# TODO Ciprian or Maria: many of the computations done inside this function should be shared by multiple cars in a multi-agent environment
# Please refactor a bit such this code becomes bounded as static data to class and reused between multiple cars' agents
# Interaction test class for one paramtere toy car. Might include import to tensorflow so under RL to not mess with imports-
#class RLCarRealTimeEnv(NullRealTimeEnv):
    # def __init__(self, car, cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample,
    #              people_sample, reconstruction, seq_len, car_dim, max_speed, car_input, set_up_done, car_goal_closer):
class ValidInitializationData():
    def __init__(self, shape):
        self.car_keys=[]
        self.valid_car_keys = [] # This is the set of cars that appear from the first frame
        self.valid_positions = np.ones(shape[1:3])  # VALID POSITOONS FOR FIRST FRAME
        self.valid_positions_cars = np.zeros(shape[1:3])  # VALID POSITOONS FOR FIRST FRAME
        self.valid_directions_cars = np.zeros((shape[1], shape[2], 3))  # VALID POSITOONS FOR FIRST FRAME

        self.car_vel_dict = {}
        self.valid_positions_copy= np.ones(shape[1:3])

class RLCarRealTimeEnv:
    def __init__(self,prefix, pos_x, pos_y, settings, isOnline, offlineData, trainableCars, reconstruction, seq_len, car_dim, max_speed, min_speed,
                 car_goal_closer, physicalCarsDict = None, physicalWalkersDict = None,
                 getRealTimeEnvWaypointPosFunctor = None,seq_len_pfnn=-1):
        #car,cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample, people_sample, tensor)
        self.trainableCars=trainableCars
        self.reconstruction=reconstruction # This should not be needed if online environment and car is on CARLA side!
        # print (" Before initialization : " + str(np.sum(self.reconstruction[:, :, :, CHANNELS.cars_trajectory])))
        # print(np.sum(reconstruction_semantic))


        # This is the keys already assigned to trainable car agents in the current session (episode)
        self.physicalCarsDict = physicalCarsDict
        self.physicalWalkersDict = physicalWalkersDict
        self.getRealTimeEnvWaypointPosFunctor = getRealTimeEnvWaypointPosFunctor


        # self.people = self.people[self.first_frame:]
        #
        # self.cars = self.cars[self.first_frame:]

        self.isOnline = isOnline
        self.settings = settings
        self.DTYPE = np.float64
        self.offlineData = offlineData
        assert self.isOnline or offlineData is not None

        self.seq_len=seq_len

        self.car_dim=car_dim
        # self.car_input=car_input
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.car_goal_closer=car_goal_closer

        self.time_ranges = {}


        self.valid_car_keys_trainableagents = []    # This is the cars that have been created as heros in online environment (if any used). If not it is a copy of the above

        self.pedestrian_data = []

        for id in range(settings.number_of_agents):
            self.pedestrian_data.append(PedestrianDataHolder(seq_len, seq_len_pfnn, self.settings.pfnn, self.DTYPE))


        self.cars = self.offlineData.cars
        self.people = self.offlineData.people

        self.cars_dict = self.offlineData.cars_dict
        self.people_dict = self.offlineData.people_dict

        self.init_frames = self.offlineData.init_frames
        self.init_frames_cars = self.offlineData.init_frames_cars


        self.target_initCache_Path=self.settings.target_initCache_Path
        self.valid_init=self.tryReloadFromCache( prefix, pos_x, pos_y)

        if self.valid_init==None:
            self.valid_init=ValidInitializationData(self.reconstruction.shape)
            self.get_valid_initializations()
            self.trySaveToCache( self.valid_init,  prefix, pos_x, pos_y)
        self.car_init_data=[]
        if self.settings.learn_init_car:
            for agent_id in range(self.settings.number_of_car_agents):
                self.car_init_data.append(InitializerCarDataHolder)


    def getInitCachePathByParams(self, episodeIndex, cameraPosX, cameraPosY):
        targetInitFile = os.path.join(self.target_initCache_Path, "{0}_x{1:0.1f}_y{2:0.1f}.pkl".format(episodeIndex, cameraPosX, cameraPosY))
        return targetInitFile

    def tryReloadFromCache(self, episodeIndex, cameraPosX, cameraPosY):
        init_unpickled = None
        targetInitFile = self.getInitCachePathByParams(episodeIndex, cameraPosX, cameraPosY)
        print(targetInitFile)
        try:
            if os.path.exists(targetInitFile):
                print(("Loading {} from cache".format(targetInitFile)))
                with open(targetInitFile, "rb") as testFile:
                    start_load = time.time()
                    #episode_unpickled = pickle.load(testFile)
                    init_unpickled = joblib.load(testFile)
                    load_time = time.time() - start_load
                    print(("Load time from cache (FILE) in s {:0.2f}s".format(load_time)))
            else:
                print(("File {} not found in cache...going to take a while to process".format(targetInitFile)))
        except ImportError:
            print("import error ")

        return init_unpickled

    def trySaveToCache(self, init, episodeIndex, cameraPosX, cameraPosY):
        targetInitFile = self.getInitCachePathByParams(episodeIndex, cameraPosX, cameraPosY)
        with open(targetInitFile, "wb") as testFile:
            print(("Storing {} to cache".format(targetInitFile)))
            serial_start = time.time()
            #episode_serialized = pickle.dump(episode, testFile, protocol=pickle.HIGHEST_PROTOCOL)
            joblib.dump(init, testFile, protocol=pickle.HIGHEST_PROTOCOL)
            serial_time = time.time() - serial_start
            print(("Serialization time (FILE) in s {:0.2f}s".format(serial_time)))

    def get_valid_initializations(self):
        self.valid_init.heatmap = get_heatmap(self.people, self.reconstruction)
        self.valid_init.road=get_road(self.reconstruction)
        for pedestrian_key in self.people_dict.keys():
            diff = self.init_frames[pedestrian_key]
            if diff <= 0 and len(self.people_dict[pedestrian_key]) > abs(diff):
                person_flat = self.people_dict[pedestrian_key][0].flatten()
                person_bbox = [0, 0, 0, 0]
                for pos in range(2, 6):
                    person_bbox[pos - 2] = int(
                        round(max(min(person_flat[pos], self.valid_init.valid_positions.shape[(pos - 2) // 2]), 0)))
                self.valid_init.valid_positions[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3]] = np.zeros_like(
                    self.valid_init.valid_positions[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3]])
        for car_key in self.cars_dict.keys():
            diff = self.init_frames_cars[car_key]
            if diff <= 0 and len(self.cars_dict[car_key]) > abs(diff):
                person_flat = self.cars_dict[car_key][0]
                person_bbox = [0, 0, 0, 0]
                for pos in range(2, 6):
                    person_bbox[pos - 2] = int(
                        round(max(min(person_flat[pos], self.valid_init.valid_positions.shape[(pos - 2) // 2]), 0)))
                self.valid_init.valid_positions[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3]] = np.zeros_like(
                    self.valid_init.valid_positions[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3]])
                if diff == 0:
                    # print (" Append valid car "+str(car_key)+" diff "+str(diff )+" pso "+str(person_bbox))
                    self.valid_init.valid_car_keys.append(car_key)
        # This is in voxels per frame !

        if self.isOnline:
            # In an online settings, we take the velocity directly from the environment data, since we reuse for the trainable cars one of the available spawned cars
            for car_key in self.cars_dict.keys():
                assert car_key in self.physicalCarsDict, f"The car you are looking {car_key} for is not registered. Registered cars are {self.physicalCarsDict.keys()}"
                carActor = self.physicalCarsDict[car_key]
                carActor_forwardVector = carActor.get_transform().get_forward_vector()
                car_forward_vector = np.array([carActor_forwardVector.z, carActor_forwardVector.y,
                                               carActor_forwardVector.x])  # put into the voxel space
                car_forward_vector = car_forward_vector / np.linalg.norm(
                    car_forward_vector)  # Normalize it, so since doesn't actually matter, i.e. voxel vs meters

                # Using max speed for now, but note that the car speed will be randomized later in the initialization of the car method
                car_desired_speed = self.settings.car_reference_speed
                car_vel = car_forward_vector * car_desired_speed
                self.valid_init.car_vel_dict[car_key] = car_vel


        else:

            for car_key in self.cars_dict.keys():
                # print (" Key "+str(car_key)+" len "+str(len(self.cars_dict[car_key])))
                if len(self.cars_dict[car_key]) >= 2:
                    self.valid_init.car_keys.append(car_key)
                    car_current = self.cars_dict[car_key][0]
                    car_next = self.cars_dict[car_key][1]
                    diff = [[], [], []]
                    for i in range(len(car_current)):
                        diff[int(i / 2)].append(car_next[i] - car_current[i])
                    self.valid_init.car_vel_dict[car_key] = np.mean(np.array(diff), axis=1)
                    prev_car = []
                    vel = self.valid_init.car_vel_dict[car_key]
                    for car in self.cars_dict[car_key]:
                        if len(prev_car) > 0:
                            diff = [[], [], []]
                            for i in range(len(car_current)):
                                diff[int(i / 2)].append(car[i] - prev_car[i])
                            vel = np.mean(np.array(diff), axis=1)
                        self.valid_init.valid_positions_cars[int(round(car[2])):int(round(car[3])),
                        int(round(car[4])):int(round(car[5]))] = 1
                        car_dims = self.valid_init.valid_positions_cars[int(round(car[2])):int(round(car[3])),
                                   int(round(car[4])):int(round(car[5]))].shape
                        self.valid_init.valid_directions_cars[int(round(car[2])):int(round(car[3])),
                        int(round(car[4])):int(round(car[5])), :] = np.tile(vel[np.newaxis, np.newaxis, :],
                                                                            (car_dims[0], car_dims[1], 1))
                        prev_car = copy.copy(car)

                        car_pos = np.array([np.mean(car[0:2]), np.mean(car[2:4]), np.mean(car[4:])])
                        car_prev_pos = np.array([np.mean(prev_car[0:2]), np.mean(prev_car[2:4]), np.mean(prev_car[4:])])
                        speed = np.linalg.norm(vel[1:])
                        if speed > 1e-2:
                            ortogonal_car_vel = np.array([0, vel[2] / speed, -vel[1] / speed])
                            min_road = find_extreme_road_pos(car, car_prev_pos, ortogonal_car_vel, self.settings.car_dim, self.valid_init.road)
                            max_road = find_extreme_road_pos(car, car_prev_pos, -ortogonal_car_vel, self.settings.car_dim, self.valid_init.road)
                            # if abs(max_road[1]-min_road[1])>2*min(self.settings.car_dim[1:]) or abs(max_road[2]-min_road[2])>2*min(self.settings.car_dim[1:]):
                            mean_road = (min_road + max_road) * 0.5
                            vector_to_car_pos = car_prev_pos - mean_road
                            car_opposite = mean_road - vector_to_car_pos
                            car_opposite_bbox = get_bbox_of_car(car_opposite, car, self.settings.car_dim)
                            if np.all(self.valid_init.valid_positions[car_opposite_bbox[2]:car_opposite_bbox[3],
                                      car_opposite_bbox[4]:car_opposite_bbox[5]]) and not np.all(
                                self.valid_init.valid_positions_cars[car_opposite_bbox[2]:car_opposite_bbox[3],
                                car_opposite_bbox[4]:car_opposite_bbox[5]]) and iou_sidewalk_car(car_opposite_bbox,
                                                                                                 self.reconstruction,
                                                                                                 no_height=True) == 0:
                                self.valid_init.valid_positions_cars[car_opposite_bbox[2]:car_opposite_bbox[3],
                                car_opposite_bbox[4]:car_opposite_bbox[5]] = 1
                                car_dims = self.valid_init.valid_positions_cars[car_opposite_bbox[2]:car_opposite_bbox[3],
                                           car_opposite_bbox[4]:car_opposite_bbox[5]].shape
                                self.valid_init.valid_directions_cars[car_opposite_bbox[2]:car_opposite_bbox[3],
                                car_opposite_bbox[4]:car_opposite_bbox[5], :] = -np.tile(vel[np.newaxis, np.newaxis, :],
                                                                                         (car_dims[0], car_dims[1], 1))

                                 #car_pos, goal_car, car_dir, car_key, online_car_key, on_car
        self.valid_init.valid_positions_copy = self.valid_init.valid_positions.copy()

    def remove_cars_from_valid_pos(self):
        for car in self.cars[0]:
            dir = self.valid_init.valid_directions_cars[int(np.mean(car[2:4])), int(np.mean(car[4:6])), :]

            if abs(dir[0]) > abs(dir[1]):
                x_lim = max(self.settings.car_dim[1:]) + 2
                y_lim = min(self.settings.car_dim[1:]) + 2
            else:
                x_lim = min(self.settings.car_dim[1:]) + 2
                y_lim = max(self.settings.car_dim[1:]) + 2
            carlimits = [max(int(car[2] - x_lim), 0), max(int(car[3] + x_lim), 0), max(int(car[4] - y_lim), 0),
                         max(int(car[5] + y_lim), 0)]
            carlimits = [min(carlimits[0], self.reconstruction.shape[1]),
                         min(carlimits[1], self.reconstruction.shape[1]),
                         min(carlimits[2], self.reconstruction.shape[2]),
                         min(carlimits[3], self.reconstruction.shape[2])]
            self.valid_init.valid_positions_cars[carlimits[0]:carlimits[1], carlimits[2]:carlimits[3]] = np.zeros_like(
                self.valid_init.valid_positions_cars[carlimits[0]:carlimits[1], carlimits[2]:carlimits[3]])

    def agent_initialization(self, episode,training=False,  viz=False, manual_dict={}):

        for car in self.trainableCars:
            observation_dict = {}
            # Note that car_dir is actually the velocity (voxels) per frame to get to the goal_car starting from car_pos in seq_len
            car_pos, goal_car, car_dir, car_key, online_car_key, on_car = car.init_agent( episode, training=training,  viz=viz)
            if len(manual_dict)>0:
                car_pos=manual_dict.car_pos[car.id]
                goal_car = manual_dict.goal_car[car.id]
                car_dir= manual_dict.car_dir[car.id]
            # Associate online environment actor ids (that physically exist in the world) with the logical agent in our environment
            car.onlinerealtime_agentId = online_car_key
            # print ("Car initial position "+str(car_pos)+" goal car "+str(goal_car)+" car dir "+str(car_dir))
            car.initial_position(car_pos, goal_car, self.seq_len, init_dir=car_dir, car_id=car_key, on_car=on_car)
            observation_dict[car]=self.get_car_initial_observation(car)

            episode.add_car_agents_to_reconstruction( observation_dict, 0)
        #episode.predict_car_agents()

    def get_car_initial_observation(self, car):
        car_observation = DotMap()
        car_observation.heroCarPos = car.car[0]
        car_observation.heroCarBBox = car.bbox[0]
        car_observation.heroCarGoal = car.goal
        car_observation.heroCarGoal = car.goal
        car_observation.car_init_dir = car.init_dir
        return car_observation

    def reset(self, alreadyAssignedCarKeys, initDict):
        for car in self.trainableCars:
            # Note that car_dir is actually the velocity (voxels) per frame to get to the goal_car starting from car_pos in seq_len
            car_pos, goal_car, car_dir, car_key, online_car_key, on_car =self.init_car( alreadyAssignedCarKeys, car.id, initDict.on_car)

            # Associate online environment actor ids (that physically exist in the world) with the logical agent in our environment
            car.onlinerealtime_agentId = online_car_key
            # print ("Car initial position "+str(car_pos)+" goal car "+str(goal_car)+" car dir "+str(car_dir))
            car.initial_position(car_pos, goal_car,  self.seq_len, init_dir=car_dir, car_id=car_key, on_car=on_car)





    def old_car_init(self,id, on_car=False, manual=False):

        car=[]
        goal_car = None
        car_dir = None
        car_id = -1
        if len(self.valid_init.car_keys) > 0:
            unused_car_keys=[]
            if on_car and len(self.valid_init.valid_car_keys) > 0:
                unused_car_keys =self.valid_init.valid_car_keys
            else:
                unused_car_keys=self.valid_init.car_keys
            np.random.shuffle(unused_car_keys)
            for car_key in unused_car_keys:
                if on_car and len(self.valid_init.valid_car_keys) > 0:
                    car_id = car_key
                    on_car=True
                else:
                    car_id = car_key
                    on_car = False
                    all_keys = self.valid_init.car_keys

                if manual:
                    print("Car keys " +str(all_keys)+" key "+str(car_key))
                    car_key = eval(input("Car key:"))

                car = self.cars_dict[car_key][0]

                car_init_pos = np.array([np.mean(car[0:2]), np.mean(car[2:4]), np.mean(car[4:])])
                car_vel = self.valid_init.car_vel_dict[car_key]
                if manual:
                    print ("Car pos " + str(car_init_pos))
                    print("Car vel "+str(car_vel))

                if on_car and len(self.valid_init.valid_car_keys) > 0:
                    car_dir = car_vel
                    goal_car = self.cars_dict[car_key][min(len(self.cars_dict[car_key]) - 1, self.seq_len + 3)]
                    car_goal_pos = np.array([np.mean(goal_car[0:2]), np.mean(goal_car[2:4]), np.mean(goal_car[4:])])
                    return car_init_pos, car_goal_pos, car_dir, car_key,0, True
                #car_pos, goal_car, car_dir, car_key, online_car_key, on_car

                min_speed = 5 * 5 / (17 * 3.6)  # 5km/h
                if np.linalg.norm(car_vel[1:]) < min_speed ** 2:  # 5km/h
                    max_speed = self.max_speed  # *5/(17*3.6)# 60 km/h
                    car_speed = np.random.rand(1) * (max_speed - min_speed) + min_speed
                    if np.linalg.norm(car_vel[1:]) < 1e-5:
                        car_vel = np.zeros_like(car_vel)
                        car_speed = 0
                    else:
                        car_vel = car_vel * (car_speed) / np.linalg.norm(car_vel)

                t = []
                for i in range(2):
                    if car_vel[i + 1] > 0:
                        t.append((self.reconstruction.shape[i + 1] - car_init_pos[i + 1]) / car_vel[i + 1])
                    else:
                        t.append((-car_init_pos[i + 1]) / car_vel[i + 1])

                t_max = min(t)
                t_min = np.max(self.car_dim[1:]) * 2 / min_speed
                if t_max > t_min:
                    car_bbox=[]
                    if not car_key in self.time_ranges:
                        self.time_ranges[car_key]=[[t_min,t_max ]]

                    if  len(self.time_ranges[car_key])>0:
                        selected_range_indx=np.random.randint(0, len(self.time_ranges[car_key]))

                        t_min_local=self.time_ranges[car_key][selected_range_indx][0]
                        t_max_local = self.time_ranges[car_key][selected_range_indx][1]
                        if t_min < len(self.cars_dict[car_key]):
                            time = np.random.rand() * (len(self.cars_dict[car_key]) - t_min_local) + t_min_local
                        else:
                            time = np.random.rand() * (t_max_local - t_min_local) + t_min_local

                        self.time_ranges[car_key].pop(selected_range_indx)
                        t_new_min=[t_min_local, time-(2*max(self.car_dim[1:])/ min_speed)]
                        if t_new_min[0]<t_new_min[1]:
                            self.time_ranges[car_key].append(t_new_min)
                        t_new_max=[time+(2*max(self.car_dim[1:])/ min_speed), t_max_local]
                        if t_new_max[0] < t_new_max[1]:
                            self.time_ranges[car_key].append(t_new_max)

                        if manual:
                            time = eval(input("time:"))
                        if round(time) < len(self.cars_dict[car_key]):
                            car_initial_bbox = self.cars_dict[car_key][min(round(time), len(self.cars_dict[car_key])-1)]#car_init_pos + (time * car_vel)
                            car_initial = np.array([np.mean(car_initial_bbox[0:2]), np.mean(car_initial_bbox[2:4]),
                                                    np.mean(car_initial_bbox[4:])])
                            goal_car_bbox = self.cars_dict[car_key][len(self.cars_dict[car_key]) - 1]
                            goal_car = np.array([np.mean(goal_car_bbox[0:2]), np.mean(goal_car_bbox[2:4]),
                                                 np.mean(goal_car_bbox[4:])])
                            car_vel_bbox = np.array(car_initial_bbox) - np.array(
                                self.cars_dict[car_key][min(round(time), len(self.cars_dict[car_key]) - 1) - 1])
                            if np.linalg.norm(car_vel_bbox[3:4]) > min_speed:
                                car_vel = np.array(
                                    [np.mean(car_vel_bbox[0:2]), np.mean(car_vel_bbox[2:4]), np.mean(car_vel_bbox[4:])])
                                car_dir = car_vel * 1 / np.linalg.norm(car_vel)
                                car_bbox = get_bbox_of_car_vel(car_initial, car_vel, self.settings.car_dim)
                            else:
                                car_bbox = car_initial_bbox
                                car_vel = goal_car - car_initial
                                if np.linalg.norm(car_vel[1:]) > min_speed:
                                    car_dir = car_vel * 1 / np.linalg.norm(car_vel)
                                else:
                                    car_dir=car_vel

                        else:
                            car_initial =car_init_pos + (time * car_vel)
                            t_goal = t_max_local - time
                            goal_car = car_initial + (t_goal * car_vel)
                            car_dir = car_vel
                            car_bbox = get_bbox_of_car_vel(car_initial, car_vel, self.settings.car_dim)

                        self.valid_init.valid_positions[int(car_bbox[2]):int(car_bbox[3]),int(car_bbox[4]):int(car_bbox[5])]=False

                        return car_initial, goal_car, car_dir,car_id,0, on_car
                        # car_pos, goal_car, car_dir, car_key, online_car_key, on_car
            for car_key in unused_car_keys:

                car_id = -1
                on_car = False

                for frame in range(len(self.cars_dict[car_key])-1):
                    car = np.array(self.cars_dict[car_key][frame])
                    next_car_pos=np.array(self.cars_dict[car_key][frame+1])
                    if is_car_on_road(next_car_pos.astype(int), self.valid_init.road):
                        car_init_pos = np.array([np.mean(car[0:2]), np.mean(car[2:4]), np.mean(car[4:])])
                        car_next_pos = np.array([np.mean(next_car_pos[0:2]), np.mean(next_car_pos[2:4]), np.mean(next_car_pos[4:])])
                        car_vel = car_next_pos-car_init_pos
                        speed=np.linalg.norm(car_vel[1:])
                        if speed>1e-2:
                            ortogonal_car_vel=np.array([0,car_vel[2]/speed, -car_vel[1]/speed])
                            min_road=find_extreme_road_pos(car, car_init_pos, ortogonal_car_vel, self.settings.car_dim, self.valid_init.road)
                            max_road = find_extreme_road_pos(car, car_init_pos, -ortogonal_car_vel, self.settings.car_dim, self.valid_init.road)
                            #if abs(max_road[1]-min_road[1])>2*min(self.settings.car_dim[1:]) or abs(max_road[2]-min_road[2])>2*min(self.settings.car_dim[1:]):
                            mean_road=(min_road+max_road)*0.5
                            vector_to_car_pos=car_init_pos-mean_road
                            car_opposite=mean_road-vector_to_car_pos
                            car_opposite_vel=-car_vel
                            car_opposite_bbox=get_bbox_of_car(car_opposite, car, self.settings.car_dim)
                            if np.all(self.valid_init.valid_positions[car_opposite_bbox[2]:car_opposite_bbox[3],car_opposite_bbox[4]:car_opposite_bbox[5]]):
                                goal_car=car_opposite+(self.seq_len*car_opposite_vel) # eventually change this
                                self.valid_init.valid_positions[car_opposite_bbox[2]:car_opposite_bbox[3],car_opposite_bbox[4]:car_opposite_bbox[5]] = False
                                return car_opposite, goal_car, car_opposite_vel, car_id,0,on_car
                            #  #car_pos, goal_car, car_dir, car_key, online_car_key, on_car

            car_id = -1
            on_car = False
            valid_road = np.logical_and(self.valid_init.valid_positions, self.valid_init.road)
            positions = np.where(valid_road)
            indx = np.random.randint(len(positions[0]))
            car = np.array([0, positions[0][indx],
                            positions[1][indx]])

            dist = int(self.seq_len * 0.75 * self.max_speed)
            limits = [int(max(car[1] - dist, 0)),
                      int(min(car[1] + dist, self.reconstruction.shape[1])),
                      int(max(car[2] - dist, 0)),
                      int(min(car[2] + dist, self.reconstruction.shape[2]))]
            positions_goal = np.where(valid_road[limits[0]:limits[1], limits[2]:limits[3]])
            goal_indx = np.random.randint(len(positions_goal[0]))
            goal_car = np.array([0, limits[0] + positions_goal[0][goal_indx], limits[2] + positions_goal[1][goal_indx]])
            i = 0
            while np.linalg.norm(goal_car[1:] - car[1:]) < self.seq_len * 0.25 * self.max_speed and i < 5:
                goal_indx = np.random.randint(len(positions_goal[0]))
                goal_car = [0, limits[0] + positions_goal[0][goal_indx], limits[2] + positions_goal[1][goal_indx]]
                i = i + 1
            car_dir = np.array(goal_car) - np.array(car)
            car_dir = car_dir * (1 / self.seq_len)
        return car, goal_car, car_dir,car_id,0,on_car
        # car_pos, goal_car, car_dir, car_key, online_car_key, on_car
    # Given the trainableCar given for setup and if it should force stick to one of the existing cars or not,
    # find an initialization position and return the position of car, goal, direction (velocity per frame to get to the goal from position) and the key car ID chosen if any
    def init_car(self,alreadyAssignedCarKeys, id, on_car=False):
        # Find positions where cars could be initialized. This should be prefferable done in CARLA.
        # Just choosing a  future car location as an initial spot
        # for key, value in self.cars_dict.items():
        #     print (" Key "+str(key)+" car "+str(value[0])+" vel "+str(self.valid_init.car_vel_dict[key]))

        # print (" On car? RL Interaction Env "+str(on_car)+" valid cars? "+str(self.valid_car_keys_trainableagents))
        car_pos = []
        goal_car = None
        car_dir = None
        car_key=-1
        # This is the set of car keys that are unselected yet and valid from the first frame
        if self.isOnline:
            unselected_valid_car_keys = [valid_car_key for valid_car_key in self.valid_car_keys_trainableagents
                                             if valid_car_key not in alreadyAssignedCarKeys]
        else:

            if id == 0:
                self.valid_init.valid_positions = self.valid_init.valid_positions_copy.copy()
                self.time_ranges = {}
            if self.settings.learn_init_car:
                return car_pos, None, None, None, None, False
            return self.old_car_init(id, manual=self.settings.manual_init, on_car=on_car)  # None, None, None, None

            # Attent to base on one of the existing cars
        online_car_key = None
        if len(unselected_valid_car_keys)>0:
            if len(unselected_valid_car_keys)>1:
                online_car_key=unselected_valid_car_keys[np.random.randint(len(unselected_valid_car_keys))]
            else:
                online_car_key = unselected_valid_car_keys[0]

        if online_car_key is None:
            return car_pos, None, None, None, None, False
        #car_pos, goal_car, car_dir, car_key, online_car_key, on_car

        alreadyAssignedCarKeys.add(online_car_key)

        car_pos = self.cars_dict[car_key][0] if online_car_key in self.cars_dict else [0, 0, 0, 0, 0, 0]#- the car key must exist in the dictionary!

        car_init_pos=np.array([np.mean(car_pos[0:2]),np.mean(car_pos[2:4]), np.mean(car_pos[4:])])
        # print ("Car pos " + str(car_init_pos))

        # Note: car_vel is in voxels per frame
        car_vel = self.valid_init.car_vel_dict[online_car_key] if online_car_key in self.valid_init.car_vel_dict else [0, 0, 0] #[1]-self.cars_dict[car_key][0]

        # print ("Car velocity " + str(car_vel) + " speed " + str(np.linalg.norm(car_vel[1:])))
        # Always randomize the target speed for the environment online case. Otherwise just check if it is smaller than minimum
        if self.isOnline or np.linalg.norm(car_vel[1:]) < self.min_speed**2:
            # print ("Readjust speed")
            car_speed = np.random.rand(1)*(self.max_speed - self.min_speed) + self.min_speed
            car_vel_normalized = car_vel / np.linalg.norm(car_vel) if np.linalg.norm(car_vel)!=0 else car_vel
            car_vel = car_vel_normalized * car_speed
        # print ("Car velocity "+str(car_vel)+" speed "+str(np.linalg.norm(car_vel[1:])))

        # Stick to the same car_vel, project the target goal in front of the car depending on the seq_len
        # Notice that a random factor here comes from the speed that is always randomized in this case. So no need to randomize again here locally
        timeToProjectUpfront = self.seq_len # Time in number of frames as above logic !
        goal_car = car_init_pos + car_vel * timeToProjectUpfront

        # Find the corresponding waypoint for the projected goal, such that it is a traffic realistic goal position take from environment
        goal_car = self.getRealTimeEnvWaypointPosFunctor(goal_car)
        # Fix the car direction according to the goal. It must get to the goal in seq_len
        car_dir = (np.array(goal_car) - np.array(car_init_pos)) * (1.0/self.seq_len) # velocity basically in voxels per frame, we must get to the goal in seq_len frames

        return car_init_pos, goal_car, car_dir, car_key, online_car_key, False


    #@profilemem(stream=memoryLogFP_decisions)
    def next_action(self, observation_dict, episode, training, manual=False):

        # Fill in first the pedestrian agent observations

        # self.car.measures[self.trainableCars[0].frame, CAR_MEASURES_INDX.inverse_dist_to_closest_car]=observation.inverse_dist_to_car
        # print (" Car observes distance to pedestrian "+str(self.car.measures[self.trainableCars[0].frame, CAR_MEASURES_INDX.inverse_dist_to_closest_car]))
        realTimeEnvObservation = DotMap()
        # print (" Update real time environment frame "+str(self.trainableCars[0].frame)+" value "+str(observation.agentPos))
        self.update_car_episode(observation_dict)


        # Construct a fake episode environment where the car will take its decision
        self.fake_episode = DotMap()
        self.fake_episode.pedestrian_data = self.pedestrian_data# TODO: Is it an issue that no copy here. SHould not be an issue!!!

        self.fake_episode.reconstruction = self.reconstruction
        self.fake_episode.useRealTimeEnv = episode.useRealTimeEnv
        self.fake_episode.environmentInteraction = episode.environmentInteraction
        self.fake_episode.reconstruction=self.reconstruction
        #self.fake_episode.valid_positions = self.valid_init.valid_positions
        
        # TODO ?? What is this for ?- This is to simulate only seeing one step at a time of the data
        if self.settings.ignore_external_cars_and_pedestrians==True:
            self.fake_episode.people =[]
            self.fake_episode.cars = []
            for frame in range(self.trainableCars[0].frame + 2):
                self.fake_episode.people.append([])
                self.fake_episode.cars.append([])
        else:
            self.fake_episode.people = self.people[0:self.trainableCars[0].frame + 2]
            self.fake_episode.cars = self.cars[0:self.trainableCars[0].frame + 2]
        car_ids=[]
        self.fake_episode.car_data=[]
        for id, car in enumerate(self.trainableCars):
            self.fake_episode.car_data.append(DotMap())
            
            if car.car_id>0:
                if car.on_car and len(self.cars_dict[car.car_id])>self.trainableCars[0].frame+1: # For supervised training of cars

                    car_ids.append(car.car_id)
                    self.fake_episode.car_data[id].supervised_car_vel = np.array(
                        self.cars_dict[car.car_id][self.trainableCars[0].frame + 1]) - np.array(
                        self.cars_dict[car.car_id][self.trainableCars[0].frame])
                    self.fake_episode.car_data[id].supervised_car_vel = np.array([np.mean(self.fake_episode.car_data[id].supervised_car_vel[0:2]),
                                                                     np.mean(self.fake_episode.car_data[id].supervised_car_vel[2:4]),
                                                                     np.mean(self.fake_episode.car_data[id].supervised_car_vel[4:])])
                    # print (" Car vel "+str(fake_episode.car_data[id].supervised_car_vel))
                    if self.trainableCars[0].frame > 0:
                        self.fake_episode.car_data[id].supervised_car_vel_prev = np.array(
                            self.cars_dict[id][self.trainableCars[0].frame]) - \
                                                                    np.array(
                                                                        self.cars_dict[car.car_id][self.trainableCars[0].frame - 1])
                        self.fake_episode.car_data[id].supervised_car_vel_prev = np.array(
                            [np.mean(self.fake_episode.car_data[id].supervised_car_vel_prev[0:2]),
                             np.mean(self.fake_episode.car_data[id].supervised_car_vel_prev[2:4]),
                             np.mean(self.fake_episode.car_data[id].supervised_car_vel_prev[4:])])
                else:
                    closest_car_dist=max(self.reconstruction.shape)
                    closest_t=-1
                    for t, extr_car in enumerate(self.cars_dict[car.car_id]):
                        dist=np.linalg.norm(car.car[self.trainableCars[0].frame]-np.array([np.mean(extr_car[:2]),np.mean(extr_car[2:4]),np.mean(extr_car[4:])]))
                        if closest_car_dist>dist:
                            closest_t=t
                            closest_car_dist=dist
                    if closest_t>0:
                        if closest_car_dist<10:
                            cur_car=np.array(self.cars_dict[car.car_id][closest_t])
                            prev_car=np.array(self.cars_dict[car.car_id][max(closest_t-1,0)])
                            car_vel = np.array([np.mean(cur_car[0:2]-prev_car[0:2]), np.mean(cur_car[2:4]-prev_car[2:4]),
                                                    np.mean(cur_car[4:]-prev_car[4:])])
                            if np.linalg.norm(car_vel)>1e-4:
                                self.fake_episode.car_data[id].external_car_vel=car_vel*(1/np.linalg.norm(car_vel))

                        else:
                            if self.trainableCars[0].frame>0:
                                self.fake_episode.car_data[id].external_car_vel =car.external_car_vel[self.trainableCars[0].frame-1]

                            else:
                                self.fake_episode.car_data[id].external_car_vel= car.init_dir


        if len(car_ids)>0  and self.settings.ignore_external_cars_and_pedestrians==False:
            # print ("Add all cars except for id car "+str(len(self.cars_dict[self.car_id]))+" "+str(self.car_id))
            self.fake_episode.cars =[]
            for frame in range(self.trainableCars[0].frame +2):
                self.fake_episode.cars.append([])
            for key in self.valid_init.car_keys:
                init_frame=self.init_frames_cars[key]
                if key not in car_ids and init_frame<self.trainableCars[0].frame +2:
                    # print("Add id "+str(key)+" in range "+str((init_frame, min(self.trainableCars[0].frame +2,len(self.cars_dict[key])))))
                    for frame in range(init_frame, min(self.trainableCars[0].frame +2,len(self.cars_dict[key]))):
                        self.fake_episode.cars[frame].append(self.cars_dict[key][frame-init_frame])
        for car in self.trainableCars:
            car.update_episode_with_car_data(self.fake_episode)
        self.fake_episode.valid_directions=self.valid_init.valid_directions_cars
        actions={}
        for car in self.trainableCars:
            actions[car]=car.next_action(self.fake_episode, training,manual)
        return actions

    def update_car_episode(self, observation_dict):
        for pedestrian, observation in observation_dict.items():
            if len(self.pedestrian_data[pedestrian.id].agent) <= self.trainableCars[0].frame:
                self.pedestrian_data[pedestrian.id].agent.append(observation.agentPos)
            else:
                self.pedestrian_data[pedestrian.id].agent[self.trainableCars[0].frame] = observation.agentPos

            if len(self.pedestrian_data[pedestrian.id].velocity) <= self.trainableCars[0].frame:
                self.pedestrian_data[pedestrian.id].velocity.append(observation.agentVel)
            else:
                self.pedestrian_data[pedestrian.id].velocity[self.trainableCars[0].frame] = observation.agentVel
            # print("IN RL env after update pos "+str(self.agent)+" vel "+str(self.agent_velocity))
            self.pedestrian_data[pedestrian.id].init_dir = observation.init_dir
            # episode.agent, episode.car, episode.velocity, episode.velocity_car, agent_frame

    def perform_action(self, action, episode, prob=0):

        # Note, next_action function above is called before this, when the decision for the car is needed,
        # this is the reason for using the fake_episode here.
        for car in self.trainableCars:
            assert (action[car] == self.car.velocity[
                self.trainableCars[0].frame]).all(), "Just a sanity check that we called next_action and we previously took the same decision as we expected"

            car.perform_action(action[car], self.fake_episode, prob)
        #print(" After performing action "+str(self.car.velocity)+" car position "+str(self.car.car[self.trainableCars[0].frame])+" frame "+str(self.trainableCars[0].frame))


    def on_post_tick(self, episode, next_frame):
        for car in self.trainableCars:
            car.on_post_tick(episode)
            assert car.getFrame() == next_frame, "Sanity check failed, we are not on the same frame"
        if self.settings.ignore_external_cars_and_pedestrians == True:
            self.fake_episode.people = []
            self.fake_episode.cars = []
            for frame in range(next_frame):
                self.fake_episode.people.append([])
                self.fake_episode.cars.append([])
        else:
            self.fake_episode.people = episode.people[0:self.trainableCars[0].frame+1]
            self.fake_episode.cars = episode.cars[0:self.trainableCars[0].frame+1]


    # def getFrame(self):
    #     return self.car.getFrame()
    #
    def getIsPedestrian(self):
        return False
    #
    # def getOnlineRealtimeAgentId(self):
    #     return self.car.getOnlineRealtimeAgentId()

    def update_agent_pos_in_episode(self, episode, updated_frame):
        for car in self.trainableCars:
            assert car.getFrame() == updated_frame - 1, "We are not targeting the correct frame !  THe current car frame is {self.car.getFrame()} and we are updating for frame next frame as being {updated_frame}"
            car.update_agent_pos_in_episode(episode, updated_frame)
            self.car_data[id].car[updated_frame]=episode.car_data[id].car[updated_frame]

    def update_metrics(self, episode):
        # The logic is this: we advanced meantime the frame with +1. So we get that data from the environment (online or offline) then use that
        # info to compute the rewards and metrics

        # Now update the metric
        for car in self.trainableCars:
            car.update_metrics(episode)

        updatedFrame = max(self.trainableCars[0].frame - 1, 0)

        # Evaluate reward
        init_pos = np.zeros((self.settings.number_of_agents, 2))
        for car in self.trainableCars:
            #car.evaluate_car_reward(updatedFrame)
           # Put the metrics in the episode for the car since they are separate and we updated them internally not in episode like we do for pedestrian
            #print ("People predicted shape in episode update " + str(self.people_predicted[-1].shape)+" pos "+str(len(self.people_predicted)-1))
            episode.car_data[car.id].measures_car[updatedFrame,:]= car.measures[updatedFrame]
            # print ("Car measures after update  " + str(np.sum(self.measures_car[max(frame-1,0),:])))
            #episode.car_data[car.id].reward_car[updatedFrame] = car.reward[updatedFrame]
            # print ("Car reward after update  " + str(np.sum(self.reward_car[max(frame-1,0)]))+" reward "+str(car_reward)+" frame "+str(max(frame-1,0)))
            episode.car_data[car.id].probabilities_car[updatedFrame]=car.probabilities[updatedFrame]
            # print ("Car probabilities after update  " + str(np.sum(self.probabilities_car[max(frame-1,0), :])))
            episode.car_data[car.id].calculate_reward(updatedFrame, self.settings.reward_weights_car,
                                                     self.settings.car_reference_speed,
                                                     self.settings.car_max_speed_voxelperframe, self.settings.allow_car_to_live_through_collisions)
            if updatedFrame==0:
                init_pos[car.id,:]=episode.car_data[car.id].car[0][1:]
            if self.settings.learn_init_car:
                episode.initializer_car_data[car.id].car= episode.car_data[car.id].car # To do: Do I need to copy these over from the agent.
                # Could teh agent contain a car holder insteda that deals with allof this?
                episode.initializer_car_data[car.id].car_goal = episode.car_data[car.id].car_goal
                episode.initializer_car_data[car.id].speed_car = episode.car_data[car.id].speed_car
                episode.initializer_car_data[car.id].measures_car[updatedFrame, :] = episode.car_data[car.id].measures_car[updatedFrame,:]

                episode.initializer_car_data[car.id].calculate_reward(updatedFrame,self.settings.reward_weights_car_initializer, self.settings.car_reference_speed,self.settings.car_max_speed_voxelperframe,self.settings.allow_car_to_live_through_collisions)

                if updatedFrame == 0 and self.settings.reward_weights_car_initializer[CAR_REWARD_INDX.init_variance] != 0:
                    variance = np.var(init_pos, axis=0) / (max(self.reconstruction.shape) ** 2)

                    # print( self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.init_variance]*(variance[0]+ variance[1]))
                    episode.initializer_car_data[self.settings.number_of_car_agents - 1].reward_car[0] = +(self.settings.reward_weights_car_initializer[CAR_REWARD_INDX.init_variance] * (variance[0] + variance[1]))

    def train(self, ep_itr, statistics, episode, filename, filename_weights, poses, priors, statistics_car, last_frame):
        for car in self.trainableCars:
            car.train( ep_itr, statistics, episode, filename, filename_weights, poses, priors, statistics_car,last_frame)  # Shell of the other class

    def evaluate(self, ep_itr, statistics, episode, poses, priors, statistics_car,last_frame):
        for car in self.trainableCars:
            car.evaluate(ep_itr, statistics, episode, poses, priors, statistics_car,last_frame)  # Shell of the other class

