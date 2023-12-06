
from datetime import datetime
import os
import sys

import shlex
import subprocess
import time
import glob
import json
from dotmap import DotMap
import numpy as np
import copy



import sys
sys.path.insert(0,os.path.join(os.path.dirname( os.path.abspath(__file__)), '..'))
from commonUtils.ReconstructionUtils import METER_TO_VOXEL, KMH_TO_MS, LAST_CITYSCAPES_SEMLABEL


RANDOM_SEED=1234#2610#1234#8018- random seed for training
RANDOM_SEED_NP=RANDOM_SEED


DEFAULT_SKIP_RATE_ON_EVALUATION_CARLA = 4

# Define some paths, names for models and datasets
CARLA_CACHE_PREFIX_EVALUATE = "carla_evaluate"
CARLA_CACHE_PREFIX_EVALUATE_TOY = "carla_evaluate_toy"
CARLA_CACHE_PREFIX_EVALUATE_NEW = "carla_evaluate_new"
CARLA_CACHE_PREFIX_EVALUATE_REALTIME = "carla_evaluate_realtime"
CARLA_CACHE_PREFIX_EVALUATE_REALTIME_NO_EXTR_CARS_OR_PEDS="carla_evaluate_realtime_no_extr"
CARLA_CACHE_PREFIX_EVALUATE_ONLINE="carla_evaluate_online"
CARLA_CACHE_PREFIX_EVALUATE_NO_EXTR_CARS_OR_PEDS="carla_evaluate_no_extr"


CARLA_CACHE_PREFIX_TRAIN = "carla_train"
CARLA_CACHE_PREFIX_TRAIN_TOY = "carla_train_toy"
CARLA_CACHE_PREFIX_TRAIN_NEW = "carla_train_new"
CARLA_CACHE_PREFIX_TRAIN_NO_EXTR_CARS_OR_PEDS="carla_train_no_extr"
CARLA_CACHE_PREFIX_TRAIN_REALTIME = "carla_train_realtime"
CARLA_CACHE_PREFIX_TRAIN_ONLINE="carla_train_online"
CARLA_CACHE_PREFIX_TRAIN_REALTIME_NO_EXTR_CARS_OR_PEDS = "carla_train_realtime_no_extr"

CARLA_CACHE_PREFIX_TEST = "carla_test"
CARLA_CACHE_PREFIX_TEST_TOY = "carla_test_toy"
CARLA_CACHE_PREFIX_TEST_NEW = "carla_test_new"
CARLA_CACHE_PREFIX_TEST_NO_EXTR_CARS_OR_PEDS = "carla_test_no_extr"
CARLA_CACHE_PREFIX_TEST_REALTIME = "carla_test_realtime"
CARLA_CACHE_PREFIX_TEST_REALTIME_NO_EXTR_CARS_OR_PEDS = "carla_test_realtime_no_extr"
CARLA_CACHE_PREFIX_TEST_ONLINE="carla_test_online"


CARLA_CACHE_PREFIX_TEST_SUPERVISED = "carla_test_sup"



REAL_TIME_ONLINE_DATASET ="./DatasetCustom_templateFolder"

METER_TO_VOXEL = 5.0
VOXEL_TO_METER = 0.2
AGENT_MAX_HEIGHT_VOXELS =  10 # 2meters
NUM_SEM_CLASSES = float(LAST_CITYSCAPES_SEMLABEL)
NUM_SEM_CLASSES_ASINT = LAST_CITYSCAPES_SEMLABEL
POSE_DIM=512


class RLCarlaOnlineEnv():

    def __init__(self):
        self.width = 128 # Width and height, depth by default for the scene under online env
        self.height = 32# 256
        self.depth=256
        self.numFramesPerEpisode = 500 # TODO: this is from gatherdataset param. Instead provide metadataoutput or just put this at least in the json config !!!
        self.no_client_rendering = True
        self.numCarlaVehicles = 3
        self.numCarlaPedestrians = 5
        self.shouldTrainableAgentsIgnoreTrafficLights = True

        # Parameters for how to control the debugging editor when simulating real time or data gathering
        self.isDataGatherAndStoreEnabled = False
        self.useFixedWorldOffsets = True # This should be used in conjuction with the variable above. when gathering data, there is no need to do that but it is nice to do.
        self.fixedWorld_minX = -449.40484 # I took these values from the translation.txt file produced from Unreal raycaster by pressing F12 on our customized map levels.
        self.fixedWorld_minY = -388.37113
        self.fixedWorld_maxX = 382.90000
        self.fixedWorld_maxY = 375.00000
        self.fixedPixelsPerMeter = 5 # Because every pixel is a voxel. A voxel has 20 cm => 5 pixels per meter

        self.no_server_rendering = False # Should be no rendering on server ?
        self.no_client_rendering = False  # If true, there will be no/or simplified rendering on client side
        self.client_simplifiedRendering = True  # If true (and above also true), it will draw a debugging perspective of the world in 2D top view
        self.host = "localhost"  # Host of Carla Server
        self.port = 2000  # Port opened by Carla Server for clients connection
        self.scenesConfigFile = "scenesConfig.json"
        #simulationSceneName = None # "scene104"
        #simulationReplayPath = None # "RL/localUserData/Datasets/carla-realtime/"  # "OnlineEnvPlayData"
        self.forceSceneReconstruction = False

        self.useSegmentationMapForOnlineLevel = 0
        self.renderTrafficLights = 0
        self.renderSpeedLimits = 0

def get_flat_indx_list(stat):
    values_list=[]
    for value in  stat:
        if type(value) == list:
            for element in value:
                values_list.append(element)
        else:
            values_list.append(value)
    return values_list

# Add external code to the python path such that you can run the scripts from external models

# --------------------------------------------------- Initialization indexes
allExternalCodePaths = []
allExternalCodePaths.extend([STGCNN_CODEPATHS])

PEDESTRIAN_INITIALIZATION_CODE=DotMap()
PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian=1
PEDESTRIAN_INITIALIZATION_CODE.by_car=2
PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian=3
PEDESTRIAN_INITIALIZATION_CODE.randomly=4
PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car=5
PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian=6
PEDESTRIAN_INITIALIZATION_CODE.learn_initialization=7 # i.e. learnt by init_net
PEDESTRIAN_INITIALIZATION_CODE.on_pavement=8
PEDESTRIAN_INITIALIZATION_CODE.near_obstacle=9
PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory=10

# ---------------------------------------------------  Measures indexes
PEDESTRIAN_MEASURES_INDX=DotMap()
PEDESTRIAN_MEASURES_INDX.hit_by_car=0
PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory=1
PEDESTRIAN_MEASURES_INDX.iou_pavement=2
PEDESTRIAN_MEASURES_INDX.hit_obstacles=3
PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init=4
PEDESTRIAN_MEASURES_INDX.out_of_axis=5
PEDESTRIAN_MEASURES_INDX.dist_to_final_pos=6
PEDESTRIAN_MEASURES_INDX.dist_to_goal=7
PEDESTRIAN_MEASURES_INDX.hit_pedestrians=8
PEDESTRIAN_MEASURES_INDX.dist_to_goal_from_current=9
PEDESTRIAN_MEASURES_INDX.one_step_prediction_error=9
PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap=10
PEDESTRIAN_MEASURES_INDX.change_in_direction=11
PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init=12
PEDESTRIAN_MEASURES_INDX.goal_reached=13
PEDESTRIAN_MEASURES_INDX.agent_dead=14
PEDESTRIAN_MEASURES_INDX.difference_to_goal_time=15
PEDESTRIAN_MEASURES_INDX.change_in_pose=16
PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car=17
PEDESTRIAN_MEASURES_INDX.distracted=18
PEDESTRIAN_MEASURES_INDX.hit_by_hero_car=19

# ---- Car ------
CAR_MEASURES_INDX = PEDESTRIAN_MEASURES_INDX # Change if necessary!
CAR_MEASURES_INDX.dist_to_closest_pedestrian=1
CAR_MEASURES_INDX.dist_to_closest_car=10
CAR_MEASURES_INDX.id_closest_agent=17
CAR_MEASURES_INDX.hit_by_agent=19

# ---- Number of measures ------
NBR_MEASURES=max(PEDESTRIAN_MEASURES_INDX.values())+1
NBR_MEASURES_CAR = max(CAR_MEASURES_INDX.values())+1

# ---------------------------------------------------  Rewards indexes
PEDESTRIAN_REWARD_INDX=DotMap()
PEDESTRIAN_REWARD_INDX.collision_with_car=0
PEDESTRIAN_REWARD_INDX.pedestrian_heatmap=1
PEDESTRIAN_REWARD_INDX.on_pavement=2
PEDESTRIAN_REWARD_INDX.collision_with_objects=3
PEDESTRIAN_REWARD_INDX.distance_travelled=4
PEDESTRIAN_REWARD_INDX.out_of_axis=5
PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal=6
PEDESTRIAN_REWARD_INDX.collision_with_pedestrian=7
PEDESTRIAN_REWARD_INDX.reached_goal=8
PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory=9
PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently=10
PEDESTRIAN_REWARD_INDX.one_step_prediction_error=11
PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance=12
PEDESTRIAN_REWARD_INDX.not_on_time_at_goal=13
PEDESTRIAN_REWARD_INDX.large_change_in_pose=14
PEDESTRIAN_REWARD_INDX.inverse_dist_to_car=15
PEDESTRIAN_REWARD_INDX.collision_with_car_agent=16
PEDESTRIAN_REWARD_INDX.init_variance=17

# ---- Car ------
CAR_REWARD_INDX=DotMap()
CAR_REWARD_INDX.distance_travelled=0
CAR_REWARD_INDX.collision_pedestrian_with_car=1
CAR_REWARD_INDX.distance_travelled_towards_goal=2
CAR_REWARD_INDX.reached_goal=3
CAR_REWARD_INDX.collision_car_with_car=4
CAR_REWARD_INDX.penalty_for_intersection_with_sidewalk=5
CAR_REWARD_INDX.collision_car_with_objects=6
CAR_REWARD_INDX.penalty_for_speeding=7
CAR_REWARD_INDX.init_variance=8

# ---- Number of reward weights ------
NBR_REWARD_WEIGHTS=max(PEDESTRIAN_REWARD_INDX.values())+1
NBR_REWARD_CAR = max(CAR_REWARD_INDX.values())+1

# ---------------------------------------------------  Statistics indexes pedestrian agent
STATISTICS_INDX=DotMap()
STATISTICS_INDX.agent_pos=[0, 3]
STATISTICS_INDX.velocity=[3, 6]
STATISTICS_INDX.action=6
STATISTICS_INDX.probabilities=[7,34]
STATISTICS_INDX.angle=33
STATISTICS_INDX.reward=34
STATISTICS_INDX.reward_d=35
STATISTICS_INDX.loss=36
STATISTICS_INDX.speed=37
STATISTICS_INDX.measures=[38, 38+NBR_MEASURES]
STATISTICS_INDX.init_method=38+NBR_MEASURES
STATISTICS_INDX.frames_of_goal_change=38+NBR_MEASURES
STATISTICS_INDX.reward_initializer=38+NBR_MEASURES+1
STATISTICS_INDX.reward_initializer_d=38+NBR_MEASURES+2
STATISTICS_INDX.loss_initializer=38+NBR_MEASURES+3
STATISTICS_INDX.goal=[38+NBR_MEASURES+4,38+NBR_MEASURES+6]
STATISTICS_INDX.goal_time=38+NBR_MEASURES+6
STATISTICS_INDX.goal_person_id=STATISTICS_INDX.init_method
#STATISTICS_INDX.loss_car_initializer=38+NBR_MEASURES+7
NBR_STATS=max(get_flat_indx_list(STATISTICS_INDX.values()))+1


# ---- Pose ------
STATISTICS_INDX_POSE=DotMap()
STATISTICS_INDX_POSE.pose=[0, 93]
STATISTICS_INDX_POSE.agent_high_frq_pos=[93, 95]
STATISTICS_INDX_POSE.agent_pose_frames=95
STATISTICS_INDX_POSE.avg_speed=96
STATISTICS_INDX_POSE.agent_pose_hidden=[97, 97+512]
NBR_POSES=max(get_flat_indx_list(STATISTICS_INDX_POSE.values()))+1

# ---------------------------------------------------  Statistics indexes pedestrian initializer
# ---- Initializer Heatmap ------
STATISTICS_INDX_MAP=DotMap()
STATISTICS_INDX_MAP.prior=0
STATISTICS_INDX_MAP.init_distribution=1
# STATISTICS_INDX_MAP.goal_prior=2
# STATISTICS_INDX_MAP.goal_distribution=3

NBR_MAPS=max(get_flat_indx_list(STATISTICS_INDX_MAP.values()))+1
STATISTICS_INDX_MAP_GOAL=DotMap()
STATISTICS_INDX_MAP_GOAL.goal_prior=0
STATISTICS_INDX_MAP_GOAL.goal_distribution=1
NBR_MAPS_GOAL=max(get_flat_indx_list(STATISTICS_INDX_MAP_GOAL.values()))+1

# ---- Initializer Heatmap ------
STATISTICS_INDX_MAP_CAR=DotMap()
STATISTICS_INDX_MAP_CAR.prior=0
STATISTICS_INDX_MAP_CAR.init_distribution=1
NBR_MAPS_CAR=max(get_flat_indx_list(STATISTICS_INDX_MAP_CAR.values()))+1

# ---- Initializer Heatmap statistics (takes less space than saving initializer output) ------

STATISTICS_INDX_MAP_STAT=DotMap()

STATISTICS_INDX_MAP_STAT.init_position_mode=[0,2]
STATISTICS_INDX_MAP_STAT.init_prior_mode=[2,4]
STATISTICS_INDX_MAP_STAT.entropy=4
STATISTICS_INDX_MAP_STAT.entropy_prior=5
STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior=6
STATISTICS_INDX_MAP_STAT.prior_init_difference=7
NBR_MAP_STATS=max(get_flat_indx_list(STATISTICS_INDX_MAP_STAT.values()))+1
STATISTICS_INDX_MAP_STAT_GOAL = STATISTICS_INDX_MAP_STAT
STATISTICS_INDX_MAP_STAT_GOAL.frame = 8
NBR_MAP_STATS_GOAL = max(get_flat_indx_list(STATISTICS_INDX_MAP_STAT_GOAL.values())) + 1

# ---- Initializer statistics on car that is being targeted by initializer ------
STATISTICS_INDX_CAR_INIT=DotMap()
STATISTICS_INDX_CAR_INIT.car_id=0
STATISTICS_INDX_CAR_INIT.car_pos=[1,3]
STATISTICS_INDX_CAR_INIT.car_vel=[3,5]
STATISTICS_INDX_CAR_INIT.manual_goal=[5,7]
NBR_CAR_MAP=max(get_flat_indx_list(STATISTICS_INDX_CAR_INIT.values()))+1

# ---------------------------------------------------  Statistics indexes car model
STATISTICS_INDX_CAR=DotMap()
STATISTICS_INDX_CAR.agent_pos=[0, 3]
STATISTICS_INDX_CAR.velocity=[3, 6]
STATISTICS_INDX_CAR.action=6
STATISTICS_INDX_CAR.probabilities=[7,9]
STATISTICS_INDX_CAR.goal=9
STATISTICS_INDX_CAR.reward=10
STATISTICS_INDX_CAR.reward_d=11
STATISTICS_INDX_CAR.speed=13
STATISTICS_INDX_CAR.bbox=[14, 20]
STATISTICS_INDX_CAR.measures=[20, 20+NBR_MEASURES_CAR]
STATISTICS_INDX_CAR.dist_to_agent=20+max(get_flat_indx_list(CAR_MEASURES_INDX.values()))
STATISTICS_INDX_CAR.angle=20+max(get_flat_indx_list(CAR_MEASURES_INDX.values()))+1
STATISTICS_INDX_CAR.reward_initializer=20+max(get_flat_indx_list(CAR_MEASURES_INDX.values()))+2
STATISTICS_INDX_CAR.reward_initializer_d=20+max(get_flat_indx_list(CAR_MEASURES_INDX.values()))+3
STATISTICS_INDX_CAR.loss=20+max(get_flat_indx_list(CAR_MEASURES_INDX.values()))+4
STATISTICS_INDX_CAR.loss_initializer=20+max(get_flat_indx_list(CAR_MEASURES_INDX.values()))+5
NBR_STATS_CAR=max(get_flat_indx_list(STATISTICS_INDX_CAR.values()))+1

# Some epsilon constants
CONST_MIN_PROB_EPSILON = 0.000000001
CONST_MINVEL_EPS = 0.00000001


for path in allExternalCodePaths:
    sys.path.append(path)

realTimeEnvOnline = False


class run_settings(object):

    # DEBUG LOG VARIABLES
    DEBUG_LOG_AGENT_INPUTS = False # Trainable agents inputs show

    memoryProfile = False
    printdebug_network_input = False
    printdebug_network="place_car"

    # Scaling of environment:
    # leave it as static please ! #To do: look up when this is wever used!
    scale_y = 5
    scale_x = 5
    scale_z = 5

    # TODO: should put some setting as static variables to avoid initializing this costly object at runtime just to investigate some arguments...
    realTimeEnvOnline = realTimeEnvOnline  # Should frame by frame simulation comeup from a real time interaction of from a pre-recorded data source

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't include states that shouldn't be saved
        del state["update_batch_init"]
        del state["update_batch_car"]
        del state["update_batch_goal"]
        del state["update_batch"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # TODO: add the things back if needed !
        self.baz = 0

    def __init__(self, evaluate=False, name_movie="", env_width=None, env_depth=None, env_height=None, likelihood=False,
                 evaluatedModel="", datasetToUse="carla", pfnn=True, forceRealTimeEnvOnline=False, forcedPathToCurrentDataset=None,
                 forced_camera_pos_x = None, forced_camera_pos_y = None):

        # Main settings, defining type of run.
        self.goal_dir=True # Does pedestrian agent have goal?
        self.learn_init = True # Do we need to learn initialization
        self.learn_init_car=False
        self.learn_goal=False # Do we need to learn where to place the goal of the pedestrian at initialization
        self.learn_time=False # Do we need to learn temporal constraint given to pedestrian

        # Debugging, profiling, caching
        self.fastTrainEvalDebug = False
        self.debugFullSceneRaycastRun = False
        self.profile =False
        self.useCaching = True
        self.ignore_external_cars_and_pedestrians=False
        self.test_online_dataset = False

        # Number of pedestrian and car agents
        self.number_of_car_agents=4#3 # 2
        self.number_of_agents =8

        if forceRealTimeEnvOnline is True:
            self.realTimeEnvOnline = forceRealTimeEnvOnline



        # ---------------------------------------------------  Pedestrian reward weights
        # Reward weights.- weighst for different reward terms.
        # cars, people, pavement, objs, dist, out_of_axis
        self.reward_weights_pedestrian = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]
        # cars
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.collision_with_car] =-2
        # Pedestrian heatmap reward
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap]=0.01
        # Pavement
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.on_pavement] =0#0.1# 0
        # Objs
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.collision_with_objects] =-0.02
        # Reward for dist travelled
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.distance_travelled] =0#1#0#0.0001#-0.001#0.001#0.001#-
        # Out of axis
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.out_of_axis] = 0
        # linear reward for dist, gradual improvement towards goal

        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] = 0.0 if self.goal_dir == False else 0.1#0.1# 0.1#0.1
        # Neg reward bump into people
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] =-0.1#15# -0.15
        # Positive reward for reaching goal
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.reached_goal] = 0.0 if self.goal_dir == False else 2#-2#1

        # Reward for being on top of pedestrian trajectory
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] =0.01 #
        # Reward for not changing direction
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = 0#-0.01  #
        # Reward for being close to following agent
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] = 0#0.1
        # Penalty for flight of distance
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] =0#-0.001 #-0.1#-0.1# -0.0001 # 0.1

        # Penalty for not reaching goal on time
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] =0#-0.0001  # 0#-0.1#-0.1# -0.0001 # 0.1

        # Penalty for large change in poses
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.large_change_in_pose] =0#-0.0001  # 0#-0.1#-0.1# -0.0001 # 0.1
        # distance to car reward
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.inverse_dist_to_car] =0#0.001#0#1  # -0.0001  # 0#-0.1#-0.1# -0.0001 # 0.1
        # collision with car agent
        #self.reward_weights_pedestrian = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.collision_with_car_agent] =-2

        #---------------------------------------------------- Init weights
        self.reward_weights_initializer=copy.deepcopy(self.reward_weights_pedestrian)
        # for indx, value in enumerate( self.reward_weights_initializer):
        #     self.reward_weights_initializer[indx]=0
        self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.collision_with_car_agent] = 2*self.number_of_agents
        self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.collision_with_car]=0#-2#0#2*self.number_of_agents
        self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.reached_goal]=2/(self.number_of_agents*100)#self.seq_len_train)
        self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal]=self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal]/(self.number_of_agents*100)

        self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.init_variance]=0.1
        # ---------------------------------------------------  Car reward weights
        self.reward_weights_car=np.zeros(NBR_REWARD_CAR)
        # Reward for distnace travelled by car
        self.reward_weights_car[CAR_REWARD_INDX.distance_travelled] = 0.01#7/4#0.0001
        self.reward_weights_car[CAR_REWARD_INDX.collision_pedestrian_with_car] = -2
        self.reward_weights_car[CAR_REWARD_INDX.distance_travelled_towards_goal] =0#0.1
        self.reward_weights_car[CAR_REWARD_INDX.reached_goal] = 0#2
        self.reward_weights_car[CAR_REWARD_INDX.collision_car_with_car]=-2
        self.reward_weights_car[CAR_REWARD_INDX.penalty_for_intersection_with_sidewalk]=-.1
        self.reward_weights_car[CAR_REWARD_INDX.collision_car_with_objects] = -2
        self.reward_weights_car[CAR_REWARD_INDX.penalty_for_speeding]=-0.1

        # ---------------------------------------------------- Car Init weights
        self.reward_weights_car_initializer = copy.deepcopy(self.reward_weights_car)
        # for indx, value in enumerate( self.reward_weights_initializer):
        #     self.reward_weights_initializer[indx]=0
        self.reward_weights_car_initializer[CAR_REWARD_INDX.collision_pedestrian_with_car] = 2
        self.reward_weights_car_initializer[CAR_REWARD_INDX.init_variance] = 0#0.1

        # ---------------------------------------------------  Load weights

        experiment_name = ""

        # When running on trained or pretrained weights.
        init_model=""
        model =""
        car_model =""

        # ---------------------------------------------------  Pedestrian agent parameters
        # Pedestrian agent parameters
        self.curriculum_goal = False
        self.curriculum_reward = False
        self.curriculum_seq_len = False

        # Model input and architecture
        self.predict_future =True
        self.temporal = True
        self.lstm = True
        self.extend_lstm_net = True
        self.extend_lstm_net_further = True
        self.old_lstm = False
        self.old = False
        self.old_fc = False
        self.old_mem = False
        self.conv_bias = True
        self.past_filler = False
        self.N = 10

        # Alternative larger scale model choices
        self.resnet = False
        self.mem_2d = False
        self.mininet = False
        self.estimate_advantages = False

        # PFNN related paremeters
        self.pfnn = True#pfnn  # Is possible use the init parameter
        self.max_frames_to_init_pfnnvelocity = 300 # How many frames to allow PFNN init
        self.sigma_vel = 0.5#2.0#2.0#1.8#0.1#0.5#2.0_ca
        self.action_freq = 1#4
        self.pose =False

        # Action space choices
        self.velocity = True
        self.velocity_sigmoid = True
        self.acceleration = False
        self.detect_turning_points = False
        self.continous = False
        self.speed_input=False
        if self.learn_time:
            self.speed_input = True
        self.angular=False
        self.actions_3d = False
        self.constrain_move = False
        self.attention = False
        self.confined_actions = False
        self.reorder_actions = False

        # When to end episode
        self.end_on_bit_by_pedestrians = False  # End episode on collision with pedestrians.
        self.stop_on_goal = False
        self.longer_goal=False
        self.stop_on_goal_car = False
        self.end_episode_on_collision=True


        # Training type
        self.refine_follow = False
        self.supervised = True #applies only when initialised on pedestrians
        self.heatmap = False

        # -------------------------------------------------- Dataset settings
        self.carla=True # Run on CARLA?
        self.new_carla=False
        self.new_carla_net=False # new semantic channels.
        self.realtime_carla = True
        self.realtime_carla_only = False
        self.waymo = False  # Run on waymo?

        self.dataset_frameRate, self.dataset_frameTime = self.getFrameRateAndTime()

            # Other environemnt settings
        self.temp_case = False
        self.paralell = False
        self.pop = False
        self.social_lstm = False

        # self.random_seed_np = RANDOM_SEED
        # self.random_seed_tf = self.random_seed_np
        self.timing = False

        # ---------------------------------------------------  Initializer parameters
        # Initializer parameters
        self.manual_init=False # Should the pedestrian initialization be given manually?
        # Architecture parameters
        self.inilitializer_interpolate=True # interpolate convolutional layers?
        self.inilitializer_interpolated_add=True # add interpolates convolutional layers?
        self.sem_arch_init=False # david's architecture
        self.gaussian_init_net=False
        self.separate_goal_net=False
        self.goal_gaussian=False # TO DO: look up what was used last
        self.init_std=2.0
        self.goal_std = 10.0#2.0#0.01#.5#2.0
        # Occlusion?
        self.use_occlusion=False
        self.lidar_occlusion=False
        self.evaluate_prior=False
        self.random_init=False
        self.evaluate_prior_car=False
        self.prior_smoothing=False
        self.prior_smoothing_sigma=15
        self.occlude_some_pedestrians=False
        self.add_pavement_to_prior=False
        self.assume_known_ped_stats=True
        self.assume_known_ped_stats_in_prior = False
        self.car_occlusion_prior=True
        self.attack_random_car=False

        # ---------------------------------------------------  Car parameters
        # Car model
        self.manual_car=False # Does the car follow manually given actions or of the network?
        self.car_input = "distance_to_car_and_pavement_intersection"  # "distance"# time_to_collision# scalar_product "difference" # "distance_to_car_and_pavement_intersection"
        self.supervised_car = False
        self.supervised_and_rl_car = False
        self.linear_car = True
        self.car_constant_speed = False
        self.angular_car=False
        self.angular_car_add_previous_angle=False
        self.sigma_car = 0.1  # variance for linear car
        self.sigma_car_angular = 0.1 # variance for linear car
        self.car_motion_smoothing=True

        self.ped_reference_speed = (3.2 * KMH_TO_MS * METER_TO_VOXEL) / self.dataset_frameRate # Pedestrian reference speed in voxels per frame --- To DO: Where is this used????
        self.car_reference_speed = (40 * KMH_TO_MS * METER_TO_VOXEL) / self.dataset_frameRate  # Car reference speed in voxels per frame
        self.car_max_speed_km_h = 70
        self.car_min_speed_km_h = 5
        self.car_max_speed_voxelperframe = ((self.car_max_speed_km_h * KMH_TO_MS) * METER_TO_VOXEL) / self.dataset_frameRate
        self.car_min_speed_voxelperframe = ((self.car_min_speed_km_h * KMH_TO_MS) * METER_TO_VOXEL) / self.dataset_frameRate
        self.allow_car_to_live_through_collisions = False
        self.lr_car = 0.1/6.0  # 0.1#0.005#0.005#0.0001#.5 * 1e-3  # learning rate # 5*1e-3

        # ---------------------------------------------------  Real time/ Hero car run settings
        # Real time env settings
        self.useRealTimeEnv = True # Should we use a frame by frame simulation ?

        if self.realTimeEnvOnline or forceRealTimeEnvOnline:
            self.useRealTimeEnv = True

        # Note: If you use online environments or datagathering from online envs, then one of these must be true to have support for training / capturing vehicles
        # If your scene defintion (see scenesConfig.json in the dataset !) contains "numTrainableVehicles" > 0, and these are both False, it SHOULD assert you.
        # If you want to train only pedestrians, just put numTrainablevehicles = 0 in your scene config.
        self.useHeroCar = True
        self.useRLToyCar = True
        self.train_init_and_pedestrian=False



        self.onlineEnvSettings = None

        # Alternative training of initializer and car
        self.train_car_and_initialize = "alternatively"#"according_to_stats"  # "simultaneously", "alternatively" ,"according_to_stats" or ""
        self.car_success_rate_threshold =1.0#0.7
        self.initializer_collision_rate_threshold = 1.0
        if self.train_car_and_initialize == "according_to_stats" and  self.car_success_rate_threshold ==1.0 and self.initializer_collision_rate_threshold== 1.0:
            self.train_only_initializer=True
        else:
            self.train_only_initializer = False
        self.num_init_epochs = 1
        self.num_car_epochs = 2
        if len(self.train_car_and_initialize)==0:
            self.keep_init_net_constant = True
            self.save_init_stats = False
        else:
            self.keep_init_net_constant = False
            self.save_init_stats = True  # instead of saving init maps, save only statistics. takes much less space.

        # Noise
        self.add_noise_to_car_input = False
        self.car_noise_probability = 0.3
        self.pedestrian_noise_probability = 0.3
        self.distracted_pedestrian = False
        self.avg_distraction_length = 2
        self.pedestrian_view_occluded = True
        self.field_of_view=114/180*np.pi
        self.field_of_view_car=90/180*np.pi

        # ---------------------------------------------------  General learning parameters (common to car, agent, initializer)
        # Run types
        self.stop_for_errors = False
        if self.fastTrainEvalDebug:
            self.stop_for_errors=True
        self.overfit = False
        self.save_frequency = 100  # How frequently to save stats when overfitting?
        self.likelihood = likelihood
        self.run_evaluation_on_validation_set = False

        # Training  model hyper parameters


        self.multiplicative_reward_pedestrian = False
        self.multiplicative_reward_initializer = True

        self.entr_par=0#.1#.2 # Entropy of pedestrian
        self.entr_par_init = 0.1  # .1#.2 # Entropy of initializer
        self.entr_par_goal = 0#0.001  # .1#.2 # Entropy of goal-initializer
        self.replay_buffer = 0  # 10
        self.polynomial_lr = False


        # Learning rate and batch size
        self.lr =0.1 #0.005  # 0.1#0.005#0.005#0.0001#.5 * 1e-3  # learning rate # 5*1e-3
        self.batch_size = 1  # 50
        #self.epsilon = 5  # 0.2

        # Discount rate
        self.gamma = .99  # 1
        self.initializer_gamma=1
        self.people_traj_gamma = .95
        self.people_traj_tresh = 0.1

        # Test other types of agents.
        self.random_agent=False
        self.goal_agent=False
        self.pedestrian_agent=False
        self.carla_agent=False
        self.cv_agent =False
        self.toy_case=False


        # Sample actions from a distribution or sample the most likely action
        self.deterministic = True # Determinstic environment (iei always same outcome at collisions).
        self.deterministic_test = True  # evaluate deterministrically
        if not self.deterministic:
            self.prob = 1

        # ----------------Dataset----------------------------------------------
        self.carla=True # Run on CARLA?
        self.new_carla=False
        self.new_carla_net=False
        self.realtime_carla=True
        self.realtime_carla_only = False
        self.waymo=False # Run on waymo?

        if self.realtime_carla or self.realtime_carla_only or self.useRealTimeEnv:
            self.onlineEnvSettings = RLCarlaOnlineEnv()


        # Other environemnt settings
        self.temp_case=False
        self.paralell=False
        self.pop=False
        self.social_lstm=False

        # self.random_seed_np = RANDOM_SEED
        # self.random_seed_tf = self.random_seed_np
        self.timing = False

        # ---------------------------------------------------  Pedestrian model architecture specific paramteres

        # Channels of pedestrian agent
        self.run_2D = True
        self.min_seg = True
        self.no_seg = False
        self.mem = False
        self.action_mem = True
        if self.mem or self.action_mem:
            self.nbr_timesteps = 10  # 32
        self.sem = 1  # [1-semantic channels ,0- one semantic channel,-1- no semantic channel]# 1
        self.cars = True  # Car channel?
        self.people = True  # People channel
        self.avoiding_people_step = 10000000
        self.car_var = True

        # Network layers -pedestrian agent and initializer
        self.pooling = [1, 2]
        self.num_layers = 2
        self.num_layers_init = 2
        # self.fully_connected_layer=0
        self.outfilters = [1, 1]  # 64,128]
        self.controller_len = -1  # -1
        self.sigma = 0.1
        self.max_speed = 2
        # Fully connected layers
        self.fc = [0, 0]  # 1052, 512
        self.fully_connected_layer = -1  # 128
        self.pool = False
        self.batch_normalize = False
        self.clip_gradients = False
        self.regularize = False
        self.normalize = False
        self.normalize_channels_init = True

        # ------------------------------------------------------- Set Run name
        if evaluate:
            self.carla = self.waymo = False
            if datasetToUse == "carla":  # default
                self.carla = True  # Run on CARLA?
                self.realtime_carla=False
            elif datasetToUse == "waymo":
                self.waymo = True
            else:
                assert datasetToUse == "cityscapes"
            # Note: if not, we run citiscapes...

        runMode = "evaluate_" if evaluate else "run_"
        self.name_movie = runMode

        # Put the name of the external model to run comparisons against, or None if working as standalone on our model only
        self.evaluateExternalModel = None if evaluatedModel == "" else evaluatedModel
        self.lastAgentExternalProcess = None
        # Basename
        if self.likelihood:
            self.name_movie += "likelihood_"
        else:
            self.name_movie += "agent_"

        # Which dataset ?
        if self.carla == True:
            self.name_movie += "_carla"
        elif self.waymo == True:
            self.name_movie += "_waymo"
        else:
            self.name_movie += "_cityscapes"

        # Agent type:
        if self.evaluateExternalModel:
            self.name_movie += "_" + self.evaluateExternalModel

        # Using pfnn ?
        if self.pfnn:
            self.name_movie += "_pfnn"

        if not evaluate:
            self.name_movie += experiment_name
        print ("Movie name " + self.name_movie)
        self.TEST_JUMP_STEP_CARLA = 1
        self.VAL_JUMP_STEP_CARLA = 4#4 #1 # How many to skip to do faster evaluation
        self.VAL_JUMP_STEP_WAYMO = 1

        # ------------------------------------------------------- Evaluation parameters
        if evaluate:
            # Correct settings for evaluation!
            self.temporal = True
            self.predict_future = True
            self.continous = False
            self.overfit = False
            self.pfnn = True
            self.speed_input = False
            self.end_on_bit_by_pedestrians = False
            self.angular = False
            #self.stop_on_goal = True
            self.replay_buffer = 0  # 10
            self.entr_par = 0  # .1#.2
            self.stop_for_errors = True
            self.overfit = False
            self.mininet = False
            self.refine_follow = False
            self.heatmap = False
            self.confined_actions = False
            self.reorder_actions = False
            self.run_evaluation_on_validation_set = False
            self.attention = False
            self.settings = False
            self.supervised = True
            self.random_agent = False
            self.goal_agent = False
            self.pedestrian_agent = False
            self.carla_agent = False
            self.cv_agent = False
            self.toy_case = False
            self.temp_case = False
            self.paralell = False
            self.pop = False
        # ------------------------------------------------------- PFNN parameters
        # PFNN and continous agent special paramters
        if self.pfnn and not self.continous:  # and self.lr:
            self.pose = True
            self.velocity = True
            self.velocity_sigmoid = True
            self.action_freq = 4
            if self.goal_dir:
                self.action_freq = 1#4
                self.reward_weights_pedestrian[6] = 0.1
                self.reward_weights_pedestrian[8] = 2
            else:
                self.action_freq = 4
                self.reward_weights_pedestrian[6] = 0
                self.reward_weights_pedestrian[8] = 0
            self.sigma_vel = 0.1#2.0
            self.lr = 0.001
            self.reward_weights_pedestrian[14] = -0.0001
        # ------------------------------------------------------- PATHS
        # Where to save model
        self.model_path = ""

        # Where to save statistics file
        # Statistics dir and img_dir are computed below. Please don't hardcode let it be data driven by settings..

        self.profiler_file_path = "./"
        self.statistics_dir = "RL/localUserData/Results/statistics"
        self.img_dir = "RL/localUserData/Results/agent/"

        # Episode cache settings
        self.target_episodesCache_Path ="RL/localUserData/CachedEpisodes"
        self.target_initCache_Path =os.path.join(self.target_episodesCache_Path, 'init')
        self.target_realCache_Path = os.path.join(self.target_episodesCache_Path, 'real')

        self.deleteCacheAtEachRun = False

        self.timestamp=datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')

        ### Change paths from here onwards
        self.pfnn_path="commonUtils/PFNNBaseCode/"

        # Where to save current experiment's settings file.
        self.path_settings_file = "RL/localUserData/some_results/"

        # Where to save statistics files from runs?
        self.eval_main="RL/localUserData/Models/eval/"
        self.evaluation_path="RL/localUserData/Models/eval/" + self.name_movie+"_"+self.timestamp

        # Where to save simple visualizations of agent behaviour?
        self.name_movie_eval = "RL/localUserData/Datasets/cityscapes/pointClouds/val/"

        # Where to save statistics files from runs?
        self.evaluation_path = "RL/localUserData/Models/eval/" + self.name_movie + "_" + self.timestamp
        self.eval_main ="RL/localUserData/Models/eval/"

        # Where to save simple visualizations of agent behaviour?
        self.name_movie_eval = "RL/localUserData/Datasets/cityscapes/pointClouds/val/"

        self.LocalResultsPath ="RL/localUserData/visualization-results/manual/"


        if self.realTimeEnvOnline:
            self.carla_main = "RL/localUserData/Datasets/carla-realtime/"
        else:
            self.carla_main = "RL/localUserData/Datasets/carla-sync/"

        self.carla_path = os.path.join(self.carla_main, "train/")
        self.carla_path_test = os.path.join(self.carla_main, "val/")
        self.carla_path_viz = os.path.join(self.carla_main,"new_data-viz/")

        if self.new_carla:
            # Path to CARLA dataset

            if self.realTimeEnvOnline:
                self.carla_main = REAL_TIME_ONLINE_DATASET
            else:
                self.carla_main = "RL/localUserData/Datasets/carla-new"


            self.carla_path = os.path.join(self.carla_main,"Town03/")  # "Packages/CARLA_0.8.2/unrealistic_dataset/"

            # Path to CARLA test set
            self.carla_path_test = os.path.join(self.carla_main, "Town03/")
            # Path to CARLA visualization set
            self.carla_path_viz = os.path.join(self.carla_main, "Town03/")
        if self.realtime_carla:
            self.carla_main_realtime = "RL/localUserData/Datasets/carla-realtime/"
            self.name_movie_eval_realtime = os.path.join(self.carla_main_realtime, "eval/")
            self.carla_path_realtime = os.path.join(self.carla_main_realtime, "train/")
            self.carla_path_test_realtime = os.path.join(self.carla_main_realtime, "val/")
            self.realtime_freq=90

        if self.realtime_carla_only:
            self.realtime_carla=False
            self.carla_main = REAL_TIME_ONLINE_DATASET
            self.carla_path_viz = os.path.join(self.carla_main, "viz/")
            self.name_movie_eval = os.path.join(self.carla_main, "eval/")
            self.carla_path= os.path.join(self.carla_main, "train/")
            self.carla_path_test= os.path.join(self.carla_main, "val/")
            self.realtime_freq = 0
            self.VAL_JUMP_STEP_CARLA = 1  # 1


        self.mode_name = "train"  # mode in cityscapes data

        if len(model)>0:
            self.load_weights = os.path.join(self.img_dir,model)  # "Results/agent/run_agent__carlalearn_initializer_multiplicative_2020-12-29-20-58-38.182501/"
            print(" Load "+str(self.load_weights ))
        else:
            self.load_weights =""

        if len(car_model)>0:
            self.load_weights_car = os.path.join(self.img_dir,car_model)  # "Results/agent/run_agent__carlalearn_initializer_multiplicative_2020-12-29-20-58-38.182501/"
            print(" Load " + str(self.load_weights))
        else:
            self.load_weights_car = ""

        if len(init_model) > 0:
            self.load_weights_init = os.path.join(self.img_dir,
                                                 init_model)  # "Results/agent/run_agent__carlalearn_initializer_multiplicative_2020-12-29-20-58-38.182501/"
            print(" Load " + str(self.load_weights))
        else:
            self.load_weights_init = ""
        # ---------------------------------------------------  Simulation specific parameters------------------------
        # --- Agent, car and environment size----
        # Size of agent-not used currently.
        self.height_agent = 4
        self.width_agent = 2
        self.depth_agent = 2
        self.agent_shape=[self.height_agent, self.width_agent,self.depth_agent]

        # How many voxels beyond oneself can the agent see.
        self.height_agent_s = 1
        self.width_agent_s = 10
        self.depth_agent_s = 10
        # if self.fastTrainEvalDebug:
        #     self.width_agent_s = 10
        #     self.depth_agent_s = 10
        self.agent_shape_s = [self.height_agent_s,  self.width_agent_s,self.depth_agent_s]
        self.net_size=[1+2*(self.height_agent_s+self.height_agent), 1+2*(self.depth_agent_s+self.depth_agent), 1+2*(self.width_agent_s+self.width_agent)]
        self.agent_s = [self.height_agent_s + self.height_agent, self.depth_agent_s + self.depth_agent,
                        self.width_agent_s + self.width_agent]

        # Size of car agent in voxels
        self.car_dim = [self.height_agent_s, 4, 7]

        # Get the dimensions of the environment:
        self.pathToCurrentDataset = None
        if forcedPathToCurrentDataset is not None:
            self.pathToCurrentDataset = forcedPathToCurrentDataset
        elif self.carla:
            self.pathToCurrentDataset = self.carla_main
        elif self.waymo:
            self.pathToCurrentDataset = self.waymo_path
        else:
            self.pathToCurrentDataset = self.colmap_path

        if not self.realTimeEnvOnline and not self.waymo: # this is to be able to run on the old data.
            env_width = 128
            env_depth = 256
            env_height = 32
            if self.new_carla:
                print("New CARLA ")
                env_depth = 1024
                env_height = 250
                env_width = 1024
            if self.onlineEnvSettings:
                print("Online Env Settings")
                env_height = self.onlineEnvSettings.height
                env_depth = self.onlineEnvSettings.depth
                env_width = self.onlineEnvSettings.width
            camera_pos_x=0
            camera_pos_y=(-env_width // 2)

        if env_height is None or env_width is None or env_depth is None: # If not forced take it from data
            self.pathToDatasetConfigFile = os.path.join(self.pathToCurrentDataset, "scenesConfig.json")

            print(f"### Dataset path config file loaded: {self.pathToDatasetConfigFile} -- be sure it is the correct one !")
            #assert os.path.exists(pathToDatasetConfigFile, "The configuration file for the dataset doesn't exist. "
            #                                                 "Please provide one in your folder AT LEAST with the info required as the variables reading below !")

            with open(self.pathToDatasetConfigFile, 'r') as scenesConfigStream:
                data = json.load(scenesConfigStream)
                env_width = data['env_width']
                env_depth = data['env_depth']
                env_height = data['env_height']
                camera_pos_x = data['camera_pos_x']
                camera_pos_y = data['camera_pos_y']
        else:
            print("You are forcing the environment size instead of taking it from dataset config directly, are you sure you want to do this ?")

        # If forcing some camera position, overwrite the defaults
        if forced_camera_pos_y != None and forced_camera_pos_x != None:
            camera_pos_x = forced_camera_pos_x
            camera_pos_y = forced_camera_pos_y

        self.height = env_height
        self.width = env_width
        self.depth = env_depth
        print("In settings file "+str(self.height)+" "+str(self.width)+" "+str(self.depth))
        self.env_shape = [self.height, self.width, self.depth]

        self.camera_pos_x = camera_pos_x
        self.camera_pos_y = camera_pos_y

        # --- Sequence length, testing, network update and visualization frequencies----

        # Sequence length
        self.seq_len_train =30*self.action_freq#10*self.action_freq
        self.seq_len_train_final =30*self.action_freq#100
        self.seq_len_evaluate = 30*self.action_freq#600#300#300
        self.seq_len_test =30*self.action_freq
        self.seq_len_test_final = 30*self.action_freq#450#100#*self.action_freq#200
        if (self.pfnn or self.goal_dir) and not self.useRLToyCar and not self.learn_init:
            self.seq_len_train = 10*self.action_freq  # *self.action_freq#10*self.action_freq
            self.seq_len_train_final = 30*self.action_freq  # *self.action_freq#100
            self.seq_len_test = 10*self.action_freq  # 30#10#*self.action_freq
            self.seq_len_test_final = 30*self.action_freq  # 30#400#100*self.action_freq#200
        self.threshold_dist =100#231#100#*self.action_freq # Longest distance to goal
        if  (not self.pfnn and self.goal_dir) and not self.useRLToyCar and not self.learn_init:
            self.threshold_dist =self.seq_len_train
        train_only_initializer=self.train_car_and_initialize == "according_to_stats" and self.initializer_collision_rate_threshold==1.0
        if self.learn_init and (not self.useRLToyCar or train_only_initializer):
            self.seq_len_train = 100 #450
            self.seq_len_train_final = 100#450
            self.seq_len_evaluate =100#450
            self.seq_len_test = 100#450
            self.seq_len_test_final =100#450


        # Update network gradients with this frequency.
        self.update_frequency = round(30/self.number_of_agents)
        self.update_frequency_init=30
        self.update_frequency_test=5#round(10/self.number_of_agents)#10

        # visualization frequency
        self.vizualization_freq=50
        self.vizualization_freq_test =15#50
        self.vizualization_freq_car=5#1

        # Test with this frequency.1
        self.test_freq =100#100
        if self.waymo:
            self.test_freq = 20
        if self.realtime_carla_only:
            self.test_freq = 15
        self.car_test_frq=50 # test on toy car environemnt with this frequency

        # Save with this frequency.
        self.save_freq = 100  # 200  # 50

        # Evaluation specific frequencies
        if evaluate:
            if self.likelihood:
                if self.waymo:
                    self.seq_len_evaluate = 5200
                else:
                    self.seq_len_evaluate =5450
            else:
                self.seq_len_evaluate = 5300
                self.threshold_dist = 155# 31 m or average speed 1.75m/s
            self.update_frequency_test = 10

        # Override some values to get fast feedback
        if self.fastTrainEvalDebug:
            self.update_frequency = 2#5
            self.update_frequency_test=2#10#10

            self.vizualization_freq=2
            self.vizualization_freq_test =2#50
            self.vizualization_freq_car=1#1

            self.test_freq = 10#2#100#1
            if self.waymo:
                self.test_freq = 1#20
            #self.test_frequency=2#3
            self.car_test_frq=1
            self.seq_len_train = 10 * self.action_freq  # *self.action_freq#10*self.action_freq
            self.seq_len_train_final = 10 * self.action_freq  # *self.action_freq#100
            self.seq_len_test = 10 * self.action_freq  # 30#10#*self.action_freq
            self.seq_len_test_final = 10 * self.action_freq  # 30#400#100*self.action_freq#200


        # --- Initializations ----
        # Which initializations to use. Note overriden in environment class.
        self.widths = [7]#[2, 4, 6, 7, 10] # road width in toy class
        #self.repeat = len(self.widths) * len(self.train_nbrs) * 50

        # POP optimization specififc parameters
        if self.pop:
            self.normalize = True
            self.fully_connected_layer = 128
            self.estimate_advantages=True
            self.vf_par=0.5
            self.entr_par=0
        # Initialize on pavement?
        self.init_on_pavement = 0

        self.sem_class = []

    @staticmethod
    def getWaymoFramerate():
        return 10.0

    @staticmethod
    def getCarlaFramerate():
        return 17.0


    def getFrameRateAndTime(self):
        frame_rate = None
        if self.waymo:
            frame_rate = self.getWaymoFramerate()
        else:
            frame_rate = self.getCarlaFramerate()
        frame_time = 1 / frame_rate
        return int(frame_rate), frame_time


    def save_settings(self, filepath):
        self.path_settings_file = os.path.join(self.path_settings_file, self.name_movie)
        if not os.path.exists(self.path_settings_file):
            os.makedirs(self.path_settings_file)
        with  open(os.path.join(self.path_settings_file,self.timestamp+"_"+self.name_movie+"_settings.txt"), 'w+') as file:
            file.write("Name of frames: " + str(self.name_movie) + "\n")
            file.write("Timestamp: "+str(self.timestamp)+ "\n")
            file.write("Model path: " + str(self.model_path) + "\n")
            file.write("Overfit: " + str(self.overfit) + "\n")
            #file.write("Only people histogram reward: " + str(self.heatmap) + "\n")
            file.write("Curriculum goal: " + str(self.curriculum_goal) + "\n")
            file.write("Controller length: " + str(self.controller_len) + "\n")
            file.write("Rewards: " + str(self.reward_weights_pedestrian) + "\n")
            file.write("Initializer Rewards: " + str(self.reward_weights_initializer) + "\n")
            file.write("Initializer Car Rewards: " + str(self.reward_weights_car_initializer) + "\n")
            file.write("Car reward  : " + str(self.reward_weights_car) + "\n")

            file.write("Pop: " + str(self.pop) + "\n")
            file.write("Confined actions : " + str(self.confined_actions) + "\n")

            file.write("Velocity : " + str(self.velocity) + "\n")
            file.write("Velocity sigmoid : " + str(self.velocity_sigmoid) + "\n")
            file.write("Velocity std: " + str(self.sigma_vel) + "\n")
            file.write("LSTM: " + str(self.lstm) + "\n")
            file.write("PFNN: " + str(self.pfnn) + "\n")
            file.write("Pose: " + str(self.pose) + "\n")
            file.write("Continous: " + str(self.continous) + "\n")
            file.write("Acceleration : " + str(self.acceleration) + "\n")
            file.write("CV agent: " + str(self.cv_agent) + "\n")
            file.write("Random agent : " + str(self.random_agent) + "\n")
            file.write("Distracted agent agent : " + str(self.distracted_pedestrian) + "\n")

            file.write("reachin goal : " + str(self.goal_dir) + "\n")
            file.write("Action frequency : " + str(self.action_freq) + "\n")
            file.write("Semantic goal : " + str(self.sem_class) + "\n")
            file.write("Number of measures: " + str(NBR_MEASURES) + "\n")
            file.write("Number of measures car : " + str(NBR_MEASURES_CAR) + "\n")
            file.write("Toy case : " + str(self.toy_case) + "\n")
            file.write("CARLA : " + str(self.carla) + "\n")
            file.write("Temporal case : " + str(self.temp_case) + "\n")
            file.write("Temporal model : " + str(self.temporal) + "\n")
            file.write("Temporal prediction: " + str(self.predict_future) + "\n")
            file.write("Old conv modelling: " + str(self.old) + "\n")
            file.write("Old lstm modelling: " + str(self.old_lstm) + "\n")
            file.write("Fully connected size : " + str(self.fc) + "\n")
            file.write("Switch goal after #test steps : " + str(self.avoiding_people_step) + "\n")
            file.write("Penalty cars: " + str(self.reward_weights_pedestrian[0]) + "\n")
            file.write("Reward people: " + str(self.reward_weights_pedestrian[1]) + "\n")
            file.write("Reward pavement: " + str(self.reward_weights_pedestrian[2]) + "\n")
            file.write("Penalty objects: " + str(self.reward_weights_pedestrian[3]) + "\n")
            file.write("Reward distance travelled: " + str(self.reward_weights_pedestrian[4]) + "\n")
            file.write("Penalty out of axis: " + str(self.reward_weights_pedestrian[5]) + "\n")
            file.write("Reward distance left: " + str(self.reward_weights_pedestrian[6]) + "\n")
            file.write("Initialize on pavement: " + str(self.init_on_pavement) + "\n")
            file.write("Random seed of numpy: " + str(RANDOM_SEED_NP) + "\n")
            file.write("Random seed of tensorflow: " + str(RANDOM_SEED) + "\n")
            file.write("Timing: " + str(self.timing) + "\n")
            file.write("2D input to network: " + str(self.run_2D) + "\n")
            file.write("Semantic channels separated: " + str(self.sem) + "\n")
            file.write("Minimal semantic channels : " + str(self.min_seg) + "\n")
            file.write("People channel: " + str(self.people) + "\n")
            file.write("Cars channel: " + str(self.cars) + "\n")
            file.write("Car variable: " + str(self.car_var) + "\n")
            file.write("Network memory: " + str(self.mem) + "\n")
            file.write("Network action memory: " + str(self.action_mem) + "\n")
            if self.mem:
                file.write("Number of timesteps: " + str(self.nbr_timesteps) + "\n")
            file.write("Load init weights from file: " + self.load_weights_init + "\n")
            file.write("Load weights from file: " + self.load_weights + "\n")
            file.write("Load car weights from file: " + self.load_weights_car + "\n")
            file.write("Learning rate: " + str(self.lr) + "\n")
            file.write("Batch size: " + str(self.batch_size) + "\n")
            file.write("Agents shape: " + str(self.agent_shape) + "\n")
            file.write("Agents sight: " + str(self.agent_shape_s) + "\n")
            file.write("Environment shape: " + str(self.env_shape) + "\n")
            file.write("Controller length: " + str(self.controller_len) + "\n")
            file.write("Discount rate: " + str(self.gamma) + "\n")
            file.write("Goal distance: " + str(self.threshold_dist) + "\n")
            file.write("Sequence length train: " + str(self.seq_len_train) + "\n")
            file.write("Sequence length test: " + str(self.seq_len_test) + "\n")
            file.write("Sequence length train final: " + str(self.seq_len_train_final) + "\n")
            file.write("Sequence length test final: " + str(self.seq_len_test_final) + "\n")
            file.write("Weights update frequency: " + str(self.update_frequency) + "\n")
            file.write("Network saved with frequency: " + str(self.save_freq) + "\n")
            file.write("Test frequency: " + str(self.test_freq) + "\n")
            file.write("Deterministic: " + str(self.deterministic) + "\n")
            file.write("Deterministic testing: " + str(self.deterministic_test) + "\n")
            # if not self.deterministic:
            #     file.write("Probability: " + str(self.prob) + "\n")
            file.write("Curriculum learning reward: " + str(self.curriculum_reward) + "\n")
            file.write("Curriculum seq_len reward: " + str(self.curriculum_seq_len) + "\n")
            # file.write("Training environments: " + str(self.train_nbrs) + "\n")
            # file.write("Testing environments: " + str(self.test_nbrs) + "\n")
            #file.write("Repeat for number of iterations: " + str(self.repeat) + "\n")
            file.write("Network pooling: " + str(self.pooling) + "\n")
            file.write("Normalize batches: " + str(self.batch_normalize) + "\n")
            file.write("Clip gradients: " + str(self.clip_gradients) + "\n")
            file.write("Regularize loss: " + str(self.regularize) + "\n")
            file.write("Allow 3D actions: " + str(self.actions_3d) + "\n")
            file.write("Normalize reward: " + str(self.normalize) + "\n")
            file.write("Number of layers in network: " + str(self.num_layers) + "\n")
            file.write("Number of out filters: " + str(self.outfilters) + "\n")

            file.write("Gaussian init net: " + str(self.gaussian_init_net) + "\n")
            file.write("Std of init net : " + str(self.init_std) + "\n")

            file.write("Train RL toy car : " + str(self.useRLToyCar) + "\n")
            file.write("Train CARLA env live : " + str(self.useRealTimeEnv) + "\n")
            file.write("Car input : " + str(self.car_input) + "\n")

            file.write("Learn initialization:  " + str(self.learn_init) + "\n")
            file.write("Learn initialization:  " + str(self.learn_init_car) + "\n")
            file.write("Learn goal : " + str(self.learn_goal) + "\n")
            file.write("Learn time : " + str(self.learn_time) + "\n")
            file.write("Learn init and pedestrian "+str(self.train_init_and_pedestrian) + "\n")
            file.write("Fast debugging training : " + str(self.fastTrainEvalDebug) + "\n")
            file.write("Use Caching : " + str(self.useCaching) + "\n")
            file.write("Prior smoothing : " + str(self.prior_smoothing) + "\n")
            file.write("Prior smoothing sigma : " + str(self.prior_smoothing_sigma) + "\n")
            file.write("Use heatmap in prior : " + str(self.add_pavement_to_prior) + "\n")
            file.write("Assume known pedestrian trajectories in prior : " + str(self.assume_known_ped_stats_in_prior) + "\n")
            file.write(
                "Assume known pedestrian trajectories in reward : " + str(self.assume_known_ped_stats) + "\n")
            file.write("Ignore external pedestrians and cars : " + str(self.ignore_external_cars_and_pedestrians) + "\n")
            file.write( "Occlusion promoting prior in car init : " + str(self.car_occlusion_prior) + "\n")
            file.write("Attack random car agent : " + str(self.attack_random_car) + "\n")

            #file.write("Initialization accv architecture : " + str(self.accv_arch_init) + "\n")
            file.write("Initialization semantic architecture : " + str(self.sem_arch_init) + "\n")
            file.write("Separate Goal Net : " + str(self.separate_goal_net) + "\n")
            file.write("Gaussian Initializer : " + str(self.gaussian_init_net) + "\n")
            file.write("Gaussian Goal Net : " + str(self.goal_gaussian) + "\n")
            file.write("Initializer std : " + str(self.init_std) + "\n")
            file.write("Goal std : " + str(self.goal_std) + "\n")
            file.write("Multiplicative reward pedestrian : " + str(self.multiplicative_reward_pedestrian) + "\n")
            file.write("Multiplicative reward initializer : " + str(self.multiplicative_reward_initializer) + "\n")
            file.write("Use occlusion : " + str(self.use_occlusion) + "\n")
            file.write("Entropy parameter agent : " + str(self.entr_par) + "\n")
            file.write("Entropy parameter initializer : " + str(self.entr_par_init) + "\n")
            file.write("Entropy parameter goal : " + str(self.entr_par_goal) + "\n")

            file.write("Add noise to car input : " + str(self.add_noise_to_car_input) + "\n")
            file.write("Car noise probability : " + str(self.car_noise_probability) + "\n")
            file.write("Pedestrian view occluded : " + str(self.pedestrian_view_occluded) + "\n")
            file.write("Toy Car input : " + str(self.car_input) + "\n")
            file.write("Toy Car linear : " + str(self.linear_car) + "\n")
            file.write("Toy Car linear std : " + str(self.sigma_car) + "\n")
            file.write("Car reference speed : " + str(self.car_reference_speed) + "\n")

            file.write("New carla training set  : " + str(self.new_carla) + "\n")
            file.write("Train car and initializer : " + self.train_car_and_initialize + "\n")
            file.write("Car success rate threshold : " + str(self.car_success_rate_threshold) + "\n")
            file.write("Initializer collision rate theshold : " + str(self.initializer_collision_rate_threshold) + "\n")
            file.write("Number of iteration to train initializer : " + str(self.num_init_epochs) + "\n")
            file.write("Number of iterations to train car : " + str(self.num_car_epochs) + "\n")

