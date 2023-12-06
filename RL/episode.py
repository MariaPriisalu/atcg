import RL.settings
from RL.car_measures import collide_with_people, find_closest_car_in_list,find_closest_controllable_car
from RL.extract_tensor import frame_reconstruction,reconstruct_static_objects
from utils.utils_functions import overlap, find_border, mark_out_car_cv_trajectory, find_occlusions, is_car_on_road, find_extreme_road_pos, get_bbox_of_car, get_bbox_of_car_vel,get_heatmap,get_road

from RL.settings import STATISTICS_INDX_MAP_CAR,NBR_MAPS_GOAL, STATISTICS_INDX_MAP_GOAL, PEDESTRIAN_MEASURES_INDX, CAR_MEASURES_INDX,PEDESTRIAN_INITIALIZATION_CODE,PEDESTRIAN_REWARD_INDX, NBR_MEASURES, NBR_MEASURES_CAR,STATISTICS_INDX, STATISTICS_INDX_POSE, STATISTICS_INDX_CAR,STATISTICS_INDX_CAR_INIT, STATISTICS_INDX_MAP, RANDOM_SEED_NP, RANDOM_SEED, realTimeEnvOnline
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES, CHANNELS

from commonUtils.ReconstructionUtils import ROAD_LABELS,SIDEWALK_LABELS, OBSTACLE_LABELS_NEW, OBSTACLE_LABELS, MOVING_OBSTACLE_LABELS
from pedestrian_data_holder import PedestrianDataHolder
from car_data_holder import CarDataHolder
from initializer_data_holder import InitializerDataHolder
from initializer_car_data_holder import InitializerCarDataHolder
from car_measures import iou_sidewalk_car
if RL.settings.run_settings.realTimeEnvOnline:
    from commonUtils.RealTimeEnv.CarlaRealTimeUtils import *
    from memory_profiler import profile as profilemem
    memoryLogFP_update_pedestrians_and_cars = None
    if run_settings.memoryProfile:
        memoryLogFP_update_pedestrians_and_cars = open("report_mem_usage_update_pedestrians_and_cars.log", "w+")

    memoryLogFP_statisticssave = None
    if run_settings.memoryProfile:
        memoryLogFP_statisticssave = open("memoryLogFP_statisticssave.log", "w+")

    from commonUtils.RealTimeEnv.RealTimeEnvInteraction import *
else:
    import numpy as np
    import random
    import copy
np.random.seed( RANDOM_SEED_NP)
random.seed(RANDOM_SEED)

from scipy.ndimage import gaussian_filter
class SimpleEpisode(object):
    def __init__(self, tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights_pedestrian,reward_weights_initializer, agent_size=(2, 2, 2),
                 people_dict={}, cars_dict={}, people_vel={}, cars_vel={}, init_frames={}, adjust_first_frame=False,
                 follow_goal=False, action_reorder=False, threshold_dist=-1, init_frames_cars={}, temporal=False,
                 predict_future=False, run_2D=True, agent_init_velocity=False, velocity_actions=False, seq_len_pfnn=0,
                 end_collide_ped=False, stop_on_goal=True, waymo=False, defaultSettings=None, agent_height=4,
                 multiplicative_reward_pedestrian=False, multiplicative_reward_initializer=True,  learn_goal=False, use_occlusion=False,
                 useRealTimeEnv=None,  car_vel_dict=None, people_vel_dict=None, car_dim=None,
                 new_carla=False, lidar_occlusion=False, centering=None,use_car_agent=False,use_pfnn_agent=False,
                 people_dict_trainable=None, cars_dict_trainable=None,number_of_agents=1, number_of_car_agents=0,
                 initializer_gamma=1, prior_smoothing=False, prior_smoothing_sigma=0, occlude_some_pedestrians=False, add_pavement_to_prior=False,
                 assume_known_ped_stats=False,assume_known_ped_stats_in_prior=False, precalculated_init_data=None, learn_init_car=False, car_occlusion_prior=False): # heroCarDetails=None,


        # Environment variables
        self.DTYPE = np.float64
        self.pos = [pos_x, pos_y]
        self.centering = centering
        assert self.centering is not None
        self.reconstruction = tensor
        self.pavement = []  # [6, 7, 8, 9, 10]
        for label in SIDEWALK_LABELS:
            self.pavement.append(label)
        for label in ROAD_LABELS:
            self.pavement.append(label)
        self.reconstruction_2D = []
        self.test_positions = []
        self.heights = []


        # Static variables
        self.actions = []
        v = [-1, 0, 1]
        j = 0
        for y in range(3):
            for x in range(3):
                self.actions.append([0, v[y], v[x]])
                j += 1
        self.actions_ordered = [4, 1, 0, 3, 6, 7, 8, 5, 2]

        # Realtime variables
        self.environmentInteraction = None
        self.useRealTimeEnv = useRealTimeEnv
        self.use_car_agent = use_car_agent
        self.use_pfnn_agent = use_pfnn_agent

        # External People and cars
        self.cars = cars_e
        self.people = people_e

        self.init_frames = init_frames
        self.init_frames_car = init_frames_cars

        self.people_dict = people_dict
        self.cars_dict = cars_dict

        self.people_vel = people_vel
        self.cars_vel = cars_vel

        # Trainable cars and people in realtime runs
        self.people_dict_trainable = people_dict_trainable
        self.cars_dict_trainable = cars_dict_trainable

        # Predictions of motion of dynamic objects
        self.car_agent_predicted = None
        self.cars_predicted = []
        self.people_predicted = []

        # Mapping of pedestrian keys to small numbers
        self.key_map = []
        if self.people_dict:
            for counter, val in enumerate(self.people_dict.keys()):
                self.key_map.append(val)
                # print self.key_map

        # Valid positions
        self.valid_people = []  # valid people positions, without keys
        if len(tensor) > 0:
            self.valid_positions = np.ones(self.reconstruction.shape[1:3])
        else:
            self.valid_positions = []
        self.valid_keys = []  # people the central 75% of the scene, present in the first frame
        self.valid_people_tracks = {}  # people in the central 75% of the scene

        # Valid inits for cars
        self.init_cars = []
        self.car_dim = car_dim

        # Run specific variables
        self.temporal = temporal
        self.predict_future = predict_future
        self.run_2D = run_2D
        self.seq_len = seq_len

        self.stop_on_goal = stop_on_goal
        self.follow_goal = follow_goal
        self.threshold_dist = threshold_dist # max distance to goal
        self.gamma = gamma
        self.initializer_gamma = initializer_gamma
        self.agent_size = agent_size
        self.reward_weights_pedestrian = reward_weights_pedestrian  # cars, people, pavement, objs, dist, out_of_axis
        self.reward_weights_initializer=reward_weights_initializer
        self.end_on_hit_by_pedestrians = end_collide_ped
        self.velocity_agent = False#agent_init_velocity
        self.past_move_len=2
        self.evaluation = False  # Set goal according to evaluation?
        self.first_frame = 0

        self.pedestrian_data=[]

        for id in range(number_of_agents):
            self.pedestrian_data.append(PedestrianDataHolder(seq_len, seq_len_pfnn, use_pfnn_agent, self.DTYPE))


        self.initOtherParameters(seq_len, velocity_actions, defaultSettings,agent_height,multiplicative_reward_pedestrian,multiplicative_reward_initializer, learn_goal,use_occlusion, car_vel_dict, people_vel_dict, car_dim,new_carla, lidar_occlusion,use_car_agent,use_pfnn_agent,number_of_agents, number_of_car_agents, prior_smoothing, prior_smoothing_sigma,occlude_some_pedestrians,add_pavement_to_prior, assume_known_ped_stats,learn_init_car, assume_known_ped_stats_in_prior ,car_occlusion_prior)


        if action_reorder:
            self.actions =[self.actions[k] for k in [4,1,0,3,6,7,8,5,2]]#[8, 7, 6, 5, 0, 1, 2, 3, 4]]
            self.actions_ordered = [0,1,2,3,4,5,6,7,8]

        self.entitiesRecordedDataSource=None
        # INITIALIZATION
        if len(tensor)>0:
            self.initialization_set_up(seq_len, adjust_first_frame, precalculated_init_data)
        self.border = []
        self.border = find_border(self.valid_positions)
        for id in range(number_of_agents):
            self.initializer_data[id].valid_positions=self.valid_positions.copy()


    def initOtherParameters(self,seq_len, velocity_actions, defaultSettings, agent_height,multiplicative_reward_pedestrian,multiplicative_reward_initializer,  learn_goal,use_occlusion,  car_vel_dict, people_vel_dict, car_dim, new_carla, lidar_occlusion,use_car_agent,use_pfnn_agent,number_of_agents, number_of_car_agents, prior_smoothing, prior_smoothing_sigma,occlude_some_pedestrians, add_pavement_to_prior, assume_known_ped_stats,learn_init_car,assume_known_ped_stats_in_prior,car_occlusion_prior):

        # Constants
        self.frame_rate, self.frame_time = defaultSettings.getFrameRateAndTime()
        self.add_pavement_to_prior =add_pavement_to_prior
        self.assume_known_ped_stats =assume_known_ped_stats
        self.assume_known_ped_stats_in_prior=assume_known_ped_stats_in_prior
        if velocity_actions:
            self.max_step = 3 / self.frame_rate * 5
        else:
            self.max_step = np.sqrt(2)

        self.agent_height = agent_height
        self.new_carla = new_carla
        self.car_occlusion_prior=car_occlusion_prior
        # Run specific
        self.use_car_agent = use_car_agent
        self.use_pfnn_agent=use_pfnn_agent
        self.multiplicative_reward_pedestrian=multiplicative_reward_pedestrian
        self.multiplicative_reward_initializer=multiplicative_reward_initializer

        # Initializer variables
        self.lidar_occlusion=lidar_occlusion
        self.use_occlusions = use_occlusion
        self.learn_goal = learn_goal
        self.initializer_data=[]
        for id in range(number_of_agents):
            self.initializer_data.append(InitializerDataHolder(self.valid_positions, seq_len, self.DTYPE))


        # Predictions
        self.occlude_some_pedestrians=occlude_some_pedestrians
        self.prior_smoothing=prior_smoothing
        self.prior_smoothing_sigma=prior_smoothing_sigma
        self.car_dim = car_dim
        self.number_of_agents = number_of_agents
        self.number_of_car_agents = number_of_car_agents
        # Variables needed to keep track of car statistics. So we can have a joint overview.
        self.car_data = []
        self.initializer_car_data = []
        self.learn_init_car=learn_init_car
        if self.use_car_agent:
            for id in range(number_of_car_agents):
                self.car_data.append(CarDataHolder(seq_len, self.DTYPE))


        if self.useRealTimeEnv:
            self.people_vel_dict=[None]*seq_len
            self.car_vel_dict=[None]*seq_len

            self.car_vel_dict[0]=car_vel_dict
            self.people_vel_dict[0] = people_vel_dict

            self.people_predicted_init=np.zeros_like(self.reconstruction_2D)
            self.cars_predicted_init = np.zeros_like(self.reconstruction_2D)
            print("Initialize people predicted init "+str(self.people_predicted_init.shape))

        else:
            try:
                self.cars_predicted=self.car_agent_predicted
            except AttributeError:
                print ("car_agent_predicted does not exist in episode")


    # These two functions are used to avoid serializing in the cached version some things that we don't need.
    # Probably need to add more !
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't include states that shouldn't be saved
        if "environmentInteraction" in state:
            del state["environmentInteraction"]

        # Do not save the dictionary in this case since we are not expecting things to be deterministic on the simulation side
        # For instance, at least the actors ids assigned from environment will be different probably
        # So it makes sense only for real time but not online simulation of episodes
        if self.environmentInteraction.isOnline:
            if "entitiesRecordedDataSource" in state:
                del state["entitiesRecordedDataSource"]
        elif "entitiesRecordedDataSource" not in state:
            #print(state.keys())
            assert False, "You are not saving the offline dataset this will cost you performance !!s"

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # TODO: add the things back if needed !
        self.baz = 0


    ####################################### Post initialization set up
    def set_correct_run_settings(self, run_2D, seq_len,  stop_on_goal,follow_goal,threshold_dist, gamma,
                                 reward_weights_pedestrian,reward_weights_initializer, end_collide_ped, agent_init_velocity, waymo,action_reorder,seq_len_pfnn,
                                 velocity_actions,agent_height, evaluation=False, defaultSettings = None,
                                 multiplicative_reward_pedestrian=False,multiplicative_reward_initializer=True, learn_goal=False, use_occlusion=False,
                                 useRealTimeEnv=False,  car_vel_dict=None, people_vel_dict=None, car_dim=None, new_carla=False,
                                 lidar_occlusion=False, use_car_agent=False, use_pfnn_agent=False,number_of_agents=1,
                                 number_of_car_agents=0, initializer_gamma=1, prior_smoothing=False, prior_smoothing_sigma=0,
                                 occlude_some_pedestrians=False, add_pavement_to_prior=False, assume_known_ped_stats=False, learn_init_car=False,
                                 assume_known_ped_stats_in_prior=False, car_occlusion_prior=False):#heroCarDetails=None,
        # print("Set values")
        # print(("Seq len in episode: " + str(self.seq_len) + " seq len " + str(seq_len)))
        # Run specific variables
        self.run_2D = run_2D
        # print(("RUn on 2D "+str(self.run_2D)))
        self.stop_on_goal = stop_on_goal
        # print(("Stop on goal " + str(self.stop_on_goal)))
        self.follow_goal = follow_goal
        # print(("Follow goal  " + str(self.follow_goal)))
        self.threshold_dist = threshold_dist  # max distance to goal
        # print(("Threshold dist   " + str(self.threshold_dist)))
        self.gamma = gamma
        # print(("Gamma   " + str(self.gamma)))
        self.reward_weights_pedestrian = reward_weights_pedestrian  # cars, people, pavement, objs, dist, out_of_axis
        self.reward_weights_initializer = reward_weights_initializer  # cars, people, pavement, objs, dist, out_of_axis
        # print(("Reward weights    " + str(self.reward_weights)))
        self.end_on_hit_by_pedestrians = end_collide_ped

        # print(("end_on_hit_by_pedestrians    " + str(self.end_on_hit_by_pedestrians)))
        self.velocity_agent = agent_init_velocity
        # print(("velocity_agent    " + str(self.velocity_agent)))
        self.initializer_gamma = initializer_gamma
        self.past_move_len = 2
        self.evaluation = evaluation  # Set goal according to evaluation?
        # print(("evaluation    " + str(self.evaluation)))
        self.first_frame = 0

        self.number_of_agents = number_of_agents
        self.number_of_car_agents = number_of_car_agents


        self.useRealTimeEnv = useRealTimeEnv
        print("IN set correct run settings  "+str( self.useRealTimeEnv))

        self.initOtherParameters(seq_len, velocity_actions, defaultSettings, agent_height, multiplicative_reward_pedestrian,multiplicative_reward_initializer, learn_goal, use_occlusion,car_vel_dict, people_vel_dict,car_dim, new_carla,lidar_occlusion, use_car_agent, use_pfnn_agent,number_of_agents, number_of_car_agents, prior_smoothing, prior_smoothing_sigma,occlude_some_pedestrians,add_pavement_to_prior, assume_known_ped_stats,learn_init_car,assume_known_ped_stats_in_prior,car_occlusion_prior)
        # print(("Maximum step " + str(self.max_step)))
        if action_reorder:
            self.actions = [self.actions[k] for k in [4, 1, 0, 3, 6, 7, 8, 5, 2]]  # [8, 7, 6, 5, 0, 1, 2, 3, 4]]
            self.actions_ordered = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            # print(("Actions "+str(self.actions)))

        vector_len = max(seq_len - 1, 1)
        # print(("Seq len in episode: " + str(self.seq_len) + " seq len " + str(seq_len)))
        if self.seq_len!=seq_len:


            # print(("Seq len in episode: "+str(self.seq_len)+" seq len "+str(seq_len)))
            self.seq_len = seq_len

            for valid_key in self.valid_keys[:]:
                if valid_key in self.init_frames:
                    if self.init_frames[valid_key]> seq_len:
                        self.valid_keys.remove(valid_key)
                else:
                    print("Key "+str(valid_key)+" not in valid keys")
                    # print(("Remove "+str(valid_key)+" frame init "+str(self.init_frames[valid_key])))

            for valid_key in self.init_cars[:]:
                if valid_key in self.init_frames_car:
                    if self.init_frames_car[valid_key] > seq_len:
                        self.init_cars.remove(valid_key)
                        # print(("Remove " + str(valid_key) + " frame init " + str(self.init_frames_car[valid_key])))
                else:
                    print("Key "+str(valid_key)+" not in valid keys")
            for valid_key in list(self.valid_people_tracks.keys()):
                if valid_key in self.init_frames:
                    if self.init_frames[valid_key]> seq_len:
                        self.valid_people_tracks.pop(valid_key)
                    # print(("Remove " + str(valid_key) + " frame init " + str(self.init_frames[valid_key])))
            self.key_map = self.valid_keys
            # print(("Key map " + str(self.key_map)))
            # Variables for gathering statistics/ agent movement

        for id in range(self.number_of_agents):
            self.pedestrian_data.append(PedestrianDataHolder(seq_len, seq_len_pfnn, use_pfnn_agent, self.DTYPE))




    ######################################## Initializations

    def set_entities_recorded(self ,entitiesRecordedDataSource):
        self.entitiesRecordedDataSource=entitiesRecordedDataSource
        self.heatmap=get_heatmap(self.entitiesRecordedDataSource.people, self.reconstruction)
        self.get_car_valid_init(self.entitiesRecordedDataSource.cars_dict)
        self.add_car_init_data()
    # Set up for initialization
    # Q: save the postprocessed reconstruction, car_agent_predicted, people_predicted either on disk or RAM as an option / memory budget
    # for each episode and have a fast reload option that takes data from cache and avoid calls to frame_reconstruction, predict_cars_and_people, get_heatmap
    def initialization_set_up(self,  seq_len, adjust_first_frame,precalculated_init_data):

        # Correct frame when the episode is inirtialized
        if not self.useRealTimeEnv:
            self.correct_init_frame(adjust_first_frame, seq_len) # Only needed for saved data!

        # Get reconstruction and people and car prediction maps
        if self.useRealTimeEnv:
            self.reconstruction, self.reconstruction_2D = reconstruct_static_objects(self.reconstruction,  run_2D=self.run_2D)

            self.people_predicted=[]
            self.cars_predicted=[]
            for frame in range(self.seq_len):
                self.people_predicted.append([])
                self.cars_predicted.append([])
            self.people_predicted[0]=np.zeros(self.reconstruction_2D.shape)
            self.cars_predicted[0] = np.zeros(self.reconstruction_2D.shape)
        else:

            self.reconstruction, self.cars_predicted, self.people_predicted, self.reconstruction_2D = frame_reconstruction(
                self.reconstruction, self.cars, self.people, False, temporal=self.temporal,
                predict_future=self.predict_future, run_2D=self.run_2D)
        # print(self.reconstruction)
        #print(self.reconstruction_2D)


        # Get valid pedestrian trajectory positions
        if not self.useRealTimeEnv:
            # Find valid people if no trajectories are provided.
            self.get_valid_positions(mark_out_people=False)
            self.get_valid_traj()
            self.mark_out_people_in_valid_pos()
        else:
            self.valid_people_tracks=self.people_dict
            self.valid_keys=list(self.people_dict.keys())
            self.init_cars=list(self.cars_dict.keys())

            # Find valid people if no trajectories are provided.
            self.get_valid_positions()

        # Predict car positions
        if not self.useRealTimeEnv:
            print (" Predict cars and people")
            self.predict_cars_and_people()

        # Find sidewalk
        self.valid_pavement = self.find_sidewalk(True)

        # Get pedestrian heatmap for reward calculation
        #if not self.useRealTimeEnv:
        if precalculated_init_data==None:
            if not self.useRealTimeEnv:
                self.heatmap = get_heatmap(self.people, self.reconstruction)
                self.get_car_valid_init(self.cars_dict )

            elif self.entitiesRecordedDataSource!=None:
                self.heatmap=get_heatmap(self.entitiesRecordedDataSource.people, self.reconstruction)
                self.get_car_valid_init( self.entitiesRecordedDataSource.cars_dict)
                self.add_car_init_data()
        else:
            self.heatmap =precalculated_init_data.heatmap
            self.valid_positions_cars=precalculated_init_data.valid_positions_cars
            self.valid_directions_cars = precalculated_init_data.valid_directions_cars
            self.road=precalculated_init_data.road
            self.add_car_init_data()

        # Save valid keys
        self.key_map=self.valid_keys
        for id in range(self.number_of_agents):
            self.initializer_data[id].initBorderOcclusionPrior(self.valid_positions)


        # Set up for initialization

    def add_car_init_data(self):
        self.initializer_car_data = []
        if self.learn_init_car:
            for id in range(self.number_of_car_agents):
                self.initializer_car_data.append(
                    InitializerCarDataHolder(self.valid_positions_cars, self.valid_directions_cars, self.seq_len,
                                             self.DTYPE))

    def initialization_set_up_fast(self, seq_len, adjust_first_frame):
        print("Fast set up")
        if not self.useRealTimeEnv:
            self.correct_init_frame(adjust_first_frame, seq_len)
            self.get_valid_traj()

    def car_outside_of_bounding_box(self, car_bbox):
        for i in range(1,3):
            if min(car_bbox[i*2:i*2+2])>self.reconstruction.shape[i] or max(car_bbox[i*2:i*2+2])<0:
                return True
        return False

    def pedestrian_outside_of_bounding_box(self, pedestrian):
        for i in range(1,3):
            if min(pedestrian[i]) > self.reconstruction.shape[i] or max(pedestrian[i]) < 0:
                return True
        return False

    # To update cars and pedestrians in episode from the real time environment.
    def add_car_agents_to_reconstruction(self, observation_dict, frame):
        for car, car_values in observation_dict.items():
            if len(car_values.heroCarPos)==0:
                return
            self.car_data[car.id].car[frame] = car_values.heroCarPos

            self.car_data[car.id].car_bbox[frame] = car_values.heroCarBBox
            # print ("Car bbox pos after update  " + str( self.car_bbox[frame]))
            self.car_data[car.id].car_goal = np.array(car_values.heroCarGoal)
            if not self.car_outside_of_bounding_box(self.car_data[car.id].car_bbox[frame]):
                car_limits = np.zeros(6, dtype=np.int)
                if self.car_data[car.id].car_bbox != None and self.car_data[car.id].car_bbox[frame] != None:

                    for dim in range(len(self.car_data[car.id].car_bbox[frame])):
                        d = 0
                        if dim % 2 == 1:
                            d = 1
                        car_limits[dim] = int(round(min(max(self.car_data[car.id].car_bbox[frame][dim], 0),
                                                        self.reconstruction.shape[dim // 2] - 1) + d))
                # print (" Car bounding box " + str(
                #     [car_limits[0], car_limits[1], car_limits[2], car_limits[3], car_limits[4], car_limits[5]]))

                # To do: handle different trainable cars and pedestrians somehow

                self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3],
                car_limits[4]:car_limits[5],
                CHANNELS.cars_trajectory] = (frame + 1) * np.ones_like(
                    self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3],
                    car_limits[4]:car_limits[5],
                    CHANNELS.cars_trajectory])
                # print (" Car bounding box " + str(np.sum(
                #     self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5],
                #     5])))
                if self.run_2D:
                    self.reconstruction_2D[car_limits[2]:car_limits[3], car_limits[4]:car_limits[5],
                    CHANNELS.cars_trajectory] = (frame + 1) * np.ones_like(
                        self.reconstruction_2D[car_limits[2]:car_limits[3], car_limits[4]:car_limits[5],
                        CHANNELS.cars_trajectory])
    def update_pedestrians_and_cars(self,frame,observation_dict, people_dict,cars_dict,people_vel_dict,car_vel_dict):

        # Remove any past
        if frame==0:
            # print("Update pedestrians and car: people predicted init at frame 0 "+str(self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory]))
            self.people_predicted_init = np.zeros_like(self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory])
            self.cars_predicted_init = np.zeros_like(self.reconstruction_2D[:,:,CHANNELS.cars_trajectory])

            self.reconstruction[:, :, :, CHANNELS.pedestrian_trajectory] = np.zeros_like(self.reconstruction[:, :, :, CHANNELS.pedestrian_trajectory])
            self.reconstruction[:,:,:,CHANNELS.cars_trajectory]=np.zeros_like(self.reconstruction[:,:,:,CHANNELS.cars_trajectory])

            if self.run_2D:
                self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory] = np.zeros_like(self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory])
                self.reconstruction_2D[:,:, CHANNELS.cars_trajectory] =np.zeros_like(self.reconstruction_2D[:,:, CHANNELS.cars_trajectory] )
            self.people=[]
            self.cars=[]

            self.people_dict={}
            self.cars_dict={}

            self.people_vel_dict = [None] * self.seq_len
            self.car_vel_dict = [None] * self.seq_len

            self.car_vel_dict[0] = car_vel_dict
            self.people_vel_dict[0] = people_vel_dict

            self.cars_predicted=[]
            self.people_predicted=[]

            self.car_data = []
            if self.use_car_agent:
                for id in range(self.number_of_car_agents):
                    self.car_data.append(CarDataHolder(self.seq_len, self.DTYPE))

        if self.use_car_agent:
            if frame==0 :
                for car, car_values in observation_dict.items():
                    if len(car_values.heroCarPos) >0:
                        self.car_data[car.id].car_dir = car_values.car_init_dir
            else:

                assert frame > 0, "We were expecting to come in through the init branch above since frame is 0 !"
                # print (" Update car velocity frame "+str(frame)+" vel "+str(car_vel))
                for car, car_values in observation_dict.items():
                    # print (" Update car velocity frame "+str(frame)+" vel "+str(car_values.heroCarVel))
                    self.car_data[car.id].velocity_car[frame-1] = np.asarray(car_values.heroCarVel)
                    # print("After update " + str(self.car_data[car.id].velocity_car[frame-1]))
                    # print(self.car_data[car.id].velocity_car)
                    self.car_data[car.id].speed_car[frame-1] = np.linalg.norm(car_values.heroCarVel[1:])
                    self.car_data[car.id].action_car[frame - 1]=car_values.heroCarAction
                    self.car_data[car.id].car_angle[frame-1] = car_values.heroCarAngle
                # print(" Update episode action "+str(self.action_car[frame - 1])+" frame "+str(frame-1))

        # print(" Update frame "+str(frame)+" -----------------------------------------------------------_-----------------")
        # print ("People " + str(self.people))
        # print ("Update to " + str(people_dict))
        if len(self.people)<=frame:
            self.people.append(list(people_dict.values()))
        else:
            self.people[frame]=list(people_dict.values())
        # print ("After update  " + str(self.people[frame]))
        # print ("Update pedestrians and cars ")
        # print ("Cars " + str(self.cars))
        # print ("Update to " + str(cars_dict))
        if len(self.cars) <= frame:
            self.cars.append(list(cars_dict.values()))
        else:
            self.cars[frame]=list(cars_dict.values())

        # for car_traj in cars_dict.values():
        #     if self.init_cars[]
        #self.cars[frame]=#list(cars_dict.values())
        # print ("After update  " + str(self.cars[frame]))
        if self.use_car_agent:
            self.add_car_agents_to_reconstruction(observation_dict, frame)

                # print ("Car bbox pos after update  " + str(self.car_bbox[frame]))


        # print ("Car goal after update  " + str(self.car_goal))

        self.people_vel_dict[frame]=people_vel_dict
        # print ("People velocities update  " + str(self.people_vel_dict[frame]))
        # print("Car vel dict farme: "+str(frame)+" ------------------------------------------------------------------------------------")
        # print(car_vel_dict)
        self.car_vel_dict[frame]=car_vel_dict
        # print ("Cars velocities update  " + str(self.car_vel_dict[frame]))

        # print ("Cars dict before update  " + str(self.cars_dict))
        # print ("Cars Reconstruction before update  " + str(np.sum(self.reconstruction[:,:,:, 5])))
        # print ("Cars Reconstruction 2D before update  " + str(np.sum(self.reconstruction_2D[ :, :, 5])))
        #print ("Cars Init frames car before update " + str(self.init_frames_car))
        for car_key in cars_dict.keys():
            car_a = cars_dict[car_key]
            if not self.car_outside_of_bounding_box(car_a):
                if car_key not in self.cars_dict or frame==0:
                    self.cars_dict[car_key]=[]
                    self.init_frames_car[car_key]=frame

                if len(self.cars_dict[car_key])<=frame:
                    self.cars_dict[car_key].append(car_a)
                    # print("Append car "+str(car_a)+" key "+str(car_key))

                car_limits=np.zeros(6, dtype=np.int)
                for dim in range(len(car_a)):
                    car_limits[dim] = int(round(min(max(car_a[dim], 0), self.reconstruction.shape[dim // 2] )))

                # print (" Car bounding box "+str([car_limits[0],car_limits[1], car_limits[2],car_limits[3], car_limits[4],car_limits[5]]))
                self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory] = (frame+1) * np.ones_like(
                    self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory])
                # print (" Car bounding box " + str(np.sum(self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], 5])))
                if self.run_2D:
                    self.reconstruction_2D[car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory] = (frame+1) * np.ones_like(
                        self.reconstruction_2D[car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory])

        # print ("Cars Reconstruction after update  " + str(np.sum(self.reconstruction[:, :, :, 5])))
        # print ("Cars Reconstruction 2D after update  " + str(np.sum(self.reconstruction_2D[:, :, 5])))
        # #print ("Cars Init frames car after update "+str(self.init_frames_car))
        #
        # print ("People dict before update  " + str(self.people_dict))
        # print ("People Reconstruction before update  " + str(np.sum(self.reconstruction[:, :, :, 4])))
        # print ("People Reconstruction 2D before update  " + str(np.sum(self.reconstruction_2D[:, :, 4])))
        #print ("People Init frames before update " + str(self.init_frames))
        for ped_key in people_dict.keys():
            x_pers = people_dict[ped_key]
            if not self.pedestrian_outside_of_bounding_box(x_pers):
                if ped_key not in self.people_dict or frame==0:
                    self.people_dict[ped_key] = []
                    self.init_frames[ped_key] = frame

                if len(self.people_dict[ped_key])<=frame:
                    self.people_dict[ped_key].append(x_pers)

                self.add_pedestrian_to_reconstruction(frame, x_pers)

        # print ("People dict after update  " + str(self.people_dict))
        # print ("People Reconstruction after update  " + str(np.sum(self.reconstruction[:, :, :, 4])))
        # print ("People Reconstruction 2D after update  " + str(np.sum(self.reconstruction_2D[:, :, 4])))
        #print ("People Init frames after update " + str(self.init_frames))

        if frame >= len(self.cars_predicted):
            self.cars_predicted.append(copy.copy(self.reconstruction_2D[:, :, CHANNELS.cars_trajectory]))#copy.copy()

        else:
            self.cars_predicted[frame]= copy.copy(self.reconstruction_2D[:, :, CHANNELS.cars_trajectory])
        # print ("Car predicted len " + str(len(self.cars_predicted)) + " sum " + str(np.sum(self.cars_predicted[frame])))
        # mask = self.cars_predicted[frame] > 0
        # self.cars_predicted[frame][mask] = self.cars_predicted[frame][mask] - (frame + 1)

        if frame >= len(self.people_predicted):

            self.people_predicted.append(copy.copy(self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory]))#copy.copy()
        else:
            self.people_predicted[frame] = copy.copy(self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory])
        # mask=self.people_predicted[frame]>0
        # self.people_predicted[frame][mask]=self.people_predicted[frame][mask]-(frame+1)


        # print ("People predicted len " + str(len(self.people_predicted)) + " sum " + str(np.sum(self.people_predicted[frame])))
        if self.use_car_agent:
            self.update_car_agent_measures(frame, observation_dict)

        if frame==0 :
            self.predict_cars_and_people()
            self.remove_cars_from_valid_pos()
            # if not self.learn_init_car:
            #     self.predict_car_agents()
        # print (" People predicted init size "+str(self.people_predicted_init.shape)+" and sum "+str(np.sum(self.people_predicted_init)))
        # print (" Cars predicted init size " + str(self.cars_predicted_init.shape)+ " and sum "+str(np.sum(self.cars_predicted_init)))

    def remove_cars_from_valid_pos(self):
        for car in self.cars[0]:
            dir = self.valid_directions_cars[int(np.mean(car[2:4])), int(np.mean(car[4:6])), :]

            if abs(dir[0]) > abs(dir[1]):
                x_lim = max(self.car_dim[1:]) + 2
                y_lim = min(self.car_dim[1:]) + 2
            else:
                x_lim = min(self.car_dim[1:]) + 2
                y_lim = max(self.car_dim[1:]) + 2
            carlimits = [max(int(car[2] - x_lim), 0), max(int(car[3] + x_lim), 0), max(int(car[4] - y_lim), 0),
                         max(int(car[5] + y_lim), 0)]
            carlimits = [min(carlimits[0], self.reconstruction.shape[1]),
                         min(carlimits[1], self.reconstruction.shape[1]),
                         min(carlimits[2], self.reconstruction.shape[2]),
                         min(carlimits[3], self.reconstruction.shape[2])]
            self.valid_positions_cars[carlimits[0]:carlimits[1], carlimits[2]:carlimits[3]] = np.zeros_like(
                self.valid_positions_cars[carlimits[0]:carlimits[1], carlimits[2]:carlimits[3]])

    def add_pedestrian_to_reconstruction(self, frame, x_pers):
        person = np.zeros(6, dtype=np.int)
        for i in range(x_pers.shape[0]):
            for j in range(x_pers.shape[1]):
                # print("i "+str(i)+" j "+str(j)+" shape 0 "+str(x_pers.shape[1])+" dim "+str(i*x_pers.shape[1]+j))
                d = 0
                if j == 1:
                    d = 1
                person[i * x_pers.shape[1] + j] = int(
                    round(min(max(x_pers[i][j], 0), self.reconstruction.shape[i] - 1)) + d)
        # print (" person "+str(person))
        if self.run_2D:
            self.reconstruction_2D[person[2]:person[3], person[4]:person[5], CHANNELS.pedestrian_trajectory] = (
                                                                                                                           frame + 1) * np.ones_like(
                self.reconstruction_2D[person[2]:person[3], person[4]:person[5], CHANNELS.pedestrian_trajectory])
        self.reconstruction[person[0]:person[1], person[2]:person[3], person[4]:person[5],
        CHANNELS.pedestrian_trajectory] = (frame + 1) * np.ones_like(
            self.reconstruction[person[0]:person[1], person[2]:person[3], person[4]:person[5],
            CHANNELS.pedestrian_trajectory])

    def update_car_agent_measures(self, frame, observation_dict):
        for car, car_values in observation_dict.items():
            if len(car_values.heroCarPos) == 0:
                return
            # print ("People predicted shape in episode update " + str(self.people_predicted[-1].shape)+" pos "+str(len(self.people_predicted)-1))
            self.car_data[car.id].measures_car[max(frame - 1, 0), :] = car_values.measures
            # print ("Car measures after update  " + str(car_data[car.id].measures_car[max(frame-1,0),:]))
            # print ("Episode Saved car is distracted "+str(car_data[car.id].measures_car[max(frame-1,0),CAR_MEASURES_INDX.distracted]))
            self.car_data[car.id].reward_car[max(frame - 1, 0)] = car_values.reward
            # print ("Car reward after update  " + str(np.sum(car_data[car.id].reward_car[max(frame-1,0)]))+" reward "+str(car_reward)+" frame "+str(max(frame-1,0)))
            self.car_data[car.id].probabilities_car[max(frame - 1, 0), :] = car_values.probabilities
            # print ("Car probabilities after update  " + str(np.sum(self.probabilities_car[max(frame-1,0), :])))

    # def predict_car_agents(self):
    #     if self.use_car_agent:
    #         for id in range(len(self.car_data)):
    #             self.cars_predicted_init = mark_out_car_cv_trajectory(self.cars_predicted_init,
    #                                                                   self.car_data[id].car_bbox[0],
    #                                                                   self.car_data[id].car_goal,
    #                                                                   self.car_data[id].car_dir, self.agent_size,
    #                                                                   time=1)

    # Find cars and pedestrians that are at valid locations. i.e. can be used ofr initialization.
    def get_valid_traj(self):
        heights=[]
        print ("Valid trajectories")
        if len(self.people_dict) == 0 and len(self.cars_dict) == 0:
            print ("No dictionary! " )

            for sublist in self.people:
                for item in sublist:
                    person_middle = np.mean(item, axis=1).astype(int)

                    if person_middle[1] >= 0 and person_middle[0] >= 0 and person_middle[1] < \
                            self.valid_positions.shape[0] and person_middle[2] < self.valid_positions.shape[
                        1] and self.valid_positions[person_middle[1], person_middle[2]] == 1:
                        self.valid_people.append(item)
                        heights.append(person_middle[0])

        else:
            # Find valid people tracks.- People who are present during frame 0.

            for key in list(self.people_dict.keys()):
                fine_people = []
                valid = False

                # Pedestrian present before first frame.
                if len(self.people_dict[key])>0:
                    for j, person in enumerate(self.people_dict[key], 0):

                        person_middle = np.mean(person, axis=1).astype(int)

                        if person_middle[1] >= 0 and person_middle[1] < self.reconstruction.shape[1] and person_middle[
                            2] >= 0 and person_middle[2] < self.reconstruction.shape[2] and self.valid_positions[
                            person_middle[1], person_middle[2]] == 1:
                            fine_people.append(person)

                            if j == 0 and self.init_frames[key] == 0:
                                valid = True


                            heights.append(person_middle[0])

                    if len(fine_people) > 1 or self.useRealTimeEnv:# To DO: Why chichking her efor useRealTimeEnv?
                        self.valid_people_tracks[key] = list(fine_people)

                        if valid:
                            self.valid_keys.append(key)
            # print ("Cars dictionary "+str(len(self.cars_dict)))
            for key in list(self.cars_dict.keys()):
                # Pedestrian present before first frame.
                # print ("First frame  "+str(self.first_frame))
                if len(self.cars_dict[key])>0:
                    if self.init_frames_car[key]==0:
                        car=self.cars_dict[key][0]
                        if self.useRealTimeEnv:
                            self.init_cars.append(key)
                        elif len(self.cars_dict[key]) > 10:
                            car = self.cars_dict[key][10]
                            if car[3] >= 0 and car[2] < self.reconstruction.shape[1] and car[5] >= 0 and car[4] < self.reconstruction.shape[2]:
                                self.init_cars.append(key)

        self.heights=heights


    def find_time_in_boundingbox(self, pos_init_average, vel, bounding_box):
        t_crossing=[]

        for i in range(2):
            if abs(vel[i]) > 0:
                for j in range(2):
                    #print ("Calculate t "+str(bounding_box[i][j]-pos_init_average[i])+" divided by "+str( vel[i]))
                    t=(bounding_box[i][j]-pos_init_average[i])/ vel[i]
                    second_coordinate=(t*vel[i-1])+pos_init_average[i-1]
                    #print (" Time "+str(t)+" i "+str(i)+" j "+str(j)+" second_coordinate "+str(second_coordinate)+" compare to "+str(bounding_box[i-1][0])+" and "+str(bounding_box[i-1][1]))
                    if t>=0 and bounding_box[i-1][0]<=second_coordinate and second_coordinate<=bounding_box[i-1][1]:
                        t_crossing.append(copy.copy(t))
        return t_crossing

    def person_bounding_box(self, pos,no_height=False, channel=3, bbox=[]):

        # print "Position "+str(pos)+ " "+str(x_range)+" "+str(y_range)+" "+str(z_range)
        if len(bbox)>0:
            x_range=bbox[0:2]
            y_range=bbox[2:4]
            z_range=bbox[4:]
        else:
            x_range, y_range, z_range = self.pos_range(pos, as_int=True)
        if no_height:
            x_range = [0, self.reconstruction.shape[0]]
        segmentation = (
            self.reconstruction[int(x_range[0]):int(x_range[1]) + 1, int(y_range[0]):int(y_range[1]) + 1,
            int(z_range[0]):int(z_range[1]) + 1, channel] * int(NUM_SEM_CLASSES)).astype(
            np.int32)

        return segmentation


    def predict_cars_and_people_in_a_bounding_box(self, id, prediction_people,prediction_car,bounding_box_of_prediction, start_pos,frame):

        if len(bounding_box_of_prediction) == 0:
            bounding_box_of_prediction = [[0, self.reconstruction.shape[1]], [0, self.reconstruction.shape[2]]]
            frame = self.seq_len
        bounding_box_reshape=[[bounding_box_of_prediction[0],bounding_box_of_prediction[1]],[bounding_box_of_prediction[2],bounding_box_of_prediction[3]]]
        # print ("Predict cars and people frame "+str(frame)+" bbox "+str(bounding_box_of_prediction))
        # print ("Number of People "+str(len(self.people_vel_dict[frame].keys()))+" cars "+str(len(self.car_vel_dict[frame].keys())))
        for key, track in list(self.people_dict.items()):  # Go through all pedestrians
            frame_init = self.init_frames[key]
            previous_pos = np.array([0, 0, 0])
            # print(" Person "+str(key)+" init frame "+str(frame_init)+" frame "+str(frame))
            if frame_init<=frame: # Find position of pedestrian in current frame
                frame_diff=frame-frame_init
                #print(" len " + str(len(self.people_dict[key])))
                if len(self.people_dict[key])>frame_diff:
                    pos_init=self.people_dict[key][frame_diff].copy()
                    pos_init_average = np.array([np.mean(pos_init[0]), np.mean(pos_init[1]), np.mean(pos_init[2])])
                    pos_init[0,1]=pos_init[0,1]+1
                    pos_init[1, 1] = pos_init[1, 1] + 1
                    pos_init[2, 1] = pos_init[2, 1] + 1
                    # in_reconstruction_dim1 = np.max(pos_init[1]) >= 0 and np.min(pos_init[1]) < \
                    #                          self.reconstruction_2D.shape[0]
                    # in_reconstruction_dim2 = np.max(pos_init[2]) >= 0 and np.min(pos_init[2]) < \
                    #                          self.reconstruction_2D.shape[1]

                    in_reconstruction = True#in_reconstruction_dim1 and in_reconstruction_dim2
                    #in_reconstruction=(pos_init_average[1]>=0 and pos_init_average[1]<self.reconstruction_2D.shape[0] ) and (pos_init_average[2]>=0 and pos_init_average[2]<self.reconstruction_2D.shape[1] )
                    if in_reconstruction and key in self.people_vel_dict[frame].keys():
                        vel= self.people_vel_dict[frame][key]
                        # print (" person "+str(key)+" frame "+str(frame))
                        # print(" Average pos "+str(pos_init_average)+" vel "+str(vel))
                        # Find out if pedestrian's tracks come into the bounding box

                        self.add_predicted_car_to_matrix(bounding_box_reshape, pos_init, pos_init_average,
                                                         prediction_people,
                                                         vel,start_pos)
        self.add_predicted_people(id, frame, prediction_people)
        # print (" After prediction people " + str(np.sum(prediction_people)))
        # print("Before prediction--car : "+str(np.sum(prediction_car)))
        for key, track in list(self.cars_dict.items()):  # Go through all pedestrians
            frame_init = self.init_frames_car[key]
            # print(" Car " + str(key) + " init frame " + str(frame_init) + " frame " + str(frame))
            if frame_init <= frame:  # Find position of pedestrian in current frame
                frame_diff = frame - frame_init
                if len(self.cars_dict[key]) > frame_diff:
                    pos_init_car = self.cars_dict[key][frame_diff]
                    pos_init_average = np.array([np.mean(pos_init_car[:2]), np.mean(pos_init_car[2:4]), np.mean(pos_init_car[4:])])
                    #
                    # in_reconstruction_dim1 = np.max(pos_init_car[2:4]) >= 0 and np.min(pos_init_car[2:4]) <self.reconstruction_2D.shape[0]
                    # in_reconstruction_dim2 = np.max(pos_init_car[4:]) >= 0 and np.min(pos_init_car[4:]) <self.reconstruction_2D.shape[1]


                    in_reconstruction=True#in_reconstruction_dim1 and in_reconstruction_dim2

                    if in_reconstruction and key in self.car_vel_dict[frame].keys():
                        vel = self.car_vel_dict[frame][key]
                        # print (" car " + str(key)+" frame "+str(frame))
                        # print(" Average pos " + str(pos_init_average) + " vel " + str(vel))
                        # Find out if pedestrian's tracks come into the bounding box
                        self.add_predicted_car_to_matrix(bounding_box_reshape, np.reshape(pos_init_car,(3,2)), pos_init_average, prediction_car,
                                                         vel,start_pos)
        # print(" After prediction cars " + str(np.sum(prediction_car)))
        if self.use_car_agent:
            for car in self.car_data:
                pos_init_car = car.car_bbox[frame]

                pos_init_average = np.array([np.mean(pos_init_car[:2]), np.mean(pos_init_car[2:4]),np.mean(pos_init_car[4:])]) if pos_init_car else None
                if frame==0:
                    vel=car.car_dir
                else:
                    vel = car.velocity_car[frame-1]
                # print(" Add car to prediction " + str(pos_init) + " velocity " + str(vel))
                if not self.car_outside_of_bounding_box(car.car_bbox[frame]):
                    self.add_predicted_car_to_matrix(bounding_box_reshape, np.reshape(pos_init_car,(3,2)), pos_init_average, prediction_car,vel,start_pos)

        for ped_id,pedestrian in enumerate(self.pedestrian_data):
            if id!=ped_id:
                frame_local=frame
                pos_ped, bbox = self.get_pedestrian_bbox(frame_local, pedestrian)
                intercept=self.pedestrian_in_bounding_box(bounding_box_reshape, bbox)
                frame_local = frame_local - 1
                while frame_local>0 and len(intercept)>0:
                    prediction_people[intercept[0]-bounding_box_reshape[0][0]:intercept[1]-bounding_box_reshape[0][0], intercept[2]-bounding_box_reshape[1][0]:intercept[3]-bounding_box_reshape[1][0]]=(frame_local+1)*np.ones((intercept[1]-intercept[0],intercept[3]-intercept[2]))
                    pos_ped, bbox = self.get_pedestrian_bbox(frame_local, pedestrian)
                    intercept = self.pedestrian_in_bounding_box(bounding_box_reshape, bbox)
                    frame_local = frame_local - 1


                pos_init_average = np.array(pedestrian.agent[frame])
                pos_init_ped = [pos_init_average[0]-self.agent_size[0], pos_init_average[0]+self.agent_size[0]+1,pos_init_average[1]-self.agent_size[1],pos_init_average[1]+self.agent_size[1]+1,pos_init_average[2]-self.agent_size[2],pos_init_average[2]+self.agent_size[2]+1]
                if frame == 0:
                    vel = pedestrian.vel_init
                else:
                    vel = pedestrian.velocity[frame-1]
                # print(" Add car to prediction " + str(pos_init) + " velocity " + str(vel))
                if not self.car_outside_of_bounding_box(pos_init_ped):
                    self.add_predicted_car_to_matrix(bounding_box_reshape, np.reshape(pos_init_ped, (3, 2)), pos_init_average,
                                                     prediction_people, vel, start_pos)
        # print ("After prediction cars " + str(np.sum(prediction_car)))

        #self.agent_prediction_car.append(prediction_car)
        # print ("Added to predicted people list " + str(len(self.agent_prediction_people))+" frame "+str(frame))
        return prediction_people, prediction_car

    def pedestrian_in_bounding_box(self,bounding_box_reshape, bbox):
        intercept=[max(bounding_box_reshape[0][0], bbox[0]), min(bounding_box_reshape[0][1], bbox[1]),max(bounding_box_reshape[1][1], bbox[2]), min(bounding_box_reshape[1][1], bbox[3])]

        if  intercept[0]<intercept[1] and  intercept[2]<intercept[3]:
            return intercept

        return []

    def get_pedestrian_bbox(self, frame, pedestrian):
        pos_ped = np.array(pedestrian.agent[frame])
        bbox_ped = [pos_ped[1] - self.agent_size[1], pos_ped[1] + self.agent_size[1] + 1,
                    pos_ped[2] - self.agent_size[2], pos_ped[2] + self.agent_size[2] + 1]
        return pos_ped, bbox_ped

    def add_predicted_people(self, id, frame, prediction_people):
        if len(self.pedestrian_data[id].agent_prediction_people) <= frame:
            self.pedestrian_data[id].agent_prediction_people.append(prediction_people.copy())
        else:
            self.pedestrian_data[id].agent_prediction_people[frame] = prediction_people.copy()

    def add_predicted_car_to_matrix(self, bounding_box_of_prediction, pos_init, pos_init_average, prediction_car, vel, start_pos):
        t_crossing = self.find_time_in_boundingbox(pos_init_average[1:], vel[1:], bounding_box_of_prediction)

        if len(t_crossing)==1:
            t_crossing.append(0)
        # print (" Time in bounding box " + str(t_crossing))
        if len(t_crossing) == 2:  # if yes then we can predict
            t_min = int(min(t_crossing))

            t_max = int(max(t_crossing))#, self.seq_len+10])
            if np.linalg.norm(vel[1:])< 0.01:
                t_max=t_min+1
                # print("Velocity is too small "+str(np.linalg.norm(vel[1:]))+" set tmax to "+str(t_max))

            t_diff=1
            if max(abs(vel[1:]))<1:
                t_diff=int(1/max(abs(vel[1:])))
                # print (" Adapt t dif "+str(t_diff))

            # print (" t_min "+str(t_min)+" t max "+str(t_max)+" location "+str(np.mean(pos_init, axis=1))+" final "+str(np.mean(pos_init + np.tile(t_max * vel, [2, 1]).T, axis=1))+" tdiff "+str(t_diff)+" vel "+str(vel))
            for t in range(t_max, t_min, -t_diff):
                location = pos_init + np.tile(t * vel, [2, 1]).T
                average_location = pos_init_average + t * vel
                # print (" Average location "+str(average_location)+" location "+str(location[1:])+" bounding box "+str(bounding_box_of_prediction))
                dir = [start_pos[0]+max(int(location[0][0]), 0),
                       start_pos[0] +min(int(location[0][1]) + 1, self.reconstruction.shape[0]),
                       start_pos[1] +max(int(location[1][0])-bounding_box_of_prediction[0][0], 0),
                       start_pos[1] +min(int(location[1][1])-bounding_box_of_prediction[0][0], prediction_car.shape[0]),
                       start_pos[2] +max(int(location[2][0])-bounding_box_of_prediction[1][0], 0),
                       start_pos[2] +min(int(location[2][1])-bounding_box_of_prediction[1][0], prediction_car.shape[1])]
                # print (" Before change " + str(np.sum(prediction_car[dir[2]:dir[3],
                #                                      dir[4]:dir[5]])))
                prediction_car[dir[2]:dir[3],
                dir[4]:dir[5]] = (t+1) * np.ones_like(prediction_car[dir[2]:dir[3], dir[4]:dir[5]])
                # print (" After change " + str(np.sum(prediction_car[dir[2]:dir[3],
                # dir[4]:dir[5]])) +" shape of input "+str(prediction_car.shape)+"  dims "+str([dir[2],dir[3], dir[4],dir[5]])+" "+str(t))
    def predict_pedestrians(self, people_predicted_init, frame=0):
        if self.predict_future:
            for key, track in list(self.people_dict.items()):  # Go through all pedestrians
                frame_init = self.init_frames[key]
                track_frame=frame-frame_init
                if len(track)>=track_frame:
                    pos = track[[track_frame]]
                    if not self.pedestrian_outside_of_bounding_box(pos):
                        vel = self.people_vel_dict[frame][track_frame]
                        average_location=  np.array([np.mean(pos[0]), np.mean(pos[1]), np.mean(pos[2])])
                        if self.useRealTimeEnv and self.temporal and self.run_2D and np.linalg.norm(vel[1:]) > 0.1:
                            loc_frame = 2
                            # Do linear predictions
                            while np.all(average_location[1:] > 1) and np.all(
                                    self.reconstruction.shape[1:3] - average_location[1:] > 1):
                                location, average_location, dir = self.get_next_pedestrian_location(average_location, location,
                                                                                          vel)

                                people_predicted_init[dir[2]:dir[3],
                                dir[4]:dir[5]] = loc_frame * np.ones_like(
                                    people_predicted_init[dir[2]:dir[3], dir[4]:dir[5]])

                                loc_frame = loc_frame + 1
        return people_predicted_init

    def predict_cars(self, people_predicted_init, frame=0):
        if self.predict_future:
            for key, track in list(self.cars_dict.items()):  # Go through all pedestrians
                frame_init = self.init_frames_car[key]
                track_frame=frame-frame_init
                if len(track)>=track_frame:
                    pos = track[[track_frame]]
                    if not self.car_outside_of_bounding_box(pos):
                        vel = self.car_vel_dict[frame][track_frame]
                        average_location=  np.array([np.mean(pos[0:2]), np.mean(pos[2:4]), np.mean(pos[4:])])
                        if self.useRealTimeEnv and self.temporal and self.run_2D and np.linalg.norm(vel[1:]) > 0.1:
                            loc_frame = 2
                            # Do linear predictions
                            while np.all(average_location[1:] > 1) and np.all(self.reconstruction.shape[1:3] - average_location[1:] > 1):
                                location, average_location, dir = self.get_next_car_location(average_location, location,vel)
                                people_predicted_init[dir[2]:dir[3],
                                dir[4]:dir[5]] = loc_frame * np.ones_like(
                                    people_predicted_init[dir[2]:dir[3], dir[4]:dir[5]])

                                loc_frame = loc_frame + 1
        return people_predicted_init

    def predict_controllable_pedestrian(self, id, people_predicted_init, frame=0, add_all=False):
        if self.predict_future and id>0:
            max_id=id
            if add_all:
                max_id=self.number_of_agents
            for ped_id in range(max_id):  # Go through all pedestrians
                if ped_id!=id:
                    if frame==0:
                        vel=np.array(self.pedestrian_data[ped_id].vel_init)
                    else:
                        vel = np.array(self.pedestrian_data[ped_id].velocity[frame-1])

                    average_location = self.pedestrian_data[ped_id].agent[frame]
                    pos = np.array([[average_location[0]-self.agent_size[0], average_location[0]+self.agent_size[0]+1],[average_location[1]-self.agent_size[1], average_location[1]+self.agent_size[1]+1],[average_location[2]-self.agent_size[2], average_location[2]+self.agent_size[2]+1]])
                    location = pos
                    # print (" Init frame " + str(frame)+ " average location "+str(average_location))
                    if self.useRealTimeEnv and self.temporal and self.run_2D and np.linalg.norm(vel[1:]) > 0.1:
                        loc_frame = 2
                        # Do linear predictions
                        while np.all(average_location[1:] > 1) and np.all(
                                self.reconstruction.shape[1:3] - average_location[1:] > 1):
                            location, average_location, dir = self.get_next_pedestrian_location(average_location, location,
                                                                                      vel)

                            people_predicted_init[dir[2]:dir[3],
                            dir[4]:dir[5]] = loc_frame * np.ones_like(
                                people_predicted_init[dir[2]:dir[3], dir[4]:dir[5]])

                            loc_frame = loc_frame + 1
        return people_predicted_init

    def predict_controllable_car(self, id, cars_predicted_init, frame=0):
        if self.predict_future and id > 0:
            for car_id in range(id):  # Go through all pedestrians

                if frame==0:
                    vel = np.array(self.car_data[car_id].car_dir)
                else:
                    vel = np.array(self.car_data[car_id].car_velocity[frame-1])
                pos = np.array(self.car_data[car_id].car_bbox[frame])
                location = pos
                average_location = self.car_data[car_id].car[frame]
                # print (" Init frame " + str(frame)+ " average location "+str(average_location))
                if self.useRealTimeEnv and self.temporal and self.run_2D and np.linalg.norm(vel[1:]) > 0.1:

                    loc_frame = 2
                    # Do linear predictions
                    while np.all(average_location[1:] > 1) and np.all(
                            self.reconstruction.shape[1:3] - average_location[1:] > 1):
                        location, average_location, dir = self.get_next_car_location(average_location, location,vel)

                        cars_predicted_init[dir[2]:dir[3],
                        dir[4]:dir[5]] = loc_frame * np.ones_like(
                            cars_predicted_init[dir[2]:dir[3], dir[4]:dir[5]])

                        loc_frame = loc_frame + 1
        return cars_predicted_init

    def predict_cars_and_people(self, bounding_box_of_prediction=[], frame=-1):
        if self.predict_future:

            for key, track in list(self.people_dict.items()): # Go through all pedestrians
                if not self.pedestrian_outside_of_bounding_box(track[0]):
                    frame_init = self.init_frames[key]
                    previous_pos = np.array([0, 0, 0])
                    for index, pos in enumerate(track): # Go through all frames of pedestrian
                        frame = frame_init + index

                        location = pos
                        average_location = np.array([np.mean(pos[0]), np.mean(pos[1]), np.mean(pos[2])])
                        #print (" Init frame " + str(frame)+ " average location "+str(average_location))
                        if self.useRealTimeEnv or index > 0:
                            if self.useRealTimeEnv:

                                #if self.people_vel_dict[frame]!=None:
                                vel= self.people_vel_dict[frame][key]
                                #print ("person vel "+str(vel))
                            else:
                                vel = average_location - previous_pos
                                vel[0] = 0
                            if np.linalg.norm(vel[1:]) > 0.1:
                                loc_frame = 1
                                if self.useRealTimeEnv:
                                    loc_frame=2
                                #Do linear predictions
                                while np.all(average_location[1:] > 1) and np.all(
                                                        self.reconstruction.shape[1:3] - average_location[1:] > 1):
                                    location, average_location, dir = self.get_next_pedestrian_location(average_location, location,
                                                                                              vel)
                                    #print (" Bounding box "+str(dir)+ " size "+str(dir[3]-dir[2])+" "+str(dir[5]-dir[4]))
                                    if self.temporal:# Fill with frame number
                                        if self.run_2D:
                                            if self.useRealTimeEnv:
                                                # print(" Shape "+str(self.people_predicted_init.shape)+" bbox "+str([dir[2],dir[3],
                                                # dir[4],dir[5]]))
                                                self.people_predicted_init[dir[2]:dir[3],
                                                dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                    self.people_predicted_init[dir[2]:dir[3], dir[4]:dir[5]])
                                            else:
                                                self.people_predicted[frame][dir[2]:dir[3],
                                                dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                    self.people_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]])
                                        else:
                                            self.people_predicted[frame][dir[0]:dir[1], dir[2]:dir[3],
                                            dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                self.people_predicted[frame][dir[0]:dir[1], dir[2]:dir[3],
                                                dir[4]:dir[5]])
                                    else:
                                        if self.run_2D:  # Fill with 0.1s
                                            self.people_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]] = \
                                                self.people_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]] + 0.1
                                        else:
                                            self.people_predicted[frame][dir[0]:dir[1], dir[2]:dir[3], dir[4]:dir[5]] = \
                                                self.people_predicted[frame][dir[0]:dir[1], dir[2]:dir[3],
                                                dir[4]:dir[5]] + 0.1
                                    # print (" Bounding box " + str(dir)+" "+str(np.mean(self.people_predicted[frame][dir[2]:dir[3],
                                    #         dir[4]:dir[5]]))+" init sum "+str(np.mean(self.people_predicted_init[dir[2]:dir[3],
                                    #         dir[4]:dir[5]])))
                                    loc_frame = loc_frame + 1
                        previous_pos = np.array([np.mean(pos[0]), np.mean(pos[1]), np.mean(pos[2])])
                    # print ("After  --------------------------------------------")
                    # for frame, pred in enumerate(self.people_predicted):
                    #      print (str(frame))
                    #      print (np.sum(pred))
            #print (" Cars dict "+str(self.cars_dict))
            for key, track in list(self.cars_dict.items()):
                frame_init = self.init_frames_car[key]
                previous_pos = np.array([0, 0, 0])
                if not self.car_outside_of_bounding_box(track[0]):
                    for index, pos in enumerate(track):
                        frame = frame_init + index
                        location = pos
                        average_location = np.array(
                            [np.mean([pos[0], pos[1]]), np.mean([pos[2], pos[3]]), np.mean([pos[4], pos[5]])])
                        #print (" Init frame " + str(frame) + " average location " + str(average_location))
                        if  self.useRealTimeEnv or index > 0:
                            if self.useRealTimeEnv:
                                vel= self.car_vel_dict[frame][key]
                                #print ("car vel " + str(vel))
                            else:
                                vel = average_location - previous_pos
                                vel[0] = 0
                            if np.linalg.norm(vel[1:]) > 0.1:
                                loc_frame = 1
                                if self.useRealTimeEnv:
                                    loc_frame = 2
                                while np.all(average_location[1:] > 1) and np.all(
                                                        self.reconstruction.shape[1:3] - average_location[1:] > 1):
                                    location, average_location, dir = self.get_next_car_location(average_location, location, vel)
                                    #print (" Bounding box " + str(dir) + " size " + str(dir[3] - dir[2]) + " " + str(
                                    #    dir[5] - dir[4]))
                                    if self.temporal:
                                        if self.run_2D:
                                            if self.useRealTimeEnv:
                                                self.cars_predicted_init[dir[2]:dir[3],
                                                dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                    self.cars_predicted_init[dir[2]:dir[3], dir[4]:dir[5]])
                                                loc_frame = loc_frame + 1
                                            else:
                                                self.cars_predicted[frame][dir[2]:dir[3],
                                                dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                    self.cars_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]])
                                                loc_frame = loc_frame + 1
                                        else:
                                            self.cars_predicted[frame][dir[0]:dir[1], dir[2]:dir[3],
                                            dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                self.cars_predicted[frame][dir[0]:dir[1], dir[2]:dir[3], dir[4]:dir[5]])
                                    else:
                                        if self.run_2D:
                                            self.cars_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]] = \
                                                self.cars_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]] + 0.1

                                        else:
                                            self.cars_predicted[frame][dir[0]:dir[1], dir[2]:dir[3], dir[4]:dir[5]] = \
                                                self.cars_predicted[frame][dir[0]:dir[1], dir[2]:dir[3], dir[4]:dir[5]] + 0.1
                                    # print (" Bounding box " + str(dir) + " " + str(
                                    #     np.mean(self.cars_predicted[frame][dir[2]:dir[3],
                                    #             dir[4]:dir[5]])) + " init sum " + str(
                                    #     np.mean(self.cars_predicted_init[dir[2]:dir[3],
                                    #             dir[4]:dir[5]])))
                                    loc_frame = loc_frame + 1
                        previous_pos = np.array(
                            [np.mean([pos[0], pos[1]]), np.mean([pos[2], pos[3]]), np.mean([pos[4], pos[5]])])
        # print("Done with predict people " + str(np.sum(self.people_predicted_init)))

    def get_next_car_location(self, average_location,  location, vel):
        location = location + np.array([0, 0, vel[1], vel[1], vel[2], vel[2]])
        average_location = average_location + vel
        # print (" Init frame " + str(frame) + " average location " + str(average_location))
        dir = [max(location[0].astype(int), 0),
               min(location[1].astype(int) + 1, self.reconstruction.shape[0]),
               max(location[2].astype(int), 0),
               min(location[3].astype(int) + 1, self.reconstruction.shape[1]),
               max(location[4].astype(int), 0),
               min(location[5].astype(int) + 1, self.reconstruction.shape[2])]
        return location, average_location, dir

    def get_next_pedestrian_location(self, average_location, location, vel):
        location = location + np.tile(vel, [2, 1]).T
        average_location = average_location + vel
        # print (" Init frame " + str(frame) + " average location " + str(average_location))
        dir = [max(location[0][0].astype(int), 0),
               min(location[0][1].astype(int) + 1, self.reconstruction.shape[0]),
               max(location[1][0].astype(int), 0),
               min(location[1][1].astype(int) + 1, self.reconstruction.shape[1]),
               max(location[2][0].astype(int), 0),
               min(location[2][1].astype(int) + 1, self.reconstruction.shape[2])]
        return location, average_location, dir

    def get_valid_positions(self, mark_out_people=True):
        # Find valid positions in tensor.
        for x in range(self.valid_positions.shape[1]):
            for y in range(self.valid_positions.shape[0]):
                if self.valid_position([0, y, x], no_height=True):  # and np.sum(self.reconstruction[:, y, x, 5]) == 0:
                    self.valid_positions[y, x] = 1
                else:
                    self.valid_positions[y, x] = 0
                    #print(" Valid pos x: "+str(x)+" y: "+str(y))
        # print("Static valid pos "+str(np.sum(np.abs( self.valid_positions))))
        self.mark_out_cars_in_valid_pos()
        # print(" After mark out cars "+str(np.sum(self.valid_positions)))
        if mark_out_people:
            self.mark_out_people_in_valid_pos()
            # print(" After mark out people " + str(np.sum(self.valid_positions)))

    def mark_out_cars_in_valid_pos(self):
        if self.useRealTimeEnv:

            for car_id in self.init_cars:
                t = 0
                dif = max(t - 1, 0) - 1
                car, car_vel = self.get_car_pos_and_vel(car_id)
                limits = self.get_car_limits(car, dif)
                continue_in_loop=True
                # print("Mark car " + str(car_id))
                if limits[0] < limits[1] and limits[2] < limits[3] and continue_in_loop:
                    self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.valid_positions[
                                                                                         limits[0]:limits[1],
                                                                                         limits[2]:limits[3]]
                    t = t + 1
                    dif = max(t - 1, 0) - 1
                    car = self.extract_car_pos(car, car_vel)
                    limits = self.get_car_limits(car, dif)
                    if np.linalg.norm(car_vel)<=0.1:
                        continue_in_loop=False


        else:
            # for t in range(0, min(max(abs(self.cars[0][0][2] - self.cars[0][0][3]), abs(self.cars[0][0][4] - self.cars[0][0][5])) + 2,len(self.cars))):
            for t in range(len(self.cars)):
                for car in self.cars[t]:
                    dif = max(t - 1, 0) - 1
                    limits = self.get_car_limits(car, dif)
                    if limits[0] < limits[1] and limits[2] < limits[3]:
                        self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.valid_positions[
                                                                                             limits[0]:limits[1],
                                                                                             limits[2]:limits[3]]

    def mark_out_people_in_valid_pos(self):
        if self.useRealTimeEnv:
            for pedestrian_id in self.valid_keys:
                t = 0
                dif = max(t - 1, 0) #- 1
                person, pedestrian_vel = self.get_pedestrian_pos_and_vel(pedestrian_id)
                limits = self.get_pedestrian_limits(person, dif)
                while limits[0] < limits[1] and limits[2] < limits[3]:
                    self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.valid_positions[
                                                                                         limits[0]:limits[1],
                                                                                         limits[2]:limits[3]]
                    t = t + 1
                    dif = max(t - 1, 0)# - 1
                    person_new = self.extract_pedestrian_pos(person, pedestrian_vel)
                    limits = self.get_car_limits(person_new, dif)


        else:
            # for t in range(0, min(max(abs(self.cars[0][0][2] - self.cars[0][0][3]), abs(self.cars[0][0][4] - self.cars[0][0][5])) + 2,len(self.cars))):
            for t in range(len(self.people)):
                for person in self.people[t]:
                    dif = max(t - 1, 0) #- 1
                    limits = self.get_pedestrian_limits(person, dif)
                    if limits[0] < limits[1] and limits[2] < limits[3]:
                        self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.valid_positions[
                                                                                             limits[0]:limits[1],
                                                                                             limits[2]:limits[3]]

    def get_car_pos_and_vel(self, car_id, frame=0):
        car = self.cars_dict[car_id][frame]
        car_vel = self.car_vel_dict[frame][car_id] #if frame in self.car_vel_dict else [0,0,0]#- The frame should always be in the dictionary

        return car, car_vel

    def get_pedestrian_pos_and_vel(self, pedestrian_id,frame=0):
        pedestrian = self.people_dict[pedestrian_id][frame]
        pedestrian_vel = self.people_vel_dict[frame][pedestrian_id] #if frame in self.people_vel_dict else [0,0,0]

        return pedestrian, pedestrian_vel

    def extract_car_pos(self, car, car_vel):
        #print ("Car "+str(car)+" car vel "+str(car_vel))
        return [car[0] + car_vel[0], car[1] + car_vel[0], car[2] + car_vel[1], car[3] + car_vel[2],
                car[4] + car_vel[2], car[5] + car_vel[2]]

    def extract_pedestrian_pos(self, person, ped_vel):
        return [person[0][0] + ped_vel[0], person[0][1] + ped_vel[0], person[1][0] + ped_vel[1], person[1][1] + ped_vel[1],
                person[2][0] + ped_vel[2], person[2][1] + ped_vel[2]]

    def get_car_limits(self, car, dif):
        dif=max(dif,0) # To do. Check that this is necessary
        x_min_value=int(car[2]) + dif - self.agent_size[1]
        x_max_value=int(car[3]) - dif + self.agent_size[1] + 1
        y_min_value = int(car[4]) - self.agent_size[2] + dif
        y_max_value = int(car[5]) - dif + self.agent_size[2] + 1
        x_min=max(x_min_value, 0)
        x_max =  min(x_max_value, self.valid_positions.shape[0])
        y_min =  max(y_min_value, 0)
        y_max =min(y_max_value, self.valid_positions.shape[1])

        limits = [x_min,
                  x_max,
                  y_min,
                  y_max]
        return limits

    def get_pedestrian_limits(self, person, dif):
        # print(" Agent size "+str(self.agent_size)+" person "+str(person))
        limits = [max(int(person[1][0]) + dif - self.agent_size[1], 0),
                          min(int(person[1][1]) - dif + self.agent_size[1]+1 , self.valid_positions.shape[0]),
                          max(int(person[2][0]) - self.agent_size[2] + dif, 0),
                          min(int(person[2][1]) - dif + self.agent_size[2]+1, self.valid_positions.shape[1])]
        # print ("Limits "+str(limits )+" dif "+str(dif))

        return limits

    def get_controllablepedestrian_limits(self, person, dif):
        # print(" Agent size "+str(self.agent_size)+" person "+str(person))
        limits = [max(int(person[1]) + dif - (2 *self.agent_size[1]), 0),
                  min(int(person[1]) - dif + (2*self.agent_size[1]) + 1, self.valid_positions.shape[0]),
                  max(int(person[2]) - (2*self.agent_size[2]) + dif, 0),
                  min(int(person[2]) - dif + (2*self.agent_size[2]) + 1, self.valid_positions.shape[1])]
        # print ("Limits "+str(limits )+" dif "+str(dif))

        return limits

    def correct_init_frame(self, adjust_first_frame, seq_len):
        self.first_frame = 50
        if len(self.people) < 500 or len(self.cars) < 500: # If a carla run adapt the first frame
            self.first_frame = 0
        if adjust_first_frame:
            if len(self.people[self.first_frame]) == 0:
                while (self.first_frame < len(self.people) and len(self.people[self.first_frame]) == 0):
                    self.first_frame += 1
                if self.first_frame == len(self.people):
                    self.first_frame = 0
                if self.first_frame >= len(self.people) - seq_len:
                    self.first_frame = len(self.people) - seq_len
        self.people = self.people[self.first_frame:]
        self.cars = self.cars[self.first_frame:]
        for frame in range(self.first_frame):
            self.people.append([])
            self.cars.append([])
        for key in list(self.people_dict.keys()):
            fine_people = []
            valid = False

            # Pedestrian present before first frame.

            if self.init_frames[key] + len(self.people_dict[key]) < self.first_frame:
                self.people_dict[key] = []

            else:

                dif = self.first_frame - self.init_frames[key]
                #  If agent present before first frame. Then update agent's initialization ro first frame.
                if self.init_frames[key] < self.first_frame:
                    self.people_dict[key] = self.people_dict[key][dif:]
                    self.init_frames[key] = 0
                else:  # Otherwise update when agent appears.
                    self.init_frames[key] = self.init_frames[key] - self.first_frame
        for key in list(self.cars_dict.keys()):
            # Pedestrian present before first frame.
            # print ("First frame  "+str(self.first_frame))
            if self.init_frames_car[key] + len(self.cars_dict[key]) <= self.first_frame:
                self.cars_dict[key] = []

            else:
                dif = self.first_frame - self.init_frames_car[key]
                # print (" dif "+str(dif)+" key "+str(key)+" init frame "+str(self.init_frames_car[key]))
                #  If agent present before first frame. Then update agent's initialization ro first frame.
                if self.init_frames_car[key] <= self.first_frame:
                    self.cars_dict[key] = self.cars_dict[key][dif:]
                    self.init_frames_car[key] = 0
                else:  # Otherwise update when agent appears.
                    self.init_frames_car[key] = self.init_frames_car[key] - self.first_frame




    def initial_position(self, id, poses_db, initialization=-1, training=True, init_key=-1, set_vel_init=True):
        agent=self.pedestrian_data[id]
        if set_vel_init:
            maxAgentSpeed_voxels = 300 / 15 * self.frame_time
            random_speed = np.random.rand() * maxAgentSpeed_voxels
            dir = np.array(self.actions[random.randint(0, len(self.actions) - 1)])
            if np.linalg.norm(dir) > 1:
                dir_norm = dir / np.linalg.norm(dir)
            else:
                dir_norm = dir
            self.pedestrian_data[id].vel_init = dir_norm * random_speed
            print(("Initial vel init "+str(self.pedestrian_data[id].vel_init)+" follow goal "+str(self.follow_goal)))
        # Reset measures.
        i = 0
        self.update_valid_positions(id)
        self.pedestrian_data[id].goal_person_id=-1
        self.pedestrian_data[id].goal=np.zeros_like(self.pedestrian_data[id].goal)
        self.pedestrian_data[id].measures = np.zeros(self.pedestrian_data[id].measures.shape)
        # Check if goal can be found

        # If given specification on initialization type.
        if initialization==PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian:

            if self.people_dict and  len(self.initializer_data[id].valid_keys)>0:
                self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
                print("initialize on pedestrian ")
                if init_key < 0:
                    init_key=random.randint(0,len(self.initializer_data[id].valid_keys)-1)
                self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
                return self.initial_position_key(id, init_key)
            else:
                print("No valid people in frame")
                return [], -1, self.pedestrian_data[id].vel_init

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.by_car:
            self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.by_car
            print(("initialize by car " + str(self.pedestrian_data[id].init_method)))
            return self.initialize_by_car(id)

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian:
            self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian
            print(("initialize by pedestrian " + str(self.pedestrian_data[id].init_method)))
            if self.people_dict and  len(list(self.valid_people_tracks.keys()))>0:
                return self.initialize_by_ped_dict(id)
            else:
                print("No valid pedestrians")
                return [], -1, self.pedestrian_data[id].vel_init

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.randomly:
            self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.randomly
            print(("initialize randomly " + str(self.pedestrian_data[id].init_method)))
            return self.initialize_randomly(id)

        elif initialization == PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car:
            self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car
            print(("initialize in front of car " + str(self.pedestrian_data[id].init_method)))
            if self.init_cars:
                return self.initialize_by_car_dict(id)
            else:
                return self.initialize_by_car(id)

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian :
            self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian
            print(("initialize by pedestrian "+str(initialization)+" " + str(self.pedestrian_data[id].init_method)))
            if self.people_dict and len(list(self.valid_people_tracks.keys())) > 0:
                return self.initialize_on_pedestrian_with_goal(id,on_pedestrian=False)
            else:
                print("No valid pedestrians")
            return [], -1, self.pedestrian_data[id].vel_init

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.on_pavement:
            self.pedestrian_data[id].init_method =PEDESTRIAN_INITIALIZATION_CODE.on_pavement
            print("Initialize on pavement")
            return self.initialize_on_pavement(id)

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.near_obstacle:
            self.pedestrian_data[id].init_method=PEDESTRIAN_INITIALIZATION_CODE.near_obstacle
            print(("initialize near obstacles " + str(self.pedestrian_data[id].init_method)+" border len "+str(len(self.border))))
            if len(self.border)>0:
                return self.initialize_near_obstacles(id)
            return self.initialize_randomly(id)

        elif initialization == PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory: #and not self.useRealTimeEnv:
            self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory
            print(("initialize on pedestrian trajectory" + str(self.pedestrian_data[id].init_method)))
            keys=self.initializer_data[id].valid_keys
            if len(keys)==0:
                return [], -1, self.pedestrian_data[id].vel_init
            init_id=np.random.randint(len(keys))
            init_key=keys[init_id]
            init_frame=0
            if len(self.valid_people_tracks[init_key])>5:
                init_frame=5+np.random.randint(len(self.valid_people_tracks[init_key])-5)
            else:
                return [], -1, self.pedestrian_data[id].vel_init
            return self.initial_position_key(id, init_id, init_frame)



        # Otherwise if training: initialize randomly, when testing initialize near people
        if training:
            u=random.uniform(0,1) # Random number.
            if len(self.valid_people) > 0:
                if u > 0.5:
                    if u<0.25:
                        self.pedestrian_data[id].init_method=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
                        return self.initialize_agent_pedestrian(id)
                    else:
                        self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.by_pedestria
                        return self.initialize_agent_pedestrian(id, on_pedestrian=False)
            if u> 0.75:
                self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.by_car
                return self.initialize_by_car(id)
            self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.randomly
            return self.initialize_randomly(id)
        else:
            print(("Init key "+str(init_key)))
            if init_key >=0 and not self.useRealTimeEnv:
                self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
                return self.initial_position_key(id,init_key)
            else:
                print("No People")
                return [], -1, self.pedestrian_data[id].vel_init


    def initial_position_key(self,id, initialization_key, frame_init=0):
        agent= self.pedestrian_data[id]
        initialization_key_val=initialization_key
        if self.key_map and len(self.key_map)>0:
            initialization_key_val=self.key_map[initialization_key]
        person=self.valid_people_tracks[initialization_key_val][frame_init]
        agent.agent[0] = np.mean(person, axis=1)

        # get the velocity of the pedestrian agent based on recorded data
        velOfPed = np.array([0,0,0])
        # print( self.valid_people_tracks[initialization_key_val])
        if (frame_init + 1) < len(self.valid_people_tracks[initialization_key_val]):
            velOfPed = np.mean(self.valid_people_tracks[initialization_key_val][frame_init + 1], axis=1) - np.mean(person, axis=1)
        elif (initialization_key_val in self.people_vel) and (frame_init in self.people_vel[initialization_key_val]) :
            velOfPed = self.people_vel[initialization_key_val][frame_init]


        agent.vel_init= velOfPed

        print(("Vel init "+str(agent.vel_init)))

        if self.follow_goal:
            agent.goal = np.mean(self.valid_people_tracks[initialization_key_val][min(frame_init+self.seq_len,len(self.valid_people_tracks[initialization_key_val])-1)], axis=1).astype(int)

        if frame_init==0:
            print(" Set initialization key to "+str(initialization_key_val))
            agent.goal_person_id=initialization_key
            agent.goal_person_id_val=initialization_key_val
        print((agent.goal_person_id))
        print(("Starting at "+str(agent.agent[0])+" "+str(agent.goal)))
        # print (str(len(self.initializer_data))+" id "+str(id))
        # print(str(len(self.initializer_data[id].valid_keys))+" value "+str(initialization_key))
        #self.remove_valid_pavement_pos(id, initialization_key)
        np.delete(self.initializer_data[id].valid_keys, initialization_key)
        return agent.agent[0], agent.goal_person_id,agent.vel_init


    def initialize_agent_pedestrian(self,id, on_pedestrian=True):
        if self.people_dict and len(list(self.valid_people_tracks.keys())) > 0:  # Initialize on pedestrian track
            self.initialize_on_pedestrian_with_goal(id,on_pedestrian=on_pedestrian)
            return self.pedestrian_data[id].agent[0], self.pedestrian_data[id].goal_person_id, self.pedestrian_data[id].vel_init
        else:
            self.initialize_on_pedestrian(id,on_pedestrian=on_pedestrian)
        return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init

    def initialize_car_environment(self,id, training):
        height = np.mean(self.cars[0][0][0:1])
        if training == False:
            pos = random.randint(0, len(self.test_positions)-1)

            self.pedestrian_data[id].agent[0] = np.array([height, self.test_positions[pos][0], self.test_positions[pos][1]])
            self.pedestrian_data[id].agent[0] = self.pedestrian_data[id].agent[0].astype(int)

            return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init
        valid_places = np.where(self.initializer_data[id].valid_positions)
        pos = random.randint(0, len(valid_places[0])-1)
        self.pedestrian_data[id].agent[0] = np.array([height, valid_places[0][pos], valid_places[1][pos]])
        if self.direction == 0:
            self.pedestrian_data[id].goal[0,:] = self.pedestrian_data[id].agent[0]
            self.pedestrian_data[id].goal[0,1] = abs(self.reconstruction.shape[1] - self.pedestrian_data[id].goal[0,1])
        elif self.direction == 1:
            self.pedestrian_data[id].goal[0,:] = self.pedestrian_data[id].agent[0]
            self.pedestrian_data[id].goal[0,2] = abs(self.reconstruction.shape[2] - self.pedestrian_data[id].goal[0,2])
        else:
            self.pedestrian_data[id].goal[0,:] = self.pedestrian_data[id].agent[0]
            self.pedestrian_data[id].goal[0,1] = abs(self.reconstruction.shape[1] - self.pedestrian_data[id].goal[0,1])
            self.pedestrian_data[id].goal[0,2] = abs(self.reconstruction.shape[2] - self.pedestrian_data[id].goal[0,2])
        self.pedestrian_data[id].agent[0]=self.pedestrian_data[id].agent[0].astype(int)
        self.pedestrian_data[id].goal = self.pedestrian_data[id].goal.astype(int)
        return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init

    def initialize_on_pedestrian_with_goal(self,id, on_pedestrian=True):

        if len(self.initializer_data[id].valid_keys)>0:
            # print ("Person ids :"+str(self.valid_keys))
            # for key in range(len(self.valid_keys)):
            #     print (str(key)+" true "+str(self.key_map[key]))
            self.pedestrian_data[id].goal_person_id = np.random.randint(len(self.initializer_data[id].valid_keys))
            print(("goal person id "+str(self.pedestrian_data[id].goal_person_id)))
            self.pedestrian_data[id].goal_person_id_val=self.key_map[self.pedestrian_data[id].goal_person_id]
            person =  self.valid_people_tracks[self.pedestrian_data[id].goal_person_id_val][0].astype(int)

            self.pedestrian_data[id].agent[0] = np.mean(person, axis=1).astype(int)
            self.initializer_data[id].valid_keys.remove(self.pedestrian_data[id].goal_person_id)
            if self.follow_goal:
                if not self.useRealTimeEnv:
                    self.pedestrian_data[id].goal=np.mean( self.valid_people_tracks[self.pedestrian_data[id].goal_person_id_val][-1], axis=1).astype(int)
                else:
                    self.pedestrian_data[id].goal =self.pedestrian_data[id].agent[0]+ self.seq_len*self.people_vel_dict[0][self.pedestrian_data[id].goal_person_id_val]
        elif len(list(self.valid_people_tracks.keys()))>0:
            keys=list(self.valid_people_tracks.keys())
            self.pedestrian_data[id].goal_person_id = np.random.randint(len(keys))
            self.pedestrian_data[id].goal_person_id_val =keys[self.pedestrian_data[id].goal_person_id]
            self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian
            start_frame=np.random.randint(len(self.valid_people_tracks[self.pedestrian_data[id].goal_person_id_val])-1)
            person =  self.valid_people_tracks[self.pedestrian_data[id].goal_person_id_val][start_frame].astype(int)
            self.pedestrian_data[id].agent[0] = np.mean(person, axis=1).astype(int)
            while not self.initializer_data[id].valid_positions[ self.pedestrian_data[id].agent[0] [1:]]:
                self.pedestrian_data[id].goal_person_id = np.random.randint(len(keys))
                self.pedestrian_data[id].goal_person_id_val = keys[self.pedestrian_data[id].goal_person_id]
                self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian
                start_frame = np.random.randint(
                    len(self.valid_people_tracks[self.pedestrian_data[id].goal_person_id_val]) - 1)
                person = self.valid_people_tracks[self.pedestrian_data[id].goal_person_id_val][start_frame].astype(int)
                self.pedestrian_data[id].agent[0] = np.mean(person, axis=1).astype(int)

            if self.follow_goal:
                if not self.useRealTimeEnv:
                    self.pedestrian_data[id].goal = np.mean( self.valid_people_tracks[self.pedestrian_data[id].goal_person_id_val][int(len( self.valid_people_tracks[self.pedestrian_data[id].goal_person_id_val])-1 )], axis=1).astype(int)
                else:
                    self.pedestrian_data[id].goal = self.pedestrian_data[id].agent[0] + self.seq_len * self.people_vel_dict[0][self.pedestrian_data[id].goal_person_id_val]
        #print("Initialize on pedestrian with goal before "+str(self.pedestrian_data[id].goal_person_id)+" "+str(self.pedestrian_data[id].agent[0]))
        if not on_pedestrian:
            self.pedestrian_data[id].init_method = PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian
            self.pedestrian_data[id].goal_person_id=-1

            width = max((person[1][1] - person[1][0])*2,5)

            depth = max((person[2][1] - person[2][0])*2,5)

            lims = [max(person[1][0] - width, 0), max(person[2][0] - depth, 0)]

            cut_out = self.initializer_data[id].valid_positions[max(person[1][0] - width, 0):min(person[1][1] + width, self.initializer_data[id].valid_positions.shape[0]),
                      max(person[2][0] - depth, 0):min(person[2][1] + depth+1, self.initializer_data[id].valid_positions.shape[1])]
            #print("Cut out  " + str(cut_out))
            while(np.sum(cut_out)==0 and width<self.initializer_data[id].valid_positions.shape[0]/2 and depth<self.initializer_data[id].valid_positions.shape[1]/2):
                width = width* 2
                depth =depth * 2
                lims = [max(person[1][0] - width, 0), max(person[2][0] - depth, 0)]
                cut_out = self.initializer_data[id].valid_positions[
                          max(person[1][0] - width, 0):min(person[1][1] + width, self.initializer_data[id].valid_positions.shape[0]),
                          max(person[2][0] - depth, 0):min(person[2][1] + depth+1, self.initializer_data[id].valid_positions.shape[1])]

            test_pos = find_border(cut_out)

            #print("Test pos  " + str(test_pos))
            if len(test_pos)>0:
                i = random.choice(list(range(len(test_pos))))
                #print ("Random test pos "+str(i)+" "+str(test_pos[i][0])+" "+str(test_pos[i][1]))
                self.pedestrian_data[id].agent[0] = np.array([self.pedestrian_data[id].agent[0][0], lims[0] + test_pos[i][0], lims[1] + test_pos[i][1]]).astype(int)
                #print ("Agent " + str(self.pedestrian_data[id].agent[0]))



            else:
                self.add_rand_int_to_inital_position(id,width+depth)
                validPosDimX = self.initializer_data[id].valid_positions.shape[0]
                validPosDimY = self.initializer_data[id].valid_positions.shape[1]
                agentIsInRange = validPosDimX > self.pedestrian_data[id].agent[0][1] and validPosDimY > self.pedestrian_data[id].agent[0][2]
                if agentIsInRange == False or self.initializer_data[id].valid_positions[self.pedestrian_data[id].agent[0][1], self.pedestrian_data[id].agent[0][2]]==False:
                    height = self.get_height_init()
                    matrix = np.logical_and(self.people_predicted[0] > 0, self.initializer_data[id].valid_positions)
                    limits = [self.reconstruction.shape[1] // 4, self.reconstruction.shape[1] * 3 // 4,
                              self.reconstruction.shape[2] // 4, self.reconstruction.shape[2] * 3 // 4]
                    car_traj = np.where(matrix[limits[0]:limits[1], limits[2]:limits[3]])
                    if len(car_traj)>0 and len(car_traj[0])>1:
                        pos = random.randint(0, len(car_traj[0])-1)
                        self.pedestrian_data[id].agent[0] = np.array([height, car_traj[0][pos], car_traj[1][pos]]).astype(int)
                    else:
                        print("No people")
                        return [], -1, self.pedestrian_data[id].vel_init

        self.pedestrian_data[id].agent[0]=self.pedestrian_data[id].agent[0].astype(int)
        if self.follow_goal:
            self.set_goal(id)

        #print(("Starting at " + str(self.pedestrian_data[id].agent[0]) + " goal is : " + str(self.pedestrian_data[id].goal)+" "+str(self.pedestrian_data[id].goal_person_id)+" "+str(self.pedestrian_data[id].goal_person_id_val)))
        return self.pedestrian_data[id].agent[0], self.pedestrian_data[id].goal_person_id, self.pedestrian_data[id].vel_init

    def initialize_randomly(self, id):
        height = self.get_height_init()# Set limit so that random spot is chosen in the middle of the scene.
        limits=[self.reconstruction.shape[1] // 4, self.reconstruction.shape[1]*3 // 4, self.reconstruction.shape[2] // 4, self.reconstruction.shape[2]*3 // 4]

        # Find valid places
        valid_places = np.where(self.initializer_data[id].valid_positions[limits[0]:limits[1], limits[2]:limits[3]])
        len_valid = len(valid_places[0])
        print ("Valid places "+str(valid_places)+" len "+str(len_valid))

        if len_valid>0:
            # Select spot to set agent.
            if len_valid >1:

                pos = random.randint(0, len_valid-1)
            else:
                pos=0

            print ("Random pos "+str(pos)+" to value "+str(valid_places[0][pos])+" "+str(valid_places[1][pos])+" limits "+str(limits))
            self.pedestrian_data[id].agent[0] = np.array([height, limits[0]+valid_places[0][pos], limits[2]+valid_places[1][pos]])

            # Select goal position.
            pos_goal = random.randint(0, len(valid_places[0])-1)

            self.pedestrian_data[id].agent[0] = self.pedestrian_data[id].agent[0].astype(int)
            if self.follow_goal:
                self.pedestrian_data[id].goal[0,:] = np.array(
                    [height, limits[0] + valid_places[0][pos_goal], limits[2] + valid_places[1][pos_goal]])
                self.set_goal(id)



            return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init
        print("No valid places")
        return [], -1, self.pedestrian_data[id].vel_init

    def get_height_init(self):

        height = 0
        if len(self.valid_pavement[0]) > 1 and len(self.heights)==0:
            #indx = np.random.randint(len(self.valid_pavement[0]) - 1)
            height = np.mean(self.valid_pavement[0])+self.agent_height#self.valid_pavement[0][indx]
        else:
            height=np.mean(self.heights)
        if np.isnan(height) or height<0:
            height=0
        elif height>32:
            height=31

        return height
    def update_valid_positions_car(self,id):

        if id == 0:

            self.initializer_car_data[id].valid_positions = np.copy(self.valid_positions_cars)


        else:
            self.initializer_car_data[id].valid_positions = np.copy(self.initializer_car_data[id-1].valid_positions)

            car = self.car_data[id-1].car_bbox[0]

            dir = self.car_data[id-1].car_dir[1:]

            if abs(dir[0]) > abs(dir[1]):
                x_lim = max(self.car_dim[1:]) + 2
                y_lim = min(self.car_dim[1:]) + 2
            else:
                x_lim = min(self.car_dim[1:]) + 2
                y_lim = max(self.car_dim[1:]) + 2
            carlimits = [max(int(car[2] - x_lim), 0), max(int(car[3] + x_lim), 0), max(int(car[4] - y_lim), 0),
                         max(int(car[5] + y_lim), 0)]
            carlimits = [min(carlimits[0], self.reconstruction.shape[1]),
                         min(carlimits[1], self.reconstruction.shape[1]),
                         min(carlimits[2], self.reconstruction.shape[2]),
                         min(carlimits[3], self.reconstruction.shape[2])]

            self.initializer_car_data[id].valid_positions[carlimits[0]:carlimits[1], carlimits[2]:carlimits[3]]= 0 *  self.initializer_car_data[id].valid_positions[carlimits[0]:carlimits[1], carlimits[2]:carlimits[3]]

    def calculate_car_prior(self, id, field_of_view_car):
        self.update_valid_positions_car(id)
        if id==0:
            car_pos=[]
            car_vel=[]
        else:
            car_pos=self.car_data[0].car[0][1:]
            car_vel =self.car_data[0].car_dir[1:]
        self.initializer_car_data[id].calculate_prior(car_pos, car_vel, field_of_view_car,  occlusion_prior=self.car_occlusion_prior)#self.agent_size, self.use_occlusions, self.lidar_occlusion,field_of_view_car)

    def calculate_prior(self,id, field_of_view_car):# id, car_pos, car_vel, car_max_dim=0, car_min_dim=0):
        self.update_valid_positions(id)
        pavement=[]
        if self.add_pavement_to_prior:
            if self.assume_known_ped_stats_in_prior:
                pavement=self.heatmap
            else:
                segmentation = (self.reconstruction[:, :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(int)
                pavement_map = np.zeros_like(segmentation)
                for label in SIDEWALK_LABELS:
                    pavement_map = np.logical_or(pavement_map, segmentation == label)
                pavement_map_2D = np.any(pavement_map, axis=0)
                pavement=np.ones_like(self.valid_positions)
                pavement[pavement_map_2D]=2
        if not self.use_occlusions:
            if id==0:
                self.initializer_data[id].calculate_prior(self.agent_size, self.use_occlusions, self.lidar_occlusion, field_of_view_car, heatmap=pavement)
            else:
                self.initializer_data[id].prior=np.copy(self.initializer_data[id-1].prior)
                self.initializer_data[id].prior=self.initializer_data[id].prior*self.initializer_data[id].valid_positions
                if self.prior_smoothing:
                    self.initializer_data[id].prior=gaussian_filter(self.initializer_data[id].prior,self.prior_smoothing_sigma )
                    self.initializer_data[id].prior = self.initializer_data[id].prior * self.initializer_data[id].valid_positions
                    self.initializer_data[id].prior=self.initializer_data[id].prior*(1/np.sum(self.initializer_data[id].prior[:]))
        else:
            if self.occlude_some_pedestrians==False or id<self.number_of_agents/2:
                self.initializer_data[id].calculate_prior(self.agent_size, self.use_occlusions, self.lidar_occlusion,
                                                          field_of_view_car, heatmap=pavement)
            else:
                self.initializer_data[id].calculate_prior(self.agent_size, True, self.lidar_occlusion,
                                                          field_of_view_car, heatmap=pavement)




    def calculate_goal_prior(self, id, frame=0):

        return self.initializer_data[id].calculate_goal_prior( self.pedestrian_data[id].agent[frame][1:],  self.frame_time, self.agent_size, frame)

    def update_valid_positions(self,id):
        self.initializer_data[id].valid_positions=np.copy(self.valid_positions)
        if id==0:
            self.initializer_data[id].valid_keys=np.copy(self.valid_keys)
            self.initializer_data[id].init_cars=np.copy(self.init_cars)
            self.initializer_data[id].valid_pavement=np.copy(self.valid_pavement)
            self.initializer_data[id].border=np.copy(self.border)
        else:
            self.initializer_data[id].valid_keys = np.copy(self.initializer_data[id-1].valid_keys)
            self.initializer_data[id].init_cars = np.copy(self.initializer_data[id-1].init_cars)
            self.initializer_data[id].valid_pavement =np.copy(self.initializer_data[id-1].valid_pavement)
            self.initializer_data[id].border = np.copy(self.initializer_data[id-1].border )
        if self.useRealTimeEnv:
            for ped_id in range(id):
                person=self.pedestrian_data[ped_id]
                t = 0
                dif = max(t - 1, 0)  # - 1
                person_pos=person.agent[0]
                pedestrian_vel=person.vel_init
                limits = self.get_controllablepedestrian_limits(person_pos, dif)
                while limits[0] < limits[1] and limits[2] < limits[3]:
                    self.initializer_data[id].valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.initializer_data[id].valid_positions[
                                                                                         limits[0]:limits[1],
                                                                                         limits[2]:limits[3]]
                    t = t + 1
                    dif = max(t - 1, 0)  # - 1
                    person_pos_new = person_pos+pedestrian_vel
                    limits = self.get_controllablepedestrian_limits(person_pos_new, dif)

            for car_id in range(self.number_of_car_agents):
                car=self.car_data[car_id]
                t = 0
                dif = max(t - 1, 0)  # - 1
                car_pos=car.car_bbox[0]
                car_vel=car.car_dir
                limits = self.get_car_limits(car_pos, dif)
                while limits[0] < limits[1] and limits[2] < limits[3]:
                    self.initializer_data[id].valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.initializer_data[id].valid_positions[
                                                                                         limits[0]:limits[1],
                                                                                         limits[2]:limits[3]]
                    t = t + 1
                    dif = max(t - 1, 0)  # - 1
                    car_pos_new =self.extract_car_pos(car_pos, car_vel)
                    limits = self.get_car_limits(car_pos_new, dif)


    # Car vel is voxels per second

    def initialize_by_car(self, id):

        if len(self.cars[0]) > 0:
            previous_pos=list(self.pedestrian_data[id].agent[0])
            if len(self.cars[0]) > 1:

                indx=list(range(len(self.cars[0])))
                random.shuffle(indx)
            else:
                indx=[0]

            for car_indx in indx:
                car= self.cars[0][car_indx]

                width=(car[3]-car[2])*2
                depth=(car[5]-car[4])*2
                lims=[max(car[2]- width,0), max(car[4]- depth,0)]
                cut_out=self.initializer_data[id].valid_positions[int(max(car[2]- width,0)):int(min(car[3]+ width, self.initializer_data[id].valid_positions.shape[0])),
                        int(max(car[4]- depth,0)):int(min(car[5]+ depth, self.initializer_data[id].valid_positions.shape[1])) ]
                test_pos=find_border(cut_out)


                if len(test_pos)>0:
                    i=random.choice(list(range(len(test_pos))))
                    self.pedestrian_data[id].agent[0] =[(car[0]+car[1])/2, lims[0]+test_pos[i][0],lims[1]+ test_pos[i][1] ]
                    if self.follow_goal:
                        self.set_goal(id)

                    return self.pedestrian_data[id].agent[0], -1,self.pedestrian_data[id].vel_init

                #take out bounding box around the car
                if car[2]<self.reconstruction.shape[1] and car[4]<self.reconstruction.shape[2] and car[0]>=0 and car[1]>=0:

                    choices=[1,2,3,4]
                    if car[4]<0:
                        choices.remove(1)
                    if car[2]<0:
                        choices.remove(2)
                    if car[5]>=self.reconstruction.shape[2]:
                        choices.remove(3)
                    if car[3]>=self.reconstruction.shape[1]:
                        choices.remove(4)
                    if len(choices)>0:
                        choice=random.choice(choices)
                        # if self.init_on_pavement>0:
                        #     x=np.random.randint(self.init_on_pavement)+self.agent_size[1]+1
                        #     y=np.random.randint(self.init_on_pavement) + self.agent_size[2]+1
                        # else:
                        x = self.agent_size[1] + 1
                        y = self.agent_size[2] + 1
                        if choice==1:
                            self.pedestrian_data[id].agent[0]=[self.pedestrian_data[id].agent[0][0], car[2] +x, car[4] -y ]
                        if choice==2:
                            self.pedestrian_data[id].agent[0]=[self.pedestrian_data[id].agent[0][0], car[2] -x, car[4] +y ]
                        if choice == 3:
                            self.pedestrian_data[id].agent[0] = [self.pedestrian_data[id].agent[0][0], car[2]+ x, car[5] + y]
                        if choice == 4:
                            self.pedestrian_data[id].agent[0] = [self.pedestrian_data[id].agent[0][0], car[3] + x, car[4] + y]
                    # print("take out bounding box around the car")
                    # pos, frame_in,
                if self.intercept_car(self.pedestrian_data[id].agent[0],0, all_frames=False):
                    # print("Agent intercepts car ")
                    height = self.get_height_init()
                    matrix = np.logical_and(self.cars_predicted[0], self.initializer_data[id].valid_positions)
                    limits = [self.reconstruction.shape[1] // 4, self.reconstruction.shape[1] * 3 // 4,
                              self.reconstruction.shape[2] // 4, self.reconstruction.shape[2] * 3 // 4]
                    car_traj = np.where(matrix[limits[0]:limits[1], limits[2]:limits[3]])
                    if len(car_traj) > 0 and len(car_traj[0]) > 1:
                        pos = random.randint(0, len(car_traj[0]) - 1)
                        self.pedestrian_data[id].agent[0] = np.array([height, car_traj[0][pos], car_traj[1][pos]]).astype(int)
                    else:
                        print("No cars")

                        return [], -1, self.pedestrian_data[id].vel_init
            self.pedestrian_data[id].agent[0]=np.array(self.pedestrian_data[id].agent[0]).astype(int)
            if self.follow_goal:
                self.set_goal(id)

        else:
            # print("Final initialization")
            height = self.get_height_init()
            matrix=[]
            if self.useRealTimeEnv and len(self.car_data)>0:
                car_id=np.random.randint(0, len(self.car_data))
                cars_predicted_init = mark_out_car_cv_trajectory(self.cars_predicted_init.copy(), self.car_data[car_id].car_bbox[0],
                                                                           self.car_data[car_id].car_goal, self.car_data[car_id].car_dir, self.agent_size, time=1)

                matrix = np.logical_and(cars_predicted_init> 0, self.initializer_data[id].valid_positions)
            else:
                matrix=np.logical_and(self.cars_predicted[0] > 0, self.initializer_data[id].valid_positions)
            limits = [self.reconstruction.shape[1] // 4, self.reconstruction.shape[1] * 3 // 4,
                      self.reconstruction.shape[2] // 4, self.reconstruction.shape[2] * 3 // 4]
            car_traj=np.where(matrix[limits[0]:limits[1], limits[2]:limits[3]])
            if len(car_traj) > 0 and len(car_traj[0]) > 1:
                pos = random.randint(0, len(car_traj[0])-1)
                self.pedestrian_data[id].agent[0] = np.array([height, car_traj[0][pos], car_traj[1][pos]]).astype(int)
            else:
                print("No cars")
                return [], -1, self.pedestrian_data[id].vel_init
        if self.follow_goal:
            self.set_goal(id)

        return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init

    def initialize_by_car_dict(self, id):
        print("Initialize by car dict ")
        if len(self.initializer_data[id].init_cars) > 1:
            indx=self.initializer_data[id].init_cars
            random.shuffle(indx)
        else:
            indx=self.initializer_data[id].init_cars
        agent_width = max(self.agent_size[1:]) + 1
        for car_indx in indx:
            if len(self.cars_dict[car_indx])>1:
                car= self.cars_dict[car_indx][0]
                width = min([car[3] - car[2], car[5] - car[4]])
                height = max([car[3] - car[2], car[5] - car[4]])
                max_step=1
                if self.max_step!=np.sqrt(2):
                    max_step=self.max_step
                timestep = int(np.ceil((width/2.0 + agent_width) / (max_step )))

                car_pos=np.array([np.mean(car[0:2]),np.mean(car[2:4]), np.mean(car[4:])])
                if not self.useRealTimeEnv:
                    car_next= self.cars_dict[car_indx][min(len(self.cars_dict[car_indx])-1, timestep)]
                    car_pos_next = np.array([np.mean(car[0:2]),np.mean(car_next[2:4]), np.mean(car_next[4:])])
                    vel = car_pos_next - car_pos
                else:
                    vel = self.car_vel_dict[0][car_indx]*timestep
                    car_pos_next=car_pos+vel

                vel[0] = 0
                vel = vel * (1 / np.linalg.norm(vel))


                self.pedestrian_data[id].agent[0] = vel * np.ceil(height / 2.0 + max(self.agent_size[1:]) + 1) + car_pos_next
                self.pedestrian_data[id].agent[0] = np.array(self.pedestrian_data[id].agent[0]).astype(int)
                self.initializer_data[id].init_cars.remove(car_indx)
                if self.follow_goal:
                    self.set_goal(id)
                return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init

        self.pedestrian_data[id].agent[0]=np.array(self.pedestrian_data[id].agent[0]).astype(int)
        if self.follow_goal:
            print( "Set goal")
            self.set_goal(id)
        return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init


    def initialize_by_ped_dict(self, id):
        print(("Initialize by pedestrian "+ str(self.initializer_data[id].valid_keys)))
        if len(self.initializer_data[id].valid_keys) > 1:
            valid_keys=self.initializer_data[id].valid_keys
            random.shuffle(valid_keys)
        else:
            valid_keys=self.initializer_data[id].valid_keys
        agent_width = max(self.agent_size[1:]) + 1
        for pedestrian_indx in valid_keys:
            if len(self.people_dict[pedestrian_indx])>1:

                person= self.people_dict[pedestrian_indx][0]
                car_pos=np.mean(person, axis=1)
                #width = min([car[1][1] - car[1][0], car[2][1] - car[2][0]])
                height=max([person[1][1] - person[1][0], person[2][1] - person[2][0]])
                max_step = 1
                if self.max_step != np.sqrt(2):
                    max_step = self.max_step
                timestep=int(np.ceil((height/2.0+agent_width)/max_step))
                if not self.useRealTimeEnv:
                    person_next_pos = self.people_dict[pedestrian_indx][min(len(self.people_dict[pedestrian_indx])-1, timestep)]
                    car_pos_next = np.mean(person_next_pos, axis=1)
                    vel = car_pos_next - car_pos
                else:
                    vel=self.people_vel_dict[0][pedestrian_indx]*timestep
                    car_pos_next=  car_pos+ vel
                vel[0]=0
                vel=vel*(1/np.linalg.norm(vel))

                self.pedestrian_data[id].agent[0]=vel*np.ceil(height/2.0+max(self.agent_size[1:])+1)+car_pos_next
                self.pedestrian_data[id].agent[0] = np.array(self.pedestrian_data[id].agent[0]).astype(int)
                self.initializer_data[id].valid_keys.remove(pedestrian_indx)
                if self.follow_goal:
                    self.set_goal(id)
                return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init

        self.pedestrian_data[id].agent[0]=np.array(self.pedestrian_data[id].agent[0]).astype(int)
        if self.follow_goal:
            self.set_goal(id)
        return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init

    def initialize_near_obstacles(self, id):
        print("Initialize by obstacle")
        height = self.get_height_init()
        if len(self.initializer_data[id].border) > 0:
            i = random.choice(list(range(len(self.initializer_data[id].border))))
            self.pedestrian_data[id].agent[0]=np.array([height,  self.initializer_data[id].border[i][0],self.initializer_data[id].border[i][1]])
            self.initializer_data[id].border.remove(i)
        if self.follow_goal:
            self.set_goal(id)
        return self.pedestrian_data[id].agent[0], -1, self.pedestrian_data[id].vel_init

    def initialize_on_pedestrian(self,id, on_pedestrian=True):
        #self.add_rand_int_to_inital_position()
        if len(self.valid_people)>0:
            if len(self.valid_people)>1:
                i = np.random.randint(len(self.valid_people) - 1)
            else:
                i=0
            #self.pedestrian_data[id].goal_person_id =i
            person = self.valid_people[i]
            self.pedestrian_data[id].agent[0] = np.mean(person, axis=1).astype(int)
        else:
            return [],-1, self.pedestrian_data[id].vel_init
            # j = np.random.randint(len(self.valid_people)-1)
            # while j==i:
            #     j = np.random.randint(len(self.valid_people)-1)
            #self.pedestrian_data[id].goal = np.mean(self.valid_people[j], axis=1).astype(int)
        if on_pedestrian==False:

            width = person[1][1] - person[1][0]
            depth = person[2][1] - person[2][0]
            lims = [max(person[1][0] - width, 0), max(person[2][0] - depth, 0)]
            cut_out = self.initializer_data[id].valid_positions[
                      max(person[1][0] - width, 0):min(person[1][1] + width, self.initializer_data[id].valid_positions.shape[0]),
                      max(person[2][0] - depth, 0):min(person[2][1] + depth, self.initializer_data[id].valid_positions.shape[1])]
            test_pos = find_border(cut_out)


            if len(test_pos)>0:
                if len(test_pos) > 1:
                    i = np.random.randint(len(test_pos)-1)
                else:
                    i=0
                self.pedestrian_data[id].agent[0] =np.array( [self.pedestrian_data[id].agent[0][0], lims[0] + test_pos[i][0], lims[1] + test_pos[i][1]])
            else:

                self.add_rand_int_to_inital_position(id,5)
                if self.initializer_data[id].valid_positions[self.pedestrian_data[id].agent[0][1], self.pedestrian_data[id].agent[0][2]] == False:
                    height = self.get_height_init()
                    matrix = np.logical_and(np.sum(self.reconstruction[:, :, :, 5], axis=0) > 0, self.initializer_data[id].valid_positions)
                    limits = [self.reconstruction.shape[1] // 4, self.reconstruction.shape[1] * 3 // 4,
                              self.reconstruction.shape[2] // 4, self.reconstruction.shape[2] * 3 // 4]
                    car_traj = np.where(matrix[limits[0]:limits[1], limits[2]:limits[3]])
                    if len(car_traj)>0 and len(car_traj[0])>1:
                        pos = random.randint(0, len(car_traj[0])-1)
                        self.pedestrian_data[id].agent[0] = np.array([height, car_traj[0][pos], car_traj[1][pos]]).astype(int)
                    else:
                        print("No people")
                        return [], -1, self.pedestrian_data[id].vel_init
        self.pedestrian_data[id].agent[0] = self.pedestrian_data[id].agent[0].astype(int)
        if self.follow_goal:
            self.set_goal(id)

        return self.pedestrian_data[id].agent[0],i, self.pedestrian_data[id].vel_init



    def initialize_on_pavement(self, id):
        if len(self.initializer_data[id].valid_pavement[0])==0:
            return [], -1, self.pedestrian_data[id].vel_init
        if len(self.initializer_data[id].valid_pavement[0])==1:
            indx=0
        else:
            indx = np.random.randint(len(self.initializer_data[id].valid_pavement[0])-1)

        pos_ground = [self.initializer_data[id].valid_pavement[0][indx], self.initializer_data[id].valid_pavement[1][indx],
                      self.initializer_data[id].valid_pavement[2][indx]]  # Check this!
        self.remove_valid_pavement_pos(id, indx)
        self.pedestrian_data[id].agent[0]=np.array(pos_ground).astype(int)

        if self.follow_goal:
            if self.useRealTimeEnv:
                goal_loc=[]
                for i in range(len(self.initializer_data[id].valid_pavement[0])):
                    if np.linalg.norm(np.array([self.initializer_data[id].valid_pavement[0][i],self.initializer_data[id].valid_pavement[1][i], self.initializer_data[id].valid_pavement[2][i]])-pos_ground)>200:
                        goal_loc.append(np.array([self.initializer_data[id].valid_pavement[0][i],self.initializer_data[id].valid_pavement[1][i], self.initializer_data[id].valid_pavement[2][i]]).astype(int))

                if len(goal_loc)==0:
                    self.set_goal(id)
                elif len(goal_loc)==1:
                    self.pedestrian_data[id].goal=goal_loc[0]
                else:
                    j=np.random.randint(len(goal_loc))
                    self.pedestrian_data[id].goal = goal_loc[j]
                print(" Goal "+str(self.pedestrian_data[id].goal))
            else:
                self.set_goal(id)
        self.pedestrian_data[id].agent[0] = self.pedestrian_data[id].agent[0].astype(int)
        return self.pedestrian_data[id].agent[0],-1, self.pedestrian_data[id].vel_init

    # def remove_valid_track_pos(self, id, indx):
    #     new_valid_positions = []
    #     for i in range(3):
    #         # print("Before removal " + str(self.initializer_data[id].valid_pavement[i]))
    #         shorter_array = np.delete(self.initializer_data[id].valid_pavement[i], indx)
    #         new_valid_positions.append(shorter_array)
    #         # print("After " + str(shorter_array))
    #     self.initializer_data[id].valid_pavement = new_valid_positions

    def remove_valid_pavement_pos(self, id, indx):
        new_valid_positions=[]
        for i in range(3):
            #print("Before removal " + str(self.initializer_data[id].valid_pavement[i]))
            shorter_array = np.delete(self.initializer_data[id].valid_pavement[i], indx)
            new_valid_positions.append(shorter_array)
            #print("After " + str(shorter_array))
        self.initializer_data[id].valid_pavement = new_valid_positions

    def get_new_goal(self, id, frame):
        if frame == 0:
            goal_frame = frame
        else:
            goal_frame = frame + 1
        number_of_goals_reached=np.sum(self.pedestrian_data[id].measures[:, PEDESTRIAN_MEASURES_INDX.goal_reached])
        frame_in_seq_len=goal_frame < self.pedestrian_data[id].goal.shape[0] - 1
        if number_of_goals_reached==1 and frame_in_seq_len:
            dir=self.pedestrian_data[id].goal[0,1:]- self.pedestrian_data[id].agent[0][1:]
            if np.linalg.norm(dir)<1e-5:
                dir=self.pedestrian_data[id].vel_init
            if np.linalg.norm(dir) < 1e-5:
                dir=np.array([1,0])
            dir_unit=dir*(1/np.linalg.norm(dir))
            if np.round(np.linalg.norm(dir_unit))!=1:
                print("Direction norm "+str(np.linalg.norm(dir_unit)))
            self.pedestrian_data[id].goal[goal_frame, 1:]=self.get_last_valid_pos_along_dir(dir_unit,id, frame)
            if self.velocity_agent:
                self.get_init_speed(frame, goal_frame, id)
            return self.pedestrian_data[id].goal[goal_frame, :]
        elif number_of_goals_reached==2 and frame_in_seq_len:
            dir = self.pedestrian_data[id].goal[0, 1:] - self.pedestrian_data[id].agent[0][1:]
            if np.linalg.norm(dir) < 1e-5:
                dir = self.pedestrian_data[id].vel_init
            if np.linalg.norm(dir) < 1e-5:
                dir = np.array([1, 0])
            dir_unit = dir * (1 / np.linalg.norm(dir))

            dir_unit_ortg=np.array([-dir_unit[1], dir_unit[0]])


            if np.random.rand(1)>0.5:
                dir_unit_ortg=-dir_unit_ortg
            if np.round(np.linalg.norm(dir_unit_ortg)) != 1:
                print("Second goal Direction norm " + str(np.linalg.norm(dir_unit_ortg)))
            self.pedestrian_data[id].goal[goal_frame, 1:] = self.get_last_valid_pos_along_dir(dir_unit_ortg, id, frame)
            if self.velocity_agent:
                self.get_init_speed(frame, goal_frame, id)
            return self.pedestrian_data[id].goal[goal_frame, :]
        # else:
        #     dir = self.pedestrian_data[id].goal[frame, 1:] - self.pedestrian_data[id].agent[0][1:]
        #     dir_unit = dir * (1 / np.linalg.norm(dir))
        #     self.pedestrian_data[id].goal[goal_frame, 1:]=self.pedestrian_data[id].goal[frame, 1:]+((self.seq_len-frame)*dir_unit)
        #     return self.pedestrian_data[id].goal[goal_frame, :]

        return self.set_goal(id, frame=frame)

    def get_last_valid_pos_along_dir(self, dir_unit, id, frame):
        new_goal_pos = self.pedestrian_data[id].agent[frame][1:]
        new_goal_pos_int = self.pedestrian_data[id].agent[frame][1:].astype(int)

        # y = np.linspace(new_goal_pos[0], x1, num)
        # y=np.linspace(y0, y1, num)
        counter=0
        while self.is_valid_position(new_goal_pos_int) and counter< self.seq_len-frame:
            new_goal_pos = new_goal_pos + dir_unit
            new_goal_pos_int = np.round(new_goal_pos)
            counter=counter+1
        if abs(np.linalg.norm(self.pedestrian_data[id].goal[frame,1:]- new_goal_pos)-counter)>5:
            print("Number of itr "+str(counter)+ " dist to goal "+str(np.linalg.norm(self.pedestrian_data[id].goal[frame,1:]- new_goal_pos)))
        new_goal_pos = new_goal_pos - dir_unit
        return new_goal_pos

    def is_valid_position(self, new_goal_pos_int):
        pos_x=int(new_goal_pos_int[0])
        pos_y=int(new_goal_pos_int[1])
        within_lower_bound = pos_x > 0 and pos_y > 0
        within_upper_bound_1 = pos_x < self.valid_positions.shape[0]
        within_upper_bound_2 = pos_y< self.valid_positions.shape[1]
        within_bounds = within_lower_bound and within_upper_bound_1 and within_upper_bound_2
        if within_bounds:
            if self.valid_positions[pos_x, pos_y]:
                return True
            else:
                return self.valid_position([0,new_goal_pos_int[0], new_goal_pos_int[1]])
        return False

    def set_goal(self, id, frame=0):
        dist =int(min(self.seq_len *0.44, self.threshold_dist))
        limits = [int(max(self.pedestrian_data[id].agent[frame][1] - dist, 0)), int(min(self.pedestrian_data[id].agent[frame][1] + dist, self.reconstruction.shape[1])),
                  int(max(self.pedestrian_data[id].agent[frame][2] - dist, 0)), int(min(self.pedestrian_data[id].agent[frame][2] + dist, self.reconstruction.shape[2]))]

        valid_places = np.where(self.initializer_data[id].valid_positions[limits[0]:limits[1], limits[2]:limits[3]])  # ,
        if self.evaluation:
            correct_places=[[],[],[]]
            if len(valid_places[0])>0:
                for i in range(0,len(valid_places[0])):
                    #print str(i)+" "+str(valid_places[0][i])
                    if np.linalg.norm(np.array([valid_places[0][i], valid_places[1][i]]))>dist/2.0:
                        correct_places[0].append(valid_places[0][i])
                        correct_places[1].append(valid_places[1][i])

                if len(correct_places[0])>0:
                    valid_places=correct_places

        if len(valid_places[0]) > 0:
            # Select spot to set goal.
            if len(valid_places[0]) ==1:
                pos = 0
            else:
                #print "Valid places len "+str(len(valid_places[0]) )
                pos = random.randint(0, len(valid_places[0]) - 1)
            if frame==0:
                goal_frame=frame
            else:
                goal_frame=frame+1
            if goal_frame<self.pedestrian_data[id].goal.shape[0]-1:
                self.pedestrian_data[id].goal[goal_frame,:] = np.array([self.pedestrian_data[id].agent[frame][0], limits[0] + valid_places[0][pos], limits[2] + valid_places[1][pos]])


                if self.velocity_agent:
                    self.get_init_speed(frame, goal_frame, id)

                return  self.pedestrian_data[id].goal[goal_frame,:]
            return  self.pedestrian_data[id].goal[goal_frame-1,:]
        else:
            print("No goal")
            return self.pedestrian_data[id].goal[frame,:]

    def get_init_speed(self, frame, goal_frame, id):
        self.pedestrian_data[id].speed_init = 1
        if np.linalg.norm(self.pedestrian_data[id].vel_init) > 1e-5:
            self.pedestrian_data[id].speed_init = np.linalg.norm(self.pedestrian_data[id].vel_init)
        self.pedestrian_data[id].goal_time[goal_frame] = min(self.seq_len - 1, np.linalg.norm(
            self.pedestrian_data[id].goal[goal_frame, :] - self.pedestrian_data[id].agent[frame]) /
                                                             self.pedestrian_data[id].speed_init)

    def add_rand_int_to_inital_position(self,id,  steps=-1):
        sign = 1
        if np.random.rand(1) > 0.5:
            sign = -1
        if steps<0:
            steps=5#self.init_on_pavement

        sz = self.pedestrian_data[id].agent[0].shape
        if steps > 0:
            changed=False
            tries=0
            while not changed:
                rand_init = np.random.randint(max(1, steps), size=sz[0])
                tmp=self.pedestrian_data[id].agent[0] + rand_init
                if tmp.all()>0 and tmp[0]<self.reconstruction.shape[0]and tmp[1]<self.reconstruction.shape[1]and tmp[2]<self.reconstruction.shape[2] and self.initializer_data[id].valid_positions[tmp[1], tmp[2]]==True:
                    self.pedestrian_data[id].agent[0][1:2] = self.pedestrian_data[id].agent[0][1:2] + rand_init[1:2]
                    changed=True
                if tries>5:
                    changed = True
                    print(("Could not add random init "+str(self.pedestrian_data[id].agent[0])))
                tries=tries+1

    def find_sidewalk(self, init):
        segmentation = (self.reconstruction[:, :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(int)
        pavement_map = np.zeros_like(segmentation)
        for label in SIDEWALK_LABELS:
            pavement_map =np.logical_or( pavement_map,segmentation == label)
        pavement=np.where(pavement_map)
        if init:
            pos_init=[[],[],[]]
            for indx, _ in enumerate(pavement[0]):
                if self.valid_positions[pavement[1][indx], pavement[2][indx]]:#, no_height=False):
                    pos_init[0].append(pavement[0][indx])
                    pos_init[1].append(pavement[1][indx])
                    pos_init[2].append(pavement[2][indx])
            return pos_init

        return pavement



    def iou_sidewalk(self, pos, no_height=True):
        segmentation = self.person_bounding_box(pos, no_height=no_height)

        # 6-ground, 7-road, 8- sidewalk, 9-paring, 10- rail track
        # area = 0
        # for val in [8]:#self.pavement:
        #    area = max(np.sum(segmentation == val), area)
        area=0#(segmentation == 8).sum()

        for label in SIDEWALK_LABELS:
            area += np.sum(segmentation == label)
        return area * 1.0 / ((self.agent_size[0] * 2.0 + 1)*(self.agent_size[1] * 2.0 + 1)*(self.agent_size[2] * 2.0 + 1) )



    def calculate_reward(self, frame, person_id=-1, episode_done=False, supervised=False, print_reward_components=False, last_frame=-1):

        rewards=[]
        init_pos=np.zeros((self.number_of_agents,2))
        for id in range(self.number_of_agents):
            agent = self.pedestrian_data[id]
            pos = agent.agent[frame + 1]
            if not len(pos) == 0:

                if not self.people_dict or agent.init_method!=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian:
                    person_id=-1
                prev_mul_reward=False
                self.time_dependent_evaluation(id,frame, person_id)  # Agent on Pavement?

                # Done up to here

                self.evaluate_measures(agent, frame, episode_done, person_id)
                self.measures_into_future(id,episode_done, frame, last_frame=last_frame)
                if print_reward_components:
                    print("Reward of agent "+str(id)+" previous mul "+str(prev_mul_reward))
                rewards.append(agent.calculate_reward(frame, self.multiplicative_reward_pedestrian, prev_mul_reward, self.reward_weights_pedestrian,print_reward_components, self.max_step, self.end_on_hit_by_pedestrians, self.stop_on_goal))
                self.initializer_data[id].measures = self.pedestrian_data[id].measures
                self.initializer_data[id].set_values(self.pedestrian_data[id])
                if print_reward_components:
                    print("Reward of initializer "+str(id)+" previous mul "+str(prev_mul_reward))
                self.initializer_data[id].calculate_reward(frame, self.multiplicative_reward_initializer, prev_mul_reward, self.reward_weights_initializer,print_reward_components, self.max_step, self.end_on_hit_by_pedestrians, self.stop_on_goal)
            if frame==0:
                init_pos[id,:]=self.pedestrian_data[id].agent[0][1:]
        if frame==0 and self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.init_variance]!=0:
            variance=np.var(init_pos, axis=0)/(max(self.reconstruction.shape)**2)

            #print( self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.init_variance]*(variance[0]+ variance[1]))
            self.initializer_data[self.number_of_agents-1].reward[frame]=+( self.reward_weights_initializer[PEDESTRIAN_REWARD_INDX.init_variance]*(variance[0]+ variance[1]))
        return rewards

    def evaluate_measures(self, agent, frame, episode_done=False, person_in=-1):
        pos = agent.agent[frame + 1]
        if isinstance(pos, np.ndarray) is False and (isinstance(pos, list) and len(pos) == 0):
            return
        agent.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement ] = 0
        agent.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement ] = self.iou_sidewalk(pos)

        # Intercept with objects.
        if agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles] ==0:
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]=self.intercept_objects(pos)

        # Distance travelled
        #print "Frame "+str(frame+1)+" init pos "+str(agent.agent[0])+" Pos "+str(agent.agent[frame + 1])+" Distance travelled: "+str(np.linalg.norm(agent.agent[frame + 1] - agent.agent[0]))
        agent.measures[frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init] = np.linalg.norm(agent.agent[frame + 1] - agent.agent[0])


        # Out of axis & Heatmap
        current_pos = np.round(pos).astype(int)
        if self.out_of_axis(current_pos): #or self.heatmap.shape[0]<=current_pos[0] or self.heatmap.shape[0]<=current_pos[1]:
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.out_of_axis] = 1  # out of axis
        elif not self.useRealTimeEnv or self.assume_known_ped_stats:
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]=self.heatmap[current_pos[1],current_pos[2] ]

        # Changing direction
        agent.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction] =self.get_measure_for_zig_zagging(agent, frame)

        if self.follow_goal: # If using a general goal

            # Distance to goal.
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal] = np.linalg.norm(np.array(agent.goal[frame,1:]) - agent.agent[frame + 1][1:])

            #print "Distance to goal: " + str(agent.measures[frame, 7]) + " goal pos: " + str(self.pedestrian_data[id].goal[1:]) + "current pos: " + str(agent.agent[frame+1][1:])+" sqt(2) "+str(np.sqrt(2))+" "+str(agent.measures[frame, 7]<=np.sqrt(2))

            # Reached goal?
            if agent.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]<=np.sqrt(2):
                #if self.stop_on_goal:
                agent.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] = 1
                # else:
                #     agent.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached]=agent.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached]+1
                #print "Reached goal"

            if frame==0 or (agent.goal[frame-1,1]!=agent.goal[frame,1] or agent.goal[frame-1,2]!=agent.goal[frame,2]):
                agent.goal_to_agent_init_dist=np.linalg.norm(np.array(agent.goal[frame,1:]) - agent.agent[frame][1:])


        # One- time step prediction error in supervised case. otherwise error so far.
        if agent.goal_person_id>=0:
            goal_next_frame = min(frame + 1, len(self.people_dict[agent.goal_person_id_val]) - 1)
            goal = np.mean(self.people_dict[agent.goal_person_id_val][goal_next_frame], axis=1).astype(int)

            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.one_step_prediction_error] = np.linalg.norm(goal[1:] - pos[1:])
            #previous_pos=np.mean(self.people_dict[self.pedestrian_data[id].goal_person_id_val][goal_next_frame-1], axis=1).astype(int)
            # print "Pedestrian frame "+str(goal_next_frame)+" pos "+str(goal[1:])+" previous pos: "+str( previous_pos)
            # print "Agent "+str(frame + 1)+" pos "+str( agent.agent[frame + 1][1:])+" velocity "+str(self.velocity[frame])
            # print " error: "+str(goal[1:] - agent.agent[frame + 1][1:])+" Measure "+str(agent.measures[frame, 9])
        if agent.goal_time[frame] > 0:
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time] = min(abs(agent.goal_time[frame] - frame) / agent.goal_time[frame], 1)

        #agent.measures[frame, 15] = min(abs(self.pedestrian_data[id].goal_time - frame) / self.pedestrian_data[id].goal_time, 1)

    def get_measure_for_zig_zagging(self,agent, frame):
        start = agent.agent[0]  # Start position of agent
        max_penalty = 0
        cur_vel = np.array(agent.velocity[max(frame, 0)])  # np.array(self.actions[int(self.action[max(frame, 0)])])
        for t in range(min(self.past_move_len, frame)):

            old_vel = np.array(agent.velocity[max(frame - t - 1,
                                                 0)])  # ) #np.array(agent.agent[frame])-np.array(agent.agent[max(frame-3,0)])
            older_vel = np.array(agent.velocity[max(frame - t - 2, 0)])  #
            if len(old_vel) > 0 and len(older_vel) > 0 and len(cur_vel) > 0:

                dif_1 = cur_vel - old_vel
                dif_2 = old_vel - older_vel
                #
                # print "Previous  " +str(t)+"  "+ str(prev_action)+" frame: "+str(frame)
                # print str(cur_vel)+" "+str(old_vel)+" "+str(older_vel)+" dif1: "+str(dif_1)+" dif2: "+str(dif_2)
                #
                # print "Change direction  "+str(np.dot(dif_1, dif_2))+" "+str(dif_1)+" "+str(dif_2)+" "
                if np.linalg.norm(old_vel) > 10 ** -6:

                    old_vel_norm = old_vel * min((1.0 / np.linalg.norm(old_vel)), 1)
                else:
                    old_vel_norm = np.zeros_like(old_vel)
                if np.linalg.norm(cur_vel) > 10 ** -6:
                    cur_vel_norm = cur_vel * min((1.0 / np.linalg.norm(cur_vel)), 1)
                else:
                    cur_vel_norm = np.zeros_like(cur_vel)

                if np.linalg.norm(cur_vel_norm - old_vel_norm) == 0 and np.linalg.norm(old_vel_norm) > 0:
                    max_penalty = max_penalty

                elif np.linalg.norm(cur_vel_norm) > 0 and np.linalg.norm(old_vel_norm) > 0:

                    if np.dot(dif_1, dif_2) < 0:
                        cos_theta = np.dot(cur_vel_norm, old_vel_norm) / (
                        np.linalg.norm(old_vel_norm) * np.linalg.norm(cur_vel_norm))
                        # print "Cos  theta "+str(cos_theta)+" "+str(np.dot(cur_vel_norm, old_vel_norm))+" "+str((np.linalg.norm(old_vel_norm) * np.linalg.norm(cur_vel_norm)))
                        if cos_theta < 1.0:
                            max_penalty = np.sqrt((1 - cos_theta) / 2)
                            # else:
                            #     max_penalty =np.sin(np.pi/4)
        return max_penalty

    def measures_into_future(self,id, episode_done, frame, last_frame):
        agent = self.pedestrian_data[id]
        pos = agent.agent[frame + 1]
        if last_frame<0:
            last_frame=self.seq_len-1
        if self.useRealTimeEnv==False or episode_done:
        # Total travelled distance
        # Intercepting pedestrian_trajectory_fixed!
            if self.intercept_pedestrian_trajectory(id, pos, frame,
                                                    no_height=True):  # self.intercept_person(frame + 1, no_height=True, person_id=per_id):
                agent.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory] = 1
        if self.useRealTimeEnv and episode_done and not self.assume_known_ped_stats:
            # convolutional solution
            # import scipy.ndimage
            # self.heatmap = scipy.ndimage.gaussian_filter(np.sum(self.reconstruction[:, :, :, CHANNELS.pedestrian_trajectory], axis=0), 15)
            # Use estimate in place of heatmap!
            # print(" Frame "+str(frame)+" len of predictions "+str(len(self.agent_prediction_people)))

            current_prediction = None
            current_predcition_size = 0
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap] = None
            if len(agent.agent_prediction_people) > frame + 1:
                current_prediction = agent.agent_prediction_people[frame + 1]
                current_predcition_size = current_prediction.shape[0] * current_prediction.shape[1]
                agent.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap] = np.sum(
                    current_prediction != 0) / current_predcition_size
            else:
                if frame<last_frame-1:
                    print("Issue!!! Pedestrian is not observing pedestrian forecasts!!!-----------------------------------"+str(frame))

        if episode_done:
            if frame < self.seq_len + 1:
                for f in range(len(agent.agent) - 1, frame, -1):
                    if len(agent.agent[f]) > 0 and len(agent.agent[f - 1]) > 0:
                        agent.measures[frame, PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init] += np.linalg.norm(agent.agent[f] - agent.agent[f - 1])

            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_final_pos] = np.linalg.norm(agent.agent[-1][1:] - agent.agent[frame][1:]) if len(agent.agent[-1][1:]) > 0 and len(agent.agent[frame][1:]) > 0 else 0


    def end_of_episode_measures(self,frame, id):
        # Intercepting cars
        agent=self.pedestrian_data[id]
        pos=agent.agent[frame+1]

        intercept_cars_in_frame=self.intercept_car(pos, frame + 1, all_frames=False)

        agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car] = max(agent.measures[max(frame - 1, 0),PEDESTRIAN_MEASURES_INDX.hit_by_car],
                                      intercept_cars_in_frame)
        #print(" End of episode measures: "+str(frame)+" hit by car in current frame? " + str(intercept_cars_in_frame) + " hit by car measure: "+str(agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]) )
        #print (" Agent evaluate end of episode measures hit by car "+str(agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]))
        if self.follow_goal:  # If using a general goal

            # Distance to goal.
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal] = np.linalg.norm(np.array(agent.goal[frame,1:]) - agent.agent[frame + 1][1:])

            #print "Distance to goal: " + str(agent.measures[frame, 7]) + " goal pos: " + str(self.goal[1:]) + "current pos: " + str(agent.agent[frame][1:])

            # Reached goal?
            if agent.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal] <= np.sqrt(2):
                agent.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] = 1
                # else:
                #     agent.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] = agent.measures[
                #                                                                        frame, PEDESTRIAN_MEASURES_INDX.goal_reached] + 1
                # # print "Reached goal"


        per_id=-1
        if agent.init_method == PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian or agent.init_method == PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian:
            per_id = agent.goal_person_id


        collide_pedestrians = len(self.collide_with_pedestrians(id, pos,frame + 1, no_height=True, per_id=per_id)) > 0
        if collide_pedestrians:
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = 1
        if self.end_on_hit_by_pedestrians:
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]=max(agent.measures[max(frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_pedestrians],agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians])
        if self.useRealTimeEnv:
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_hero_car] = max(
                agent.measures[max(frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_by_hero_car],
                self.intercept_agent_car(pos,frame + 1, all_frames=False))
        # print(" End of episode measures-2 : " + str(frame) + " hit by car in current frame? " + str(
        #     intercept_cars_in_frame) + " hit by car measure: " + str(
        #     agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]))

    def time_dependent_evaluation(self,id, frame, person_id=-1):
        # Intercepting cars - nothing needs to change
        agent = self.pedestrian_data[id]
        pos = agent.agent[frame + 1]
        intercept_cars_in_frame=self.intercept_car(pos, frame + 1, all_frames=False)
        agent.measures[frame,PEDESTRIAN_MEASURES_INDX.hit_by_car] = max(agent.measures[max(frame - 1, 0),PEDESTRIAN_MEASURES_INDX.hit_by_car],intercept_cars_in_frame)
        #print(" Time dependent  measures: hit by car in current frame? " + str(intercept_cars_in_frame) + " hit by car measure: " + str(agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]))

        # print ("Frame "+str(frame)+" hit car: "+str(agent.measures[frame, measures_indx["hit_by_car"]]))
        per_id = -1
        if len(self.people_dict) > 0:
            per_id = agent.goal_person_id

        # Coincide with human trajectory
        agent.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory] = 0

        # Hit pedestrians
        if agent.init_method == PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian or agent.init_method == PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian :
            # ok assuming one frame
            if len(self.collide_with_pedestrians(id, pos, frame + 1, no_height=True,per_id=per_id)) > 0:  # self.intercept_person(frame + 1, no_height=True, person_id=per_id, frame_input=frame):

                agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = 1
        else:
            # ok assuming one frame
            if len(self.collide_with_pedestrians(id, pos,frame + 1, no_height=True)) > 0:
                agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = 1

        # If agent will be dead after hitting a pedestrian, then the measure i set to 1 after it has occured once.
        if self.end_on_hit_by_pedestrians:
            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = max(agent.measures[max(frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_pedestrians], agent.measures[frame, CAR_MEASURES_INDX.hit_pedestrians])

        # Measure distance to closest car (if using hero car distance to hero car!)
        agent.measures[frame, PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car]=self.evaluate_inverse_dist_to_car(pos, frame, agent.velocity[frame], 2*np.pi)#print ("Measure   " + str(agent.measures[frame, 17]))

        # Evaluate heatmap!
        if  self.useRealTimeEnv:
            # convolutional solution
            # import scipy.ndimage
            # self.heatmap = scipy.ndimage.gaussian_filter(np.sum(self.reconstruction[:, :, :, CHANNELS.pedestrian_trajectory], axis=0), 15)
            # Use estimate in place of heatmap!
            #print(" Frame "+str(frame)+" len of predictions "+str(len(self.agent_prediction_people)))

            # current_prediction = None
            # current_predcition_size = 0
            # agent.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap] = None
            # if len(agent.agent_prediction_people) > frame+1:
            #     current_prediction = agent.agent_prediction_people[frame+1]
            #     current_predcition_size = current_prediction.shape[0]*current_prediction.shape[1]
            #     agent.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap] = np.sum(current_prediction != 0) / current_predcition_size
            # else:
            #     print("Issue!!! Pedestrian is not observing pedestrian forecasts!!!-----------------------------------")
            #

            agent.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_hero_car] = max(
                agent.measures[max(frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_by_hero_car],
                self.intercept_agent_car(pos,frame + 1, all_frames=False))


    def evaluate_inverse_dist_to_car(self, pos,frame, velocity, field_of_view):
        closest_car, min_dist = self.find_closest_car(frame+1,pos, velocity, field_of_view)
        if len(closest_car) == 0:
            return 0
        elif min_dist <= self.agent_size[1]:
            return 1
        else:
            return 1.0 / copy.copy(min_dist)

    def on_pavement(self, pos):
        segmentation = self.person_bounding_box(pos, no_height=True)

        # 6-ground, 7-road, 8- sidewalk, 9-paring, 10- rail track
        for val in self.pavement:
            if np.any(segmentation == val):
                return True
        return False

    def intercept_agent_car(self, pos, frame_in, all_frames=False, agent_frame=-1, bbox=None,car=None):
        if bbox == None:
            if agent_frame < 0:
                x_range, y_range, z_range = self.pos_range(pos)
            else:
                x_range, y_range, z_range = self.pos_range(pos)
            overlapped = 0
            frames = []
            if all_frames or frame_in >= len(self.cars):
                frames = list(range(len(self.cars)))
            else:
                frames.append(frame_in)
            person_bbox = [x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]]
        else:
            person_bbox = bbox


        if car == None and self.use_car_agent:
            for car in self.car_data:
                car_bbx = car.car_bbox
                # print (" Car bbox "+str(self.car_bbox))

                # print(" Cars "+str(cars))
                for frame in frames:
                        # if evaluate_hero_car and len(self.car)> frame:
                    # print (" Hero  car " + str(car[frame][2:6])+" person bbox "+str(person_bbox[2:]))
                    if car_bbx[frame] != None and (overlap(car_bbx[frame][2:6], person_bbox[2:], 1) or overlap( person_bbox[2:],car_bbx[frame][2:6], 1)):
                        overlapped += 1

                # overlap_hero_car=True
        # print ("Overlapped "+str(overlapped))
        return overlapped

    def intercept_car(self, pos, frame_in, all_frames=False, agent_frame=-1, bbox=None,cars=None, car=None ):

        if bbox == None:
            if agent_frame<0:
                #print ("Intecept car get pedestrian frame "+str(frame_in) )
                x_range, y_range, z_range = self.pos_range(pos)
            else:
                x_range, y_range, z_range = self.pos_range(pos)
            overlapped = 0
            frames=[]
            if all_frames or frame_in>=len(self.cars):
                frames=list(range(len(self.cars)))
            else:
                frames.append(frame_in)
            person_bbox=[x_range[0],x_range[1], y_range[0],y_range[1],z_range[0], z_range[1]]
        else:
            person_bbox=bbox

        if cars==None:
            cars=self.cars
            #print (" Car bbox "+str(self.car_bbox))


        #print("Cars in frame "+str(frames))
        for frame in frames:
            for car_local in cars[frame]:
                car_locally=np.array(car_local).copy()
                car_locally[1]=car_locally[1]-1
                car_locally[3] = car_locally[3] - 1
                car_locally[5] = car_locally[5] - 1
                #print ("  car " + str(car_local[2:6])+" person bbox "+str(person_bbox[2:]))  #print ("  car " + str(car[frame][2:6])+" person bbox "+str(person_bbox[2:]))
                if overlap(car_locally[2:6],person_bbox[2:],1) or overlap(person_bbox[2:],car_locally[2:6],1):
                    overlapped += 1
                    #print ("Overlapped")
                # if evaluate_hero_car and len(self.car)> frame:


            #if self.use_car_agent:
            for car in self.car_data:
            # print(" Intercept hero car ? " + str(car[frame][2:6]))
                if car.car_bbox[frame] != None and (overlap(car.car_bbox[frame][2:6],person_bbox[2:],1) or overlap(person_bbox[2:],car.car_bbox[frame][2:6],1)) :
                    overlapped+=1
                        # print("Overlapped")

                #overlap_hero_car=True
        # print ("Final overlapped "+str(overlapped))
        return overlapped



    def pos_range(self, pos, as_int=True):

        if not len(pos) == 3:
            print(("Position "+str(pos)+" "+str(len(pos))))
        #print ("Position " + str(pos) + " ")
        # if (len(pos) == 0):
        #     pos = [0, 0, 0]

        x_range = [max(pos[0] - self.agent_size[0],0), min(pos[0] + self.agent_size[0], self.reconstruction.shape[0])]
        y_range = [max(pos[1] - self.agent_size[1], 0),min(pos[1] + self.agent_size[1], self.reconstruction.shape[1])]
        z_range = [max(pos[2] - self.agent_size[2],0),min(pos[2] + self.agent_size[2], self.reconstruction.shape[2])]
        if as_int:
            for i in range(2):
                x_range[i]=int(round(x_range[i]))
                y_range[i] = int(round(y_range[i]))
                z_range[i] = int(round(z_range[i]))

        return x_range, y_range, z_range

    def pose_range_car(self, frame_in, pos, as_int=True):

        if not len(pos) == 3:
            print(("Position "+str(pos)+" "+str(len(pos))))
        #print "Position " + str(pos) + " "
        x_range = [max(pos[0] - self.agent_size[0],0), min(pos[0] + self.agent_size[0], self.reconstruction.shape[0])]
        y_range = [max(pos[1] - self.agent_size[1], 0),min(pos[1] + self.agent_size[1], self.reconstruction.shape[1])]
        z_range = [max(pos[2] - self.agent_size[2],0),min(pos[2] + self.agent_size[2], self.reconstruction.shape[2])]
        if as_int:
            for i in range(2):
                x_range[i]=int(round(x_range[i]))
                y_range[i] = int(round(y_range[i]))
                z_range[i] = int(round(z_range[i]))

        return x_range, y_range, z_range

    def collide_with_people_dict(self,frame_in, people_overlapped, x_range, y_range, z_range, per_id):

        for person_key in list(self.people_dict.keys()):

            if self.key_map[per_id] != person_key:
                dif=frame_in-self.init_frames[person_key]

                if dif>=0 and len(self.people_dict[person_key])>dif:
                    person=self.people_dict[person_key][dif]

                    x_pers = [ min(person[1, :]), max(person[1, :]),
                              min(person[2, :]),
                              max(person[2, :])]

                    if overlap(x_pers, [ y_range[0], y_range[1], z_range[0], z_range[1]], 1):

                        people_overlapped.append(person)

                        return people_overlapped
        return people_overlapped

    def intercept_pedestrian_trajectory(self,id, pos, frame_in, no_height=False):
        x_range, y_range, z_range = self.pos_range(pos)
        if no_height:
            x_range = [0, self.reconstruction.shape[0]]
        if self.useRealTimeEnv: # how to make it per person? Avreage by agent size?
            if frame_in+1<len(self.pedestrian_data[id].agent_prediction_people):
                trajectory_current_frame=self.pedestrian_data[id].agent_prediction_people[frame_in+1]
                center=[int(trajectory_current_frame.shape[0]*0.5),int(trajectory_current_frame.shape[1]*0.5 )]
                agent_view=trajectory_current_frame[center[0]-self.agent_size[1]:center[0]+self.agent_size[1]+1, center[1]-self.agent_size[2]:center[1]-self.agent_size[2]+1]
                traj=agent_view!=0
                traj_sum=np.sum(traj)
                return traj_sum
            return 0
        people_overlapped = []
        for frame in range(len(self.people)):
            people_overlapped,collide_with_agents=collide_with_people( frame, people_overlapped, x_range, y_range, z_range, self.people,[], self.agent_size)
        return people_overlapped

    def intercept_pedestrian_trajectory_tensor(self, frame_in, no_height=False):
        x_range, y_range, z_range = self.pos_range(frame_in, as_int=True)
        if no_height:
            x_range = [0, self.reconstruction.shape[0]]

        return np.sum(self.reconstruction[ x_range[0]: x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1],CHANNELS.pedestrian_trajectory ])>0

    def collide_with_pedestrians(self, id, pos, frame_in, no_height=False, per_id=-1, agent_frame=-1):
        people_overlapped = []
        x_range, y_range, z_range = self.pos_range(pos)
        if no_height:
            x_range = [0, self.reconstruction.shape[0]]
        for local_per_id, person in enumerate(self.pedestrian_data):
            if id!=local_per_id:
                bbox=[person.agent[frame_in][0]-self.agent_size[0],
                      person.agent[frame_in][0]+self.agent_size[0],
                      person.agent[frame_in][1]-self.agent_size[1],
                      person.agent[frame_in][1]+self.agent_size[1],
                      person.agent[frame_in][2]-self.agent_size[2],
                      person.agent[frame_in][2]+self.agent_size[2]]
                if overlap(bbox[2:], [y_range[0], y_range[1], z_range[0], z_range[1]], 1):
                    people_overlapped.append(bbox)
        if per_id<0:
            people_overlapped, collide_with_agents =collide_with_people(frame_in, people_overlapped, x_range, y_range, z_range, self.people, [], self.agent_size)
            #frame_input, people_overlapped, x_range, y_range, z_range, people,pedestrian_data,agent_size
            return people_overlapped

        return self.collide_with_people_dict(frame_in, people_overlapped, x_range, y_range, z_range, per_id)


    def pose_dist(self, pose1, pose2):
        dif = pose1 - np.mean(pose2)
        return np.linalg.norm(dif) / 14

    def intercept_objects(self, pos, no_height=False, cars=False, reconstruction=None,bbox=[]):  # Normalisera/ binar!
        segmentation = self.person_bounding_box(pos, no_height=no_height, bbox=bbox)

        #count=((10 < segmentation) & (segmentation < 21) ).sum()
        if self.new_carla:
            obstacles = OBSTACLE_LABELS_NEW
        else:
            obstacles = OBSTACLE_LABELS
        #print(" Obstacle class " + str(obstacles))
        count = 0
        for label in obstacles:
            count += (segmentation == label).sum()

        if cars:
            for label in MOVING_OBSTACLE_LABELS:
                count += (segmentation == label).sum()
            print(" Moving class " + str(MOVING_OBSTACLE_LABELS))
            #count += (segmentation > 25).sum()
        if count > 0:
            if len(bbox)>0:
                return count
            x_range, y_range, z_range = self.pos_range(pos)
            if (x_range[1]+2-x_range[0])*(y_range[1]+2-y_range[0])>0:
                #print np.histogram(segmentation.flatten())
                return count *1.0/((x_range[1]+2-x_range[0])*(y_range[1]+2-y_range[0]))#*(z_range[1]+2-z_range[0]))
            else:
                return 0
        return 0

        # To do: add cars!

    def valid_position(self, pos, no_height=True,bbox=[]):

        objs = self.intercept_objects(pos=pos, no_height=no_height,bbox=bbox)
        #print ("Intercept objects " + str(objs))
        return objs <=0  #Was not 0 due to noise previously

    def out_of_axis(self,current_pos):
        directions=[False, False]
        for indx in range(1,3):
            #print "Dir "+str(indx)+" Max agent: "+str(agent.agent[frame_in][indx]+self.agent_size[indx])+" Min: "+str(agent.agent[frame_in][indx]-self.agent_size[indx])
            if (round(current_pos[indx]+self.agent_size[indx]))>=self.reconstruction.shape[indx] or round(current_pos[indx]-self.agent_size[indx])<0:
                directions=True

        return np.array(directions).any()


    def discounted_reward(self, frame):
       for pedestrian in self.pedestrian_data:
           pedestrian.discounted_reward(self.gamma,frame)

       for init in self.initializer_data:
           init.discounted_reward(self.gamma,frame)

       for init_car in self.initializer_car_data:
           init_car.discounted_reward(self.gamma, frame)

       for car in self.car_data:
           car.discounted_reward(self.gamma,frame)

    # Note: This is outdated!!!
    def get_people_in_frame(self, frame_in, needPeopleNames = False):
        people=[]
        people_names=[]
        for person_key in list(self.people_dict.keys()):

            if self.pedestrian_data[id].goal_person_id_val != person_key:

                dif = frame_in - self.init_frames[person_key]

                if dif >= 0 and len(self.people_dict[person_key]) > dif:
                    person = self.people_dict[person_key][dif]
                    person_prev= self.people_dict[person_key][max(dif-1, 0)]

                    x_pers = [np.mean(person[2, :]),np.mean(person[1, :]),np.mean(person[2, :]-person_prev[2,:]),
                              np.mean(person[1, :]-person_prev[1,:]),np.mean([person[2, 1]-person[2, 0], person[1, 1]-person[1, 0]]) ]
                    people.append(np.array(x_pers).astype(int))
                    people_names.append(person_key)
        if needPeopleNames: # To keep compatibility...
            return people, people_names
        else:
            return people


    def remove_current_frame(self, image, frame):
        current_timestep_mask= image==frame
        mask=image>0
        current_timestep_mask=np.logical_and(current_timestep_mask, mask)
        image[mask]=image[mask]-frame
        image[current_timestep_mask]=1
        return image

    def get_agent_neighbourhood(self,id, pos, breadth,frame_in,vel=[], training=True, eval=True, pedestrian_view_occluded=False,temporal_scaling=0.1*0.3, field_of_view=2*np.pi):
        #temporal_scaling=0.1*0.3 # what scaling to have on temporal input to stay in range [-1,1]
        # print("Episode get agent neighborhood! ")
        start_pos = np.zeros(3, np.int)
        min_pos=np.zeros(3, np.int)
        max_pos = np.zeros(3, np.int)
        if len(breadth)==2:
            breadth=[self.reconstruction.shape[0]/2+1, breadth[0], breadth[1]]

        # Get 2D or 3D input? Create holders
        if self.run_2D:
            mini_ten = np.zeros(( breadth[1] * 2 + 1, breadth[2] * 2 + 1, 6), dtype=np.float)
            tmp_people = np.zeros(( mini_ten.shape[0], mini_ten.shape[1], 1))
            tmp_cars = np.zeros(( mini_ten.shape[0], mini_ten.shape[1], 1))
            tmp_people_cv = np.zeros(( mini_ten.shape[0], mini_ten.shape[1], 1))
            tmp_cars_cv =np.zeros(( mini_ten.shape[0], mini_ten.shape[1], 1))
        else:
            mini_ten = np.zeros((breadth[0] * 2 + 1, breadth[1] * 2 + 1, breadth[2] * 2 + 1, 6), dtype=np.float)
            tmp_people = np.zeros(( mini_ten.shape[0],  mini_ten.shape[1],  mini_ten.shape[2], 1))
            tmp_cars = np.zeros(( mini_ten.shape[0],  mini_ten.shape[1],  mini_ten.shape[2], 1))
            tmp_people_cv = np.zeros((mini_ten.shape[0], mini_ten.shape[1], mini_ten.shape[2], 1))
            tmp_cars_cv = np.zeros((mini_ten.shape[0], mini_ten.shape[1], mini_ten.shape[2], 1))

        # Range of state

        for i in range(3):
            min_pos[i]=round(pos[i])-breadth[i]
            max_pos[i] = round(pos[i]) + breadth[i] +1
            # print("Rounded positions "+str(round(pos[i]))+" breadth "+str(breadth[i]))

            if min_pos[i]<0: # agent below rectangle of visible.
                start_pos[i] = -min_pos[i]
                min_pos[i]=0
                if max_pos[i]<0:
                    if self.useRealTimeEnv:
                        self.add_predicted_people(id,frame_in, tmp_people)
                    return mini_ten, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv
            if max_pos[i]>self.reconstruction.shape[i]:
                max_pos[i]=self.reconstruction.shape[i]
                if min_pos[i]>self.reconstruction.shape[i]:
                    if self.useRealTimeEnv:
                        self.add_predicted_people(id,frame_in, tmp_people)
                    return mini_ten, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv
        # print (" Agent dimensions: "+str([ min_pos[1],max_pos[1], min_pos[2],max_pos[2]]))
        # if self.useRealTimeEnv:
        #     print (" cars "+str(self.car_bbox[frame_in])+" velocity "+str(self.velocity_car[frame_in-1 ]))
        # print (" cars "+str(self.cars[frame_in]))
        # print (" pedestrians " + " cars " + str(self.people[frame_in]))
        # Get reconstruction cut out
        if self.run_2D:
            tmp = self.reconstruction_2D[ min_pos[1]:max_pos[1], min_pos[2]:max_pos[2], :].copy()
        else:
            tmp=self.reconstruction[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], min_pos[2]:max_pos[2], :].copy()

        max_len_people=len(self.people_predicted)-1
        # constant velocity prediction
        if self.run_2D:
            # Get constant velocity prediction
            tmp_people_cv[ start_pos[1]:start_pos[1] + tmp.shape[0],start_pos[2]:start_pos[2] + tmp.shape[1], :] = self.people_predicted[min(frame_in, max_len_people)][
                                                           min_pos[1]:max_pos[1], min_pos[2]:max_pos[2],np.newaxis].copy()
            tmp_cars_cv[ start_pos[1]:start_pos[1] + tmp.shape[0],start_pos[2]:start_pos[2] + tmp.shape[1], :] = self.cars_predicted[min(frame_in, max_len_people)][
                                                           min_pos[1]:max_pos[1], min_pos[2]:max_pos[2], np.newaxis].copy()
            if self.useRealTimeEnv:
                tmp_people_cv=self.remove_current_frame(tmp_people_cv, frame_in+1)
                tmp_cars_cv=self.remove_current_frame(tmp_cars_cv, frame_in+1)

                bounding_box=[min_pos[1],max_pos[1], min_pos[2],max_pos[2]]
                # print (" Bounding box "+str(bounding_box))
                tmp_people_cv, tmp_cars_cv= self.predict_cars_and_people_in_a_bounding_box(id,tmp_people_cv,tmp_cars_cv,bounding_box, start_pos,frame_in)
                tmp_people_cv = np.expand_dims(tmp_people_cv, axis=2)
                tmp_cars_cv = np.expand_dims(tmp_cars_cv, axis=2)
                # print ("Sum of people traj "+str(np.sum(abs(tmp_people_cv)))+" Sum of car traj "+str(np.sum(abs(tmp_cars_cv))))
            # Insert reconstruction into holder
            mini_ten[ start_pos[1]:start_pos[1] + tmp.shape[0],start_pos[2]:start_pos[2] + tmp.shape[1], :] = tmp

        else: # This is outdated.
            # Get constant velocity prediction
            tmp_people_cv[start_pos[0]:start_pos[0]+tmp.shape[0],start_pos[1]:start_pos[1]+tmp.shape[1],start_pos[2]:start_pos[2]+tmp.shape[2], :]=self.people_predicted[min(frame_in, max_len_people)][min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], min_pos[2]:max_pos[2],np.newaxis].copy()
            tmp_cars_cv[start_pos[0]:start_pos[0] + tmp.shape[0], start_pos[1]:start_pos[1] + tmp.shape[1],
            start_pos[2]:start_pos[2] + tmp.shape[2], :] = self.cars_predicted[min(frame_in, max_len_people)][min_pos[0]:max_pos[0],
                                                           min_pos[1]:max_pos[1], min_pos[2]:max_pos[2] , np.newaxis].copy()
            # Insert reconstruction into holder
            mini_ten[start_pos[0]:start_pos[0]+tmp.shape[0],start_pos[1]:start_pos[1]+tmp.shape[1],start_pos[2]:start_pos[2]+tmp.shape[2], :]=tmp

        # Bounds of the bounding box
        bbox=[min_pos[0],max_pos[0], min_pos[1],max_pos[1], min_pos[2],max_pos[2]]

        # Add all pedestrians in loc_frame into pedestrian layer
        loc_frame=frame_in

        if self.pedestrian_data[id].goal_person_id>=0:
            for person_key in list(self.people_dict.keys()):

                dif = frame_in - self.init_frames[person_key]
                if dif >= 0 and len(self.people_dict[person_key]) > dif: # If pedestrian is in frame
                    person = self.people_dict[person_key][dif] # Get pedestrian position in current frame
                    x_pers = [min(person[0, :]), max(person[0, :])+1, min(person[1, :]), max(person[1, :])+1,
                              min(person[2, :]),
                              max(person[2, :])+1]
                    if overlap(x_pers[2:], bbox[2:], 1) or self.temporal: # Check if overlapping
                        # print ("Person goal id " + str(x_pers)+"  second "+str(bbox))
                        intersection = [max(x_pers[0], bbox[0]).astype(int), min(x_pers[1], bbox[1]).astype(int), max(x_pers[2], bbox[2]).astype(int),
                                        min(x_pers[3], bbox[3]).astype(int),
                                        max(x_pers[4], bbox[4]).astype(int), min(x_pers[5], bbox[5]).astype(int)]

                        #print (" Intercsetion before goal id " + str(intersection))
                        intersection[0:2] = intersection[0:2] - bbox[0]
                        intersection[2:4] = intersection[2:4] - bbox[2]
                        intersection[4:] = intersection[4:] - bbox[4]
                        #print (" Intercsetion goal id " + str(intersection))
                        if self.pedestrian_data[id].goal_person_id_val != person_key:
                            if self.run_2D:
                                tmp_people[ intersection[2]:intersection[3],
                                intersection[4]:intersection[5], 0] = 1
                            else:
                                tmp_people[intersection[0]:intersection[1], intersection[2]:intersection[3],
                                intersection[4]:intersection[5], 0] = 1
                        elif self.useRealTimeEnv:
                            self.pedestrian_data[id].agent_prediction_people[frame_in][ intersection[2]:intersection[3],
                                intersection[4]:intersection[5], 0] = 1


        else:
            if len(self.people) > loc_frame:
                for person in self.people[loc_frame]:
                    x_pers = [min(person[0, :]), max(person[0, :])+1, min(person[1, :]), max(person[1, :])+1, min(person[2, :]),
                              max(person[2, :])+1]

                    if overlap(x_pers[2:],bbox[2:] , 1) or self.temporal:
                        # if overlap(x_pers[2:],bbox[2:] , 1):
                        #     print ("Person " + str(x_pers)+"  second "+str(bbox)+" temporal: "+str(self.temporal)+" overlap "+str(overlap(x_pers[2:],bbox[2:] , 1)))
                        intersection = [max(x_pers[0], bbox[0]).astype(int), min(x_pers[1], bbox[1]).astype(int), max(x_pers[2], bbox[2]).astype(int), min(x_pers[3], bbox[3]).astype(int),
                                        max(x_pers[4], bbox[4]).astype(int), min(x_pers[5], bbox[5]).astype(int)]

                        #print (" Intercsetion before " + str(intersection))
                        intersection[0:2] = intersection[0:2] - bbox[0]
                        intersection[2:4] = intersection[2:4] - bbox[2]
                        intersection[4:] = intersection[4:] - bbox[4]
                        #print (" Intercsetion " + str(intersection))
                        if self.run_2D:
                            tmp_people[ intersection[2]:intersection[3],
                            intersection[4]:intersection[5], 0] = 1
                        else:
                            tmp_people[intersection[0]:intersection[1], intersection[2]:intersection[3],
                            intersection[4]:intersection[5], 0] = 1
        for per_id,pedestrian in enumerate(self.pedestrian_data):
            if id!=per_id:
                person=pedestrian.agent[loc_frame]
                x_pers = np.array([person[0]-self.agent_size[0], person[0] +self.agent_size[0]+ 1, person[1]-self.agent_size[1], person[1]+self.agent_size[1] + 1,
                          person[2]-self.agent_size[2],
                          person[2]+self.agent_size[2] + 1])

                if overlap(x_pers[2:], bbox[2:], 1) or self.temporal:
                    # if overlap(x_pers[2:],bbox[2:] , 1):
                    #     print ("Person " + str(x_pers)+"  second "+str(bbox)+" temporal: "+str(self.temporal)+" overlap "+str(overlap(x_pers[2:],bbox[2:] , 1)))
                    intersection = np.array([int(max(x_pers[0], bbox[0])), int(min(x_pers[1], bbox[1])),
                                    int(max(x_pers[2], bbox[2])), int(min(x_pers[3], bbox[3])),
                                    int(max(x_pers[4], bbox[4])), int(min(x_pers[5], bbox[5]))])

                    # print (" Intercsetion before " + str(intersection))
                    intersection[0:2] = intersection[0:2] - bbox[0]
                    intersection[2:4] = intersection[2:4] - bbox[2]
                    intersection[4:] = intersection[4:] - bbox[4]
                    # print (" Intercsetion " + str(intersection))
                    if self.run_2D:
                        tmp_people[intersection[2]:intersection[3],
                        intersection[4]:intersection[5], 0] = 1
                    else:
                        tmp_people[intersection[0]:intersection[1], intersection[2]:intersection[3],
                        intersection[4]:intersection[5], 0] = 1


        # Add colliding cars in loc_frame into colliding cars layer.
        cars_in_scene=copy.copy(self.cars[min(loc_frame, max_len_people)])

        for car in self.car_data:
            local_car=np.copy(car.car_bbox[min(loc_frame, max_len_people)])
            local_car[1]=local_car[1]+1
            local_car[3] = local_car[3] + 1
            local_car[5] = local_car[5] + 1
            cars_in_scene.append(car.car_bbox[min(loc_frame, max_len_people)])
        for obj in cars_in_scene:
            if (obj is not None) and (overlap(obj[2:], bbox[2:], 1)or self.temporal):
                intersection= [int(max(obj[0], bbox[0])),int(min(obj[1], bbox[1]+1)),int(max(obj[2], bbox[2])),int(min(obj[3], bbox[3]+1)),int(max(obj[4], bbox[4])),int(min(obj[5], bbox[5]+1))]
                intersection[0:2]=intersection[0:2]-bbox[0]
                intersection[2:4] = intersection[2:4] - bbox[2]
                intersection[4:] = intersection[4:] - bbox[4]
                if self.run_2D:
                    tmp_cars[ intersection[2]:intersection[3],intersection[4]:intersection[5], 0] = 1
                else:
                    tmp_cars[intersection[0]:intersection[1], intersection[2]:intersection[3],
                    intersection[4]:intersection[5], 0] = 1
        if self.useRealTimeEnv:
            time=(frame_in+1)
        else:
            time=frame_in
        if self.temporal:
            if self.run_2D: # GT- pedestrian locations
                temp=mini_ten[:,:,CHANNELS.cars_trajectory]>0 # All places where pedestrians have ever been

                mini_ten[:,:,CHANNELS.cars_trajectory] = mini_ten[:,:,CHANNELS.cars_trajectory]-(time*temp) # Remove current frame
                mini_ten[:,:,CHANNELS.cars_trajectory]=temporal_scaling*mini_ten[:,:,CHANNELS.cars_trajectory] # Add scaling
                temp = mini_ten[:, :, CHANNELS.pedestrian_trajectory] > 0
                mini_ten[ :, :, CHANNELS.pedestrian_trajectory] = mini_ten[ :, :, CHANNELS.pedestrian_trajectory] - (time * temp)
                mini_ten[ :, :, CHANNELS.pedestrian_trajectory] = temporal_scaling * mini_ten[ :, :, CHANNELS.pedestrian_trajectory]
            else:
                temp = mini_ten[:, :, :, CHANNELS.cars_trajectory] > 0
                mini_ten[:, :, :, CHANNELS.cars_trajectory] = mini_ten[:, :, :, CHANNELS.cars_trajectory] - (time * temp)
                mini_ten[:, :, :, CHANNELS.cars_trajectory] = temporal_scaling * mini_ten[:, :, :, CHANNELS.cars_trajectory]
                temp = mini_ten[:, :, :, CHANNELS.pedestrian_trajectory] > 0
                mini_ten[:, :, :, CHANNELS.pedestrian_trajectory] = mini_ten[:, :, :, CHANNELS.pedestrian_trajectory] - (time * temp)
                mini_ten[:, :, :, CHANNELS.pedestrian_trajectory] = temporal_scaling * mini_ten[:, :, :, CHANNELS.pedestrian_trajectory]
            tmp_people_cv=temporal_scaling*tmp_people_cv
            tmp_cars_cv=temporal_scaling*tmp_cars_cv

        # elif self.predict_future:
        #     tmp_people_cv = 10*temporal_scaling * tmp_people_cv
        #     tmp_cars_cv = 10*temporal_scaling * tmp_cars_cv
        if pedestrian_view_occluded:

            free_of_static_objs=mini_ten[:,:,CHANNELS.semantic]==0
            free_of_cars=np.squeeze(tmp_cars==0)
            free_of_people=np.squeeze(tmp_people==0)
            valid_positions=np.logical_and(free_of_static_objs,free_of_cars)
            valid_positions=np.logical_and(valid_positions, free_of_people)

            position=np.array(pos)-np.array(breadth)+1
            occlusion_map = find_occlusions(valid_positions, valid_positions, position, np.linalg.norm(vel[1:]), vel[1:], 0, mini_ten.shape, self.lidar_occlusion, max_angle=field_of_view)
            occlusion_map_tiled=np.tile(occlusion_map.reshape(occlusion_map.shape[0],occlusion_map.shape[1],1), 6)
            mini_ten[occlusion_map_tiled==1]=0
            tmp_people[occlusion_map == 1] = 0
            tmp_cars[occlusion_map == 1] = 0
            tmp_people_cv[occlusion_map == 1] = 0
            tmp_cars_cv[occlusion_map == 1] = 0

        #print(" temporal cars after scaling "+str(np.sum(np.abs(tmp_cars_cv)))+" people "+str(np.sum(np.abs(tmp_people_cv)))+" scaling "+str(temporal_scaling))

        return mini_ten, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv


    #@profilemem(stream=memoryLogFP_statisticssave)

    def save(self, statistics, num_episode, poses,initialization_map,initialization_car,statistics_car, initialization_map_car,initialization_goal, ped_seq_len=-1):
        if ped_seq_len<0:
            ped_seq_len=self.seq_len
        # TO DO: Can copys be removed?
        for id in range(self.number_of_agents):
            agent=self.pedestrian_data[id]
            initializer=self.initializer_data[id]
            statistics[num_episode,id,  :ped_seq_len, STATISTICS_INDX.agent_pos[0]:STATISTICS_INDX.agent_pos[1]] = np.vstack(agent.agent[:ped_seq_len]).copy()
            statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]] = np.vstack(agent.velocity[:ped_seq_len]).copy()
            statistics[num_episode,id, -1, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]] = copy.copy(agent.vel_init)
            statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.action] = np.vstack(agent.action[:ped_seq_len]).reshape(ped_seq_len)
            statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.probabilities[0]:STATISTICS_INDX.probabilities[1]] = np.vstack(agent.probabilities[:ped_seq_len]).copy()
            if np.sum(agent.angle)>0:
                statistics[num_episode,id,:ped_seq_len, STATISTICS_INDX.angle] = agent.angle[:ped_seq_len].copy()
            statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.reward] = agent.reward[:ped_seq_len].copy()
            statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.reward_d] = agent.reward_d[:ped_seq_len].copy()
            statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.loss] = agent.loss[:ped_seq_len]
            statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.speed] = np.vstack(agent.speed[:ped_seq_len]).reshape(ped_seq_len )
            statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.measures[0]:STATISTICS_INDX.measures[1]] = agent.measures[:ped_seq_len].copy()
            #print (" Agent distracted: "+str(statistics[num_episode, :ped_seq_len, STATISTICS_INDX.measures[0]+PEDESTRIAN_MEASURES_INDX.distracted])+" save to file ")
            # print "Save hit by car: "+str(statistics[num_episode, :, 38])
            #  if agent.goal_person_id>=0:
            #      statistics[num_episode, 0:3, 38 + NBR_MEASURES] = np.mean(self.people_dict[agent.goal_person_id][-1], axis=1)
            #print "Save init method "+str(agent.init_method)+" "+str( 38+self.nbr_measures)
            statistics[num_episode,id, 0, STATISTICS_INDX.init_method] = agent.init_method



            if self.follow_goal and len(agent.goal)>0:
                statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.goal[0]:STATISTICS_INDX.goal[1]] = np.vstack(agent.goal[:ped_seq_len,1:]).copy()
                if np.sum(agent.goal_time)>0:
                    statistics[num_episode,id, :ped_seq_len, STATISTICS_INDX.goal_time] = np.squeeze(agent.goal_time)
                if agent.goal_person_id >=0:
                    statistics[num_episode,id, 6, STATISTICS_INDX.goal_person_id] = agent.goal_person_id

            if self.use_pfnn_agent:
                poses[num_episode,id, :, STATISTICS_INDX_POSE.pose[0]:STATISTICS_INDX_POSE.pose[1]]=copy.deepcopy(agent.agent_pose)
                poses[num_episode,id, :, STATISTICS_INDX_POSE.agent_high_frq_pos[0]:STATISTICS_INDX_POSE.agent_high_frq_pos[1]]=copy.deepcopy(agent.agent_high_frq_pos)
                poses[num_episode,id, :, STATISTICS_INDX_POSE.agent_pose_frames] = copy.deepcopy(agent.agent_pose_frames)
                poses[num_episode,id, :, STATISTICS_INDX_POSE.avg_speed] = copy.deepcopy(agent.avg_speed)
                poses[num_episode,id, :, STATISTICS_INDX_POSE.agent_pose_hidden[0]:STATISTICS_INDX_POSE.agent_pose_hidden[1]] = copy.deepcopy(agent.agent_pose_hidden)
                # print("Saved poses . "+str(num_episode) +"index "+str(STATISTICS_INDX_POSE.agent_pose_hidden[0])+" - "+str(STATISTICS_INDX_POSE.agent_pose_hidden[1])+" "+str(np.sum(np.abs(poses[num_episode, 0, STATISTICS_INDX_POSE.agent_pose_hidden[0]:STATISTICS_INDX_POSE.agent_pose_hidden[1]]))) )
            # if self.env_nbr > 5 and self.env_nbr<8:
            #     statistics[num_episode, 2, 38+NBR_MEASURES] =self.num_crossings
            #     statistics[num_episode, 3:5, 38+NBR_MEASURES]=self.point
            #     statistics[num_episode, 5:5+len(self.thetas), 38+NBR_MEASURES] = self.thetas
            if len(initialization_map)>0 and agent.init_method == 7:
                statistics[num_episode, id, :ped_seq_len, STATISTICS_INDX.reward_initializer] = initializer.reward[:ped_seq_len].copy()
                statistics[num_episode, id, :ped_seq_len, STATISTICS_INDX.reward_initializer_d] = initializer.reward_d[:ped_seq_len].copy()
                initialization_map[num_episode,id, :len(initializer.prior.flatten()), STATISTICS_INDX_MAP.prior] = initializer.prior.flatten()
                initialization_map[num_episode,id, :len(initializer.init_distribution), STATISTICS_INDX_MAP.init_distribution] = initializer.init_distribution
                if self.learn_goal:
                    for frame in range(len(initializer.goal_priors)):
                        initialization_goal[num_episode][id].append(np.zeros((len(initializer.goal_priors[frame]) ,NBR_MAPS_GOAL), dtype=np.float64))
                        initialization_goal[num_episode][id][:,STATISTICS_INDX_MAP_GOAL.goal_prior]=initializer.goal_priors[frame].flatten()
                        initialization_goal[num_episode][id][:,STATISTICS_INDX_MAP_GOAL.goal_distribution]=initializer.goal_distributions[frame].flatten()
                        statistics[num_episode, id, 1:len(initializer.frames_of_goal_change),STATISTICS_INDX.frames_of_goal_change]=initializer.frames_of_goal_change[1:].flatten()
                    # initialization_map[num_episode,id, :len(initializer.goal_prior.flatten()), STATISTICS_INDX_MAP.goal_prior] = initializer.goal_prior.flatten()
                    # initialization_map[num_episode,id, :len(initializer.goal_distribution.flatten()), STATISTICS_INDX_MAP.goal_distribution] = initializer.goal_distribution.flatten()

                initialization_car[num_episode,id, STATISTICS_INDX_CAR_INIT.car_id] = initializer.init_car_id
                initialization_car[num_episode,id,
                STATISTICS_INDX_CAR_INIT.car_pos[0]:STATISTICS_INDX_CAR_INIT.car_pos[1]] = initializer.init_car_pos
                initialization_car[num_episode,id,
                STATISTICS_INDX_CAR_INIT.car_vel[0]:STATISTICS_INDX_CAR_INIT.car_vel[1]] = initializer.init_car_vel
                if self.learn_goal:
                    initialization_car[num_episode,id, 5:5 + len(initializer.manual_goal)] = initializer.manual_goal

        if self.use_car_agent:
            for id in range(self.number_of_car_agents):
                car = self.car_data[id]

                if len(initialization_map_car) > 0:
                    initializer_car =self.initializer_car_data[id]
                    statistics_car[num_episode, id, :ped_seq_len, STATISTICS_INDX_CAR.reward_initializer] = initializer_car.reward_car[
                                                                                                :ped_seq_len].copy()
                    statistics_car[num_episode, id, :ped_seq_len, STATISTICS_INDX_CAR.reward_initializer_d] = initializer_car.reward_car_d[
                                                                                                  :ped_seq_len].copy()
                    initialization_map_car[num_episode, id, :len(initializer_car.prior.flatten()),STATISTICS_INDX_MAP_CAR.prior] = initializer_car.prior.flatten()
                    initialization_map_car[num_episode, id, :len(initializer_car.init_distribution),STATISTICS_INDX_MAP_CAR.init_distribution] = initializer_car.init_distribution
                statistics_car[num_episode,id, :ped_seq_len , STATISTICS_INDX_CAR.agent_pos[0]:STATISTICS_INDX_CAR.agent_pos[1]] = np.vstack(car.car[:ped_seq_len]).copy()

                statistics_car[num_episode,id, :ped_seq_len , STATISTICS_INDX_CAR.velocity[0]:STATISTICS_INDX_CAR.velocity[1]] = np.vstack(car.velocity_car[:ped_seq_len ]).copy()
                statistics_car[num_episode,id, :ped_seq_len, STATISTICS_INDX_CAR.action] = car.action_car[:ped_seq_len].copy()
                #print (" Size before addition "+str(np.squeeze(car.probabilities_car[:ped_seq_len - 1]).shape)+" after "+str(statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.probabilities].shape))
                statistics_car[num_episode,id, :ped_seq_len , STATISTICS_INDX_CAR.probabilities[0]:STATISTICS_INDX_CAR.probabilities[1]] = np.squeeze(car.probabilities_car[:ped_seq_len ,:])


                statistics_car[num_episode,id, :len(car.car_goal), STATISTICS_INDX_CAR.goal] = car.car_goal.copy()
                statistics_car[num_episode,id, len(car.car_goal):len(car.car_goal)+len(car.car_dir), STATISTICS_INDX_CAR.goal] = car.car_dir.copy()


                statistics_car[num_episode,id, :ped_seq_len , STATISTICS_INDX_CAR.reward] = car.reward_car[:ped_seq_len ].copy()
                statistics_car[num_episode,id, :ped_seq_len , STATISTICS_INDX_CAR.reward_d] = car.reward_car_d[:ped_seq_len ].copy()
                #statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.loss] = car.loss_car[:ped_seq_len - 1]
                #print(" Save car loss "+str(statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.loss])+" pos "+str(STATISTICS_INDX_CAR.loss)+" "+str(num_episode))

                # if ped_seq_len > 1 and len(self.pedestrian_data[id].agent[1]) == 0:
                #     return
                statistics_car[num_episode,id, :ped_seq_len , STATISTICS_INDX_CAR.speed] = np.vstack(car.speed_car[:ped_seq_len ]).reshape(
                    ped_seq_len)


                statistics_car[num_episode,id, :ped_seq_len, STATISTICS_INDX_CAR.bbox[0]:STATISTICS_INDX_CAR.bbox[1]] = np.vstack(car.car_bbox[:ped_seq_len ]).copy()
                statistics_car[num_episode,id, :ped_seq_len, STATISTICS_INDX_CAR.measures[0]:STATISTICS_INDX_CAR.measures[1]] = car.measures_car[:ped_seq_len ].copy()
                statistics_car[num_episode,id, :ped_seq_len , STATISTICS_INDX_CAR.angle] = car.car_angle[:ped_seq_len ].copy()

        return statistics, poses, initialization_map, initialization_car, statistics_car

    def save_loss(self, statistics):

        statistics[:, :, 36] =self.loss#np.copy( self.loss)
        return statistics




    def get_input_cars(self, pos, frame, velocity, field_of_view, distracted=False):
        feature=np.zeros([1,len(self.actions)-1], dtype=np.float32)

        closest_car, min_dist = self.find_closest_car(frame, pos, velocity, field_of_view)
        #print(" Get input cars distracted "+str(distracted))
        if len(closest_car)==0 or distracted:
            return feature
        car_dir=closest_car-np.array(pos)
        car_dir[0] = 0
        dir = self.find_action_to_direction(car_dir, min_dist)
        if dir>4:
            dir=dir-1
        feature[0,dir]=min_dist/np.linalg.norm(self.reconstruction.shape[0:3])


        return feature

    def object_along_path(self,id, init_pos, end_pos, frame):
        occluded=False
        vec=end_pos-init_pos
        vec=vec *(1/np.linalg.norm(vec))
        cur_pos=init_pos
        while np.linalg.norm(end_pos-cur_pos)>1:
            if not self.valid_position(cur_pos):
                return True
            cur_pos=cur_pos+vec

        #for pedestrian in self.people[frame]:


    def find_closest_car(self, frame, pos, velocity, field_of_view, ignore_car_agent=False):
        closest_car, min_dist = find_closest_car_in_list(frame, pos,self, self.agent_size, velocity, field_of_view, is_fake_episode=False)

        if self.use_car_agent and not ignore_car_agent:
            closest_car_agent, min_dist_agent = find_closest_controllable_car(frame, pos,self, self.agent_size, velocity, field_of_view, is_fake_episode=False)
            if min_dist_agent<=min_dist or min_dist<0:
                return  closest_car_agent, min_dist_agent
        return closest_car, min_dist


    def get_input_cars_smooth(self, pos, frame, velocity, field_of_view, distracted=False):
        feature=np.zeros([1,len(self.actions)-1], dtype=np.float32)

        closest_car, min_dist = self.find_closest_car(frame, pos, velocity, field_of_view)
        if len(closest_car)==0 or distracted:
            return feature
        car_dir=closest_car-np.array(pos)
        car_dir[0] = 0
        feature[0,:] = self.find_action_to_direction_smooth(car_dir, min_dist)*(1.0/np.linalg.norm(self.reconstruction.shape[0:3]))
        #print(" Scale car proximty feature dividing by"+str(np.linalg.norm(self.reconstruction.shape[0:3])))
        return feature

    def get_input_cars_cont_linear(self, pos, frame, velocity, field_of_view, distracted=False):
        feature = np.zeros([1, 2], dtype=np.float32)

        closest_car, min_dist = self.find_closest_car(frame, pos, velocity, field_of_view)
        if len(closest_car) == 0 or distracted:
            return feature
        feature=(closest_car - np.array(pos))*1.0/np.sqrt(self.reconstruction.shape[1] ** 2 + self.reconstruction.shape[2] ** 2)


        return np.expand_dims(feature[1:], axis=0)

    def get_input_cars_cont_angular(self, pos, frame, velocity, field_of_view, distracted=False):
        feature = np.zeros([1, 2], dtype=np.float32)

        closest_car, min_dist = self.find_closest_car(frame, pos, velocity, field_of_view)

        if len(closest_car) == 0 or distracted:
            return feature
        dir=closest_car[1:] - np.array(pos)[1:]

        feature[0,0]=np.linalg.norm(dir)*1.0/np.sqrt(self.reconstruction.shape[1] ** 2 + self.reconstruction.shape[2] ** 2)

        feature[0,1]=np.arctan2(dir[0],dir[1])
        if feature[0,1] <= -np.pi:
            feature[0,1]=feature[0,1]+2*np.pi
        feature[0,1]=feature[0,1]/np.pi
        return feature

    def get_input_cars_cont(self, pos, frame, velocity, field_of_view, distracted=False):
        feature = np.zeros([1, 2], dtype=np.float32)

        closest_car = []
        closest_car, min_dist = self.find_closest_car(frame, pos, velocity, field_of_view)
        if len(closest_car) == 0 or distracted :
            return feature
        feature=(closest_car - np.array(pos))#*1.0/(self.reconstruction.shape[1]*self.reconstruction.shape[2])
        if abs(feature[1])<0.5:
            feature[1]=2
        else:
            feature[1] = 1 / feature[1]
        if abs(feature[2])<.5:
            feature[2] = 2
        else:
            feature[2] = 1 / feature[2]

        return np.expand_dims(feature[1:], axis=0)


    def get_goal_dir(self, pos, goal, training=True, distracted=False):
        feature=np.zeros([1,len(self.actions)+1], dtype=np.float32)
        car_dir=goal-np.array(pos)
        car_dir[0]=0
        min_dist= np.linalg.norm(car_dir)
        #print "Distance to goal: "+str(min_dist)+" seq len: "+str(self.seq_len)
        dir = self.find_action_to_direction(car_dir,min_dist)
        feature[0,dir]=1
        feature[0,-1]=min_dist*2.0/(self.seq_len*np.sqrt(2))
        return feature

    def get_goal_dir_smooth(self, pos, goal, training=True):
        feature=np.zeros([1,len(self.actions)+1], dtype=np.float32)
        car_dir=goal-np.array(pos)
        car_dir[0]=0
        min_dist= np.linalg.norm(car_dir)
        #print "Distance to goal: "+str(min_dist)+" seq len: "+str(car_dir)
        feature[0,:-2] = self.find_action_to_direction_smooth(car_dir,min_dist)
        feature[0,-1]=min_dist*2.0/(self.seq_len*np.sqrt(2))
        #print feature
        return feature

    def get_goal_dir_cont(self, pos, goal):
        feature=np.zeros([1,2], dtype=np.float32)

        car_dir=goal-np.array(pos)
        #print "Goal "+str(goal)+" pos "+str(pos)+" "+str(car_dir)+" scaling: "+str(1.0/np.sqrt(self.reconstruction.shape[1]**2+self.reconstruction.shape[2]**2))
        feature[0,0]=car_dir[1]*1.0/np.sqrt(self.reconstruction.shape[1]**2+self.reconstruction.shape[2]**2)
        feature[0,1] = car_dir[2] * 1.0 / np.sqrt(self.reconstruction.shape[1] ** 2 + self.reconstruction.shape[2] ** 2)
        #print "Feature "+str(feature)

        return feature

    def get_goal_dir_angular(self, pos, goal):
        feature=np.zeros([1,2], dtype=np.float32)

        dir=goal[1:]-np.array(pos)[1:]
        #print "Goal "+str(goal)+" pos "+str(pos)+" "+str(car_dir)+" scaling: "+str(1.0/np.sqrt(self.reconstruction.shape[1]**2+self.reconstruction.shape[2]**2))
        feature[0,0]=np.linalg.norm(dir)*1.0/np.sqrt(self.reconstruction.shape[1]**2+self.reconstruction.shape[2]**2)
        feature[0,1] = np.arctan2(dir[0],dir[1])-(np.pi/2)
        if feature[0,1]<=-np.pi:
            feature[0,1]=feature[0,1]+2*np.pi
        feature[0, 1]=feature[0,1]/np.pi
        #print "Goal direction " + str(np.arctan2(dir[0], dir[1]) / np.pi) + " after minus " + str(feature[0, 1] )
        #print "Feature "+str(feature)

        return feature

    def get_time(self,id,  pos, goal,frame, goal_time=-1):

        goal_displacement = goal - np.array(pos)

        goal_dist = np.linalg.norm(goal_displacement)

        if goal_time<0:
            goal_time=self.pedestrian_data[id].goal_time[frame]


        if frame< goal_time:
            speed=goal_dist / (goal_time - frame)
            return min(speed, 1)
        else:

            return 1

    def find_action_to_direction(self, car_dir, min_dist):
        max_cos = -1
        dir = -1
        if min_dist==0:
            return 4
        for j, action in enumerate(self.actions, 0):
            if np.linalg.norm(action)>0.1:#j != 4:

                if np.dot(action, car_dir) / (np.linalg.norm(action) * min_dist) > max_cos:
                    dir = j
                    max_cos = np.dot(action, car_dir) / (np.linalg.norm(action) * min_dist)
        return dir

    def find_action_to_direction_smooth(self, car_dir, min_dist):
        directions=np.zeros(len(self.actions)-1)

        dir = -1
        if min_dist==0:
            directions[4] =1
            return directions
        for j, action in enumerate(self.actions, 0):
            if np.linalg.norm(action)>0.1:#j != 4:
                if j>4:
                    directions[j-1] = np.dot(action, car_dir) / (np.linalg.norm(action) * min_dist)
                else:
                    directions[j ] = np.dot(action, car_dir) / (np.linalg.norm(action) * min_dist)
        #print (" Car variable directions dot product "+str(directions)+" actions "+str(self.actions) )
        return directions

    def get_car_valid_init(self, cars_dict):
        self.road = get_road(self.reconstruction)
        self.valid_positions_cars = np.zeros(self.reconstruction.shape[1:3])  # VALID POSITOONS FOR FIRST FRAME
        self.valid_directions_cars = np.zeros(
            (self.reconstruction.shape[1], self.reconstruction.shape[2], 3))  # VALID POSITOONS FOR FIRST FRAME
        for car_key in cars_dict.keys():
            # print (" Key "+str(car_key)+" len "+str(len(self.cars_dict[car_key])))
            if len(cars_dict[car_key]) >= 2:
                #self.car_keys.append(car_key)
                car_current = cars_dict[car_key][0]
                car_next = cars_dict[car_key][1]
                diff = [[], [], []]
                for i in range(len(car_current)):
                    diff[int(i / 2)].append(car_next[i] - car_current[i])
                # self.car_vel_dict[car_key] = np.mean(np.array(diff), axis=1)
                prev_car = []
                vel =np.mean(np.array(diff), axis=1)# self.car_vel_dict[car_key]
                for car in cars_dict[car_key]:
                    if len(prev_car) > 0:
                        diff = [[], [], []]
                        for i in range(len(car_current)):
                            diff[int(i / 2)].append(car[i] - prev_car[i])
                        vel = np.mean(np.array(diff), axis=1)
                    self.valid_positions_cars[int(round(car[2])):int(round(car[3])),
                    int(round(car[4])):int(round(car[5]))] = 1
                    car_dims = self.valid_positions_cars[int(round(car[2])):int(round(car[3])),
                               int(round(car[4])):int(round(car[5]))].shape
                    self.valid_directions_cars[int(round(car[2])):int(round(car[3])),
                    int(round(car[4])):int(round(car[5])), :] = np.tile(vel[np.newaxis, np.newaxis, :],
                                                                        (car_dims[0], car_dims[1], 1))
                    prev_car = copy.copy(car)

                    car_pos = np.array([np.mean(car[0:2]), np.mean(car[2:4]), np.mean(car[4:])])
                    car_prev_pos = np.array([np.mean(prev_car[0:2]), np.mean(prev_car[2:4]), np.mean(prev_car[4:])])
                    speed = np.linalg.norm(vel[1:])
                    if speed > 1e-2:
                        ortogonal_car_vel = np.array([0, vel[2] / speed, -vel[1] / speed])
                        min_road = find_extreme_road_pos(car, car_prev_pos, ortogonal_car_vel,self.car_dim, self.road)
                        max_road =find_extreme_road_pos(car, car_prev_pos, -ortogonal_car_vel, self.car_dim, self.road)
                        # if abs(max_road[1]-min_road[1])>2*min(self.settings.car_dim[1:]) or abs(max_road[2]-min_road[2])>2*min(self.settings.car_dim[1:]):
                        mean_road = (min_road + max_road) * 0.5
                        vector_to_car_pos = car_prev_pos - mean_road
                        car_opposite = mean_road - vector_to_car_pos
                        car_opposite_bbox = get_bbox_of_car(car_opposite, car, self.car_dim)
                        if np.all(self.valid_positions[car_opposite_bbox[2]:car_opposite_bbox[3], car_opposite_bbox[4]:car_opposite_bbox[5]]) and not \
                                np.all( self.valid_positions_cars[car_opposite_bbox[2]:car_opposite_bbox[3],car_opposite_bbox[4]:car_opposite_bbox[5]]) \
                                and iou_sidewalk_car(car_opposite_bbox,self.reconstruction,no_height=True) == 0:
                            self.valid_positions_cars[car_opposite_bbox[2]:car_opposite_bbox[3],
                            car_opposite_bbox[4]:car_opposite_bbox[5]] = 1
                            car_dims = self.valid_positions_cars[car_opposite_bbox[2]:car_opposite_bbox[3],
                                       car_opposite_bbox[4]:car_opposite_bbox[5]].shape
                            self.valid_directions_cars[car_opposite_bbox[2]:car_opposite_bbox[3],
                            car_opposite_bbox[4]:car_opposite_bbox[5], :] = np.tile(-vel[np.newaxis, np.newaxis, :],
                                                                                     (car_dims[0], car_dims[1], 1))



