import unittest
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import sys

from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings
from RL.net_sem_2d import Seg_2d_softmax
from RL.agent_net import NetAgent
from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE,NBR_MEASURES,PEDESTRIAN_MEASURES_INDX
from RL.environment_interaction import EntitiesRecordedDataSource, EnvironmentInteraction
from commonUtils.ReconstructionUtils import LAST_CITYSCAPES_SEMLABEL, cityscapes_labels_dict, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, SIDEWALK_LABELS,CHANNELS,OBSTACLE_LABELS_NEW, OBSTACLE_LABELS,cityscapes_labels_dict
# Test methods in episode.




class TestNet(unittest.TestCase):
    def get_reward(self, all_zeros=False):

        rewards = np.zeros(NBR_REWARD_WEIGHTS)
        if not all_zeros:
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
            rewards[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] = 1
            rewards[PEDESTRIAN_REWARD_INDX.on_pavement] = 1
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
            rewards[PEDESTRIAN_REWARD_INDX.distance_travelled] = 1
            rewards[PEDESTRIAN_REWARD_INDX.out_of_axis] = -1
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
            rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        return rewards
    def update_episode(self, environmentInteraction, episode, next_frame):
        observation, observation_dict = environmentInteraction.getObservation(frameToUse=next_frame)
        episode.update_pedestrians_and_cars(observation.frame,
                                            observation_dict,
                                            observation.people_dict,
                                            observation.cars_dict,
                                            observation.pedestrian_vel_dict,
                                            observation.car_vel_dict)

    def get_episode(self, cars, gamma, people, pos_x, pos_y, tensor, seq_len=15, rewards=[], agent_size=(0, 0, 0),
                    people_dict={}, init_frames={}, car_dict={}, init_frames_cars={}, new_carla=False):
        if len(rewards) == 0:
            rewards = self.get_reward(False)
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1

        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                         seq_len=seq_len, rewards=rewards,
                                                                         agent_size=agent_size,
                                                                         people_dict=people_dict,
                                                                         init_frames=init_frames, car_dict=car_dict,
                                                                         init_frames_cars=init_frames_cars,
                                                                         new_carla=new_carla)
        episode.agent_size = [0, 0, 0]
        return agent, episode, environmentInteraction
    # Help function. Setup for tests.

    # def initialize_episode(self,net, cars, gamma, people, pos_x, pos_y, tensor,seq_len=30, rewards=(1, 1, 1, 1, 1, 1),  width=0):
    #     #  tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights
    #     setup = self.get_settings(width)
    #
    #     # if self.fastTrainEvalDebug:
    #     #     self.width_agent_s = 10
    #     #     self.depth_agent_s = 10
    #     setup.agent_shape_s = [setup.height_agent_s, setup.width_agent_s, setup.depth_agent_s]
    #     episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards,rewards,
    #                             agent_size=[width, width, width], adjust_first_frame=False, run_2D=True, centering={}, defaultSettings=setup)
    #
    #
    #
    #     #tf.reset_default_graph()
    #     agent=SimplifiedAgent(setup)
    #     agent = NetAgent( setup,net, None)
    #     agent.id=0
    #     return agent, episode, net

    def initialize_episode(self,net, cars, gamma, people, pos_x, pos_y, tensor, seq_len=30, rewards=[],
                           agent_size=(0, 0, 0), people_dict={}, init_frames={}, car_dict={}, init_frames_cars={},
                           new_carla=False,  width=0):
        settings = self.get_settings(width)
        settings.agent_shape_s=[settings.height_agent_s, settings.width_agent_s, settings.depth_agent_s]
        settings.useRLToyCar = False
        settings.multiplicative_reward_pedestrian = False
        settings.goal_dir = False
        if len(rewards) == 0:
            rewards = self.get_reward()

        if settings.useRealTimeEnv:
            entitiesRecordedDataSource = EntitiesRecordedDataSource(init_frames=init_frames,
                                                                    init_frames_cars=init_frames_cars,
                                                                    cars_sample=cars,
                                                                    people_sample=people,
                                                                    cars_dict_sample=car_dict,
                                                                    people_dict_sample=people_dict,
                                                                    cars_vel={},
                                                                    ped_vel={},
                                                                    reconstruction=tensor,
                                                                    forced_num_frames=None)
            environmentInteraction = EnvironmentInteraction(False,
                                                            entitiesRecordedDataSource=entitiesRecordedDataSource,
                                                            parentEnvironment=None, args=settings)
            environmentInteraction.reset([], [], episode=None)
            observation, observation_dict = environmentInteraction.getObservation(frameToUse=None)
            car_vel_dict = observation.car_vel_dict
            people_vel_dict = observation.pedestrian_vel_dict
            people_dict = {}
            for key, value in observation.people_dict.items():
                people_dict[key] = [value]
            cars_dict = {}
            for key, value in observation.cars_dict.items():
                cars_dict[key] = [value]
            cars = entitiesRecordedDataSource.cars  # observation.cars
            people = entitiesRecordedDataSource.people  # observation.people
            init_frames = entitiesRecordedDataSource.init_frames  # observation.init_frames
            init_frames_cars = entitiesRecordedDataSource.init_frames_cars  # observation.init_frames_cars
            episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, settings.gamma,
                                    seq_len, rewards, rewards, agent_size=agent_size,
                                    people_dict=people_dict,
                                    cars_dict=cars_dict,
                                    people_vel=people_vel_dict,
                                    cars_vel=car_vel_dict,
                                    init_frames=init_frames, follow_goal=settings.goal_dir,
                                    action_reorder=settings.reorder_actions,
                                    threshold_dist=settings.threshold_dist, init_frames_cars=init_frames_cars,
                                    temporal=settings.temporal, predict_future=settings.predict_future,
                                    run_2D=settings.run_2D, agent_init_velocity=settings.speed_input,
                                    velocity_actions=settings.velocity or settings.continous,
                                    end_collide_ped=settings.end_on_bit_by_pedestrians,
                                    stop_on_goal=settings.stop_on_goal, waymo=settings.waymo,
                                    centering=(0, 0),
                                    defaultSettings=settings,
                                    multiplicative_reward_pedestrian=settings.multiplicative_reward_pedestrian,
                                    multiplicative_reward_initializer=settings.multiplicative_reward_initializer,
                                    learn_goal=settings.learn_goal or settings.separate_goal_net,
                                    use_occlusion=settings.use_occlusion,
                                    useRealTimeEnv=settings.useRealTimeEnv, car_vel_dict=car_vel_dict,
                                    people_vel_dict=people_vel_dict, car_dim=settings.car_dim,
                                    new_carla=new_carla, lidar_occlusion=settings.lidar_occlusion,
                                    use_car_agent=settings.useRLToyCar or settings.useHeroCar,
                                    use_pfnn_agent=settings.pfnn, number_of_agents=settings.number_of_agents,
                                    number_of_car_agents=settings.number_of_car_agents)  # To DO:  Check if needed heroCarDetails = heroCarDetails
            episode.environmentInteraction = environmentInteraction
        else:
            environmentInteraction = None
            episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards, rewards,
                                    agent_size=agent_size,
                                    people_dict=people_dict, init_frames=init_frames, cars_dict=car_dict,
                                    init_frames_cars=init_frames_cars, defaultSettings=settings, centering={},
                                    useRealTimeEnv=settings.useRealTimeEnv, new_carla=new_carla)
        agent = NetAgent( settings,net, None)
        agent.id = 0
        if settings.useRealTimeEnv:
            heroAgentPedestrians = [agent]

            realTimeEnvObservation, observation_dict = environmentInteraction.reset(heroAgentCars=[],
                                                                                    heroAgentPedestrians=heroAgentPedestrians,
                                                                                    episode=episode)
            episode.update_pedestrians_and_cars(realTimeEnvObservation.frame,
                                                observation_dict,
                                                realTimeEnvObservation.people_dict,
                                                realTimeEnvObservation.cars_dict,
                                                realTimeEnvObservation.pedestrian_vel_dict,
                                                realTimeEnvObservation.car_vel_dict)
            environmentInteraction.frame = 0
        return agent, episode, environmentInteraction
    #
    # def initialize_tensor(self, seq_len=30, size=3):
    #     tensor = np.zeros((size, size, size, 6))
    #     people = []
    #     cars = []
    #     for i in range(seq_len):
    #         people.append([])
    #         cars.append([])
    #     gamma = 0.99
    #     pos_x = 0
    #     pos_y = 0
    #     return cars, gamma, people, pos_x, pos_y, tensor

    # def initialize_pos(self, agent, episode):
    #     pos, i, vel = episode.initial_position(0, None, initialization=PEDESTRIAN_INITIALIZATION_CODE.randomly)
    #
    #     agent.initial_position(pos, episode.pedestrian_data[0].goal)

    def update_agent_and_episode(self, action, agent, environmentInteraction, episode, frame, breadth=0,action_nbr=4):
        if breadth == 0:
            breadth = episode.agent_size[1:]
        episode.pedestrian_data[0].action[frame - 1] = action_nbr
        episode.pedestrian_data[0].velocity[frame - 1] = action
        episode.get_agent_neighbourhood(0, agent.position, breadth, frame - 1)
        environmentInteraction.signal_action({agent: action}, updated_frame=frame)
        agent.perform_action(action, episode)
        # Do the simulation for next tick using decisions taken on this tick
        # If an online realtime env is used this call will fill in the data from the simulator.
        # If offline, it will take needed data from recorded/offline data.
        environmentInteraction.tick(frame)
        self.update_episode(environmentInteraction, episode, frame)
        agent.update_agent_pos_in_episode(episode, frame)
        agent.on_post_tick(episode)
        agent.update_metrics(episode)
        episode.pedestrian_data[0].action[frame - 1] = 5

    def evaluate_measure(self, frame, episode, non_zero_measure, expeceted_value_of_non_zero_measure):
        for measure in range(NBR_MEASURES):
            if measure == non_zero_measure:
                np.testing.assert_array_equal(
                    episode.pedestrian_data[0].measures[frame, non_zero_measure], expeceted_value_of_non_zero_measure)
            else:
                if measure !=PEDESTRIAN_MEASURES_INDX.change_in_direction and measure != PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal_from_current and measure != PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap and measure != PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.change_in_pose and measure != PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car:
                    print(measure)
                    np.testing.assert_array_equal(episode.pedestrian_data[0].measures[frame, measure], 0)

    def evaluate_measures(self, frame, episode, dict):
        non_zero_measures = []
        zero_measures = []
        for measure in range(NBR_MEASURES):
            if measure in dict:
                non_zero_measures.append(measure)
            else:
                zero_measures.append(measure)

        for measure in non_zero_measures:
            print(measure)
            np.testing.assert_array_equal(
                episode.pedestrian_data[0].measures[frame, measure], dict[measure])
        for measure in zero_measures:
            print(measure)
            if measure !=PEDESTRIAN_MEASURES_INDX.change_in_direction and measure != PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal_from_current and measure != PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap and measure != PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.change_in_pose and measure != PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car:
                np.testing.assert_array_equal(episode.pedestrian_data[0].measures[frame, measure], 0)

    # ----
    # Help function. Setup for tests.

    def get_settings(self, width):
        setup = run_settings()
        setup.useRLToyCar = False
        setup.useHeroCar = False
        setup.number_of_car_agents = 0  #
        setup.number_of_agents = 1  #
        setup.pedestrian_view_occluded = False
        setup.field_of_view = 2 * np.pi  # 114/180*np.pi
        setup.field_of_view_car = 2 * np.pi  # 90/180*np.pi
        setup.run_2D = True
        setup.height_agent = width
        setup.width_agent = width
        setup.depth_agent = width
        setup.agent_shape = [setup.height_agent, setup.width_agent, setup.depth_agent]
        setup.height_agent_s = 0
        setup.width_agent_s = 0
        setup.depth_agent_s = 0
        setup.net_size = [1 + 2 * (setup.height_agent_s + setup.height_agent),
                          1 + 2 * (setup.depth_agent_s + setup.depth_agent),
                          1 + 2 * (setup.width_agent_s + setup.width_agent)]
        setup.agent_s = [setup.height_agent_s + setup.height_agent, setup.depth_agent_s + setup.depth_agent,
                         setup.width_agent_s + setup.width_agent]
        return setup

    def initialize_tensor(self, seq_len, size=3):

        tensor = np.zeros((size,size, size, 6))
        people = []
        cars = []
        for i in range(seq_len):
            people.append([])
            cars.append([])
        gamma = 0.99
        pos_x = 0
        pos_y = 0
        return cars, gamma, people, pos_x, pos_y, tensor

    def initialize_pos(self, agent, episode):
        pos, i , vel= episode.initial_position(0,None)

        agent.initial_position(pos, episode.pedestrian_data[0].goal[0,:])

    # Test correct empty initialization.
    def test_walk_into_objs(self):
        width = 0
        setup = self.get_settings(width)

        net = Seg_2d_softmax(setup)
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1,1,1,CHANNELS.semantic]=cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES # pavement
        tensor[2, 1, 2,CHANNELS.semantic] =cityscapes_labels_dict['building']/NUM_SEM_CLASSES # person/obj?
        agent, episode, environmentInteraction=self.initialize_episode(net, cars, gamma, people, pos_x, pos_y, tensor, seq_len=11, rewards=(0, 0, 0, 1, 0, 0), width=0)
        episode.agent_size = [0, 0, 0]
        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 1)
        self.initialize_pos(agent, episode)
        tensor=net.get_input(0,episode, [1,1.2,.9])
        expected=np.zeros((1,1,1,42))
        expected[:,:,:,cityscapes_labels_dict['sidewalk']+net.channels.semantic]=1
        #expected[:, :, :, 2] = 1
        np.testing.assert_array_equal(tensor, expected)

        tensor = net.get_input(0,episode, [0,.9,2.1])
        expected = np.zeros((1, 1, 1, 42))
        expected[:, :, :,  cityscapes_labels_dict['building']+net.channels.semantic] = 1
        #expected[:, :, :, 2] = 1
        np.testing.assert_array_equal(tensor, expected)
        tf.compat.v1.reset_default_graph()

    def test_walk_into_objs2(self):
        width = 1
        setup = self.get_settings(width)
        # Test correct empty initialization.
        net = Seg_2d_softmax(setup)
        # def test_walk_into_objs2(self):
        width = 11
        setup = self.get_settings(width)
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1, 1, 1,CHANNELS.semantic] = cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        tensor[2, 1, 2,CHANNELS.semantic] = cityscapes_labels_dict['building']/NUM_SEM_CLASSES
        agent, episode, environmentInteraction= self.initialize_episode(net, cars, gamma, people, pos_x, pos_y, tensor, seq_len=11,
                                                      width=1)
        episode.agent_size = [1, 1, 1]
        self.initialize_pos(agent, episode)
        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 1)
        tensor = net.get_input(0,episode, [1, 0.8, 1.1])
        expected = np.zeros((1, 3, 3, 42))
        expected[:, 1, 1, cityscapes_labels_dict['sidewalk']+net.channels.semantic] = 1
        expected[0, 1, 2, cityscapes_labels_dict['building']+net.channels.semantic] = 1
        #expected[:, :, :, 2] = 1
        np.testing.assert_array_equal(tensor, expected)

        tensor = net.get_input(0,episode, [0, 1.1, 2+1e-15])
        expected = np.zeros((1, 3, 3, 42))
        expected[:, 1, 1, cityscapes_labels_dict['building']+net.channels.semantic] = 1
        #expected[:, :, :, 2] = 1
        expected[:, 1, 0, cityscapes_labels_dict['sidewalk']+net.channels.semantic] = 1
        np.testing.assert_array_equal(tensor, expected)
