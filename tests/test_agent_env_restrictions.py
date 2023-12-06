import unittest
import numpy as np
import sys

from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX

from RL.episode import SimpleEpisode
from RL.agent_pfnn import AgentPFNN
from RL.agent import ContinousAgent, SimplifiedAgent
from RL.settings import run_settings,NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_MEASURES_INDX
from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE,NBR_MEASURES,PEDESTRIAN_MEASURES_INDX
from RL.environment_interaction import EntitiesRecordedDataSource, EnvironmentInteraction
from commonUtils.ReconstructionUtils import LAST_CITYSCAPES_SEMLABEL, cityscapes_labels_dict, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, SIDEWALK_LABELS,CHANNELS,OBSTACLE_LABELS_NEW, OBSTACLE_LABELS,cityscapes_labels_dict
# Test methods in episode.


class TestEnv(unittest.TestCase):
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

    # # Help function. Setup for tests.
    # def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor,seq_len=30, rewards=np.ones(NBR_REWARD_WEIGHTS), agent_size=(0,0,0),people_dict={}, init_frames={}):
    #     #  tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights
    #     setup = run_settings()
    #     setup.useHeroCar = False
    #     setup.useRLToyCar = False
    #     episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards,rewards, agent_size=agent_size,
    #                             people_dict=people_dict, init_frames=init_frames, adjust_first_frame=False,
    #                             velocity_actions=False, defaultSettings=setup,centering={})
    #     episode.pedestrian_data[0].action=np.zeros(len(episode.pedestrian_data[0].action))
    #     agent=SimplifiedAgent(setup)
    #     agent.id=0
    #
    #     return agent, episode

    def update_episode(self, environmentInteraction, episode, next_frame):
        observation, observation_dict = environmentInteraction.getObservation(frameToUse=next_frame)
        episode.update_pedestrians_and_cars(observation.frame,
                                            observation_dict,
                                            observation.people_dict,
                                            observation.cars_dict,
                                            observation.pedestrian_vel_dict,
                                            observation.car_vel_dict)

    def get_episode(self, cars, gamma, people, pos_x, pos_y, tensor, seq_len=15, rewards=[], agent_size=(0, 0, 0),
                    people_dict={}, init_frames={}, car_dict={}, init_frames_cars={}, new_carla=False,continous=False):
        if len(rewards) == 0:
            rewards = self.get_reward(False)
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1

        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                         seq_len=seq_len, rewards=rewards,
                                                                         agent_size=agent_size,
                                                                         people_dict=people_dict,
                                                                         init_frames=init_frames, car_dict=car_dict,
                                                                         init_frames_cars=init_frames_cars,
                                                                         new_carla=new_carla, continous=continous)
        episode.agent_size = [0, 0, 0]
        return agent, episode, environmentInteraction
        # Help function. Setup for tests.
    def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor, seq_len=30, rewards=[],
                           agent_size=(0, 0, 0), people_dict={}, init_frames={}, car_dict={}, init_frames_cars={},
                           new_carla=False, continous=False):
        settings = run_settings()
        settings.useRLToyCar = False
        settings.useHeroCar = False
        settings.number_of_car_agents = 0  #
        settings.number_of_agents = 1  #
        settings.pedestrian_view_occluded = False
        settings.field_of_view = 2 * np.pi  # 114/180*np.pi
        settings.field_of_view_car = 2 * np.pi  # 90/180*np.pi
        settings.useRLToyCar = False
        settings.multiplicative_reward_pedestrian = False
        settings.stop_on_goal = True
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
        if continous:
            agent = ContinousAgent(settings)

        else:
            agent = SimplifiedAgent(settings)
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

    def update_agent_and_episode(self, action, agent, environmentInteraction, episode, frame, breadth=0, action_nbr=5):
        if breadth == 0:
            breadth = episode.agent_size[1:]
        if frame<episode.seq_len:
            episode.pedestrian_data[0].action[frame - 1] = action_nbr
            episode.pedestrian_data[0].velocity[frame - 1] = action
        episode.get_agent_neighbourhood(0, agent.position, breadth, frame - 1)
        if frame < episode.seq_len:
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
            #episode.pedestrian_data[0].action[frame - 1] = action_nbr

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
        pos, i,vel = episode.initial_position(0,None, initialization=8)

        agent.initial_position(pos,episode.pedestrian_data[0].goal[0,:], vel=vel)

    # Test correct empty initialization.
    def test_walk_into_objs(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1,1,1,CHANNELS.semantic]=cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        tensor[2, 1, 2,  CHANNELS.semantic]=cityscapes_labels_dict['building']/NUM_SEM_CLASSES
        rewards=self.get_reward(True)

        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects]=-1
        agent, episode, environmentInteraction =self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agent, episode)

        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])

        # Invalid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0,0,1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])


        # Invalid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])

        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)


        # Valid move
        self.update_agent_and_episode([0, 1,0], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [1, 2, 1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])
        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], -1)



        # Valid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [1, 2, 2])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)


        # Invalid move
        self.update_agent_and_episode([0, -1,0], agent, environmentInteraction, episode, 5)
        # agent.perform_action([0, -1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [1, 2, 2])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], 0)


        # Valid move
        self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode,6)
        # agent.perform_action([0, 0, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [1, 2, 1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])
        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], -1)


        # Valid move
        self.update_agent_and_episode([0, -1, 0], agent, environmentInteraction, episode, 7)
        # agent.perform_action([0, -1,0], episode)
        # agent.update_agent_pos_in_episode(episode, 7)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[6] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], 0)


        # Valid move
        self.update_agent_and_episode([0, -1,0], agent, environmentInteraction, episode, 8)
        # agent.perform_action([0, -1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 8)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[7] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[8], [1, 0, 1])
        np.testing.assert_array_equal(agent.position, [1, 0, 1])
        np.testing.assert_array_equal(episode.calculate_reward(6, episode_done=True)[0], 0)

        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 9)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 9)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[8] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[9], [1, 0, 2])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(7, episode_done=True)[0], 0)



        self.update_agent_and_episode([0, 1,0], agent, environmentInteraction, episode, 10)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 10)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[9] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[10], [1, 0, 2])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(8, episode_done=True)[0], 0)

    # Test correct empty initialization.
    def test_walk_into_objs_varied_vel(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1, 1, 1 ,CHANNELS.semantic]=cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        tensor[2, 1, 2,  CHANNELS.semantic]=cityscapes_labels_dict['building']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11,
                                                 rewards=rewards)
        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agent, episode)

        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])

        # Invalid move
        self.update_agent_and_episode([0, 0, 0.9], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0, 0.9], episode)
        #
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])


        # Invalid move
        self.update_agent_and_episode([0, 0, 0.7], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 0.7], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)


        # Valid move
        self.update_agent_and_episode([0, 0.8, 0.1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 0.8, 0.1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [1, 2, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[3], [1, 1.8, 1.1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])
        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], -1)


        # Valid move
        self.update_agent_and_episode([0, 0.1, 1.2], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, 0.1, 1.2], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [1, 2, 2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[4], [1, 1.9, 2.3])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)


        # Invalid move
        self.update_agent_and_episode([0, -1.3, 0], agent, environmentInteraction, episode, 5)
        # agent.perform_action([0, -1.3, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [1, 2, 2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[5], [1, 1.9, 2.3])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], 0)


        # Valid move
        self.update_agent_and_episode([0, 0, -1.3], agent, environmentInteraction, episode,6)
        # agent.perform_action([0, 0, -1.3], episode)
        # agent.update_agent_pos_in_episode(episode, 6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [1, 2, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[6], [1, 1.9, 1.0])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])
        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], -1)


        # Valid move
        self.update_agent_and_episode([0, -1.4,0], agent, environmentInteraction, episode, 7)
        # agent.perform_action([0, -1.4, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 7)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[6] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [1, 1, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[7], [1, .5, 1.0])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], 0)


        # Valid move
        self.update_agent_and_episode([0, -0.7, 0], agent, environmentInteraction, episode, 8)
        # agent.perform_action([0, -.7, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 8)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[7] = 5
        # #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[8], [1, 0, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[8], [1, -.2, 1.0])
        np.testing.assert_array_equal(agent.position, [1, 0, 1])
        np.testing.assert_array_equal(episode.calculate_reward(6, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, 1.1], agent, environmentInteraction, episode, 9)
        # agent.perform_action([0, 0, 1.1], episode)
        # agent.update_agent_pos_in_episode(episode, 9)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[8] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[9], [1, 0, 2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[9], [1, -.2, 2.1])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(7, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0.7,0], agent, environmentInteraction, episode, 10)
        # agent.perform_action([0, .7, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 10)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[9] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[10], [1, 0, 2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[10], [1, -.2, 2.1])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(8, episode_done=True)[0], 0)

    # Test correct empty initialization.
    def test_walk_into_objs_wide(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        tensor[1, 3, 3, CHANNELS.semantic]=cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        tensor[2, 3, 5,  CHANNELS.semantic]=cityscapes_labels_dict['building']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=14, rewards=rewards)
        episode.agent_size = [0, 1, 1]

        self.initialize_pos(agent, episode)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])

        # Invalid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])


        # Invalid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])
        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)


        self.update_agent_and_episode([0, 1,0], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [1, 4, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], -1.0)


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [1, 4, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 1,0], agent, environmentInteraction, episode, 5)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [1, 5, 3])
        np.testing.assert_array_equal(agent.position, [1, 5, 3])
        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], -1)


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 6)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [1, 5, 4])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, -1,0], agent, environmentInteraction, episode, 7)
        # agent.perform_action([0, -1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 7)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[6] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [1, 5, 4])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 8)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 8)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[7] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[8], [1, 5, 5])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(6, episode_done=True)[0], -1)


        self.update_agent_and_episode([0, -1,0], agent, environmentInteraction, episode, 9)
        # agent.perform_action([0, -1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 9)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[8] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[9], [1, 5, 5])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(7, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 10)
        # agent.perform_action([0,0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 10)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[9] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[10], [1, 5,6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(8, episode_done=True)[0], -1)


        self.update_agent_and_episode([0,-1,0], agent, environmentInteraction, episode, 11)
        # agent.perform_action([0, -1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 11)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[10] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[11], [1, 5, 6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(9, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 12)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 12)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[11] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[12], [1, 5, 7])
        np.testing.assert_array_equal(agent.position, [1, 5, 7])
        np.testing.assert_array_equal(episode.calculate_reward(10, episode_done=True)[0], -1)


        self.update_agent_and_episode([0, -1,0], agent, environmentInteraction, episode, 13)
        # agent.perform_action([0, -1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 13)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[12] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[13], [1, 4, 7])
        np.testing.assert_array_equal(agent.position, [1, 4, 7])
        np.testing.assert_array_equal(episode.calculate_reward(11, episode_done=True)[0], 0)
        #np.testing.assert_array_equal(episode.calculate_reward(12, episode_done=True)[0], 0)

        # Test correct empty initialization.

    def test_walk_into_objs_wide_varied_vel(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        tensor[1, 3, 3,CHANNELS.semantic]=cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        tensor[2, 3, 5,  CHANNELS.semantic]=cityscapes_labels_dict['building']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=14,
                                                 rewards=rewards)
        episode.agent_size = [0, 1, 1]

        self.initialize_pos(agent, episode)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])

        # Invalid move
        self.update_agent_and_episode([0, 0, 0.9], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0, .9], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [1, 3, 3])
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])


        # Invalid move
        self.update_agent_and_episode([0, 0, 1.2], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1.2], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])
        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)


        self.update_agent_and_episode([0, 0.8, 0], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, .8, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        # np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [1, 4, 3])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[3], [1, 3.8, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], -1.0)


        self.update_agent_and_episode([0, 0, .7], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, 0, .7], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [1, 4, 3])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[4], [1, 3.8, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0.9, 0], agent, environmentInteraction, episode, 5)
        # agent.perform_action([0, .9, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [1, 5, 3])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[5], [1, 4.7, 3])
        np.testing.assert_array_equal(agent.position, [1, 5, 3])
        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], -1)


        self.update_agent_and_episode([0, 0, .8], agent, environmentInteraction, episode, 6)
        # agent.perform_action([0, 0, .8], episode)
        # agent.update_agent_pos_in_episode(episode, 6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [1, 5, 4])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[6], [1, 4.7, 3.8])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, -0.7, 0], agent, environmentInteraction, episode, 7)
        # agent.perform_action([0, -.7, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 7)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[6] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [1, 5, 4])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[7], [1, 4.7, 3.8])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, .9], agent, environmentInteraction, episode, 8)
        # agent.perform_action([0, 0, .9], episode)
        # agent.update_agent_pos_in_episode(episode, 8)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[7] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[8], [1, 5, 5])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[8], [1, 4.7, 4.7])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(6, episode_done=True)[0], -1)


        self.update_agent_and_episode([0, -1.3, 0], agent, environmentInteraction, episode, 9)
        # agent.perform_action([0, -1.3, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 9)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[8] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[9], [1, 5, 5])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[9], [1, 4.7, 4.7])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(7, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, 1.3], agent, environmentInteraction, episode, 10)
        # agent.perform_action([0, 0, 1.3], episode)
        # agent.update_agent_pos_in_episode(episode, 10)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[9] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[10], [1, 5, 6])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[10], [1,4.7, 6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(8, episode_done=True)[0], -1)


        self.update_agent_and_episode([0,-1.1, 0], agent, environmentInteraction, episode, 11)
        # agent.perform_action([0, -1.1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 11)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[10] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[11], [1, 5, 6])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[11], [1, 4.7, 6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(9, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, 1.1], agent, environmentInteraction, episode, 12)
        # agent.perform_action([0, 0, 1.1], episode)
        # agent.update_agent_pos_in_episode(episode, 12)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[11] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[12], [1, 5, 7])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[12], [1, 4.7, 7.1])
        np.testing.assert_array_equal(agent.position, [1, 5, 7])
        np.testing.assert_array_equal(episode.calculate_reward(10, episode_done=True)[0], -1)


        self.update_agent_and_episode([0, -.7, 0], agent, environmentInteraction, episode, 13)
        # agent.perform_action([0, -.7, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 13)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[12] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[13], [1, 4, 7])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[13], [1, 4, 7.1])
        np.testing.assert_array_equal(agent.position, [1, 4, 7])
        np.testing.assert_array_equal(episode.calculate_reward(11, episode_done=True)[0], 0)
        #np.testing.assert_array_equal(episode.calculate_reward(12, episode_done=True)[0], 0)


    def test_pedestrians(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(7)
        people_list1=[]
        people_list2 = []
        people_list3 = []
        people[0].append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people_list1.append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people[1].append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people_list1.append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people[0].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people[1].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3, 2)))
        people_list2.append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people_list2.append(np.array([0, 2, 2, 2, 2, 2]).reshape((3, 2)))
        people[0].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        people[1].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        people_list3.append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        people_list3.append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))

        people_dict={0:people_list1,1:people_list2, 2:people_list3}
        init_frames={0:0, 1:0, 2:0}

        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=7,  rewards=rewards,people_dict=people_dict,init_frames=init_frames , agent_size=[0,1,1])
        episode.agent_size = [0, 0, 0]
        #episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0]=[0,0,0]

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 1,1], agent, environmentInteraction, episode, 1, breadth=[0,1,1])
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 1])


        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 2, breadth=[0,1,1])
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)


        self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode, 3, breadth=[0,1,1])
        # agent.perform_action([0, -1, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode, 4, breadth=[0,1,1])
        # agent.perform_action([0, -1, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [0, 0, 0])
        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 5, breadth=[0,1,1])
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [0, 0, 1])
        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 6, breadth=[0,1,1])
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [0, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
        #np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], 0)

    def test_pedestrians_varied_vel(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(7)
        people_list1=[]
        people_list2 = []
        people_list3 = []
        people[0].append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people_list1.append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people[1].append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people_list1.append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people[0].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people[1].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3, 2)))
        people_list2.append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people_list2.append(np.array([0, 2, 2, 2, 2, 2]).reshape((3, 2)))
        people[0].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        people[1].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        people_list3.append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        people_list3.append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))

        people_dict={0:people_list1, 1:people_list2, 2:people_list3}
        frames_init={0:0, 1:0, 2:0}

        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=7,  rewards=rewards,people_dict=people_dict ,init_frames=frames_init)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0]=[0,0,0]

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0.9, 0.9], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0.9, 0.9], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        episode.pedestrian_data[0].action[0] = 5

        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[1], [0, .9, .9])

        self.update_agent_and_episode([0, 1.1, 1.2], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 1.1, 1.2], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[2], [0, 2.0, 2.1])
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)



        self.update_agent_and_episode([0, -.9, -1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, -.9, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[3], [0, 1.1, 1.1])
        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, -.9, -1.1], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, -.9, -1.1], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [0, 0, 0])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[4], [0, .2, 0])
        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)

        self.update_agent_and_episode([0, 0, 1.2], agent, environmentInteraction, episode, 5)
        # agent.perform_action([0, 0, 1.2], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [0, 0, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[5], [0, .2, 1.2])
        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], 0)



        self.update_agent_and_episode([0, 0, .75], agent, environmentInteraction, episode, 6)
        # agent.perform_action([0, 0, .75], episode)
        # agent.update_agent_pos_in_episode(episode, 6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [0, 0, 2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[6], [0, 0.2, 1.95])
        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
        #np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], 0)



    def test_pedestrians_wide(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        people_dict= { }
        init_frames={}
        for i in range(7):
            people[0].append(np.array([0, 2, i, i, i, i]).reshape((3, 2)))
            people[1].append(np.array([0, 2, i, i, i, i]).reshape((3, 2)))
            people_dict[i]=[np.array([0, 2, i, i, i, i]).reshape((3, 2)),np.array([0, 2, i, i, i, i]).reshape((3, 2))]
            init_frames[i]=0

        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] = 1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=14, rewards=rewards, agent_size=[0,1,1],people_dict=people_dict ,init_frames=init_frames)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = [0, 0, 0]

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        for i in range(1,7):
            self.update_agent_and_episode([0, .5, .5], agent, environmentInteraction, episode, i, breadth=[0, 1, 1])
            # agent.perform_action([0, .5, .5], episode)
            # agent.update_agent_pos_in_episode(episode, i)
            # agent.on_post_tick(episode)
            # agent.update_metrics(episode)
            # episode.pedestrian_data[0].action[i-1] = 5
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[i], [0, i*0.5,i*0.5])
            #np.testing.assert_array_equal(agent.position, [0, i * 0.5, i * 0.5])
            if i>2:
                np.testing.assert_array_less( 0, episode.calculate_reward(i-2, episode_done=True)[0])
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                         seq_len=14, rewards=rewards,
                                                                         agent_size=[0, 1, 1],people_dict=people_dict ,init_frames=init_frames)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0, None)
        episode.pedestrian_data[0].agent[0] = [0, 0, 0]

        episode.pedestrian_data[0].agent[0] = np.array([0, 6, 6])
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        for i in range(1,7):
            self.update_agent_and_episode([0, -.5, -.5], agent, environmentInteraction, episode, i, breadth=[0, 1, 1])
            # agent.perform_action([0, -.5, -.5], episode)
            # agent.update_agent_pos_in_episode(episode, i)
            # agent.on_post_tick(episode)
            # agent.update_metrics(episode)
            # episode.pedestrian_data[0].action[i - 1] = 5
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[i], [0, 6-(i*.5), 6-(i*.5)])
            #np.testing.assert_array_equal(agent.position, [0, 6 - (i * .5), 6 - (i * .5)])
            if i > 2:
                np.testing.assert_array_less( 0, episode.calculate_reward(i-2, episode_done=True)[0])




    def test_dist_travelled(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.distance_travelled] = 1
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        for step in range(seq_len-2):
            self.update_agent_and_episode([0, .75, 0.75], agent, environmentInteraction, episode,step+1)
            # agent.perform_action([0, .75, 0.75], episode)
            # agent.update_agent_pos_in_episode(episode, step+1)
            # agent.on_post_tick(episode)
            # agent.update_metrics(episode)
            # episode.pedestrian_data[0].action[step] = 5
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[step+1], [0, 0.75*(step+1), 0.75*(step+1)])
            # np.testing.assert_array_equal(agent.position,
            #                               [0, 0.75 * (step + 1), 0.75 * (step + 1)])
            if step >=1:
                np.testing.assert_array_equal(episode.calculate_reward(step, episode_done=True)[0], 0)
        self.update_agent_and_episode([0, .75, 0.75], agent, environmentInteraction, episode,  seq_len-1)
        # agent.perform_action([0, .75, .75], episode)
        # agent.update_agent_pos_in_episode(episode, seq_len-1)
        # agent.on_post_tick(episode)
        # episode.pedestrian_data[0].action[seq_len-2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[seq_len-1], [0, (seq_len-1)*.75, (seq_len-1)*.75])
        self.update_agent_and_episode([0, .75, 0.75], agent, environmentInteraction, episode, seq_len )
        np.testing.assert_approx_equal(episode.calculate_reward(seq_len-2, episode_done=True)[0], (seq_len-1)*np.sqrt(2)*.75)



    def test_dist_travelled_vel(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.distance_travelled] = 1
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        for step in range(seq_len-2):
            self.update_agent_and_episode([0, 1/np.sqrt(2), 1/np.sqrt(2)], agent, environmentInteraction, episode, step + 1)
            # agent.perform_action([0, 1/np.sqrt(2), 1/np.sqrt(2)], episode)
            # agent.update_agent_pos_in_episode(episode, step+1)
            # agent.on_post_tick(episode)
            # agent.update_metrics(episode)
            # episode.pedestrian_data[0].action[step] = 5
            np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[step+1], [0, 1/np.sqrt(2)*(step+1), 1/np.sqrt(2)*(step+1)])
            # np.testing.assert_array_almost_equal(agent.position,
            #                                      [0, 1 / np.sqrt(2) * (step + 1), 1 / np.sqrt(2) * (step + 1)])
            np.testing.assert_array_equal(episode.calculate_reward(step, episode_done=True)[0], 0)
        self.update_agent_and_episode([0, 1 / np.sqrt(2), 1 / np.sqrt(2)], agent, environmentInteraction, episode,seq_len-1)
        # agent.perform_action([0,1/np.sqrt(2), 1/np.sqrt(2)], episode)
        # agent.update_agent_pos_in_episode(episode, seq_len-1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[seq_len-2] = 5
        self.update_agent_and_episode([0, 1 / np.sqrt(2), 1 / np.sqrt(2)], agent, environmentInteraction, episode,
                                      seq_len )
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[seq_len-1], [0, 1/np.sqrt(2)*(seq_len-1), 1/np.sqrt(2)*(seq_len-1)])
        np.testing.assert_approx_equal(episode.calculate_reward(seq_len-2, episode_done=True)[0], seq_len-1)

    def test_cars_hit_on_one(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list=[]
        cars[1].append([0,2+1,0,0+1,1,1+1])
        cars_list.append([0,2+1,0,0+1,1,1+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_dict={0:cars_list}
        init_frames_cars={0:1}
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        #car_dict={}, init_frames_cars={}
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards,car_dict=cars_dict,init_frames_cars=init_frames_cars)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0.1])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0.1])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0, .9], agent, environmentInteraction, episode,1)
        # agent.perform_action([0, 0, .9], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 0,1])
        np.testing.assert_array_equal(agent.position, [0, 0,1])


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 0, 1])
        np.testing.assert_array_equal(agent.position, [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], -1)
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 3)
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)


    def test_cars_hit_on_one_vel(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list = []
        cars[1].append([0,2+1,0,0+1,1,1+1])
        cars_list.append([0,2+1,0,0+1,1,1+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_dict = {0: cars_list}
        init_frames_cars = {0: 1}
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards,car_dict=cars_dict,init_frames_cars=init_frames_cars)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0, .9], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0, .9], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 0, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[1], [0, 0, .9])
        # np.testing.assert_array_equal(agent.position, [0, 0, .9])


        self.update_agent_and_episode([0, 0, .8], agent, environmentInteraction, episode,2)
        # agent.perform_action([0, 0, .8], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 0, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[2], [0, 0, .9])
        # np.testing.assert_array_equal(agent.position, [0, 0, .9])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], -1)
        self.update_agent_and_episode([0, 0, .8], agent, environmentInteraction, episode, 3)
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)


    def test_cars_hit_on_two(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list=[]
        cars[1].append([0,2+1,0,0+1,1,1+1])
        cars_list.append([0,2+1,0,0+1,1,1+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_dict = {0: cars_list}
        init_frames_cars = {0: 1}
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards,car_dict=cars_dict,init_frames_cars=init_frames_cars)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 1, 0.1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 1, 0.1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 0.1])


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 1, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 0, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], -1)


        for itr in range(3, seq_len-1):
            self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, itr+1)
            # agent.perform_action([0, 0, -1], episode)
            # agent.update_agent_pos_in_episode(episode, itr+1)
            # agent.on_post_tick(episode)
            # agent.update_metrics(episode)
            # episode.pedestrian_data[0].action[itr] = 5
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[itr+1], [0, 1, 1.1])
            np.testing.assert_approx_equal(episode.calculate_reward(itr-1, episode_done=True)[0], 0)


    def test_cars_hit_on_two_vel(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list = []
        cars[1].append([0,2+1,0,0+1,1,1+1])
        cars_list.append([0,2+1,0,0+1,1,1+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_dict = {0: cars_list}
        init_frames_cars = {0: 1}
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards,car_dict=cars_dict,init_frames_cars=init_frames_cars)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 1.1, 0], agent, environmentInteraction, episode,  1)
        # agent.perform_action([0, 1.1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5

        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 0])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[1], [0, 1.1, 0])


        self.update_agent_and_episode([0, 0, 1.2], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1.2], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 1, 1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[2], [0, 1.1, 1.2])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, -.9], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 0, -.9], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[3], [0, 1.1, 1.2])
        #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], -1)
        #np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)

        for itr in range(3, seq_len-1):
            self.update_agent_and_episode([0, 0, -.96], agent, environmentInteraction, episode, itr+1)
            # agent.perform_action([0, 0, -.96], episode)
            # agent.update_agent_pos_in_episode(episode,itr+ 1)
            # agent.on_post_tick(episode)
            # agent.update_metrics(episode)
            # episode.pedestrian_data[0].action[itr] = 5
            #np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1])
            np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[itr+1], [0, 1.1, 1.2])
            np.testing.assert_approx_equal(episode.calculate_reward(itr-1, episode_done=True)[0], 0)

    def test_cars_follow_car(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list = []
        cars[1].append([0,2+1,0,0+1,1,1+1])
        cars_list.append([0,2+1,0,0+1,1,1+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_dict = {0: cars_list}
        init_frames_cars = {0: 1}
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards,car_dict=cars_dict,init_frames_cars=init_frames_cars)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 0, 0])


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 5)
        np.testing.assert_approx_equal(episode.calculate_reward(3, episode_done=True)[0], 0)

    def test_cars_follow_car_vel(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list=[]
        cars[1].append([0, 2+1, 0, 0+1, 1, 1+1])
        cars_list.append([0, 2+1, 0, 0+1, 1, 1+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_dict = {0: cars_list}
        init_frames_cars = {0: 1}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        tensor = np.ones(tensor.shape) * cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards,car_dict=cars_dict,init_frames_cars=init_frames_cars)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0, 0.1], agent, environmentInteraction, episode,1)
        # agent.perform_action([0, 0, 0.1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[1], [0, 0, 0.1])


        self.update_agent_and_episode([0, 0, 0.8], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, .8], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[2], [0, 0, .9])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 1.1, 0], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1.1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1.1,.9])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 1.2, 0.2], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, 1.2, 0.2], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [0, 2.3, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        self.update_agent_and_episode([0, 1.2, 0.2], agent, environmentInteraction, episode, 5)
        np.testing.assert_approx_equal(episode.calculate_reward(3, episode_done=True)[0], 0)

    def test_cars_hit_on_three(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list = []

        cars[1].append([0, 2+1, 0, 0+1, 1, 1+1])
        cars_list.append([0, 2+1, 0, 0+1, 1, 1+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])

        cars_dict = {0: cars_list}
        init_frames_cars = {0: 1}
        tensor = np.ones(tensor.shape) * cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards,car_dict=cars_dict,init_frames_cars=init_frames_cars)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode,1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 1])


        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 3)
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], -1)

    def test_cars_hit_on_three_vel(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list = []
        cars[1].append([0, 2+1, 0, 0+1, 1, 1+1])
        cars_list.append([0, 2+1, 0, 0+1, 1, 1+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_dict = {0: cars_list}
        init_frames_cars = {0: 1}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        tensor = np.ones(tensor.shape) * cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards,car_dict=cars_dict,init_frames_cars=init_frames_cars)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0.9, 1.1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0.9, 1.1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[1], [0, 0.9, 1.1])


        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, .9, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 3)
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], -1)

    def test_goal_reached_on_two(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) *cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 1
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)
        episode.follow_goal=True

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])
        episode.pedestrian_data[0].goal[0,:]=np.zeros(3)
        episode.pedestrian_data[0].goal[0,1] = 2
        episode.pedestrian_data[0].goal[0,2] = 2

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        #np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 0, 1])


        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 1, 1])


        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.dist_to_goal], np.sqrt(5))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.goal_reached], 0)


        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 1)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.dist_to_goal], np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.goal_reached], 1)

        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 4)
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2,PEDESTRIAN_MEASURES_INDX.dist_to_goal], np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, PEDESTRIAN_MEASURES_INDX.goal_reached], 1)

    def test_goal_reached_on_two_vel(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)
        episode.follow_goal = True

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])
        episode.pedestrian_data[0].goal[0,:]=np.zeros(3)
        episode.pedestrian_data[0].goal[0,1] = 2
        episode.pedestrian_data[0].goal[0,2] = 2

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0, 1.1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0, 1.1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 0, 1.1])


        self.update_agent_and_episode([0, .9, 0], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, .9, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, .9, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, 7], np.sqrt(4.81))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, 13], 0)



        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1.9, 2.1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 7], np.sqrt(2.02))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 13], 0)

        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 4)
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 1)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 7], np.sqrt(0.02))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 13], 1)

    def test_neg_dist_reward(self):
        seq_len = 4
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) *cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        #np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5

        self.update_agent_and_episode([0, 1,0 ], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5

        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 1-((2+np.sqrt(2))/np.sqrt(8)))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, 12], 2 + np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0,6], np.sqrt(8))



        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 1-((1+np.sqrt(2))/np.sqrt(5)))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 12],  1+np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 6], np.sqrt(5))


        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 2, 2])
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0],0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 12],np.sqrt(2) )
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 6], np.sqrt(2))


    def test_follow_agent_reward(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        people[0]=[np.array([[0,0],[0,0], [0,0]]) ]
        people[1] = [np.array([[0, 0], [1, 1], [1, 1]]),np.array([[0,0],[0,1], [0,0]])]
        people[2] = [np.array([[0, 0], [2, 2], [2, 2]]), np.array([[0, 0], [1, 2], [0, 1]]) ]
        people[3] = [np.array([[0, 0], [3, 3],[3, 3]]), np.array([[0, 0], [2, 3], [1, 2]])]
        people[4] = [np.array([[0, 0], [4, 4], [4, 4]]), np.array([[0, 0], [3, 4], [2, 3]])]

        init_frames={1:0, 2:1}
        agent_1=[np.array([[0,0],[0,0], [0,0]]),np.array([[0, 0], [1, 1], [1, 1]]), np.array([[0, 0], [2, 2], [2, 2]]), np.array([[0, 0], [3, 3],[3, 3]]),np.array([[0, 0], [4, 4], [4, 4]])]
        agent_2 = [ np.array([[0, 0], [0, 1], [0,0]]), np.array([[0, 0], [1, 2], [0, 1]]),
                   np.array([[0, 0], [2, 3], [1, 2]]), np.array([[0, 0], [3, 4], [2, 3]])]

        people_map={1:agent_1, 2:agent_2}

        tensor = np.ones(tensor.shape) * cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] = 1
        #(0-0, 0-1, 0-2, 0-3,0-4, 0-5, 0-6,0-7,0-8,0-9,0-10,1-11,0,0,0)
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards, people_dict=people_map, init_frames=init_frames)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None, initialization=1, init_key=0)
        print (episode.pedestrian_data[0].goal_person_id)
        np.testing.assert_array_equal(episode.pedestrian_data[0].goal_person_id_val, 1)
        #episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        #np.testing.assert_array_equal(agent.position, [0, 0, 0])
        #"Next action: [ 1:'downL', 2:'down', 3:'downR', 4:'left', 5:'stand', 6:'right',7:'upL', 8:'up', 9:'upR', 0: stop excecution] "
        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0]=8

        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 1])

        self.update_agent_and_episode([0, 1,0], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 7

        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 1)
        #np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, 12], np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0,PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init], np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.hit_pedestrians], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.one_step_prediction_error], 0)


        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0],1- (1/(2*episode.max_step)))
        #np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 12], np.sqrt(2)+1)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init], np.sqrt(5))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.hit_pedestrians], 1)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.one_step_prediction_error], 1)
        #
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_metrics(episode)
        # np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 2, 2])
        # np.testing.assert_approx_equal(episode.calculate_reward(2),1-((2+np.sqrt(2))/np.sqrt(8)))
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 12], 2 + np.sqrt(2))
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 4], 2*np.sqrt(2))


    def test_mov_penalty_reward(self):
        seq_len = 10
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])

        for frame in range(4):
            self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, frame+1)
            # agent.perform_action([0, 0, 1], episode)
            # agent.update_agent_pos_in_episode(episode, frame+1)
            # agent.on_post_tick(episode)
            # agent.update_metrics(episode)
            # episode.pedestrian_data[0].action[frame] = 5
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[frame+1], [0, 0, frame+1])
            if frame>1:
                np.testing.assert_approx_equal(episode.calculate_reward(frame-1, episode_done=True)[0],0)
                np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)

        for frame in range(4, 8):
            self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, frame + 1)
            # agent.perform_action([0, 1, 0], episode)
            # agent.update_agent_pos_in_episode(episode, frame+1)
            # agent.on_post_tick(episode)
            # agent.update_metrics(episode)
            # episode.pedestrian_data[0].action[frame] = 7
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[frame], [0, frame-4,4 ])

            np.testing.assert_approx_equal(episode.calculate_reward(frame-1, episode_done=True)[0], 0)
            np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[frame-1, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)

    # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]
    def test_mov_penalty_reward_two(self):
        seq_len = 14
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) *cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # episode.pedestrian_data[0].action[0] = 7
        # episode.pedestrian_data[0].velocity[0] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1,0])


        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 8
        # episode.pedestrian_data[0].velocity[1] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0,2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0,0,1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        # episode.pedestrian_data[0].velocity[2] = np.array([0, 0, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 2, 2])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, -1, 1], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, -1, 1], episode)
        # agent.update_agent_pos_in_episode(episode,4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 2
        # episode.pedestrian_data[0].velocity[3] = np.array([0, -1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [0, 1, 3])
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[2, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, -1, 0], agent, environmentInteraction, episode, 5)
        # agent.perform_action([0, -1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 1
        # episode.pedestrian_data[0].velocity[4] = np.array([0, -1, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [0, 0, 3])
        np.testing.assert_approx_equal(episode.calculate_reward(3, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[3, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)


        self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode, 6)
        # agent.perform_action([0, -1, -1], episode)
        # agent.update_agent_pos_in_episode(episode,6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 0
        # episode.pedestrian_data[0].velocity[5] = np.array([0, -1, -1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [0, -1, 2])
        np.testing.assert_approx_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[4, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)


        self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, 7)
        # agent.perform_action([0, 0, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 7)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[6] = 3
        # episode.pedestrian_data[0].velocity[6] = np.array([0, 0, -1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [0, -1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(5, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[5, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, 1, -1], agent, environmentInteraction, episode, 8)
        # agent.perform_action([0, 1, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 8)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[7] = 6
        # episode.pedestrian_data[0].velocity[7] = np.array([0, 1, -1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[8], [0, 0, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(6, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[6, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 9)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 9)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[8] = 7
        # episode.pedestrian_data[0].velocity[8] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[9], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(7, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[7, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)


        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 10)
        # agent.perform_action([0, 0, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 10)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[9] = 4
        # episode.pedestrian_data[0].velocity[9] = np.array([0, 0, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[10], [0, 1, 0]
                                      )
        np.testing.assert_approx_equal(episode.calculate_reward(8, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[8, 11], 0)


        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 11)
        # agent.perform_action([0, 0, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 11)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[10] = 4
        # episode.pedestrian_data[0].velocity[10] = np.array([0, 0, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[11], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(9, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[9, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 12)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 12)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[11] = 8
        # episode.pedestrian_data[0].velocity[11] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[12], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(10, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[10, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)

        np.testing.assert_approx_equal(episode.calculate_reward(11)[0],0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[11, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)


        # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]

    def test_mov_penalty_reward_three(self):
        seq_len = 14
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = -1
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 1, action_nbr=7)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 7
        # episode.pedestrian_data[0].velocity[0] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 0])


        self.update_agent_and_episode([0, 1, -1], agent, environmentInteraction, episode, 2,action_nbr=6)
        # agent.perform_action([0, 1, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 6
        # episode.pedestrian_data[0].velocity[1] = np.array([0, 1, -1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 2, -1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)


        self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, 3,action_nbr=3)
        # agent.perform_action([0, 0, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 3
        # episode.pedestrian_data[0].velocity[2] = np.array([0, 0, -1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 2, -2])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0,-1, -1], agent, environmentInteraction, episode, 4,action_nbr=0)
        # agent.perform_action([0, -1, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 0
        # episode.pedestrian_data[0].velocity[3] = np.array([0, -1, -1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [0, 1, -3])
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[2, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, -1, 0], agent, environmentInteraction, episode, 5,action_nbr=1)
        # agent.perform_action([0, -1, 0], episode),
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 1
        # episode.pedestrian_data[0].velocity[4] = np.array([0, -1, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [0, 0, -3])
        np.testing.assert_approx_equal(episode.calculate_reward(3, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[3, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, -1, 1], agent, environmentInteraction, episode, 6,action_nbr=2)
        # agent.perform_action([0, -1, +1], episode)
        # agent.update_agent_pos_in_episode(episode, 6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 2
        # episode.pedestrian_data[0].velocity[5] = np.array([0, -1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [0, -1, -2])
        np.testing.assert_approx_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[4, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 7,action_nbr=5)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 7)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[6] = 5
        # episode.pedestrian_data[0].velocity[6] = np.array([0, 0, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [0, -1, -1])
        np.testing.assert_approx_equal(episode.calculate_reward(5, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[5, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 8,action_nbr=8)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 8)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[7] = 8
        # episode.pedestrian_data[0].velocity[7] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[8], [0, 0, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(6, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[6, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 9,action_nbr=7)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 9)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[8] = 7
        # episode.pedestrian_data[0].velocity[8] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[9], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(7, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[7, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 10,action_nbr=4)
        # agent.perform_action([0, 0, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 10)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[9] = 4
        # episode.pedestrian_data[0].velocity[9] = np.array([0, 0, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[10], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(8, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[8, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 11,action_nbr=8)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 11)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[10] = 8
        # episode.pedestrian_data[0].velocity[10] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[11], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(9, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[9, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)
        np.testing.assert_approx_equal(episode.calculate_reward(10)[0], -np.sin(np.pi/8))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[10, PEDESTRIAN_MEASURES_INDX.change_in_direction], np.sin(np.pi/8))

    def test_mov_penalty_zig_zag(self):
        seq_len = 14
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) *cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = -1
        agent, episode, environmentInteraction =  self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])

        # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 1, action_nbr=7)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 7
        # episode.pedestrian_data[0].velocity[0]=np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 0])


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2, action_nbr=5)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        # episode.pedestrian_data[0].velocity[1] = np.array([0, 0, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)

        self.update_agent_and_episode([0, 1,0], agent, environmentInteraction, episode, 3, action_nbr=7)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 7
        # episode.pedestrian_data[0].velocity[2] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)



        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 4, action_nbr=7)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 7
        # episode.pedestrian_data[0].velocity[3] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [0, 3, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], -1 / np.sqrt(2))
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[2, PEDESTRIAN_MEASURES_INDX.change_in_direction], 1 / np.sqrt(2))



        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 5, action_nbr=5)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 5
        # episode.pedestrian_data[0].velocity[4] = np.array([0, 0, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [0, 3, 2])
        np.testing.assert_approx_equal(episode.calculate_reward(3, episode_done=True)[0], -1 / np.sqrt(2))
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[3, PEDESTRIAN_MEASURES_INDX.change_in_direction], 1 / np.sqrt(2))



        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 6, action_nbr=8)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 8
        # episode.pedestrian_data[0].velocity[5] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [0, 4, 3])
        np.testing.assert_approx_equal(episode.calculate_reward(4, episode_done=True)[0], -1 / np.sqrt(2))
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[4, PEDESTRIAN_MEASURES_INDX.change_in_direction], 1 / np.sqrt(2))




        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 7, action_nbr=8)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 7)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[6] = 8
        # episode.pedestrian_data[0].velocity[6] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [0, 5, 4])
        np.testing.assert_approx_equal(episode.calculate_reward(5, episode_done=True)[0], -np.sin(np.pi / 8))
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[5, PEDESTRIAN_MEASURES_INDX.change_in_direction], np.sin(np.pi / 8))




        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 8, action_nbr=8)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 8)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[7] = 8
        # episode.pedestrian_data[0].velocity[7] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[8], [0, 6, 5])
        np.testing.assert_approx_equal(episode.calculate_reward(6, episode_done=True)[0], -np.sin(np.pi / 8))
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[6, PEDESTRIAN_MEASURES_INDX.change_in_direction], np.sin(np.pi / 8))




        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 9, action_nbr=8)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 9)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[8] = 8
        # episode.pedestrian_data[0].velocity[8] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[9], [0, 7, 6])
        np.testing.assert_approx_equal(episode.calculate_reward(7, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[7, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)
        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 10, action_nbr=8)
        np.testing.assert_approx_equal(episode.calculate_reward(8, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[8, PEDESTRIAN_MEASURES_INDX.change_in_direction], 0)







