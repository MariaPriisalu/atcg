import unittest
import numpy as np
import sys

from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE,NBR_MEASURES,PEDESTRIAN_MEASURES_INDX
from RL.environment_interaction import EntitiesRecordedDataSource, EnvironmentInteraction
from commonUtils.ReconstructionUtils import LAST_CITYSCAPES_SEMLABEL, cityscapes_labels_dict, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS

class TestEnv(unittest.TestCase):

    def get_reward(self, all_zeros=False):

        rewards = np.zeros(NBR_REWARD_WEIGHTS)
        if all_zeros:
            return rewards

        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        rewards[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] = 1
        rewards[PEDESTRIAN_REWARD_INDX.on_pavement] = 1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        rewards[PEDESTRIAN_REWARD_INDX.distance_travelled] = 1
        rewards[PEDESTRIAN_REWARD_INDX.out_of_axis] = -1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        return rewards

    def update_episode(self,environmentInteraction, episode, next_frame):
       
        observation, observation_dict = environmentInteraction.getObservation(frameToUse=next_frame)
        episode.update_pedestrians_and_cars(observation.frame,
                                            observation_dict,
                                            observation.people_dict,
                                            observation.cars_dict,
                                            observation.pedestrian_vel_dict,
                                            observation.car_vel_dict)

    # Help function. Setup for tests.
    def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor,seq_len=30, rewards=[], agent_size=[], people_dict={}, init_frames={},car_dict={},init_frames_cars={} ):
        settings = run_settings()
        if len(agent_size)==0:
            agent_size=[0,0,0]
        settings.agent_shape=agent_size
        settings.useRLToyCar=False
        settings.useHeroCar =False
        settings.number_of_car_agents = 0  #
        settings.number_of_agents = 1  #
        settings.pedestrian_view_occluded = False
        settings.field_of_view = 2 * np.pi  # 114/180*np.pi
        settings.field_of_view_car = 2 * np.pi  # 90/180*np.pi
        settings.multiplicative_reward_pedestrian=False
        settings.goal_dir=False
        if len(rewards)==0:
            rewards = self.get_reward()

        if settings.useRealTimeEnv:
            entitiesRecordedDataSource=EntitiesRecordedDataSource(init_frames=init_frames,
                                       init_frames_cars=init_frames_cars,
                                       cars_sample=cars,
                                       people_sample=people,
                                       cars_dict_sample=car_dict,
                                       people_dict_sample=people_dict,
                                       cars_vel={},
                                       ped_vel={},
                                       reconstruction=tensor,
                                       forced_num_frames=None)
            environmentInteraction = EnvironmentInteraction(False, entitiesRecordedDataSource=entitiesRecordedDataSource, parentEnvironment=None,args=settings)
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
            init_frames_cars =entitiesRecordedDataSource.init_frames_cars  # observation.init_frames_cars
            episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, settings.gamma,
                                    seq_len, rewards,rewards,agent_size=agent_size,
                                    people_dict=people_dict,
                                    cars_dict=cars_dict,
                                    people_vel=people_vel_dict,
                                    cars_vel=car_vel_dict,
                                    init_frames=init_frames, follow_goal=settings.goal_dir,
                                    action_reorder=settings.reorder_actions,
                                    threshold_dist=settings.threshold_dist, init_frames_cars=init_frames_cars,
                                    temporal=settings.temporal, predict_future=settings.predict_future,
                                    run_2D=settings.run_2D, agent_init_velocity=settings.speed_input,
                                    velocity_actions=settings.velocity or settings.continous, end_collide_ped=settings.end_on_bit_by_pedestrians,
                                    stop_on_goal=settings.stop_on_goal, waymo=settings.waymo,
                                    centering=(0,0),
                                    defaultSettings=settings,
                                    multiplicative_reward_pedestrian=settings.multiplicative_reward_pedestrian,
                                    multiplicative_reward_initializer=settings.multiplicative_reward_initializer,
                                    learn_goal=settings.learn_goal or settings.separate_goal_net,
                                    use_occlusion=settings.use_occlusion,
                                    useRealTimeEnv=settings.useRealTimeEnv, car_vel_dict=car_vel_dict,
                                    people_vel_dict=people_vel_dict, car_dim=settings.car_dim,
                                    new_carla=settings.new_carla, lidar_occlusion=settings.lidar_occlusion,
                                    use_car_agent=settings.useRLToyCar or settings.useHeroCar,
                                    use_pfnn_agent=settings.pfnn, number_of_agents=settings.number_of_agents,
                                    number_of_car_agents=settings.number_of_car_agents)  # To DO:  Check if needed heroCarDetails = heroCarDetails
            episode.environmentInteraction =environmentInteraction
        else:
            environmentInteraction=None
            episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards, rewards,
                                    agent_size=agent_size,
                                    people_dict=people_dict, init_frames=init_frames, cars_dict=car_dict,
                                    init_frames_cars=init_frames_cars, defaultSettings=settings, centering={},
                                    useRealTimeEnv=settings.useRealTimeEnv)
        agent=SimplifiedAgent(settings)
        agent.id=0
        if settings.useRealTimeEnv:
            heroAgentPedestrians=[agent]

            realTimeEnvObservation, observation_dict = environmentInteraction.reset(heroAgentCars=[],
                                                                                         heroAgentPedestrians=heroAgentPedestrians,
                                                                                         episode=episode)
            episode.update_pedestrians_and_cars(realTimeEnvObservation.frame,
                                                observation_dict,
                                                realTimeEnvObservation.people_dict,
                                                realTimeEnvObservation.cars_dict,
                                                realTimeEnvObservation.pedestrian_vel_dict,
                                                realTimeEnvObservation.car_vel_dict)
            environmentInteraction.frame=0
        return agent, episode, environmentInteraction

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
        pos, i, vel = episode.initial_position(0, None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)

        agent.initial_position(pos,episode.pedestrian_data[0].goal[0,:])

    def update_agent_and_episode(self, action, agent, environmentInteraction, episode, frame,action_nbr=4):
        if frame<episode.seq_len:
            episode.pedestrian_data[0].action[frame - 1] = action_nbr
            episode.pedestrian_data[0].velocity[frame - 1] = action
        episode.get_agent_neighbourhood(0, agent.position, episode.agent_size[1:], frame-1)
        if frame <episode.seq_len:
            environmentInteraction.signal_action({agent:action}, updated_frame=frame)
            agent.perform_action(action, episode)
            # Do the simulation for next tick using decisions taken on this tick
            # If an online realtime env is used this call will fill in the data from the simulator.
            # If offline, it will take needed data from recorded/offline data.
            environmentInteraction.tick(frame)
            self.update_episode(environmentInteraction, episode, frame)
            agent.update_agent_pos_in_episode(episode, frame)
            agent.on_post_tick(episode)
            agent.update_metrics(episode)
            episode.pedestrian_data[0].action[frame-1] = 5

    def evaluate_measure(self, frame, episode, non_zero_measure, expeceted_value_of_non_zero_measure):
        for measure in range(NBR_MEASURES):
            if measure == non_zero_measure:
                np.testing.assert_array_equal(
                    episode.pedestrian_data[0].measures[frame, non_zero_measure], expeceted_value_of_non_zero_measure)
            else:
                if measure!=PEDESTRIAN_MEASURES_INDX.dist_to_final_pos and measure !=PEDESTRIAN_MEASURES_INDX.change_in_direction and measure != PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal_from_current and measure != PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap and measure != PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.change_in_pose and measure != PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car:
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
            print("Measure "+str(measure)+" value "+str( episode.pedestrian_data[0].measures[frame, measure]))
            np.testing.assert_array_equal(
                episode.pedestrian_data[0].measures[frame, measure], dict[measure])
        for measure in zero_measures:
            if measure!=PEDESTRIAN_MEASURES_INDX.dist_to_final_pos and measure !=PEDESTRIAN_MEASURES_INDX.change_in_direction and measure != PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal_from_current and measure != PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap and measure != PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.change_in_pose and measure != PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car:
                print("Measure " + str(measure) + " value " + str(episode.pedestrian_data[0].measures[frame, measure]))
                np.testing.assert_array_equal(episode.pedestrian_data[0].measures[frame, measure], 0)

    # Test correct empty initialization.
    def test_walk_into_objs(self):

        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(15)
        tensor[1,1,1,CHANNELS.semantic]=cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        tensor[2, 1, 2, CHANNELS.semantic] = cityscapes_labels_dict['building'] / NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode, environmentInteraction=self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=15, rewards=rewards)

        self.initialize_pos(agent, episode)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])

        # Invalid move
        self.update_agent_and_episode([0,0,1],agent, environmentInteraction, episode, 1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])


        # Invalid move

        self.update_agent_and_episode([0, 0, 1],agent, environmentInteraction, episode,2)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])

        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)
        self.evaluate_measures(0, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1, PEDESTRIAN_MEASURES_INDX.iou_pavement: 1})


        # Valid move

        self.update_agent_and_episode([0, 1, 0],agent, environmentInteraction, episode,3)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [1, 2, 1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])

        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], -1)
        self.evaluate_measures(1, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1, PEDESTRIAN_MEASURES_INDX.iou_pavement: 1})


        # Invalid move

        self.update_agent_and_episode([0, -1, 1], agent, environmentInteraction, episode, 4)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [1, 2, 1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])

        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        self.evaluate_measures(2, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0, PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})


        # Valid move

        self.update_agent_and_episode([0, 0, 1],agent, environmentInteraction, episode, 5)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [1, 2, 2])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])

        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], -1)
        self.evaluate_measures(3, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1, PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})

        # Invalid move

        self.update_agent_and_episode([0, -1, 0],agent, environmentInteraction, episode, 6)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [1, 2, 2])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])

        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
        self.evaluate_measures(4, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0, PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})


        # Valid move

        self.update_agent_and_episode([0, 0, -1],agent, environmentInteraction, episode, 7)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [1, 2, 1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])

        np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], -1)
        self.evaluate_measures(5, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1, PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})


        # Valid move

        self.update_agent_and_episode([0, -1,0],agent, environmentInteraction, episode, 8)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[8], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])

        np.testing.assert_array_equal(episode.calculate_reward(6, episode_done=True)[0], 0)
        self.evaluate_measures(6, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0, PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})

        # Valid move

        self.update_agent_and_episode([0, -1, 0],agent, environmentInteraction, episode, 9)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[9], [1, 0, 1])
        np.testing.assert_array_equal(agent.position, [1, 0, 1])

        np.testing.assert_array_equal(episode.calculate_reward(7, episode_done=True)[0], 0)
        self.evaluate_measures(7, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0, PEDESTRIAN_MEASURES_INDX.iou_pavement: 1})

        # InValid move

        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 10)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[10], [1, 0, 1])
        np.testing.assert_array_equal(agent.position, [1, 0, 1])

        np.testing.assert_array_equal(episode.calculate_reward(8, episode_done=True)[0], 0)
        self.evaluate_measures(8, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0, PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})


        # Valid move
        self.update_agent_and_episode([0, 0, 1],agent, environmentInteraction, episode, 11)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[11], [1, 0, 2])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])

        np.testing.assert_array_equal(episode.calculate_reward(9, episode_done=True)[0], -1)
        self.evaluate_measures(9, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1, PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})


        # Invalid move
        self.update_agent_and_episode([0, 1, 0],agent, environmentInteraction, episode, 12)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[12], [1, 0, 2])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])

        np.testing.assert_array_equal(episode.calculate_reward(10, episode_done=True)[0], 0)
        self.evaluate_measures(10, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0, PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})

        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 13)
        np.testing.assert_array_equal(episode.calculate_reward(11, episode_done=True)[0], -1)
        self.evaluate_measures(11, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1, PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})


    # Test correct empty initialization.
    #[0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]
    def test_walk_into_objs_wide(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(17, size=7)
        tensor[1, 3, 3, CHANNELS.semantic] = cityscapes_labels_dict['sidewalk'] / NUM_SEM_CLASSES
        tensor[2, 3, 5, CHANNELS.semantic] = cityscapes_labels_dict['building'] / NUM_SEM_CLASSES
        rewards = self.get_reward(all_zeros=True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        print(" rewards "+str(rewards))
        agent, episode,environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=17, rewards=rewards, agent_size = [0, 1, 1])
        episode.agent_size = [0, 1, 1]

        self.initialize_pos(agent, episode)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])

        # Invalid move

        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])

        # Invalid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])

        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)
        self.evaluate_measures(0, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 1.0 / (3 * 3)})


        # Valid move
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 3)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [1, 4, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])

        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], -1.0)
        self.evaluate_measures(1, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 1.0 / (3 * 3)})


        # Invalid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode,4)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [1, 4, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])

        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        self.evaluate_measures(2, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 1.0 / (3 * 3)})



        # Valid move
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 5)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [1, 5, 3])
        np.testing.assert_array_equal(agent.position, [1, 5, 3])

        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], -1)
        self.evaluate_measures(3, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 1.0 / (3 * 3)})

        # Invalid move
        self.update_agent_and_episode([0, -1, 1], agent, environmentInteraction, episode,6)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [1, 5, 3])
        np.testing.assert_array_equal(agent.position, [1, 5, 3])

        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
        self.evaluate_measures(4, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})



        # Valid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 7)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [1, 5, 4])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])

        np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], -1)
        self.evaluate_measures(5, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})

        np.testing.assert_array_equal(episode.calculate_reward(6)[0], 0)
        self.evaluate_measures(6, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})
        # Invalid move
        self.update_agent_and_episode([0, -1, 1], agent, environmentInteraction, episode, 8)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[8], [1, 5, 4])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])

        # Invalid move
        self.update_agent_and_episode([0, -1, 0], agent, environmentInteraction, episode, 9)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[9], [1, 5, 4])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])

        np.testing.assert_array_equal(episode.calculate_reward(7, episode_done=True)[0], -1)
        self.evaluate_measures(7, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})


        # valid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 10)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[10], [1, 5, 5])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])

        np.testing.assert_array_equal(episode.calculate_reward(8, episode_done=True)[0], -1)
        self.evaluate_measures(8, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})



        # Invalid move
        self.update_agent_and_episode([0, -1, 0], agent, environmentInteraction, episode, 11)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[11], [1, 5, 5])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])

        np.testing.assert_array_equal(episode.calculate_reward(9, episode_done=True)[0], 0)
        self.evaluate_measures(9, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})



        # valid move
        self.update_agent_and_episode([0,0, 1], agent, environmentInteraction, episode, 12)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[12], [1, 5,6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])

        np.testing.assert_array_equal(episode.calculate_reward(10, episode_done=True)[0], -1)
        self.evaluate_measures(10, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0})


        # Invalid move
        self.update_agent_and_episode([0, -1, 0], agent, environmentInteraction, episode, 13)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[13], [1, 5, 6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])

        np.testing.assert_array_equal(episode.calculate_reward(11, episode_done=True)[0], 0)
        self.evaluate_measures(11, episode, {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0,
                                             PEDESTRIAN_MEASURES_INDX.iou_pavement: 0,
                                             PEDESTRIAN_MEASURES_INDX.out_of_axis: 1})



        # valid move
        self.update_agent_and_episode([0, 0, 1], agent,  environmentInteraction, episode, 14)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[14], [1, 5, 7])
        np.testing.assert_array_equal(agent.position, [1, 5, 7])

        np.testing.assert_array_equal(episode.calculate_reward(12, episode_done=True)[0], -1)
        self.evaluate_measures(12, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0, PEDESTRIAN_MEASURES_INDX.out_of_axis: 1})



        # valid move
        self.update_agent_and_episode([0, -1, 0], agent,  environmentInteraction, episode, 15)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[15], [1, 4, 7])
        np.testing.assert_array_equal(agent.position, [1, 4, 7])

        np.testing.assert_array_equal(episode.calculate_reward(13, episode_done=True)[0], 0)
        self.evaluate_measures(13, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0, PEDESTRIAN_MEASURES_INDX.out_of_axis: 1})


        # invalid move
        self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, 16)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[16], [1, 4, 7])
        np.testing.assert_array_equal(agent.position, [1, 4, 7])
        np.testing.assert_array_equal(episode.calculate_reward(14, episode_done=True)[0], 0)
        self.evaluate_measures(14, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0, PEDESTRIAN_MEASURES_INDX.out_of_axis: 1})

        self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, 17)
        np.testing.assert_array_equal(episode.calculate_reward(15, episode_done=True)[0], -1)

        self.evaluate_measures(15, episode,
                               {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1,
                                PEDESTRIAN_MEASURES_INDX.iou_pavement: 0, PEDESTRIAN_MEASURES_INDX.out_of_axis: 1})



    def test_pedestrian_trajectory_measure(self):
        # In real time setting the previous agent position is always rewarded for being on the pedestrian trajectory.
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(8)
        people[0].append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people[1].append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))

        people[0].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people[0].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        # rewards=np.zeros(15)
        # rewards[9]=1
        # rewards[7] =-1
        people_dict={0:[people[0][0],people[1][0]],1:[people[0][1]],2:[people[0][2]]}
        init_frames={0:0,1:0,2:0}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=8,  rewards=rewards, people_dict=people_dict, init_frames=init_frames)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian, init_key=0)

        episode.pedestrian_data[0].agent[0]=[0,0,0]
        # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]

        # Check that initialized on pedestrian trajectory
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        # Positive reward for on pedestrian trajectory. Following pedestrian.
        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 1])


        # Positive reward for on pedestrian trajectory. Following pedestrian.
        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 2)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 2, 2])

        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0],
                                      1)  # No negative reward because it is the agent we are trying to follow. Should be +1 reward for staying on pedestrian trajecory.
        self.evaluate_measure(0, episode, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory, 1)



        # No reward for on pedestrian trajectory (because moved closer to initial location than previous position).
        self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode, 3)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1])

        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], 1)
        self.evaluate_measure(1, episode, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory, 1)


        # No reward for on pedestrian trajectory  (because moved closer to initial location than previous position).
        self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode, 4)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [0, 0, 0])

        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        self.evaluate_measure(2, episode, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory, 1)


        # Positive reward for on pedestrian trajectory.
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 5)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[5], [0, 0, 1])

        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], 0)
        self.evaluate_measure(3, episode, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory, 1)


        # No reward for on pedestrian trajectory. Not on pedestrian trajectory.
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 6)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[6], [0, 0, 2])

        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
        self.evaluate_measure(4, episode, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory, 0)


        # Positive reward for on pedestrian trajectory.
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 7)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[7], [0, 1, 2])
        np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], 1)
        self.evaluate_measure(5, episode, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory, 1)

        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 8)
        np.testing.assert_array_equal(episode.calculate_reward(6, episode_done=True)[0], 0)
        self.evaluate_measure(6, episode, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory, 0)


    def test_pedestrians_on_diagonal_pedestrian_trajectory_measure(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        people_dict={}
        init_frames={}
        for i in range(7):
            people[0].append(np.array([0, 2, i, i, i, i]).reshape((3, 2)))
            people_dict[i]=[people[0][-1]]

            init_frames[i]=0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode,environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=14, rewards=rewards,people_dict=people_dict, init_frames=init_frames)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0, None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian)
        episode.pedestrian_data[0].agent[0] = [0, 0, 0]

        agent.initial_position( episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        for i in range(1,7):
            self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, i)
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[i], [0, i,i])
            if i>=2:
                np.testing.assert_array_equal(episode.calculate_reward(i-2, episode_done=True)[0], 1)
                self.evaluate_measure(i-2, episode, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory, 1)

    def test_pedestrians_on_diagonal_pedestrian_trajectory_measure_backwards(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        people_dict = {}
        init_frames = {}
        for i in range(7):
            people[0].append(np.array([0, 2, i, i, i, i]).reshape((3, 2)))
            # people[1].append(np.array([0, 2, i, i, i, i]).reshape((3, 2)))
            people_dict[i] = [people[0][-1]]

            init_frames[i] = 0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                         seq_len=14, rewards=rewards,
                                                                         people_dict=people_dict,
                                                                         init_frames=init_frames)
        episode.agent_size = [0, 0, 0]
        episode.pedestrian_data[0].agent[0] = [0, 6, 6]
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])

        for i in range( 1,7):
            self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode, i)
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[i], [0, 6-i, 6-i])
            if i >= 2:
                np.testing.assert_array_equal(episode.calculate_reward(i-2, episode_done=True)[0], 1)
                self.evaluate_measure(i-2, episode, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory, 1)

    def test_dist_travelled(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.distance_travelled] = 1


        agent, episode,environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0, None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        for step in range(seq_len-2):
            self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, step+1)
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[step+1], [0, step+1, step+1])
            if step>1:
                np.testing.assert_array_equal(episode.calculate_reward(step-1, episode_done=True)[0], 0)
                self.evaluate_measure(step-1, episode,PEDESTRIAN_MEASURES_INDX.iou_pavement, 7)
                np.testing.assert_approx_equal(
                    episode.pedestrian_data[0].measures[step-1, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init],
                    (step) * np.sqrt(2))
        np.testing.assert_array_equal(episode.calculate_reward(seq_len-3, episode_done=True)[0], 0)
        self.evaluate_measure(seq_len-3, episode, PEDESTRIAN_MEASURES_INDX.iou_pavement, 7)
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[seq_len-3, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init],
            (seq_len-2) * np.sqrt(2))

        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, seq_len-1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[seq_len-1], [0, seq_len-1, seq_len-1])
        np.testing.assert_approx_equal(episode.calculate_reward(seq_len-2, episode_done=True)[0], (seq_len-1)*np.sqrt(2))
        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, seq_len)
        self.evaluate_measure(seq_len-2, episode, PEDESTRIAN_MEASURES_INDX.iou_pavement, 7)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[seq_len-2, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init], (seq_len-1)*np.sqrt(2))



    def test_cars_hit_on_one(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0,2+ 1,0,0+ 1,1,1+ 1])
        cars[2].append([0, 2+ 1, 1, 1+ 1, 1, 1+ 1])
        cars[3].append([0, 2+ 1, 2, 2+ 1, 1, 1+ 1])
        car_dict={0:[[0,2+ 1,0,0+ 1,1,1+ 1],[0, 2+ 1, 1, 1+ 1, 1, 1+ 1],[0, 2+ 1, 2, 2+ 1, 1, 1+ 1]]}
        cars_init={0:1}

        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['road']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode,environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards,car_dict=car_dict,init_frames_cars=cars_init )
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0, None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0],episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        # Agent hits car
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 0, 1])

        # Agent dead
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode,2)
        episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 0, 1])

        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], -1)
        self.evaluate_measure(0, episode, PEDESTRIAN_MEASURES_INDX.hit_by_car, 1)

        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode,3)
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)
        self.evaluate_measures(1, episode, {PEDESTRIAN_MEASURES_INDX.agent_dead:1, PEDESTRIAN_MEASURES_INDX.hit_by_car:1})


    def test_cars_hit_on_two(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0,2+ 1,0,0+ 1,1,1+ 1])
        cars[2].append([0, 2+ 1, 1, 1+ 1, 1, 1+ 1])
        cars[3].append([0, 2+ 1, 2, 2+ 1, 1, 1+ 1])
        car_dict = {0: [[0, 2+ 1, 0, 0+ 1, 1, 1+ 1], [0, 2+ 1, 1, 1+ 1, 1, 1+ 1], [0, 2+ 1, 2, 2+ 1, 1, 1+ 1]]}
        cars_init = {0: 1}
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['road']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode,environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards,car_dict=car_dict,init_frames_cars=cars_init)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0, None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])
        # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        # Valid move
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 0])


        # Agent hits car
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 1, 1])

        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, 0], 0)
        self.evaluate_measure(0, episode, PEDESTRIAN_MEASURES_INDX.hit_by_car, 0)



        # Agent dead
        self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, 3)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1])

        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], -1)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 0], 1)
        self.evaluate_measure(1, episode, PEDESTRIAN_MEASURES_INDX.hit_by_car, 1)


        # Agent dead
        for itr in range(3, seq_len-1):
            self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, itr+1)
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[itr+1], [0, 1, 1])

            if itr==3:
                np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
                np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 0], 1)
                self.evaluate_measures(2, episode,
                                       {PEDESTRIAN_MEASURES_INDX.agent_dead: 1, PEDESTRIAN_MEASURES_INDX.hit_by_car: 1})
            else:

                np.testing.assert_approx_equal(episode.calculate_reward(itr-1, episode_done=True)[0], 0)
                self.evaluate_measures(itr-1, episode,
                                       {PEDESTRIAN_MEASURES_INDX.agent_dead: 1, PEDESTRIAN_MEASURES_INDX.hit_by_car: 1})


    def test_cars_follow_car(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0,2+ 1,0,0+ 1,1,1+ 1])
        cars[2].append([0, 2+ 1, 1, 1+ 1, 1, 1+ 1])
        cars[3].append([0, 2+ 1, 2, 2+ 1, 1, 1+ 1])
        car_dict = {0: [[0, 2+ 1, 0, 0+ 1, 1, 1+ 1], [0, 2+ 1, 1, 1+ 1, 1, 1+ 1], [0, 2+ 1, 2, 2+ 1, 1, 1+ 1]]}
        cars_init = {0: 1}
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['road']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode,environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                                        rewards=rewards,car_dict=car_dict,init_frames_cars=cars_init)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0, None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        #0
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        #1 Agent is one step behind car - 0 reward
        self.update_agent_and_episode([0, 0, 0], agent, environmentInteraction, episode, 1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 0, 0])


        #2 Agent is one step behind car - 0 reward
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        self.evaluate_measure(0, episode, PEDESTRIAN_MEASURES_INDX.hit_by_car, 0)


        #3 Agent is one step behind car - 0 reward
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 3)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)  # 0
        self.evaluate_measure(1, episode, PEDESTRIAN_MEASURES_INDX.hit_by_car, 0)

        #4 Agent is one step behind car - 0 reward
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 4)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)  # 0
        self.evaluate_measure(2, episode, PEDESTRIAN_MEASURES_INDX.hit_by_car, 0)

        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 5)
        np.testing.assert_approx_equal(episode.calculate_reward(3, episode_done=True)[0], 0)#0
        self.evaluate_measure(3, episode, PEDESTRIAN_MEASURES_INDX.hit_by_car, 0)

    def test_cars_hit_on_three(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0, 2+ 1, 0, 0+ 1, 1, 1+ 1])
        cars[2].append([0, 2+ 1, 1, 1+ 1, 1, 1+ 1])
        cars[3].append([0, 2+ 1, 2, 2+ 1, 1, 1+ 1])
        car_dict = {0: [[0, 2+ 1, 0, 0+ 1, 1, 1+ 1], [0, 2+ 1, 1, 1+ 1, 1, 1+ 1], [0, 2+ 1, 2, 2+ 1, 1, 1+ 1]]}
        cars_init = {0: 1}
        tensor = np.ones(tensor.shape) * cityscapes_labels_dict['road'] / NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode,environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards,car_dict=car_dict,init_frames_cars=cars_init)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0, None)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        # 1 : Valid move
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [0, 1, 0])


        # 2: Valid move
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 2)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], [0, 2, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        self.evaluate_measure(0, episode, PEDESTRIAN_MEASURES_INDX.hit_by_car, 0)


        # 2: Invalid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 3)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)
        self.evaluate_measures(1, episode,
                               {PEDESTRIAN_MEASURES_INDX.agent_dead: 0, PEDESTRIAN_MEASURES_INDX.hit_by_car: 0})
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 4)

        np.testing.assert_approx_equal(episode.calculate_reward(2,episode_done=True)[0], -1)
        self.evaluate_measures(2, episode,
                               {PEDESTRIAN_MEASURES_INDX.agent_dead:0, PEDESTRIAN_MEASURES_INDX.hit_by_car:1})