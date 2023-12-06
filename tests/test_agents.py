import unittest
import numpy as np
import sys

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
        pos, i,vel = episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
        episode.vel_init = np.zeros(3)
        agent.initial_position(pos,episode.pedestrian_data[0].goal[0,:])


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
            agent =  ContinousAgent(settings)

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
            episode.pedestrian_data[0].action[frame - 1] = action_nbr

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
    # Test correct empty initialization.
    def test_walk_into_objs(self):
        seq_len=11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        tensor[1,1,1,CHANNELS.semantic]=cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        tensor[2, 1, 2,  CHANNELS.semantic]=cityscapes_labels_dict['building']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)

        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agent, episode)

        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(agent.pos_exact, [1, 1, 1])

        # Invalid move
        self.update_agent_and_episode([0,0,1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0,0,1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[agent.id].action[0] = 5
        #print("Agent 1 " + str(episode.pedestrian_data[0].agent[0][1:]+[0,1]) +" avoid (1,2) ------------------------------------------")
        np.testing.assert_array_equal( [1,1],episode.pedestrian_data[0].agent[1][1:])
        np.testing.assert_array_equal([ 1, 1],agent.pos_exact[1:])


        # Invalid move
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[agent.id].action[1] = 5
        #print("Agent 2 " + str(episode.pedestrian_data[0].agent[1][1:] + [0, 1]) + " avoid (1,2) ------------------------------------------")
        np.testing.assert_array_equal( [1, 1], episode.pedestrian_data[0].agent[2][1:])
        np.testing.assert_array_equal([1, 1], agent.pos_exact[1:])
        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.hit_obstacles], 1)

        # Valid move
        self.update_agent_and_episode([0, 1,0], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        #print("Agent 3 " + str(episode.pedestrian_data[0].agent[2][1:] + [1, 0]) + " avoid (1,2) ------------------------------------------")
        np.testing.assert_array_less(  episode.pedestrian_data[0].agent[2][1], episode.pedestrian_data[0].agent[3][1])
        np.testing.assert_array_almost_equal( episode.pedestrian_data[0].agent[2][0], episode.pedestrian_data[0].agent[3][0],decimal=0)
        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], -1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.hit_obstacles], 1)




        # Invalid move
        self.update_agent_and_episode([0, -1, 1], agent, environmentInteraction, episode, 4)
        #agent.perform_action([0, -1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        # print (episode.pedestrian_data[0].agent)
        #print("Agent 4 " + str(episode.pedestrian_data[0].agent[3][1:] + [-1, 1]) + " avoid (1,2) ------------------------------------------")
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], episode.pedestrian_data[0].agent[3])
        np.testing.assert_array_equal(agent.pos_exact,episode.pedestrian_data[0].agent[3])
        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[2, PEDESTRIAN_MEASURES_INDX.hit_obstacles], 0)


        # valid move
        self.update_agent_and_episode([0, -1,0], agent, environmentInteraction, episode, 5)
        # agent.perform_action([0, -1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 5
        #print("Agent 5 " + str(episode.pedestrian_data[0].agent[4][1:] + [-1, 0]) + " avoid (1,2) ------------------------------------------")
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[5][1], episode.pedestrian_data[0].agent[4][1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[4][0], episode.pedestrian_data[0].agent[5][0], decimal=0)
        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], -1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[3, PEDESTRIAN_MEASURES_INDX.hit_obstacles], 1)

        self.update_agent_and_episode([0, -1, 0], agent, environmentInteraction, episode, 6)
        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[4, PEDESTRIAN_MEASURES_INDX.hit_obstacles], 0)




    def test_pedestrians(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        people_list=[]
        people[1].append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people_list.append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people[2].append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people_list.append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people[3].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people_list.append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people[4].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        people_list.append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        people[5].append(np.array([0, 2, 0, 0, 1, 1]).reshape((3, 2)))
        people_list.append(np.array([0, 2, 0, 0, 1, 1]).reshape((3, 2)))

        people_dict={0:people_list}
        people_init={0:1}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        #agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=7,  rewards=rewards)
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  seq_len=seq_len, rewards=rewards,people_dict=people_dict, init_frames=people_init)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.vel_init = np.zeros(3)
        episode.pedestrian_data[0].agent[0]=[0,0,0]

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5
        # print("Agent 1 "+str(episode.pedestrian_data[0].agent[1][1:])+" people in frame "+str( people[1])+" ------------------------------------------")

        np.testing.assert_array_less( [ 0, 0], episode.pedestrian_data[0].agent[1][1:])


        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        # print("Agent 2 " + str(episode.pedestrian_data[0].agent[1][1:]+[1, 1]) + " people in frame " + str(people[2]) + " ------------------------------------------")
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[1][1:], episode.pedestrian_data[0].agent[2][1:])
        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.hit_pedestrians],
                                      0)



        self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, -1, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # print("Agent 3 " + str(episode.pedestrian_data[0].agent[2][1:]+[-1, -1]) + " people in frame " + str(people[3]) + " ------------------------------------------")
        episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_less( episode.pedestrian_data[0].agent[3][1:], episode.pedestrian_data[0].agent[2][1:])
        np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], 0)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.hit_pedestrians],
                                      0)



        self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, -1, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # print("Agent 4 " + str(episode.pedestrian_data[0].agent[3][1:] + [-1, -1]) + " people in frame " + str(
        #     people[4]) + " ------------------------------------------")
        episode.pedestrian_data[0].action[3] = 5
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[4][1:], episode.pedestrian_data[0].agent[3][1:])
        np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[2, PEDESTRIAN_MEASURES_INDX.hit_pedestrians],
                                      0)



        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 5)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[4] = 5
        # print("Agent 5 " + str(episode.pedestrian_data[0].agent[4][1:] + [0, 1]) + " people in frame " + str(
        #     people[5]) + " ------------------------------------------")
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[4][2], episode.pedestrian_data[0].agent[5][2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[4][1], episode.pedestrian_data[0].agent[5][1], decimal=0)
        np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], 0)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[3, PEDESTRIAN_MEASURES_INDX.hit_pedestrians],
                                      0)



        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 6)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 6)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[5] = 5
        # print("Agent 6 " + str(episode.pedestrian_data[0].agent[5][1:] + [0, 1]) + " people in frame " + str(
        #     people[6]) + " ------------------------------------------")
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[5][2], episode.pedestrian_data[0].agent[6][2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[5][1], episode.pedestrian_data[0].agent[6][1], decimal=0)
        np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], -1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[4, PEDESTRIAN_MEASURES_INDX.hit_pedestrians],
                                      1)
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 7)
        np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], 0)
        np.testing.assert_array_equal(episode.pedestrian_data[0].measures[5, PEDESTRIAN_MEASURES_INDX.hit_pedestrians], 0)

    def test_cars_hit_on_one(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list=[]
        cars[1].append([0,2+1, 0,0+1,1,1+1])
        cars_list.append([0,2+1, 0,0+1,1,1+1])
        cars[1].append([0, 2+1, 0, 0+1, 0, 0+1])
        cars_list.append([0, 2+1, 0, 0+1, 0, 0+1])
        cars[2].append([0, 2+1, 0, 1+1, 1, 2+1])
        cars_list.append([0, 2+1, 0, 1+1, 1, 2+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])
        cars[3].append([0, 2+1, 0, 0+1, 1, 1+1])
        cars_list.append([0, 2+1, 0, 0+1, 1, 1+1])
        cars_dict={0:cars_list}
        cars_init={0:1}
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        # agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)

        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  seq_len=seq_len, rewards=rewards,
                                                                  car_dict=cars_dict, init_frames_cars=cars_init)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.vel_init = np.zeros(3)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0.1])

        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0.1])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0] = 5

        np.testing.assert_array_less(episode.pedestrian_data[0].agent[0][2], episode.pedestrian_data[0].agent[1][2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[0][1], episode.pedestrian_data[0].agent[1][1], decimal=0)


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], episode.pedestrian_data[0].agent[1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], -1)
        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 3)
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)



    def test_cars_hit_on_two(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars_list = []
        cars[1].append([0,2+1,0,0+1,1,1+1])
        cars_list.append([0,2+1,0,0+1,1,1+1])
        cars[2].append([0, 2+1, 1, 1+1, 1, 1+1])
        cars_list.append([0, 2+1, 1, 1+1, 1, 1+1])
        cars[3].append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_list.append([0, 2+1, 2, 2+1, 1, 1+1])
        cars_dict={0:cars_list}
        cars_init={0:1}
        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        # agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  seq_len=seq_len, rewards=rewards,
                                                                  car_dict=cars_dict, init_frames_cars=cars_init)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.vel_init = np.zeros(3)
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
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[0][1], episode.pedestrian_data[0].agent[1][1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[0][2], episode.pedestrian_data[0].agent[1][2], decimal=0)


        self.update_agent_and_episode([0, 0, 1], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 0, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[1][2], episode.pedestrian_data[0].agent[2][2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[1][1], episode.pedestrian_data[0].agent[2][1], decimal=0)
        print(episode.pedestrian_data[0].agent)
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 0, -1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], episode.pedestrian_data[0].agent[2])
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 0], 1)
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], -1)


        for itr in range(3, seq_len-1):
            self.update_agent_and_episode([0, 0, -1], agent, environmentInteraction, episode, itr+1)
            # agent.perform_action([0, 0, -1], episode)
            # agent.update_agent_pos_in_episode(episode, itr+1)
            # agent.on_post_tick(episode)
            # agent.update_metrics(episode)
            # episode.pedestrian_data[0].action[itr] = 5
            np.testing.assert_array_equal(episode.pedestrian_data[0].agent[itr+1],episode.pedestrian_data[0].agent[2])
            np.testing.assert_approx_equal(episode.calculate_reward(itr-1, episode_done=True)[0], 0)

        print(episode.pedestrian_data[0].agent)



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

        cars_dict={0:cars_list}
        cars_init={0:1}

        tensor=np.ones(tensor.shape)*cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1

        #agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  seq_len=seq_len, rewards=rewards,
                                                                  car_dict=cars_dict, init_frames_cars=cars_init)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.vel_init = np.zeros(3)
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
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[1][2], episode.pedestrian_data[0].agent[2][2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[1][1], episode.pedestrian_data[0].agent[2][1], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[2][1], episode.pedestrian_data[0].agent[3][1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[2][2], episode.pedestrian_data[0].agent[3][2], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 0)


        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[3] = 5
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[3][1], episode.pedestrian_data[0].agent[4][1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[3][2], episode.pedestrian_data[0].agent[4][2], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 5)
        np.testing.assert_approx_equal(episode.calculate_reward(3, episode_done=True)[0], 0)


    def test_goal_reached_on_two(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 1
        # agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
        #                                          rewards=rewards)
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  seq_len=seq_len, rewards=rewards)
        episode.follow_goal=True
        episode.vel_init = np.zeros(3)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None)
        episode.vel_init = np.zeros(3)
        episode.pedestrian_data[0].agent[0] = np.array([0, 0, 0])
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
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[0][2], episode.pedestrian_data[0].agent[1][2])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[0][1], episode.pedestrian_data[0].agent[1][1], decimal=0)


        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 5
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[1][1], episode.pedestrian_data[0].agent[2][1])
        np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[1][2], episode.pedestrian_data[0].agent[2][2], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, 7], np.sqrt(5))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.goal_reached], 0)



        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 3)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        #np.testing.assert_array_less(episode.pedestrian_data[0].agent[2][1:], episode.pedestrian_data[0].agent[3][1:])
        np.testing.assert_approx_equal(episode.pedestrian_data[0].agent[2][1], episode.pedestrian_data[0].agent[3][1])
        print (np.linalg.norm(episode.pedestrian_data[0].agent[3][1:]-episode.pedestrian_data[0].goal[0,1:]))
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True)[0], 1)
        # np.testing.assert_approx_equal(episode.calculate_reward(1), 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.goal_reached], 1)


        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 4)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_approx_equal(episode.pedestrian_data[0].agent[3][1], episode.pedestrian_data[0].agent[4][1])
        np.testing.assert_approx_equal(episode.pedestrian_data[0].agent[3][2], episode.pedestrian_data[0].agent[4][2])
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
        # np.testing.assert_approx_equal(episode.calculate_reward(2), 1)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, PEDESTRIAN_MEASURES_INDX.goal_reached], 1)


        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 5)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 5)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 5
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[4], episode.pedestrian_data[0].agent[5])
        np.testing.assert_approx_equal(episode.calculate_reward(3, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[3, PEDESTRIAN_MEASURES_INDX.goal_reached], 1)

        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 6)
        np.testing.assert_approx_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[4, PEDESTRIAN_MEASURES_INDX.goal_reached], 1)

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
        #rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        #(0, 0, 0, 0,0, 0, 0,0,0,0,0,1,0, 0,0,0)
        # agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
        #                                          rewards=rewards, people_dict=people_map, init_frames=init_frames)
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  seq_len=seq_len, rewards=rewards, people_dict=people_map, init_frames=init_frames)
        episode.max_step=1
        episode.agent_size = [0, 0, 0]
        episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian, init_key=0)
        np.testing.assert_array_equal(episode.pedestrian_data[0].goal_person_id_val, 1)


        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 1)
        # agent.perform_action([0, 1, 1], episode)
        # agent.update_agent_pos_in_episode(episode, 1)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[0]=8

        np.testing.assert_array_less(episode.pedestrian_data[0].agent[0][1:], episode.pedestrian_data[0].agent[1][1:])


        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 2)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 2)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[1] = 7
        np.testing.assert_approx_equal(1, episode.calculate_reward(0, episode_done=True)[0])
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, 12], np.sqrt(2))
        np.testing.assert_approx_equal(
            episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init],
            np.sqrt(episode.pedestrian_data[0].agent[1][1] ** 2 + episode.pedestrian_data[0].agent[1][2] ** 2))

        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[0, PEDESTRIAN_MEASURES_INDX.hit_pedestrians],
                                       0)
        np.testing.assert_approx_equal(0, episode.pedestrian_data[0].measures[
            0, PEDESTRIAN_MEASURES_INDX.one_step_prediction_error])



        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 3)
        np.testing.assert_approx_equal( 0.5,episode.calculate_reward(1, episode_done=True)[0])
        np.testing.assert_array_less(episode.pedestrian_data[0].agent[1][1], episode.pedestrian_data[0].agent[2][1])
        # np.testing.assert_approx_equal(episode.calculate_reward(1)[0],1- (1/np.sqrt(32)))
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 12], np.sqrt(2)+1)
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 4], np.sqrt(5))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.hit_pedestrians],
                                       1)
        # np.testing.assert_array_less(0, episode.pedestrian_data[0].measures[1, 9])
        np.testing.assert_array_less(0, episode.pedestrian_data[0].measures[1, PEDESTRIAN_MEASURES_INDX.one_step_prediction_error])
        self.update_agent_and_episode([0, 1, 0], agent, environmentInteraction, episode, 4)
        # agent.perform_action([0, 1, 0], episode)
        # agent.update_agent_pos_in_episode(episode, 3)
        # agent.on_post_tick(episode)
        # agent.update_metrics(episode)
        # episode.pedestrian_data[0].action[2] = 7
        np.testing.assert_approx_equal( episode.calculate_reward(2, episode_done=True)[0],1/3)
        np.testing.assert_array_less( episode.pedestrian_data[0].agent[2][1],episode.pedestrian_data[0].agent[3][1])
        print(episode.pedestrian_data[0].agent)
        # np.testing.assert_approx_equal(episode.calculate_reward(1)[0],1- (1/np.sqrt(32)))
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 12], np.sqrt(2)+1)
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[1, 4], np.sqrt(5))
        np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, PEDESTRIAN_MEASURES_INDX.hit_pedestrians], 1)
        np.testing.assert_array_less(0, episode.pedestrian_data[0].measures[2, PEDESTRIAN_MEASURES_INDX.one_step_prediction_error])


        # agent.perform_action([0, 1, 1], episode)
        # agent.update_metrics(episode)
        # np.testing.assert_array_equal(episode.pedestrian_data[0].agent[3], [0, 2, 2])
        # np.testing.assert_approx_equal(episode.calculate_reward(2)[0],1-((2+np.sqrt(2))/np.sqrt(8)))
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 12], 2 + np.sqrt(2))
        # np.testing.assert_approx_equal(episode.pedestrian_data[0].measures[2, 4], 2*np.sqrt(2))


    # def test_continous_agent(self):
    #     seq_len = 11
    #     cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
    #     tensor[1,0,0,CHANNELS.semantic]=cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
    #     tensor[2, 2, 1, CHANNELS.semantic]=cityscapes_labels_dict['building']/NUM_SEM_CLASSES
    #     rewards = self.get_reward(True)
    #     rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
    #     #agent, episode=self.initialize_episode_cont(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11, rewards=rewards)
    #     agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
    #                                                               seq_len=seq_len, rewards=rewards,
    #                                                               continous=True)
    #     episode.agent_size = [0, 0, 0]
    #     self.initialize_pos(agent, episode)
    #
    #     np.testing.assert_array_equal(episode.pedestrian_data[0].agent[0], [1,0,0])
    #     np.testing.assert_array_equal(agent.position, [1, 0, 0])
    #
    #     # valid move
    #
    #     self.update_agent_and_episode([0,1,1], agent, environmentInteraction, episode, 1,action_nbr=-np.pi/4)
    #     # agent.perform_action([0, 1, 1], episode)
    #     # agent.update_agent_pos_in_episode(episode, 1)
    #     # agent.on_post_tick(episode)
    #     # agent.update_metrics(episode)
    #     episode.pedestrian_data[0].action[0] = -np.pi/4
    #     np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], [1,1,1])
    #     np.testing.assert_array_equal(agent.position, [1, 1, 1])
    #
    #
    #     # valid move
    #     self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 2,action_nbr=-np.pi/4)
    #     # agent.perform_action([0, 1, 1], episode)
    #     # agent.update_agent_pos_in_episode(episode, 2)
    #     # agent.on_post_tick(episode)
    #     # agent.update_metrics(episode)
    #     episode.pedestrian_data[0].action[1] =  -np.pi/4
    #     np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[2], [1, 1,1+ np.sqrt(2)])
    #     np.testing.assert_array_almost_equal(agent.pos_exact, [1, 1, 1+np.sqrt(2)])
    #     np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], 0)
    #     np.testing.assert_equal(agent.angle, - np.pi / 4)
    #
    #
    #     # Valid move
    #     self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 3,action_nbr=-np.pi/4)
    #     # agent.perform_action([0, 1, 1], episode)
    #     # agent.update_agent_pos_in_episode(episode, 3)
    #     # agent.on_post_tick(episode)
    #     # agent.update_metrics(episode)
    #     episode.pedestrian_data[0].action[2] = -np.pi/4
    #     np.testing.assert_array_almost_equal(  episode.pedestrian_data[0].agent[3],[1, 0,2+np.sqrt(2)])
    #     np.testing.assert_array_almost_equal( agent.pos_exact,[1, 0,2+np.sqrt(2)])
    #     np.testing.assert_array_equal(episode.calculate_reward(1, episode_done=True)[0], 0)
    #     np.testing.assert_equal(agent.angle, -np.pi / 2)
    #
    #
    #
    #     # Valid move
    #     self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 4,action_nbr=-np.pi/4)
    #     # agent.perform_action([0, 1, 1], episode)
    #     # agent.update_agent_pos_in_episode(episode, 4)
    #     # agent.on_post_tick(episode)
    #     # agent.update_metrics(episode)
    #     episode.pedestrian_data[0].action[3] =  -np.pi/4
    #     np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[4], [1, - np.sqrt(2),2+np.sqrt(2)])
    #     np.testing.assert_array_almost_equal(agent.pos_exact, [1, - np.sqrt(2), 2+np.sqrt(2)])
    #     np.testing.assert_array_equal(episode.calculate_reward(2, episode_done=True)[0], 0)
    #     np.testing.assert_equal(agent.angle, -3 * np.pi / 4)
    #
    #
    #
    #     # Valid move
    #     self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 5,action_nbr=-np.pi/4)
    #     # agent.perform_action([0, 1, 1], episode)
    #     # agent.update_agent_pos_in_episode(episode, 5)
    #     # agent.on_post_tick(episode)
    #     # agent.update_metrics(episode)
    #     episode.pedestrian_data[0].action[4] =  -np.pi/4
    #     np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[5], [1, - np.sqrt(2)-1, 1+np.sqrt(2) ])
    #     np.testing.assert_array_almost_equal(agent.pos_exact, [1, - np.sqrt(2)-1, 1+np.sqrt(2) ])
    #     np.testing.assert_array_equal(episode.calculate_reward(3, episode_done=True)[0], 0)
    #     np.testing.assert_equal(agent.angle, np.pi)
    #
    #
    #
    #     # Valid move
    #     self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 6,action_nbr=-np.pi/4)
    #     # agent.perform_action([0, 1, 1], episode)
    #     # agent.update_agent_pos_in_episode(episode, 6)
    #     # agent.on_post_tick(episode)
    #     # agent.update_metrics(episode)
    #     episode.pedestrian_data[0].action[5] =  -np.pi/4
    #     np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[6], [1, - np.sqrt(2) - 1, 1 ])
    #     np.testing.assert_array_almost_equal(agent.pos_exact, [1, - np.sqrt(2) - 1, 1 ])
    #     np.testing.assert_array_equal(episode.calculate_reward(4, episode_done=True)[0], 0)
    #     np.testing.assert_approx_equal(agent.angle, 3 * np.pi / 4)
    #
    #     # Valid move
    #     self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 7,action_nbr=-np.pi/4)
    #     # agent.perform_action([0, 1, 1], episode)
    #     # agent.update_agent_pos_in_episode(episode, 7)
    #     # agent.on_post_tick(episode)
    #     # agent.update_metrics(episode)
    #     episode.pedestrian_data[0].action[6] =  -np.pi/4
    #     np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[7], [1, - np.sqrt(2) , 0])
    #     np.testing.assert_array_almost_equal(agent.pos_exact, [1, - np.sqrt(2) , 0])
    #     np.testing.assert_array_equal(episode.calculate_reward(5, episode_done=True)[0], 0)
    #     np.testing.assert_approx_equal(agent.angle, np.pi / 2)
    #
    #
    #     # Valid move
    #     self.update_agent_and_episode([0, 1, 1], agent, environmentInteraction, episode, 8,action_nbr=-np.pi/4)
    #     # agent.perform_action([0, 1, 1], episode)
    #     # agent.update_agent_pos_in_episode(episode, 8)
    #     # agent.on_post_tick(episode)
    #     # agent.update_metrics(episode)
    #     episode.pedestrian_data[0].action[7] =  -np.pi/4
    #     np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[8], [1, 0, 0])
    #     np.testing.assert_array_almost_equal(agent.pos_exact, [1, 0, 0])
    #     np.testing.assert_array_equal(episode.calculate_reward(6, episode_done=True)[0], 0)
    #     np.testing.assert_approx_equal(agent.angle, np.pi / 4)
    #
    #
    #     self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode, 6,action_nbr=-np.pi/4)
    #     # agent.perform_action([0, -1, -1], episode)
    #     # agent.update_agent_pos_in_episode(episode, 9)
    #     # agent.on_post_tick(episode)
    #     # agent.update_metrics(episode)
    #     episode.pedestrian_data[0].action[8] =  np.pi/4
    #     np.testing.assert_array_almost_equal(episode.pedestrian_data[0].agent[9], [1, -1, -1])
    #     np.testing.assert_array_almost_equal(agent.pos_exact, [1, -1, -1])
    #     np.testing.assert_array_equal(episode.calculate_reward(7, episode_done=True)[0], 0)
    #     self.assertLess(abs(agent.angle), 1e-12)
    #     self.update_agent_and_episode([0, -1, -1], agent, environmentInteraction, episode,7, action_nbr=-np.pi / 4)
    #     np.testing.assert_array_equal(episode.calculate_reward(8, episode_done=True)[0], 0)
    #     print(agent.angle/np.pi)
    #     np.testing.assert_approx_equal(agent.angle, 3*np.pi / 4)




