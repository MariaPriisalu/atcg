import unittest
import numpy as np
import sys

from RL.episode import SimpleEpisode
from RL.agent_pfnn import AgentPFNN
from RL.agent import ContinousAgent, SimplifiedAgent
from RL.settings import run_settings,NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_MEASURES_INDX
from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings, CAR_MEASURES_INDX, CAR_REWARD_INDX,NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE,NBR_MEASURES,PEDESTRIAN_MEASURES_INDX
from RL.environment_interaction import EntitiesRecordedDataSource, EnvironmentInteraction
from commonUtils.ReconstructionUtils import LAST_CITYSCAPES_SEMLABEL, cityscapes_labels_dict, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, SIDEWALK_LABELS,CHANNELS,OBSTACLE_LABELS_NEW, OBSTACLE_LABELS,cityscapes_labels_dict
# Test methods in episode.
from tests.test_multiple_agents import TestEnv
from RL.car_measures import find_closest_controllable_pedestrian

class CarTestEnv(TestEnv):

    def test_dist_to_closest_car(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)


        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.collision_car_with_car] = -1
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        init_rewards=rewards


        car_agent_dict = {'positions': {0: np.array([0, 0, 0]),1: np.array([0, 1, 0]),2: np.array([0, 2, 0]),3: np.array([0, 3, 0]), 4: np.array([0, 4, 0])}, 'goals': {0: np.array([0, 0, 4]),1: np.array([0, 1, 4]),2: np.array([0, 2, 4]),3: np.array([0, 3, 4]),4: np.array([0, 4, 4])}, 'dirs': {0: np.array([0, 1, 0]),1: np.array([0, 1, 0]),2: np.array([0, 1, 0]),3: np.array([0, 1, 0]),4: np.array([0, 1, 0])}}

        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y,
                                                                               tensor,
                                                                               seq_len=seq_len, rewards=rewards,
                                                                               num_agents=2, num_car_agents=5,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,
                                                                               init_rewards=init_rewards)



        self.initialize_pos(agents, episode, positions={0: np.array([0, 2, 2]),1: np.array([0, 4, 4])})

        closest_car, min_dist=episode.find_closest_car(0, [0,2,2], [0,1,1], 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0,  2,0])
        np.testing.assert_array_equal(min_dist, 2)

        feature=episode.get_input_cars_smooth( [0,2,2], 0, [0,1,1], 2*np.pi,  distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 3)

        closest_car, min_dist = episode.find_closest_car(0, [0, 4, 4], [0,1,1], 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0,  4,0])
        np.testing.assert_array_equal(min_dist, 4)
        feature = episode.get_input_cars_smooth([0, 4,4], 0,[0,1,1], 2*np.pi,  distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 3)
        #frame, pos,episode, agent_shape, movement_direction, field_of_view,is_fake_episode
        min_dist, id = find_closest_controllable_pedestrian(0, [0, 0, 0], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist, 2*np.sqrt(2))

        min_dist, id = find_closest_controllable_pedestrian(0, [0, 1, 0],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist, np.sqrt(5))

        min_dist, id = find_closest_controllable_pedestrian(0, [0, 2, 0], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist, 2 )

        min_dist, id = find_closest_controllable_pedestrian(0, [0, 3, 0], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist,  np.sqrt(5))

        min_dist, id = find_closest_controllable_pedestrian(0, [0, 4, 0], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist, 2 * np.sqrt(2))


        # First and second agent walk into eachother
        actions = {agents[0]: [0,-1,0],agents[1]: [0,0,-1], car_agents.trainableCars[0]: np.array([0, 0, 1]),car_agents.trainableCars[1]: np.array([0, 0, 1]),car_agents.trainableCars[2]: np.array([0, 0, 1]), car_agents.trainableCars[3]: np.array([0, 0, 1]), car_agents.trainableCars[4]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1:[4,3]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [0,1], 1:[1,1], 2:[2,1], 3:[3,1], 4:[4,1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)


        closest_car, min_dist=episode.find_closest_car(1, [0,1,2],np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0,  1,1])
        np.testing.assert_array_equal(min_dist, 1)

        feature=episode.get_input_cars_smooth( [0,1,2], 1,  np.array([0,0,0]), 2*np.pi, distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 3)

        closest_car, min_dist = episode.find_closest_car(1, [0, 4, 3],np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0,  4,1])
        np.testing.assert_array_equal(min_dist, 2)
        feature = episode.get_input_cars_smooth([0, 4,4], 1,  np.array([0,0,0]), 2*np.pi, distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 3)

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 0, 1], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, np.sqrt(2))

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 1, 1], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, 1)

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 2, 1], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 1, 2])
        np.testing.assert_array_equal(min_dist,np.sqrt( 2) )

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 3, 1], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 1, 2])
        np.testing.assert_array_equal(min_dist,  np.sqrt(5))

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 4, 1], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 4, 3])
        np.testing.assert_array_equal(min_dist, 2 )

        # First and second agent walk into eachother
        actions = {agents[0]: [0, 0, 0], agents[1]: [0, 0, -1], car_agents.trainableCars[0]: np.array([0, 0, 1]),
                   car_agents.trainableCars[1]: np.array([0, -1, 0]), car_agents.trainableCars[2]: np.array([0, 0, 1]),
                   car_agents.trainableCars[3]: np.array([0, 0, 1]), car_agents.trainableCars[4]: np.array([0, -1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1: [4, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [0, 2], 1: [0, 1], 2: [2, 2], 3: [3, 2], 4: [3, 1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        closest_car, min_dist = episode.find_closest_car(2, [0, 1, 2],np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0, 0, 2])
        np.testing.assert_array_equal(min_dist, 1)

        feature = episode.get_input_cars_smooth([0, 1, 2], 2, np.array([0,0,0]), 2*np.pi,  distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 1)

        closest_car, min_dist = episode.find_closest_car(2, [0, 4, 2],np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0, 3, 2])
        np.testing.assert_array_equal(min_dist, 1)
        feature = episode.get_input_cars_smooth([0, 4, 2], 2, np.array([0,0,0]), 2*np.pi,  distracted=False)
        np.testing.assert_array_equal(np.argmax(feature),1)

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 0, 2], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, 1)

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 0, 1], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, np.sqrt(2))

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 2, 2], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, 1)

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 3, 2], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 4, 2])
        np.testing.assert_array_equal(min_dist, 1)

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 3, 1], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 4, 2])
        np.testing.assert_array_equal(min_dist,np.sqrt(2))

    def test_dist_to_closest_car_external_car(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        cars_list = []
        cars[0].append(np.array([0, 0 + 1, 4, 4 + 1, 0, 0 + 1]))
        cars_list.append(np.array([0, 0 + 1, 4, 4 + 1, 0, 0 + 1]))

        cars[1].append(np.array([0, 0 + 1, 4, 4 + 1, 1, 1 + 1]))
        cars_list.append(np.array([0, 0 + 1, 4, 4 + 1, 1, 1 + 1]))

        cars[2].append(np.array([0, 0 + 1, 4, 4 + 1, 1, 1 + 1]))
        cars_list.append(np.array([0, 0 + 1, 4, 4 + 1, 1, 1 + 1]))

        car_dict = {0: cars_list}
        init_frames_cars = {0: 0}

        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.collision_car_with_car] = -1
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        init_rewards=rewards


        car_agent_dict = {'positions': {0: np.array([0, 0, 0]),1: np.array([0, 1, 0]),2: np.array([0, 2, 0]),3: np.array([0, 3, 0]),}, 'goals': {0: np.array([0, 0, 4]),1: np.array([0, 1, 4]),2: np.array([0, 2, 4]),3: np.array([0, 3, 4])}, 'dirs': {0: np.array([0, 1, 0]),1: np.array([0, 1, 0]),2: np.array([0, 1, 0]),3: np.array([0, 1, 0])}}

        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y,
                                                                               tensor,
                                                                               seq_len=seq_len, rewards=rewards,
                                                                               num_agents=2, num_car_agents=4,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,
                                                                               init_rewards=init_rewards,
                                                                               car_dict=car_dict,
                                                                               init_frames_cars=init_frames_cars)



        self.initialize_pos(agents, episode, positions={0: np.array([0, 2, 2]),1: np.array([0, 4, 4])})

        closest_car, min_dist=episode.find_closest_car(0, [0,2,2], np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0,  2,0])
        np.testing.assert_array_equal(min_dist, 2)

        feature=episode.get_input_cars_smooth( [0,2,2], 0, np.array([0,0,0]), 2*np.pi,  distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 3)

        closest_car, min_dist = episode.find_closest_car(0, [0, 4, 4], np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0,  4, 0])
        np.testing.assert_array_equal(min_dist, 4)
        feature = episode.get_input_cars_smooth([0, 4,4], 0,  np.array([0,0,0]), 2*np.pi, distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 3)

        min_dist, id = find_closest_controllable_pedestrian(0, [0, 0, 0], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist, 2*np.sqrt(2))

        min_dist, id = find_closest_controllable_pedestrian(0, [0, 1, 0], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist, np.sqrt(5))

        min_dist, id = find_closest_controllable_pedestrian(0, [0, 2, 0], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist, 2 )

        min_dist, id = find_closest_controllable_pedestrian(0, [0, 3, 0], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist,  np.sqrt(5))

        min_dist, id = find_closest_controllable_pedestrian(0, [0, 4, 0], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[0], [0, 2, 2])
        np.testing.assert_array_equal(min_dist, 2 * np.sqrt(2))


        # First and second agent walk into eachother
        actions = {agents[0]: [0,-1,0],agents[1]: [0,0,-1], car_agents.trainableCars[0]: np.array([0, 0, 1]),car_agents.trainableCars[1]: np.array([0, 0, 1]),car_agents.trainableCars[2]: np.array([0, 0, 1]), car_agents.trainableCars[3]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1:[4,3]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [0,1], 1:[1,1], 2:[2,1], 3:[3,1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)


        closest_car, min_dist=episode.find_closest_car(1, [0,1,2],np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0,  1,1])
        np.testing.assert_array_equal(min_dist, 1)

        feature=episode.get_input_cars_smooth( [0,1,2], 1, np.array([0,0,0]), 2*np.pi,  distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 3)

        closest_car, min_dist = episode.find_closest_car(1, [0, 4, 3],np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0,  4,1])
        np.testing.assert_array_equal(min_dist, 2)
        feature = episode.get_input_cars_smooth([0, 4,4], 1, np.array([0,0,0]), 2*np.pi,  distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 3)

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 0, 1],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, np.sqrt(2))

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 1, 1],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, 1)

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 2, 1],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 1, 2])
        np.testing.assert_array_equal(min_dist,np.sqrt( 2) )

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 3, 1],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 1, 2])
        np.testing.assert_array_equal(min_dist,  np.sqrt(5))

        min_dist, id = find_closest_controllable_pedestrian(1, [0, 4, 1], episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[1], [0, 4, 3])
        np.testing.assert_array_equal(min_dist, 2 )

        # First and second agent walk into eachother
        actions = {agents[0]: [0, 0, 0], agents[1]: [0, 0, -1], car_agents.trainableCars[0]: np.array([0, 0, 1]),
                   car_agents.trainableCars[1]: np.array([0, -1, 0]), car_agents.trainableCars[2]: np.array([0, 0, 1]),
                   car_agents.trainableCars[3]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1: [4, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [0, 2], 1: [0, 1], 2: [2, 2], 3: [3, 2]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        closest_car, min_dist = episode.find_closest_car(2, [0, 1, 2],np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0, 0, 2])
        np.testing.assert_array_equal(min_dist, 1)

        feature = episode.get_input_cars_smooth([0, 1, 2], 2, np.array([0,0,0]), 2*np.pi,  distracted=False)
        np.testing.assert_array_equal(np.argmax(feature), 1)

        closest_car, min_dist = episode.find_closest_car(2, [0, 4, 2],np.array([0,0,0]), 2*np.pi)
        np.testing.assert_array_equal(closest_car, [0, 3, 2])
        np.testing.assert_array_equal(min_dist, 1)
        feature = episode.get_input_cars_smooth([0, 4, 2], 2, np.array([0,0,0]), 2*np.pi,  distracted=False)
        np.testing.assert_array_equal(np.argmax(feature),1)

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 0, 2],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, 1)

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 0, 1],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, np.sqrt(2))

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 2, 2],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 1, 2])
        np.testing.assert_array_equal(min_dist, 1)

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 3, 2],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 4, 2])
        np.testing.assert_array_equal(min_dist, 1)

        min_dist, id = find_closest_controllable_pedestrian(2, [0, 3, 1],  episode , episode.agent_size, np.array([0,0,0]), 2*np.pi, is_fake_episode=False)
        np.testing.assert_array_equal(episode.pedestrian_data[id].agent[2], [0, 4, 2])
        np.testing.assert_array_equal(min_dist,np.sqrt(2))