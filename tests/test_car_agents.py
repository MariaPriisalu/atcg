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

class CarTestEnv(TestEnv):

    def test_drive_into_car_agents(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=3)

        cars_list = []
        cars[0].append(np.array([0, 0 + 1, 0, 0 + 1, 2, 2 + 1]))
        cars_list.append(np.array([0, 0 + 1, 0, 0 + 1, 2, 2 + 1]))

        cars[1].append(np.array([0, 0 + 1, 1, 1 + 1, 2, 2 + 1]))
        cars_list.append(np.array([0, 0 + 1, 1, 1 + 1, 2, 2 + 1]))

        cars[2].append(np.array([0, 0 + 1, 2, 2 + 1, 2, 2 + 1]))
        cars_list.append(np.array([0, 0 + 1, 2, 2 + 1, 2, 2 + 1]))

        car_dict = {0: cars_list}
        init_frames_cars = {0: 0}

        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.collision_car_with_car] = -1
        init_rewards_car=car_rewards
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        init_rewards=rewards



        car_agent_dict = {'positions': {0: np.array([0, 0, 0]),1: np.array([0, 1, 0]),2: np.array([0, 2, 0]),3: np.array([0, 0, 1])}, 'goals': {0: np.array([0, 0, 3]),1: np.array([0, 1, 3]),2: np.array([0, 2, 3]),3: np.array([0, 3, 1])}, 'dirs': {0: np.array([0, 1, 0]),1: np.array([0, 1, 0]),2: np.array([0, 1, 0]),3: np.array([0, 0, 1])}}

        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y,
                                                                               tensor,
                                                                               seq_len=seq_len, rewards=rewards,
                                                                               num_agents=1, num_car_agents=4,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,
                                                                               init_rewards=init_rewards,init_rewards_car=init_rewards_car, car_dict=car_dict, init_frames_cars=init_frames_cars)


        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode, positions={0: np.array([0, 3, 3])})

        # First and second agent walk into eacother
        actions = {agents[0]: [0,0,0], car_agents.trainableCars[0]: np.array([0, 0, 1]),car_agents.trainableCars[1]: np.array([0, 0, 1]),car_agents.trainableCars[2]: np.array([0, 0, 1]), car_agents.trainableCars[3]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [3, 3]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [0,1], 1:[1,1], 2:[2,1], 3:[1,1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: [0,0,0], car_agents.trainableCars[0]: np.array([0, 0, 1]),car_agents.trainableCars[1]: np.array([0, 0, 1]),car_agents.trainableCars[2]: np.array([0, 0, 1]), car_agents.trainableCars[3]: np.array([0, 1, 0])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [3, 3]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [0, 2], 1: [1, 1], 2: [2, 2], 3: [1, 1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        rewards_target_car = {0: 0, 1:-1, 2:0, 3:-1}
        self.check_car_reward(episode, rewards_target_car, 0)
        self.check_car_initializer_reward(episode, rewards_target_car, 0)

        measures_dict_positive = {PEDESTRIAN_REWARD_INDX.out_of_axis:1}
        pedestrian_dict = {0: measures_dict_positive}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.hit_by_car: 0}
        measures_dict_positive = {CAR_MEASURES_INDX.hit_by_car: 1}
        car_dict = {0: measures_dict_negative, 1:measures_dict_positive,2:measures_dict_negative, 3:measures_dict_positive }
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)

        actions = {agents[0]: [0, 0, 0], car_agents.trainableCars[0]: np.array([0, 0, 1]),
                   car_agents.trainableCars[1]: np.array([0, 0, 1]), car_agents.trainableCars[2]: np.array([0, 0, 1]),
                   car_agents.trainableCars[3]: np.array([0, 1, 0])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [3, 3]}
        self.check_positions(episode, agents, new_positions, 3)
        new_positions_cars = {0: [0, 3], 1: [1, 1], 2: [2, 2], 3: [1, 1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 3)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        rewards_target_car = {0: 0,1:0,2:-1,3:0}
        self.check_car_reward(episode, rewards_target_car, 1)
        self.check_car_initializer_reward(episode, rewards_target_car, 1)

        measures_dict_positive = {PEDESTRIAN_REWARD_INDX.out_of_axis: 1}
        pedestrian_dict = {0: measures_dict_positive}

        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)


        measures_dict_1 = {CAR_MEASURES_INDX.hit_by_car: 0}
        measures_dict_2 = {CAR_MEASURES_INDX.hit_by_car: 1,CAR_MEASURES_INDX.agent_dead: 1}
        measures_dict_3 = {CAR_MEASURES_INDX.hit_by_car: 1}
        car_dict = {0: measures_dict_1, 1:measures_dict_2,2:measures_dict_3, 3:measures_dict_2 }
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)

    def test_drive_into_obstacles(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=3)

        tensor[2, 1, 1, CHANNELS.semantic] = cityscapes_labels_dict['building'] / NUM_SEM_CLASSES

        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.collision_car_with_objects] = -1
        init_rewards_car = car_rewards
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        init_rewards=rewards


        car_agent_dict = {'positions': {0: np.array([0, 0, 0]),1: np.array([0, 1, 0])}, 'goals': {0: np.array([0, 0, 3]),1: np.array([0, 1, 3])}, 'dirs': {0: np.array([0, 1, 0]),1: np.array([0, 1, 0])}}

        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y,
                                                                               tensor,
                                                                               seq_len=seq_len, rewards=rewards,
                                                                               num_agents=1, num_car_agents=2,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,
                                                                               init_rewards=init_rewards,init_rewards_car=init_rewards_car)


        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode, positions={0: np.array([0, 2, 0])})

        # First and second agent walk into eacother
        actions = {agents[0]: [0,0,1], car_agents.trainableCars[0]: np.array([0, 0, 1]),car_agents.trainableCars[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [2,1]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [0,1], 1:[1,0]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: [0,0,1], car_agents.trainableCars[0]: np.array([0, 0, 1]),car_agents.trainableCars[1]: np.array([0, 0, 1])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [2, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars ={0: [0,2], 1:[1,0]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        rewards_target_car = {0: 0, 1:-1}
        self.check_car_reward(episode, rewards_target_car, 0)
        self.check_car_initializer_reward(episode, rewards_target_car, 0)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_obstacles:0}
        pedestrian_dict = {0: measures_dict_positive}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.hit_obstacles: 0}
        measures_dict_positive = {CAR_MEASURES_INDX.hit_obstacles: 1}
        car_dict = {0: measures_dict_negative, 1:measures_dict_positive}
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)

        actions = {agents[0]: [0,0,1], car_agents.trainableCars[0]: np.array([0, 0, 1]),car_agents.trainableCars[1]: np.array([0, 0, 1])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [2, 3]}
        self.check_positions(episode, agents, new_positions, 3)
        new_positions_cars = {0: [0, 3], 1: [1, 0]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 3)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        rewards_target_car = {0: 0, 1:0}
        self.check_car_reward(episode, rewards_target_car, 1)
        self.check_car_initializer_reward(episode, rewards_target_car, 1)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.out_of_axis: 0}
        pedestrian_dict = {0: measures_dict_positive}

        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.hit_obstacles: 0}
        measures_dict_positive = {CAR_MEASURES_INDX.hit_obstacles: 1, CAR_MEASURES_INDX.agent_dead: 1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)

    def test_dist_travelled(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=3)



        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.distance_travelled] = 1
        init_rewards_car=car_rewards
        rewards = self.get_reward(True)

        init_rewards=rewards


        car_agent_dict = {'positions': {0: np.array([0, 0, 0]),1: np.array([0, 1, 0])}, 'goals': {0: np.array([0, 0, 3]),1: np.array([0, 1, 3])}, 'dirs': {0: np.array([0, 1, 0]),1: np.array([0, 1, 0])}}

        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y,
                                                                               tensor,
                                                                               seq_len=seq_len, rewards=rewards,
                                                                               num_agents=1, num_car_agents=2,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,
                                                                               init_rewards=init_rewards,
                                                                               init_rewards_car=init_rewards_car)


        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode, positions={0: np.array([0, 2, 0])})

        # First and second agent walk into eacother
        actions = {agents[0]: [0,0,1], car_agents.trainableCars[0]: np.array([0, 0, 1]),car_agents.trainableCars[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [2,1]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [0,1], 1:[1,1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: [0,0,1], car_agents.trainableCars[0]: np.array([0, 0, 0]),car_agents.trainableCars[1]: np.array([0, 0, 1])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [2, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars ={0: [0,1], 1:[1,2]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        rewards_target_car = {0: 1, 1:1}
        self.check_car_reward(episode, rewards_target_car, 0)
        self.check_car_initializer_reward(episode, rewards_target_car, 0)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_obstacles:0}
        pedestrian_dict = {0: measures_dict_positive}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.distance_travelled_from_init: 0}
        measures_dict_positive = {CAR_MEASURES_INDX.distance_travelled_from_init: 1}
        car_dict = {0: measures_dict_positive, 1:measures_dict_positive}
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)

        actions = {agents[0]: [0,0,1], car_agents.trainableCars[0]: np.array([0, 0, 1]),car_agents.trainableCars[1]: np.array([0, 0, 1])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [2, 3]}
        self.check_positions(episode, agents, new_positions, 3)
        new_positions_cars = {0: [0, 2], 1: [1, 3]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 3)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        rewards_target_car = {0: 0.5, 1:1}
        self.check_car_reward(episode, rewards_target_car, 1)
        self.check_car_initializer_reward(episode, rewards_target_car, 1)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.out_of_axis: 0}
        pedestrian_dict = {0: measures_dict_positive}

        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.distance_travelled_from_init: 1}
        measures_dict_positive = {CAR_MEASURES_INDX.distance_travelled_from_init: 2}
        car_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)

    def test_iou_pavement(self): # iou reward is delayed by one step
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=3)

        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.penalty_for_intersection_with_sidewalk]= -1
        init_rewards_car = car_rewards
        rewards = self.get_reward(True)

        tensor[2, 1, 1, CHANNELS.semantic] = cityscapes_labels_dict['sidewalk'] / NUM_SEM_CLASSES

        init_rewards = rewards

        car_agent_dict = {'positions': {0: np.array([0, 0, 0]), 1: np.array([0, 1, 0])},
                          'goals': {0: np.array([0, 0, 3]), 1: np.array([0, 1, 3])},
                          'dirs': {0: np.array([0, 1, 0]), 1: np.array([0, 1, 0])}}

        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y,
                                                                               tensor,
                                                                               seq_len=seq_len, rewards=rewards,
                                                                               num_agents=1, num_car_agents=2,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,
                                                                               init_rewards=init_rewards,init_rewards_car=init_rewards_car)

        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode, positions={0: np.array([0, 2, 0])})

        # First and second agent walk into eacother
        actions = {agents[0]: [0, 0, 1], car_agents.trainableCars[0]: np.array([0, 0, 1]),
                   car_agents.trainableCars[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [2, 1]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [0, 1], 1: [1, 1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: [0, 0, 1], car_agents.trainableCars[0]: np.array([0, 0, 0]),
                   car_agents.trainableCars[1]: np.array([0, 0, 1])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [2, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [0, 1], 1: [1, 2]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        rewards_target_car = {0: 0, 1: 0}
        self.check_car_reward(episode, rewards_target_car, 0)
        self.check_car_initializer_reward(episode, rewards_target_car, 0)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0}
        pedestrian_dict = {0: measures_dict_positive}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.iou_pavement: 0}
        measures_dict_positive = {CAR_MEASURES_INDX.iou_pavement: 1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_negative}
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)

        actions = {agents[0]: [0, 0, 1], car_agents.trainableCars[0]: np.array([0, 0, 1]),
                   car_agents.trainableCars[1]: np.array([0, 0, 1])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [2, 3]}
        self.check_positions(episode, agents, new_positions, 3)
        new_positions_cars = {0: [0, 2], 1: [1, 3]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 3)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        rewards_target_car = {0: 0, 1: -1}
        self.check_car_reward(episode, rewards_target_car, 1)
        self.check_car_initializer_reward(episode, rewards_target_car, 1)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.out_of_axis: 0}
        pedestrian_dict = {0: measures_dict_positive}

        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.iou_pavement: 0}
        measures_dict_positive = {CAR_MEASURES_INDX.iou_pavement: 1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)

    def test_goal_reaching(self): # iou reward is delayed by one step
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=3)

        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.reached_goal]= 10
        car_rewards[CAR_REWARD_INDX.distance_travelled_towards_goal]=1
        init_rewards_car = car_rewards
        rewards = self.get_reward(True)



        init_rewards = rewards

        car_agent_dict = {'positions': {0: np.array([0, 0, 0]), 1: np.array([0, 1, 0])},
                          'goals': {0: np.array([0, 0, 3]), 1: np.array([0, 1, 3])},
                          'dirs': {0: np.array([0, 1, 0]), 1: np.array([0, 1, 0])}}

        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y,
                                                                               tensor,
                                                                               seq_len=seq_len, rewards=rewards,
                                                                               num_agents=1, num_car_agents=2,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,
                                                                               init_rewards=init_rewards,
                                                                               init_rewards_car=init_rewards_car)

        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode, positions={0: np.array([0, 2, 0])})

        # First and second agent walk into eacother
        actions = {agents[0]: [0, 0, 1], car_agents.trainableCars[0]: np.array([0, 0, 1]),
                   car_agents.trainableCars[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [2, 1]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [0, 1], 1: [1, 1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: [0, 0, 1], car_agents.trainableCars[0]: np.array([0, 0, 0]),
                   car_agents.trainableCars[1]: np.array([0, 0, 1])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [2, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [0, 1], 1: [1, 2]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        rewards_target_car = {0: 1, 1: 1}
        self.check_car_reward(episode, rewards_target_car, 0)
        self.check_car_initializer_reward(episode, rewards_target_car, 0)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0}
        pedestrian_dict = {0: measures_dict_positive}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 2}
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 0}
        car_dict = {0: measures_dict_negative, 1: measures_dict_negative}
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)

        actions = {agents[0]: [0, 0, 1], car_agents.trainableCars[0]: np.array([0, 0, 1]),
                   car_agents.trainableCars[1]: np.array([0, 0, 1])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [2, 3]}
        self.check_positions(episode, agents, new_positions, 3)
        new_positions_cars = {0: [0, 2], 1: [1, 3]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 3)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        rewards_target_car = {0: 0, 1: 1}
        self.check_car_reward(episode, rewards_target_car, 1)
        self.check_car_initializer_reward(episode, rewards_target_car, 1)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.out_of_axis: 0}
        pedestrian_dict = {0: measures_dict_positive}

        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)

        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.dist_to_goal:2}
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)

        actions = {agents[0]: [0, 0, 1], car_agents.trainableCars[0]: np.array([0, 0, 1]),
                   car_agents.trainableCars[1]: np.array([0, 0, 1])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 4, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [2, 4]}
        self.check_positions(episode, agents, new_positions, 4)
        new_positions_cars = {0: [0, 3], 1: [1, 3]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 4)

        rewards_target = {0: 0}
        self.check_reward(episode, rewards_target, 2)
        self.check_initializer_reward(episode, rewards_target, 2)
        rewards_target_car = {0: 1 , 1: 10}
        self.check_car_reward(episode, rewards_target_car, 2)
        self.check_car_initializer_reward(episode, rewards_target_car, 2)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.out_of_axis:1}
        pedestrian_dict = {0: measures_dict_positive}

        self.evaluate_measures(2, episode, pedestrian_dict)
        self.evaluate_measures_initializer(2, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 1}
        measures_dict_positive = {CAR_MEASURES_INDX.dist_to_goal: 0,CAR_MEASURES_INDX.goal_reached:1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_car_measures(2, episode, car_dict)
        self.evaluate_car_initializer_measures(2, episode, car_dict)



