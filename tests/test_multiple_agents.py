import unittest
import numpy as np
import sys

from RL.episode import SimpleEpisode
from RL.agent_pfnn import AgentPFNN
from RL.agent import ContinousAgent, SimplifiedAgent
from RL.settings import run_settings,NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_MEASURES_INDX,CAR_REWARD_INDX
from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings,CAR_MEASURES_INDX, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE,NBR_MEASURES,PEDESTRIAN_MEASURES_INDX,NBR_REWARD_CAR
from RL.environment_interaction import EntitiesRecordedDataSource, EnvironmentInteraction
from commonUtils.ReconstructionUtils import LAST_CITYSCAPES_SEMLABEL, cityscapes_labels_dict, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, SIDEWALK_LABELS,CHANNELS,OBSTACLE_LABELS_NEW, OBSTACLE_LABELS,cityscapes_labels_dict
# Test methods in episode.
from RL.agent_car import CarAgent
from dotmap import DotMap

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

    def get_reward_car(self):
        car_rewards = np.zeros(NBR_REWARD_CAR)
        return car_rewards

    def get_reward_initializer(self):
        reward_initializer = np.zeros(NBR_REWARD_WEIGHTS)
        return reward_initializer

    def get_reward_initializer_car(self):
        car_rewards = np.zeros(NBR_REWARD_CAR)
        return car_rewards

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

    def initialize_pos(self, agents, episode,positions={}, goals={}):
        for agent in agents:
            pos, i,vel = episode.initial_position(agent.id,None)
            if agent.id in positions:
                episode.pedestrian_data[agent.id].agent[0]=positions[agent.id]
            episode.pedestrian_data[agent.id].vel_init = np.zeros(3)
            if agent.id in goals:
                episode.pedestrian_data[agent.id].goal[0,:]=goals[agent.id]
            agent.initial_position(episode.pedestrian_data[agent.id].agent[0],episode.pedestrian_data[agent.id].goal[0,:])

    def initialize_car_pos(self, cars, seq_len, positions={}, goals={}, dirs={}):
        for car in cars.trainableCars:
            car.initial_position(positions[car.id], goals[car.id], seq_len, init_dir= dirs[car.id], car_id=car.car_id)


    def update_episode(self, environmentInteraction, episode, next_frame):
        observation, observation_dict = environmentInteraction.getObservation(frameToUse=next_frame)
        episode.update_pedestrians_and_cars(observation.frame,
                                            observation_dict,
                                            observation.people_dict,
                                            observation.cars_dict,
                                            observation.pedestrian_vel_dict,
                                            observation.car_vel_dict)

    def get_episode(self, cars, gamma, people, pos_x, pos_y, tensor, seq_len=15, rewards=[], agent_size=(0, 0, 0),
                    people_dict={}, init_frames={}, car_dict={}, init_frames_cars={}, new_carla=False,continous=False,
                    num_agents=1, num_car_agents=0, car_agent_dict={},car_rewards=[],init_rewards=[], car_dim=[],
                    goal_dir=False, max_car_step=1, stop_on_goal=True, stop_on_goal_car=True,init_rewards_car=[]):
        if len(rewards) == 0:
            rewards = self.get_reward(False)
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1

        car_agents,agents, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                         seq_len=seq_len, rewards=rewards,
                                                                         agent_size=agent_size,
                                                                         people_dict=people_dict,
                                                                         init_frames=init_frames, car_dict=car_dict,
                                                                         init_frames_cars=init_frames_cars,
                                                                         new_carla=new_carla,
                                                                                     continous=continous,num_agents=num_agents,
                                                                                     num_car_agents=num_car_agents,
                                                                                     car_agent_dict=car_agent_dict,
                                                                                     car_rewards=car_rewards, init_rewards=init_rewards,
                                                                                     car_dim=car_dim, goal_dir=goal_dir,
                                                                                     max_car_step=max_car_step,stop_on_goal=stop_on_goal,
                                                                                     stop_on_goal_car=stop_on_goal_car,init_rewards_car=init_rewards_car)
        episode.agent_size = agent_size
        return car_agents, agents, episode, environmentInteraction
        # Help function. Setup for tests.

    def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor, seq_len=30, rewards=[],
                           agent_size=(0, 0, 0), people_dict={}, init_frames={}, car_dict={}, init_frames_cars={},
                           new_carla=False, continous=False, num_agents=1, num_car_agents=0, car_agent_dict={},car_rewards=[],
                           init_rewards=[], car_dim=[],goal_dir=False,max_car_step=1, agent_view_occlusion=False, stop_on_goal=False,stop_on_goal_car=False, init_rewards_car=[]):
        settings = run_settings()

        settings.useRLToyCar = False
        settings.multiplicative_reward_pedestrian = False
        if goal_dir==True:
            settings.goal_dir = True
        else:
            settings.goal_dir = False
        settings.number_of_agents=num_agents
        settings.number_of_car_agents = num_car_agents
        settings.agent_shape=agent_size
        settings.car_max_speed_voxelperframe=max_car_step
        settings.pedestrian_view_occluded=agent_view_occlusion
        settings.stop_on_goal = stop_on_goal
        settings.stop_on_goal_car = stop_on_goal_car
        settings.learn_init_car = False
        if num_car_agents>0:
            #settings.useHeroCar=True
            settings.useRLToyCar=True
            settings.learn_init_car=True
            if len(car_dim)==0:
                settings.car_dim=[0,0,0]
            else:
                settings.car_dim=car_dim
        if len(rewards) == 0:
            rewards = self.get_reward()

        if len(car_rewards) == 0:
            car_rewards = self.get_reward_car()

        settings.reward_weights_car=car_rewards

        if len(init_rewards)==0:
            init_rewards=self.get_reward_initializer()
        settings.reward_weights_initializer=init_rewards

        if len(init_rewards_car) == 0:
            init_rewards_car = self.get_reward_initializer_car()
        settings.reward_weights_car_initializer = init_rewards_car

        car_agents = []
        for i in range(num_car_agents):
            car_agents.append(CarAgent(settings, None, None))

            car_agents[-1].id = i

        agents = []
        for i in range(num_agents):
            if continous:
                agents.append(ContinousAgent(settings))

            else:
                agents.append(SimplifiedAgent(settings))
            agents[-1].id = i

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
            if settings.useRLToyCar:
                from RL.RealTimeRLCarEnvInteraction import RLCarRealTimeEnv

                # TODO: expand this further to store the list of all agent cars
                getRealTimeEnvWaypointPosFunctor = None
                cachePrefix="test"
                pos_x=0
                pos_y=0
                newAgentCar = RLCarRealTimeEnv(cachePrefix,pos_x, pos_y,settings=settings,
                                               isOnline=settings.realTimeEnvOnline,
                                               offlineData=entitiesRecordedDataSource,
                                               trainableCars=car_agents,
                                               reconstruction=entitiesRecordedDataSource.reconstruction,
                                               seq_len=seq_len, car_dim=settings.car_dim,
                                               max_speed=settings.car_max_speed_voxelperframe,
                                               min_speed=settings.car_min_speed_voxelperframe,
                                               # The setup is considered done in online mode
                                               car_goal_closer=settings.reward_weights_car[CAR_REWARD_INDX.reached_goal],
                                               physicalCarsDict=entitiesRecordedDataSource.env_physical_cars,
                                               physicalWalkersDict=entitiesRecordedDataSource.env_physical_pedestrian,
                                               getRealTimeEnvWaypointPosFunctor=getRealTimeEnvWaypointPosFunctor,

                                               )

                # Reset the car here. TODO Ciprian do we really need this ??? It is already reset on episode start
                # initParams = DotMap()
                # initParams.on_car = False
                # newAgentCar.reset(alreadyAssignedCarKeys=alreadyAssignedCarKeys, initDict=initParams)
                self.initialize_car_pos(newAgentCar, seq_len, positions=car_agent_dict['positions'],
                                        goals=car_agent_dict['goals'],
                                        dirs=car_agent_dict['dirs'])
            else:
                newAgentCar=[]
            observation, observation_dict = environmentInteraction.reset(newAgentCar, agents, episode=None)


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
                                    seq_len, rewards, init_rewards, agent_size=agent_size,
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
                                    number_of_car_agents=settings.number_of_car_agents,learn_init_car=settings.learn_init_car)  # To DO:  Check if needed heroCarDetails = heroCarDetails
            episode.environmentInteraction = environmentInteraction
        else:
            environmentInteraction = None
            episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards, init_rewards,
                                    agent_size=agent_size,
                                    people_dict=people_dict, init_frames=init_frames, cars_dict=car_dict,
                                    init_frames_cars=init_frames_cars, defaultSettings=settings, centering={},
                                    useRealTimeEnv=settings.useRealTimeEnv, new_carla=new_carla, follow_goal=settings.goal_dir, learn_init_car=settings.learn_init_car)

        if settings.number_of_car_agents>0:
            episode.get_car_valid_init(cars_dict)
            episode.add_car_init_data()

        if settings.useRealTimeEnv:


            realTimeEnvObservation, observation_dict = environmentInteraction.reset(heroAgentCars=newAgentCar,
                                                                                    heroAgentPedestrians=agents,
                                                                                    episode=episode)
            episode.update_pedestrians_and_cars(realTimeEnvObservation.frame,
                                                observation_dict,
                                                realTimeEnvObservation.people_dict,
                                                realTimeEnvObservation.cars_dict,
                                                realTimeEnvObservation.pedestrian_vel_dict,
                                                realTimeEnvObservation.car_vel_dict)
            environmentInteraction.frame = 0
        return newAgentCar, agents, episode, environmentInteraction

    def getObservationForCar(self, episode, frame, allPedestrianAgents):
        ActionDict= {}
        for pedestrian in allPedestrianAgents:
            ActionDict[pedestrian] = DotMap()
            ActionDict[pedestrian].frame = frame
            ActionDict[pedestrian].init_dir = episode.pedestrian_data[pedestrian.id].vel_init
            ActionDict[pedestrian].agentPos = episode.pedestrian_data[pedestrian.id].agent[frame]
            ActionDict[pedestrian].agentVel = episode.pedestrian_data[pedestrian.id].velocity[frame]
            ActionDict[pedestrian].inverse_dist_to_car = episode.pedestrian_data[pedestrian.id].measures[frame, PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car]
        return ActionDict
    def update_agents_and_episode(self, actions, environmentInteraction, episode, frame, breadth=0, action_nbr=5,carAgent=None, allPedestrianAgents=[], manual_goal_dict={}):
        if breadth == 0:
            breadth = episode.agent_size[1:]
        for agent, action in actions.items():
            if agent.isPedestrian:
                episode.pedestrian_data[agent.id].action[frame - 1] = action_nbr
                episode.pedestrian_data[agent.id].velocity[frame - 1] = action
                episode.get_agent_neighbourhood(agent.id, agent.position, breadth, frame - 1)
            else:
                agent.velocity=action
                agent.velocities[frame - 1] = action
                agent.speed[frame - 1] = np.linalg.norm(action[1:])
                agent.action[frame - 1]= agent.speed[frame - 1]
                agent.angle[frame - 1]=0
                agent.get_car_net_input_to_episode(episode, frame-1)



        # trainableAgents[self.all_car_agents]= {}
        if carAgent:
            carObservation = self.getObservationForCar(episode, frame, allPedestrianAgents)
            agent_car_decision_dict = carAgent.update_car_episode(carObservation)


        environmentInteraction.signal_action(actions, updated_frame=frame)
        for agent, action in actions.items():
            agent.perform_action(action, episode)
        # Do the simulation for next tick using decisions taken on this tick
        # If an online realtime env is used this call will fill in the data from the simulator.
        # If offline, it will take needed data from recorded/offline data.
        environmentInteraction.tick(frame)
        self.update_episode(environmentInteraction, episode, frame)
        for agent, action in actions.items():
            agent.update_agent_pos_in_episode(episode, frame)
        for agent, action in actions.items():
            if agent.isPedestrian:
                manual_goal=[]
                if agent in manual_goal_dict:
                    manual_goal=manual_goal_dict[agent]

                agent.on_post_tick(episode, manual_goal=manual_goal)
            else:
                agent.on_post_tick(episode)
            if agent.isPedestrian:
                agent.update_metrics(episode)
                episode.pedestrian_data[agent.id].action[frame - 1] = action_nbr
        if carAgent:
            carAgent.update_metrics(episode)

    def evaluate_measure(self, frame, episode, non_zero_measure, expeceted_value_of_non_zero_measure):
        for measure in range(NBR_MEASURES):
            if measure == non_zero_measure:
                np.testing.assert_array_equal(
                    episode.pedestrian_data[0].measures[frame, non_zero_measure], expeceted_value_of_non_zero_measure)
            else:
                if measure !=PEDESTRIAN_MEASURES_INDX.change_in_direction and measure != PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal_from_current and measure != PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap and measure != PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.change_in_pose and measure != PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car:
                    print(measure)
                    np.testing.assert_array_equal(episode.pedestrian_data[0].measures[frame, measure], 0)
    def evaluate_measures_initializer(self, frame, episode, pedestrian_dict):
        for id, dict in pedestrian_dict.items():
            non_zero_measures = []
            zero_measures = []
            for measure in range(NBR_MEASURES):
                if measure in dict:
                    non_zero_measures.append(measure)
                else:
                    zero_measures.append(measure)

            for measure in non_zero_measures:
                print("Should be non-zero "+str(measure) + " person id: " + str(id))
                np.testing.assert_array_equal(
                    episode.initializer_data[id].measures[frame, measure], dict[measure])
            for measure in zero_measures:
                print("Should be zero "+str(measure) +" person id: "+str(id))
                if measure !=PEDESTRIAN_MEASURES_INDX.change_in_direction and measure != PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal_from_current and measure != PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap and measure != PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.change_in_pose and measure != PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car:
                    np.testing.assert_array_equal(episode.initializer_data[id].measures[frame, measure], 0)

    def evaluate_measures(self, frame, episode, pedestrian_dict):
        for id, dict in pedestrian_dict.items():
            non_zero_measures = []
            zero_measures = []
            for measure in range(NBR_MEASURES):
                if measure in dict:
                    non_zero_measures.append(measure)
                else:
                    zero_measures.append(measure)

            for measure in non_zero_measures:
                print("Should be non-zero "+str(measure) + " person id: " + str(id))
                np.testing.assert_array_equal(
                    episode.pedestrian_data[id].measures[frame, measure], dict[measure])
            for measure in zero_measures:
                print("Should be zero "+str(measure) +" person id: "+str(id))
                if measure !=PEDESTRIAN_MEASURES_INDX.change_in_direction and measure != PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal and measure != PEDESTRIAN_MEASURES_INDX.dist_to_goal_from_current and measure != PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap and measure != PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init and measure != PEDESTRIAN_MEASURES_INDX.change_in_pose and measure != PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car:
                    np.testing.assert_array_equal(episode.pedestrian_data[id].measures[frame, measure], 0)
    def evaluate_car_initializer_measures(self,frame, episode, pedestrian_dict):
        for id, dict in pedestrian_dict.items():
            non_zero_measures = []
            zero_measures = []
            for measure in range(NBR_MEASURES):
                if measure in dict:
                    non_zero_measures.append(measure)
                else:
                    zero_measures.append(measure)

            for measure in non_zero_measures:
                print("Should be non-zero "+str(measure) + " car id: " + str(id))
                np.testing.assert_array_equal(
                    episode.initializer_car_data[id].measures_car[frame, measure], dict[measure])
            for measure in zero_measures:
                print("Should be zero "+str(measure) +" car id: "+str(id))
                if measure !=CAR_MEASURES_INDX.dist_to_closest_pedestrian and measure != CAR_MEASURES_INDX.dist_to_closest_car and measure != CAR_MEASURES_INDX.dist_to_agent and measure != CAR_MEASURES_INDX.distance_travelled_from_init and measure!=CAR_MEASURES_INDX.dist_to_goal and measure!=CAR_MEASURES_INDX.id_closest_agent:
                    np.testing.assert_array_equal(episode.initializer_car_data[id].measures_car[frame, measure], 0)
    def evaluate_car_measures(self, frame, episode, pedestrian_dict):
        for id, dict in pedestrian_dict.items():
            non_zero_measures = []
            zero_measures = []
            for measure in range(NBR_MEASURES):
                if measure in dict:
                    non_zero_measures.append(measure)
                else:
                    zero_measures.append(measure)

            for measure in non_zero_measures:
                print("Should be non-zero "+str(measure) + " car id: " + str(id))
                np.testing.assert_array_equal(
                    episode.car_data[id].measures_car[frame, measure], dict[measure])
            for measure in zero_measures:
                print("Should be zero "+str(measure) +" car id: "+str(id))
                if measure !=CAR_MEASURES_INDX.dist_to_closest_pedestrian and measure != CAR_MEASURES_INDX.dist_to_closest_car and measure != CAR_MEASURES_INDX.dist_to_agent and measure != CAR_MEASURES_INDX.distance_travelled_from_init and measure!=CAR_MEASURES_INDX.dist_to_goal and measure!=CAR_MEASURES_INDX.id_closest_agent:
                    np.testing.assert_array_equal(episode.car_data[id].measures_car[frame, measure], 0)

    def check_positions(self, episode, agents, positions,frame):
        for id, pos in positions.items():
            np.testing.assert_array_equal(pos, episode.pedestrian_data[id].agent[frame][1:])
            np.testing.assert_array_equal(pos, agents[id].pos_exact[1:])

    def check_positions_car(self, episode, cars, positions,frame):
        for id, pos in positions.items():
            print("Car "+str(id)+" True pos "+str( episode.car_data[id].car[frame][1:]))
            np.testing.assert_array_equal(pos, episode.car_data[id].car[frame][1:])
            np.testing.assert_allclose(pos, cars[id].pos_exact[1:])

    def check_reward(self, episode, rewards_target, frame, episode_done=True):
        rewards = episode.calculate_reward(frame, episode_done=True)
        for indx, reward in rewards_target.items():
            print("Reward of agent "+str(indx)+" is "+str(rewards[indx]))
            np.testing.assert_allclose(reward, rewards[indx])

    def check_car_reward(self, episode, rewards_target, frame):

        for indx, reward in rewards_target.items():
            print("Reward of agent " + str(indx) + " is " + str(episode.car_data[indx].reward_car[frame]))
            np.testing.assert_allclose(reward, episode.car_data[indx].reward_car[frame])

    def check_initializer_reward(self, episode, rewards_target, frame):

        for indx, reward in rewards_target.items():
            print ("Init Reward of agent " + str(indx) + " is " + str(episode.initializer_data[indx].reward[frame]))
            np.testing.assert_allclose(reward,episode.initializer_data[indx].reward[frame])

    def check_car_initializer_reward(self, episode, rewards_target, frame):

        for indx, reward in rewards_target.items():
            print ("Init Reward of agent " + str(indx) + " is " + str(episode.initializer_car_data[indx].reward_car[frame]))
            np.testing.assert_allclose(reward,episode.initializer_car_data[indx].reward_car[frame])
    # ----
    # ----
    # Test correct empty initialization.
    def test_walk_into_objs_and_pavement(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        tensor[1, 1, 1, CHANNELS.semantic] = cityscapes_labels_dict['sidewalk'] / NUM_SEM_CLASSES
        tensor[2, 1, 2, CHANNELS.semantic] = cityscapes_labels_dict['building'] / NUM_SEM_CLASSES
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  seq_len=seq_len, rewards=rewards, num_agents=4, init_rewards=init_rewards)

        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agents, episode, positions={0:np.array([0,0,0]), 1:np.array([0,2,0]), 2:np.array([0,0,2]), 3:np.array([0,2,2])})

        # Third agent walks into wall
        actions={agents[0]:np.array([0,0,0]),agents[1]:np.array([0,0,0]), agents[2]:np.array([0,0,0]),agents[3]:np.array([0,-1,0])}
        self.update_agents_and_episode( actions, environmentInteraction, episode, 1, breadth=0, action_nbr=5)

        new_positions={0:[0,0], 1:[2,0], 2:[0,2], 3:[2,2]}
        self.check_positions(episode, agents, new_positions, 1)
        rewards_target={0:0, 1:0,2:0,3:-1}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        measures_dict_positive={PEDESTRIAN_MEASURES_INDX.hit_obstacles:1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0}
        pedestrian_dict={0:measures_dict_negative, 1:measures_dict_negative, 2:measures_dict_negative, 3:measures_dict_positive}
        self.evaluate_measures( 0, episode, pedestrian_dict)

        # Second and third person walk in the wall
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]), agents[2]: np.array([0, 1, 0]), agents[3]: np.array([0, -1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, breadth=0, action_nbr=5)

        new_positions = {0: [0, 1], 1: [2, 1], 2: [0, 2], 3: [2, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        rewards_target = {0: 0, 1: 0, 2: -1, 3: -1}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 0}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_negative, 2: measures_dict_positive,
                           3: measures_dict_positive}
        self.evaluate_measures(1, episode, pedestrian_dict)

        # First person walks on sidewalk
        actions = {agents[0]: np.array([0, 0, 0]), agents[1]: np.array([0, -1, 0]), agents[2]: np.array([0, 0, 0]),
                   agents[3]: np.array([0, 0, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, breadth=0, action_nbr=5)

        new_positions = {0: [0, 1], 1: [1, 1], 2: [0, 2], 3: [2, 2]}
        self.check_positions(episode, agents, new_positions, 3)
        rewards_target = {0: 0, 1: 0, 2: 0, 3: 0}
        self.check_reward(episode, rewards_target, 2)
        self.check_initializer_reward(episode, rewards_target, 2)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.iou_pavement: 1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.iou_pavement: 0}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_positive, 2: measures_dict_negative,
                           3: measures_dict_negative}
        self.evaluate_measures(2, episode, pedestrian_dict)

        # All people walks on into object
        actions = {agents[0]: np.array([0, 1, 1]), agents[1]: np.array([0, 0, 1]), agents[2]: np.array([0, 1, 0]),
                   agents[3]: np.array([0,-1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 4, breadth=0, action_nbr=5)

        new_positions = {0: [0, 1], 1: [1, 1], 2: [0, 2], 3: [2, 2]}
        self.check_positions(episode, agents, new_positions, 4)
        rewards_target = {0: -1, 1: -1, 2: -1, 3: -1}
        self.check_reward(episode, rewards_target, 3)
        self.check_initializer_reward(episode, rewards_target, 3)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.iou_pavement: 1,PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_obstacles: 1}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_positive, 2: measures_dict_negative,
                           3: measures_dict_negative}
        self.evaluate_measures(3, episode, pedestrian_dict)
        

    def test_walk_into_each_other(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)

        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  seq_len=seq_len, rewards=rewards, num_agents=4, init_rewards=init_rewards)

        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agents, episode, positions={0:np.array([0,0,0]), 1:np.array([0,2,0]), 2:np.array([0,0,2]), 3:np.array([0,2,2])})

        # First and second agent walk into eacother
        actions={agents[0]:np.array([0,1,0]),agents[1]:np.array([0,-1,0]), agents[2]:np.array([0,0,0]),agents[3]:np.array([0,0,0])}
        self.update_agents_and_episode( actions, environmentInteraction, episode, 1, breadth=0, action_nbr=5)

        new_positions={0:[1,0], 1:[1,0], 2:[0,2], 3:[2,2]}
        self.check_positions(episode, agents, new_positions, 1)

        # Second and third walk into eachother
        actions = {agents[0]: np.array([0, 1, 0]), agents[1]: np.array([0, -1, 0]), agents[2]: np.array([0, 1, 0]),
                   agents[3]: np.array([0, -1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, breadth=0, action_nbr=5)

        new_positions = {0: [2, 0], 1: [0, 0], 2: [1, 2], 3: [1, 2]}
        self.check_positions(episode, agents, new_positions, 2)

        rewards_target={0:-1, 1:-1,2:0,3:0}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        measures_dict_positive={PEDESTRIAN_MEASURES_INDX.hit_pedestrians:1, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:0}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:0}
        pedestrian_dict={0:measures_dict_positive, 1:measures_dict_positive, 2:measures_dict_negative, 3:measures_dict_negative}
        self.evaluate_measures( 0, episode, pedestrian_dict)

        # second person walks into third
        actions = {agents[0]: np.array([0, 0, 0]), agents[1]: np.array([0, 0, 2]), agents[2]: np.array([0, 1, 0]),
                   agents[3]: np.array([0, -1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, breadth=0, action_nbr=5)

        new_positions = {0: [2, 0], 1: [0, 2], 2: [2, 2], 3: [0, 2]}
        self.check_positions(episode, agents, new_positions, 3)

        rewards_target = {0: 0, 1: 0, 2: -1, 3: -1}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:0}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:0}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_negative, 2: measures_dict_positive,
                           3: measures_dict_positive}
        self.evaluate_measures(1, episode, pedestrian_dict)

        # All people walk into the center
        actions = {agents[0]: np.array([0, -1, 1]), agents[1]: np.array([0, 1, -1]), agents[2]: np.array([0, -1, -1]),
                   agents[3]: np.array([0, 1, -1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 4, breadth=0, action_nbr=5)

        new_positions = {0: [1, 1], 1: [1, 1], 2: [1, 1], 3: [1, 1]}
        self.check_positions(episode, agents, new_positions, 4)

        rewards_target = {0: 0, 1: -1, 2: 0, 3: -1}
        self.check_reward(episode, rewards_target, 2)
        self.check_initializer_reward(episode, rewards_target, 2)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:0}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:0}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_positive, 2: measures_dict_negative,
                           3: measures_dict_positive}
        self.evaluate_measures(2, episode, pedestrian_dict)

        # All people walk into the center
        actions = {agents[0]: np.array([0, 0, 0]), agents[1]: np.array([0,0, 0]), agents[2]: np.array([0, 0, 0]),
                   agents[3]: np.array([0, 0, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 5, breadth=0, action_nbr=5)

        new_positions = {0: [1, 1], 1: [1, 1], 2: [1, 1], 3: [1, 1]}
        self.check_positions(episode, agents, new_positions,5)


        rewards_target = {0: -1, 1: -1, 2: -1, 3: -1}
        self.check_reward(episode, rewards_target, 3)
        self.check_initializer_reward(episode, rewards_target, 3)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:0}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive, 2: measures_dict_positive,
                           3: measures_dict_positive}
        self.evaluate_measures(3, episode, pedestrian_dict)

    def test_walk_into_pedestrians(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        people_list=[]
        people[0].append(np.array([[0,0],[0,0], [2,2]]))
        people_list.append(np.array([[0,0],[0,0], [2,2]]))
        people[1].append(np.array([[0, 0], [0, 0], [2, 2]]))
        people_list.append(np.array([[0, 0], [0, 0], [2, 2]]))

        people[2].append(np.array([[0, 0], [1, 1], [2, 2]]))
        people_list.append(np.array([[0, 0], [1, 1], [2, 2]]))
        people[3].append(np.array([[0, 0], [1, 1], [2, 2]]))
        people_list.append(np.array([[0, 0], [1, 1], [2, 2]]))

        people[4].append(np.array([[0, 0], [2, 2], [2, 2]]))
        people_list.append(np.array([[0, 0], [2, 2], [2, 2]]))
        people[5].append(np.array([[0, 0], [2, 2], [2, 2]]))
        people_list.append(np.array([[0, 0], [2, 2], [2, 2]]))

        people_dict={0:people_list}
        init_frames={0:0}

        # people_dict={}, init_frames={}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                   seq_len=seq_len, rewards=rewards, num_agents=2,
                                                                   people_dict=people_dict, init_frames=init_frames, init_rewards=init_rewards)

        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agents, episode,
                            positions={0: np.array([0, 0, 0]), 1: np.array([0, 2, 0])})

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, breadth=0, action_nbr=5)

        new_positions = {0: [0, 1], 1: [2, 1]}
        self.check_positions(episode, agents, new_positions, 1)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, breadth=0, action_nbr=5)

        new_positions = {0: [0, 2], 1: [2, 2]}
        self.check_positions(episode, agents, new_positions, 2)


        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_negative}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 1, 0]), agents[1]: np.array([0, 0, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, breadth=0, action_nbr=5)

        new_positions = {0: [1, 2], 1: [2, 2]}
        self.check_positions(episode, agents, new_positions, 3)


        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory:1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 1, 0]), agents[1]: np.array([0, 0, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 4, breadth=0, action_nbr=5)

        new_positions = {0: [2, 2], 1: [2, 2]}
        self.check_positions(episode, agents, new_positions, 4)


        rewards_target = {0: -1, 1: 0}
        self.check_reward(episode, rewards_target, 2)
        self.check_initializer_reward(episode, rewards_target, 2)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1,PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory:1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(2, episode, pedestrian_dict)
        self.evaluate_measures_initializer(2, episode, pedestrian_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 0]), agents[1]: np.array([0, 0, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 5, breadth=0, action_nbr=5)

        new_positions = {0: [2, 2], 1: [2, 2]}
        self.check_positions(episode, agents, new_positions, 5)


        rewards_target = {0: -1, 1: -1}
        self.check_reward(episode, rewards_target, 3)
        self.check_initializer_reward(episode, rewards_target, 3)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1,
                                  PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}
        self.evaluate_measures(3, episode, pedestrian_dict)
        self.evaluate_measures_initializer(3, episode, pedestrian_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, -1]), agents[1]: np.array([0, -1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 6, breadth=0, action_nbr=5)

        new_positions = {0: [2, 1], 1: [1, 2]}
        self.check_positions(episode, agents, new_positions, 6)


        rewards_target = {0: -1, 1: -1}
        self.check_reward(episode, rewards_target, 4)
        self.check_initializer_reward(episode, rewards_target, 4)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1,
                                  PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}
        self.evaluate_measures(4, episode, pedestrian_dict)
        self.evaluate_measures_initializer(4, episode, pedestrian_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 0]), agents[1]: np.array([0, 0, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 7, breadth=0, action_nbr=5)

        new_positions = {0: [2, 1], 1: [1, 2]}
        self.check_positions(episode, agents, new_positions, 7)


        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 5)
        self.check_initializer_reward(episode, rewards_target, 5)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0,
                                  PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_measures(5, episode, pedestrian_dict)
        self.evaluate_measures_initializer(5, episode, pedestrian_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 0]), agents[1]: np.array([0, 0, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 8, breadth=0, action_nbr=5)

        new_positions = {0: [2, 1], 1: [1, 2]}
        self.check_positions(episode, agents, new_positions, 8)

        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 6)
        self.check_initializer_reward(episode, rewards_target, 6)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0,
                                  PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 0}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_measures(6, episode, pedestrian_dict)
        self.evaluate_measures_initializer(6, episode, pedestrian_dict)

    def test_walk_behind_pedestrians(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        people_list = []
        people[0].append(np.array([[0, 0], [0, 0], [2, 2]]))
        people_list.append(np.array([[0, 0], [0, 0], [2, 2]]))
        people[1].append(np.array([[0, 0], [0, 0], [2, 2]]))
        people_list.append(np.array([[0, 0], [0, 0], [2, 2]]))

        people[2].append(np.array([[0, 0], [1, 1], [2, 2]]))
        people_list.append(np.array([[0, 0], [1, 1], [2, 2]]))
        people[3].append(np.array([[0, 0], [1, 1], [2, 2]]))
        people_list.append(np.array([[0, 0], [1, 1], [2, 2]]))

        people[4].append(np.array([[0, 0], [2, 2], [2, 2]]))
        people_list.append(np.array([[0, 0], [2, 2], [2, 2]]))
        people[5].append(np.array([[0, 0], [2, 2], [2, 2]]))
        people_list.append(np.array([[0, 0], [2, 2], [2, 2]]))

        people_dict = {0: people_list}
        init_frames = {0: 0}

        # people_dict={}, init_frames={}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                   seq_len=seq_len, rewards=rewards, num_agents=2,
                                                                   people_dict=people_dict, init_frames=init_frames,agent_size=[0,1,1], init_rewards=init_rewards)

        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agents, episode,
                            positions={0: np.array([0, 0, 0]), 1: np.array([0, 2, 0])})

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, -1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, breadth=[0,1,1], action_nbr=5)

        new_positions = {0: [0, 1], 1: [1, 0]}
        self.check_positions(episode, agents, new_positions, 1)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, -1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, breadth=[0, 1, 1], action_nbr=5)

        new_positions = {0: [0, 2], 1: [0, 0]}
        self.check_positions(episode, agents, new_positions, 2)



        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1} # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 0}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_negative}
        self.evaluate_measures(0, episode, pedestrian_dict)


        # Did not move not, but positive reward for previous step
        actions = {agents[0]: np.array([0, 0, 0]), agents[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [0, 2], 1: [0, 1]}
        self.check_positions(episode, agents, new_positions, 3)


        rewards_target = {0: 1, 1: 0}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(1, episode, pedestrian_dict)

        # Did move positive reward -0
        actions = {agents[0]: np.array([0, 1, 0]), agents[1]: np.array([0, 0, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 4, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [1, 2], 1: [0, 1]}
        self.check_positions(episode, agents, new_positions, 4)

        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 2)
        self.check_initializer_reward(episode, rewards_target, 2)
        measures_dict_positive = {
            PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(2, episode, pedestrian_dict)

        # Did  move positive reward-1
        actions = {agents[0]: np.array([0, 0, 0]), agents[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 5, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [1, 2], 1: [0, 2]}
        self.check_positions(episode, agents, new_positions,5)

        rewards_target = {0: 1, 1: 0}
        self.check_reward(episode, rewards_target, 3)
        self.check_initializer_reward(episode, rewards_target, 3)
        measures_dict_positive = {
            PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(3, episode, pedestrian_dict)

        actions = {agents[0]: np.array([0, 0, 0]), agents[1]: np.array([0, 0, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 6, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [1, 2], 1: [0, 2]}
        self.check_positions(episode, agents, new_positions, 6)


        rewards_target = {0: 0, 1: 1}
        self.check_reward(episode, rewards_target, 4)
        self.check_initializer_reward(episode, rewards_target, 4)
        measures_dict_positive = {
            PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}
        self.evaluate_measures(4, episode, pedestrian_dict)

    def test_heatmap_walk_behind_pedestrians(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        people_list = []
        people[0].append(np.array([[0, 0], [0, 0], [2, 2]]))
        people_list.append(np.array([[0, 0], [0, 0], [2, 2]]))


        people[1].append(np.array([[0, 0], [1, 1], [2, 2]]))
        people_list.append(np.array([[0, 0], [1, 1], [2, 2]]))

        people[2].append(np.array([[0, 0], [2, 2], [2, 2]]))
        people_list.append(np.array([[0, 0], [2, 2], [2, 2]]))


        people_dict = {0: people_list}
        init_frames = {0: 0}

        # people_dict={}, init_frames={}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] = 1
        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] = 1
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                   seq_len=seq_len, rewards=rewards, num_agents=2,
                                                                   people_dict=people_dict, init_frames=init_frames,
                                                                   agent_size=[0, 1, 1],init_rewards=init_rewards)

        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agents, episode,
                            positions={0: np.array([0, 0, 0]), 1: np.array([0, 2, 0])})

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, -1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [0, 1], 1: [1, 0]}
        self.check_positions(episode, agents, new_positions, 1)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, -1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [0, 2],  1: [0, 0]}
        self.check_positions(episode, agents, new_positions, 2)

        rewards_target = {0: 2/9, 1: 0}
        self.check_reward(episode, rewards_target, 0)
        rewards_target = {0: 2 / 9, 1: 0}
        self.check_initializer_reward(episode, rewards_target, 0)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap: 2/9}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(0, episode, pedestrian_dict)

        actions = {agents[0]: np.array([0, 1, 0]), agents[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [1, 2], 1: [0, 1]}
        self.check_positions(episode, agents, new_positions, 3)

        rewards_target = {0: 2/9, 1: 0}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        measures_dict_positive = { PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap: 2/9,PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(1, episode, pedestrian_dict)

        actions = {agents[0]: np.array([0, 1, 0]), agents[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 4, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [2, 2], 1: [0, 2]}
        self.check_positions(episode, agents, new_positions, 4)

        rewards_target = {0: 3 / 9, 1: 2/9}
        self.check_reward(episode, rewards_target, 2)
        self.check_initializer_reward(episode, rewards_target, 2)
        measures_dict_positive = {
            PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap: 3 / 9,PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap: 2/9}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(2, episode, pedestrian_dict)

    def test_hit_car(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        cars_list = []
        cars[0].append(np.array([0, 0+1,0, 0+1,2, 2+1]))
        cars_list.append(np.array([0, 0+1,0, 0+1,2, 2+1]))


        cars[1].append(np.array([0, 0+1,1, 1+1,2, 2+1]))
        cars_list.append(np.array([0, 0+1,1, 1+1,2, 2+1]))

        cars[2].append(np.array([0, 0+1,2, 2+1,2, 2+1]))
        cars_list.append(np.array([0, 0+1,2, 2+1,2, 2+1]))


        car_dict = {0: cars_list}
        init_frames_cars = {0: 0}

        # people_dict={}, init_frames={}
        rewards = self.get_reward(True)
        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_car_agent] = 1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                   seq_len=seq_len, rewards=rewards, num_agents=2,
                                                                   car_dict=car_dict, init_frames_cars=init_frames_cars,
                                                                   agent_size=[0, 1, 1], init_rewards=init_rewards)

        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agents, episode,
                            positions={0: np.array([0, 0, 0]), 1: np.array([0, 2, 0])})

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [0, 1], 1: [2,1]}
        self.check_positions(episode, agents, new_positions, 1)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, breadth=[0, 1, 1], action_nbr=5)
        new_positions = {0: [0, 2],  1: [2, 2]}
        self.check_positions(episode, agents, new_positions, 2)

        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 0)
        rewards_target = {0: 0, 1: 0}
        self.check_initializer_reward(episode, rewards_target, 0)
        measures_dict_positive = {
            PEDESTRIAN_MEASURES_INDX.hit_by_car: 0}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}
        self.evaluate_measures(0, episode, pedestrian_dict)

        rewards_target = {0: 0, 1: -1}
        self.check_reward(episode, rewards_target, 1)
        rewards_target = {0: 0, 1: -1}
        self.check_initializer_reward(episode, rewards_target, 1)
        measures_dict_positive = {
            PEDESTRIAN_MEASURES_INDX.hit_by_car: 0}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(1, episode, pedestrian_dict)

    def test_hit_car_agent(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        cars_list = []
        cars[0].append(np.array([0, 0 + 1, 0, 0 + 1, 2, 2 + 1]))
        cars_list.append(np.array([0, 0 + 1, 0, 0 + 1, 2, 2 + 1]))

        cars[1].append(np.array([0, 0 + 1, 1, 1 + 1, 2, 2 + 1]))
        cars_list.append(np.array([0, 0 + 1, 1, 1 + 1, 2, 2 + 1]))

        cars[2].append(np.array([0, 0 + 1, 2, 2 + 1, 2, 2 + 1]))
        cars_list.append(np.array([0, 0 + 1, 2, 2 + 1, 2, 2 + 1]))

        car_dict = {0: cars_list}
        init_frames_cars = {0: 0}

        # people_dict={}, init_frames={}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        car_rewards= self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.collision_pedestrian_with_car]=-1
        init_rewards_car=self.get_reward_car()
        init_rewards_car[CAR_REWARD_INDX.collision_pedestrian_with_car]=1

        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_car_agent] = 1


        car_agent_dict={'positions':{0: np.array([0, 0, 1])}, 'goals':{0: np.array([0, 3, 1])},'dirs':{0: np.array([0, 1, 0])}}
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                   seq_len=seq_len, rewards=rewards, num_agents=3,num_car_agents=1,
                                                                   car_dict=car_dict,
                                                                   init_frames_cars=init_frames_cars,
                                                                               car_agent_dict=car_agent_dict,car_rewards=car_rewards,
                                                                               init_rewards=init_rewards, init_rewards_car=init_rewards_car)

        episode.agent_size = [0, 0, 0]
        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode,
                            positions={0: np.array([0, 0, 0]), 1: np.array([0, 1, 0]),2: np.array([0, 2, 0]), })


        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]), agents[2]: np.array([0, 0, 1]), car_agents.trainableCars[0]:np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [0, 1], 1: [1, 1], 2: [2, 1]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [1, 1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]), agents[2]: np.array([0, 0, 1]), car_agents.trainableCars[0]:np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [0, 2], 1: [1, 1], 2: [2, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [1, 1]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        rewards_target = {0: 0, 1: -1, 2: 0}
        self.check_reward(episode, rewards_target, 0)
        rewards_target = {0: 0, 1: 1, 2: 0}
        self.check_initializer_reward(episode, rewards_target, 0)
        rewards_target_car = {0: -1}
        self.check_car_reward(episode, rewards_target_car, 0)
        rewards_target_car_init = {0: 1}
        self.check_car_initializer_reward(episode, rewards_target_car_init, 0)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 0}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1,PEDESTRIAN_MEASURES_INDX.hit_by_car: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative,2: measures_dict_positive}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.hit_by_agent: 1, CAR_MEASURES_INDX.hit_pedestrians: 1}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)



        rewards_target = {0: 0, 1: 0, 2: -1}
        self.check_reward(episode, rewards_target, 1)
        rewards_target = {0: 0, 1: 0, 2: -1}
        self.check_initializer_reward(episode, rewards_target, 1)
        rewards_target_car = {0: 0}
        self.check_car_reward(episode, rewards_target_car, 1)
        self.check_car_initializer_reward(episode, rewards_target_car, 1)
        measures_dict_0 = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 0}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_1 = {PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1, PEDESTRIAN_MEASURES_INDX.hit_by_car: 1, PEDESTRIAN_MEASURES_INDX.agent_dead:1}
        measures_dict_2 = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 1}
        pedestrian_dict =  {0: measures_dict_0,1: measures_dict_1,2: measures_dict_2}
        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.agent_dead: 1, CAR_MEASURES_INDX.hit_by_agent: 1,
                                  CAR_MEASURES_INDX.hit_pedestrians: 1}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)


    def test_hit_car_multiple_agents(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=4)
        cars_list = []
        cars[0].append(np.array([0, 0 + 1, 0, 0 + 1, 2, 2 + 1]))
        cars_list.append(np.array([0, 0 + 1, 0, 0 + 1, 2, 2 + 1]))

        cars[1].append(np.array([0, 0 + 1, 1, 1 + 1, 2, 2 + 1]))
        cars_list.append(np.array([0, 0 + 1, 1, 1 + 1, 2, 2 + 1]))

        cars[2].append(np.array([0, 0 + 1, 2, 2 + 1, 2, 2 + 1]))
        cars_list.append(np.array([0, 0 + 1, 2, 2 + 1, 2, 2 + 1]))

        car_dict = {0: cars_list}
        init_frames_cars = {0: 0}

        # people_dict={}, init_frames={}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.collision_pedestrian_with_car] = -1
        init_rewards_car=self.get_reward_car()
        init_rewards_car[CAR_REWARD_INDX.collision_pedestrian_with_car] = 1

        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_car_agent]=1

        car_agent_dict={'positions':{0: np.array([0, 0, 1]),1: np.array([0, 0, 3])}, 'goals':{0: np.array([0, 4, 1]),1: np.array([0, 4, 3])},'dirs':{0: np.array([0, 1, 0]),1: np.array([0, 1, 0])}}
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                   seq_len=seq_len, rewards=rewards, num_agents=4,num_car_agents=2,
                                                                   car_dict=car_dict,
                                                                   init_frames_cars=init_frames_cars,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,
                                                                               init_rewards=init_rewards,init_rewards_car=init_rewards_car)

        episode.agent_size = [0, 0, 0]
        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode,
                            positions={0: np.array([0, 0, 0]), 1: np.array([0, 1, 0]),2: np.array([0, 2, 0]),3: np.array([0, 3, 0]) })


        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]), agents[2]: np.array([0, 0, 1]),agents[3]: np.array([0, 0, 1]), car_agents.trainableCars[0]:np.array([0, 1, 0]), car_agents.trainableCars[1]:np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [0, 1], 1: [1, 1], 2: [2, 1],3: [3, 1]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [1, 1],1: [1, 3]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]), agents[2]: np.array([0, 0, 1]), agents[3]: np.array([0, 0, 1]), car_agents.trainableCars[0]:np.array([0, 1, 0]), car_agents.trainableCars[1]:np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [0, 2], 1: [1, 1], 2: [2, 2],3: [3, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [1, 1],1: [2, 3]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        rewards_target = {0: 0, 1: -1, 2: 0, 3:0}
        self.check_reward(episode, rewards_target, 0)

        rewards_target_init = {0: 0, 1: 1, 2: 0, 3: 0}
        self.check_initializer_reward(episode, rewards_target_init, 0)

        rewards_target_car = {0: -1, 1: 0}
        self.check_car_reward(episode, rewards_target_car, 0)
        rewards_target_car_init = {0: 1, 1: 0}
        self.check_car_initializer_reward(episode, rewards_target_car_init, 0)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 0}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1,PEDESTRIAN_MEASURES_INDX.hit_by_car: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative,2: measures_dict_positive,3: measures_dict_positive}
        
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)
        measures_dict_positive = {CAR_MEASURES_INDX.hit_by_agent: 0}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {CAR_MEASURES_INDX.hit_by_agent: 1, CAR_MEASURES_INDX.hit_pedestrians: 1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]), agents[2]: np.array([0, 0, 1]),
                   agents[3]: np.array([0, 0, 1]), car_agents.trainableCars[0]: np.array([0, 1, 0]),
                   car_agents.trainableCars[1]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, action_nbr=5, carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [0, 3], 1: [1, 1], 2: [2, 2], 3: [3, 3]}
        self.check_positions(episode, agents, new_positions, 3)
        new_positions_cars = {0: [1, 1], 1: [3, 3]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 3)

        rewards_target = {0: 0, 1: 0, 2: -1, 3:0}
        self.check_reward(episode, rewards_target, 1)

        rewards_target_init = {0: 0, 1: 0, 2: -1, 3: 0}
        self.check_initializer_reward(episode, rewards_target_init, 1)

        rewards_target_car = {0: 0, 1:0}
        self.check_car_reward(episode, rewards_target_car, 1)

        self.check_car_initializer_reward(episode, rewards_target_car, 1)

        measures_dict_0 = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 0}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_1 = {PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1, PEDESTRIAN_MEASURES_INDX.hit_by_car: 1,
                           PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        measures_dict_2 = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 1}
        pedestrian_dict = {0: measures_dict_0, 1: measures_dict_1, 2: measures_dict_2,3: measures_dict_0}
        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)
        measures_dict_positive = {CAR_MEASURES_INDX.hit_by_agent: 0}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = { CAR_MEASURES_INDX.agent_dead:1,CAR_MEASURES_INDX.hit_by_agent: 1, CAR_MEASURES_INDX.hit_pedestrians: 1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]), agents[2]: np.array([0, 0, 1]),
                   agents[3]: np.array([0, 0, 1]), car_agents.trainableCars[0]: np.array([0, 1, 0]),
                   car_agents.trainableCars[1]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 4, action_nbr=5, carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [0, 4], 1: [1, 1], 2: [2, 2], 3: [3, 3]}
        self.check_positions(episode, agents, new_positions, 4)
        new_positions_cars = {0: [1, 1], 1: [3, 3]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 4)

        rewards_target = {0: 0, 1: 0, 2: 0, 3:-1}
        self.check_reward(episode, rewards_target, 2)

        rewards_target_init = {0: 0, 1: 0, 2: 0, 3:1}
        self.check_initializer_reward(episode, rewards_target_init, 2)

        rewards_target_car = {0: 0, 1: -1}
        self.check_car_reward(episode, rewards_target_car, 2)
        rewards_target_car_init = {0: 0, 1: 1}
        self.check_car_initializer_reward(episode, rewards_target_car_init, 2)

        measures_dict_0 = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 0}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_1 = {PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1, PEDESTRIAN_MEASURES_INDX.hit_by_car: 1,
                           PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        measures_dict_2 = { PEDESTRIAN_MEASURES_INDX.hit_by_car: 1,
                           PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        measures_dict_3 = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 1,PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1}
        pedestrian_dict = {0: measures_dict_0, 1: measures_dict_1, 2: measures_dict_2, 3:measures_dict_3}
        self.evaluate_measures(2, episode, pedestrian_dict)
        self.evaluate_measures_initializer(2, episode, pedestrian_dict)

        measures_dict_positive = {CAR_MEASURES_INDX.hit_by_agent: 1, CAR_MEASURES_INDX.hit_pedestrians: 1}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_negative = {CAR_MEASURES_INDX.hit_by_agent: 1, CAR_MEASURES_INDX.hit_pedestrians: 1,
                                  CAR_MEASURES_INDX.agent_dead: 1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_positive}
        self.evaluate_car_measures(2, episode, car_dict)
        self.evaluate_car_initializer_measures(2, episode, car_dict)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]), agents[2]: np.array([0, 0, 1]),
                   agents[3]: np.array([0, 0, 1]), car_agents.trainableCars[0]: np.array([0, 1, 0]),
                   car_agents.trainableCars[1]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 5, action_nbr=5, carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [0, 5], 1: [1, 1], 2: [2, 2], 3: [3, 3]}
        self.check_positions(episode, agents, new_positions, 5)
        new_positions_cars = {0: [1, 1], 1: [3, 3]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 5)


        rewards_target = {0: 0, 2: 0, 1: 0, 3:0}
        self.check_reward(episode, rewards_target, 3)

        rewards_target_init = {0: 0, 1: 0, 2: 0, 3: 0}
        self.check_initializer_reward(episode, rewards_target_init, 3)

        rewards_target_car = {0: 0, 1: 0}
        self.check_car_reward(episode, rewards_target_car, 3)
        self.check_car_initializer_reward(episode, rewards_target_car, 3)
        measures_dict_0 = {PEDESTRIAN_MEASURES_INDX.out_of_axis: 1}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_1 = {PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1, PEDESTRIAN_MEASURES_INDX.hit_by_car: 1,
                           PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        measures_dict_2 = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 1,PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        pedestrian_dict = {0: measures_dict_0, 1: measures_dict_1, 2: measures_dict_2, 3:measures_dict_1}
        self.evaluate_measures(3, episode, pedestrian_dict)
        self.evaluate_measures_initializer(3, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.hit_by_agent: 1, CAR_MEASURES_INDX.hit_pedestrians: 1,
                                  CAR_MEASURES_INDX.agent_dead: 1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_negative}
        self.evaluate_car_measures(3, episode, car_dict)

        rewards_target = {0: 0, 2: 0, 1: 0, 3: 0}
        self.check_reward(episode, rewards_target, 4)

        rewards_target_init = {0: 0, 1: 0, 2: 0, 3: 0}
        self.check_initializer_reward(episode, rewards_target_init, 4)

        rewards_target_car = {0: 0, 1: 0}
        self.check_car_reward(episode, rewards_target_car, 4)

        self.check_car_initializer_reward(episode, rewards_target_car, 4)

        measures_dict_0 = {PEDESTRIAN_MEASURES_INDX.out_of_axis: 1}  # PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap
        measures_dict_1 = {PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1, PEDESTRIAN_MEASURES_INDX.hit_by_car: 1,
                           PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        measures_dict_2 = {PEDESTRIAN_MEASURES_INDX.hit_by_car: 1, PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        pedestrian_dict = {0: measures_dict_0, 1: measures_dict_1, 2: measures_dict_2, 3: measures_dict_1}
        self.evaluate_measures(4, episode, pedestrian_dict)
        self.evaluate_measures_initializer(4, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.hit_by_agent: 1, CAR_MEASURES_INDX.hit_pedestrians: 1,
                                  CAR_MEASURES_INDX.agent_dead: 1}
        car_dict = {0: measures_dict_negative, 1: measures_dict_negative}
        self.evaluate_car_measures(4, episode, car_dict)
        self.evaluate_car_initializer_measures(4, episode, car_dict)


    def test_hit_car_agent_wide(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=6)


        # people_dict={}, init_frames={}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        car_rewards= self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.collision_pedestrian_with_car]=-1
        init_rewards_car=self.get_reward_car()
        init_rewards_car[CAR_REWARD_INDX.collision_pedestrian_with_car]=1
        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        init_rewards[PEDESTRIAN_REWARD_INDX.collision_with_car_agent] = 1

        car_agent_dict={'positions':{0: np.array([0, 1,4])}, 'goals':{0: np.array([0, 5, 4])},'dirs':{0: np.array([0, 1, 0])}}
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                   seq_len=seq_len, rewards=rewards, num_agents=2,num_car_agents=1,
                                                                   car_agent_dict=car_agent_dict,car_rewards=car_rewards,
                                                                               agent_size=[1, 1, 1],car_dim=[1,1,1],
                                                                               init_rewards=init_rewards,init_rewards_car=init_rewards_car)

        episode.agent_size = [0, 1, 1]
        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode,
                            positions={0: np.array([0, 1, 1]), 1: np.array([0, 4, 1])})


        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]), car_agents.trainableCars[0]:np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1: [4, 2]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [2,4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5, carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1: [4, 2]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [2, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        rewards_target = {0:-1, 1: -1}
        self.check_reward(episode, rewards_target, 0)
        rewards_target = {0: 1, 1: 1}
        self.check_initializer_reward(episode, rewards_target, 0)
        rewards_target_car = {0: -1}
        self.check_car_reward(episode, rewards_target_car, 0)
        rewards_target_car_init = {0: 1}
        self.check_car_initializer_reward(episode, rewards_target_car_init, 0)

        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1,PEDESTRIAN_MEASURES_INDX.hit_by_car: 1}
        pedestrian_dict = {0: measures_dict_negative, 1: measures_dict_negative}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.hit_by_agent: 1, CAR_MEASURES_INDX.hit_pedestrians: 1}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)


        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode,3, action_nbr=5, carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1: [4, 2]}
        self.check_positions(episode, agents, new_positions, 3)
        new_positions_cars = {0: [2, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 3)


        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 1)
        rewards_target = {0: 0, 1: 0}
        self.check_initializer_reward(episode, rewards_target, 1)
        rewards_target_car = {0: 0}
        self.check_car_reward(episode, rewards_target_car, 1)
        self.check_car_initializer_reward(episode, rewards_target_car, 1)

        measures_dict_1 = {PEDESTRIAN_MEASURES_INDX.hit_by_hero_car: 1, PEDESTRIAN_MEASURES_INDX.hit_by_car: 1, PEDESTRIAN_MEASURES_INDX.agent_dead:1}

        pedestrian_dict =  {0: measures_dict_1,1: measures_dict_1}
        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.agent_dead: 1, CAR_MEASURES_INDX.hit_by_agent: 1,
                                  CAR_MEASURES_INDX.hit_pedestrians: 1}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)




    def test_reach_goal(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=6)

        # people_dict={}, init_frames={}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 10
        rewards[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal]=1

        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.reached_goal] = 10
        car_rewards[CAR_REWARD_INDX.distance_travelled_towards_goal] = 1
        init_rewards_car=car_rewards


        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 10
        init_rewards[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] = 1

        car_agent_dict = {'positions': {0: np.array([0, 1, 4])}, 'goals': {0: np.array([0, 5, 4])},
                          'dirs': {0: np.array([0, 1, 0])}}
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y,
                                                                               tensor,
                                                                               seq_len=seq_len, rewards=rewards,
                                                                               num_agents=2, num_car_agents=1,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,init_rewards=init_rewards, init_rewards_car=init_rewards_car,goal_dir=True)

        episode.max_step=1
        car_agents.trainableCars[0].settings.car_max_speed_voxelperframe=1
        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode,positions={0: np.array([0, 1, 1]), 1: np.array([0, 4, 1])}, goals={0: np.array([0, 1, 4]), 1: np.array([0, 4, 4])})

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1: [4, 2]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [2, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [1, 3], 1: [4, 3]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [3, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)

        rewards_target = {0: 1, 1: 1}
        self.check_reward(episode, rewards_target, 0)
        self.check_initializer_reward(episode, rewards_target, 0)
        rewards_target_car = {0: 1}
        self.check_car_reward(episode, rewards_target_car, 0)
        self.check_car_initializer_reward(episode, rewards_target_car, 0)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 2,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 3, CAR_MEASURES_INDX.goal_reached: 0}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [1, 3], 1: [4, 3]}
        self.check_positions(episode, agents, new_positions, 3)
        new_positions_cars = {0: [4, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 3)

        rewards_target = {0: 10, 1: 10}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        rewards_target_car = {0: 1}
        self.check_car_reward(episode, rewards_target_car, 1)
        self.check_car_initializer_reward(episode, rewards_target_car, 1)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 1,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}

        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 2, CAR_MEASURES_INDX.goal_reached: 0}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 4, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1,3], 1: [4, 3]}
        self.check_positions(episode, agents, new_positions,4)
        new_positions_cars = {0: [5, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 4)

        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 2)
        self.check_initializer_reward(episode, rewards_target, 2)
        rewards_target_car = {0: 1}
        self.check_car_reward(episode, rewards_target_car, 2)
        self.check_car_initializer_reward(episode, rewards_target_car, 2)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 1,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 1,
                                  PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}
        self.evaluate_measures(2, episode, pedestrian_dict)
        self.evaluate_measures_initializer(2, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 1, CAR_MEASURES_INDX.goal_reached: 0}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(2, episode, car_dict)
        self.evaluate_car_initializer_measures(2, episode, car_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 5, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1, 3], 1: [4, 3]}
        self.check_positions(episode, agents, new_positions, 5)
        new_positions_cars = {0: [5, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 5)

        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 3)
        self.check_initializer_reward(episode, rewards_target, 3)
        rewards_target_car = {0: 10}
        self.check_car_reward(episode, rewards_target_car, 3)
        self.check_car_initializer_reward(episode, rewards_target_car, 3)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 1,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 1,
                                  PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}
        self.evaluate_measures(3, episode, pedestrian_dict)
        self.evaluate_measures_initializer(3, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 0, CAR_MEASURES_INDX.goal_reached: 1}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(3, episode, car_dict)
        self.evaluate_car_initializer_measures(3, episode, car_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode,6, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1, 3], 1: [4, 3]}
        self.check_positions(episode, agents, new_positions, 6)
        new_positions_cars = {0: [5, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 6)

        rewards_target = {0: 0, 1: 0}
        self.check_reward(episode, rewards_target, 4)
        self.check_initializer_reward(episode, rewards_target, 4)
        rewards_target_car = {0: 0}
        self.check_car_reward(episode, rewards_target_car, 4)
        self.check_car_initializer_reward(episode, rewards_target_car, 4)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 1,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 1,
                                  PEDESTRIAN_MEASURES_INDX.agent_dead: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}
        self.evaluate_measures(4, episode, pedestrian_dict)
        self.evaluate_measures_initializer(4, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 0, CAR_MEASURES_INDX.goal_reached: 1,  CAR_MEASURES_INDX.agent_dead: 1}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(4, episode, car_dict)
        self.evaluate_car_initializer_measures(4, episode, car_dict)

    def test_reach_goal_multiple_goals(self):
        seq_len = 11
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=6)

        # people_dict={}, init_frames={}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 10
        rewards[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] = 1

        car_rewards = self.get_reward_car()
        car_rewards[CAR_REWARD_INDX.reached_goal] = 10
        car_rewards[CAR_REWARD_INDX.distance_travelled_towards_goal] = 1
        init_rewards_car=car_rewards

        init_rewards = self.get_reward_initializer()
        init_rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 10
        init_rewards[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] = 1


        car_agent_dict = {'positions': {0: np.array([0, 1, 4])}, 'goals': {0: np.array([0, 5, 4])},
                          'dirs': {0: np.array([0, 1, 0])}}
        car_agents, agents, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y,
                                                                               tensor,
                                                                               seq_len=seq_len, rewards=rewards,
                                                                               num_agents=2, num_car_agents=1,
                                                                               car_agent_dict=car_agent_dict,
                                                                               car_rewards=car_rewards,
                                                                               init_rewards=init_rewards,
                                                                               goal_dir=True, stop_on_goal=False,init_rewards_car=init_rewards_car)

        episode.max_step = 1
        car_agents.trainableCars[0].settings.car_max_speed_voxelperframe = 1
        # positions={}, goals={}, dirs={}

        self.initialize_pos(agents, episode, positions={0: np.array([0, 1, 1]), 1: np.array([0, 4, 1])},
                            goals={0: np.array([0, 1, 4]), 1: np.array([0, 4, 4])})

        # First and second agent walk into eacother888
        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 1, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1: [4, 2]}
        self.check_positions(episode, agents, new_positions, 1)
        new_positions_cars = {0: [2, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 1)

        actions = {agents[0]: np.array([0, 0, 1]), agents[1]: np.array([0, 0, 1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        new_goal={agents[0]: np.array([0, 1, 0]), agents[1]: np.array([0, 0, 3])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 2, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents, manual_goal_dict=new_goal)
        new_positions = {0: [1, 3], 1: [4, 3]}
        self.check_positions(episode, agents, new_positions, 2)
        new_positions_cars = {0: [3, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 2)
        np.testing.assert_array_equal(episode.pedestrian_data[0].goal[2], new_goal[agents[0]])
        np.testing.assert_array_equal(episode.pedestrian_data[1].goal[2], new_goal[agents[1]])
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, episode.pedestrian_data[0].goal[1],  episode.pedestrian_data[0].goal[2])
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, episode.pedestrian_data[1].goal[1],
                                 episode.pedestrian_data[1].goal[2])


        rewards_target = {0: 1, 1: 1}
        self.check_reward(episode, rewards_target, 0)

        self.check_initializer_reward(episode, rewards_target, 0)
        rewards_target_car = {0: 1}
        self.check_car_reward(episode, rewards_target_car, 0)
        self.check_car_initializer_reward(episode, rewards_target_car, 0)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 2,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}
        self.evaluate_measures(0, episode, pedestrian_dict)
        self.evaluate_measures_initializer(0, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 3, CAR_MEASURES_INDX.goal_reached: 0}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(0, episode, car_dict)
        self.evaluate_car_initializer_measures(0, episode, car_dict)

        actions = {agents[0]: np.array([0, 0, -1]), agents[1]: np.array([0, -1, 0]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 3, action_nbr=5,
                                       carAgent=car_agents,
                                       allPedestrianAgents=agents)
        new_positions = {0: [1, 2], 1: [3, 3]}
        self.check_positions(episode, agents, new_positions, 3)
        new_positions_cars = {0: [4, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 3)

        rewards_target = {0: 10, 1: 10}
        self.check_reward(episode, rewards_target, 1)
        self.check_initializer_reward(episode, rewards_target, 1)
        rewards_target_car = {0: 1}
        self.check_car_reward(episode, rewards_target_car, 1)
        self.check_car_initializer_reward(episode, rewards_target_car, 1)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 1,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 1}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_positive}

        self.evaluate_measures(1, episode, pedestrian_dict)
        self.evaluate_measures_initializer(1, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 2, CAR_MEASURES_INDX.goal_reached: 0}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(1, episode, car_dict)
        self.evaluate_car_initializer_measures(1, episode, car_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 0, -1]), agents[1]: np.array([0, -1, 0]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        new_goal = {agents[0]: np.array([0, 5, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 4, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents, manual_goal_dict=new_goal)
        new_positions = {0: [1, 1], 1: [2, 3]}
        self.check_positions(episode, agents, new_positions, 4)
        new_positions_cars = {0: [5, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 4)

        rewards_target = {0: 1, 1: 1}
        self.check_reward(episode, rewards_target, 2)
        self.check_initializer_reward(episode, rewards_target, 2)
        rewards_target_car = {0: 1}
        self.check_car_reward(episode, rewards_target_car, 2)
        self.check_car_initializer_reward(episode, rewards_target_car, 2)
        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 2,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 0,
                                  PEDESTRIAN_MEASURES_INDX.agent_dead: 0}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 3,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 0,
                                  PEDESTRIAN_MEASURES_INDX.agent_dead: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(2, episode, pedestrian_dict)
        self.evaluate_measures_initializer(2, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 1, CAR_MEASURES_INDX.goal_reached: 0}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(2, episode, car_dict)
        self.evaluate_car_initializer_measures(2, episode, car_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 1, 0]), agents[1]: np.array([0, -1, 0]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}
        new_goal = {agents[1]: np.array([0, 5, 1])}
        self.update_agents_and_episode(actions, environmentInteraction, episode, 5, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents, manual_goal_dict=new_goal)
        new_positions = {0: [2, 1], 1: [1, 3]}
        self.check_positions(episode, agents, new_positions, 5)
        new_positions_cars = {0: [5, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 5)

        np.testing.assert_array_equal(episode.pedestrian_data[1].goal[5], new_goal[agents[1]])
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, episode.pedestrian_data[1].goal[5],
                                 episode.pedestrian_data[1].goal[4])
        np.testing.assert_array_equal( episode.pedestrian_data[0].goal[5],episode.pedestrian_data[0].goal[4])

        rewards_target = {0: 10, 1: 1}
        self.check_reward(episode, rewards_target, 3)
        self.check_initializer_reward(episode, rewards_target, 3)
        rewards_target_car = {0: 10}
        self.check_car_reward(episode, rewards_target_car, 3)
        self.check_car_initializer_reward(episode, rewards_target_car, 3)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 1,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 1,
                                  PEDESTRIAN_MEASURES_INDX.agent_dead: 0}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 2,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 0,
                                  PEDESTRIAN_MEASURES_INDX.agent_dead: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(3, episode, pedestrian_dict)
        self.evaluate_measures_initializer(3, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 0, CAR_MEASURES_INDX.goal_reached: 1}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(3, episode, car_dict)
        self.evaluate_car_initializer_measures(3, episode, car_dict)

        # First and second agent walk into eacother
        actions = {agents[0]: np.array([0, 1,0]), agents[1]: np.array([0, 1, -1]),
                   car_agents.trainableCars[0]: np.array([0, 1, 0])}

        self.update_agents_and_episode(actions, environmentInteraction, episode, 6, action_nbr=5,
                                       carAgent=car_agents, allPedestrianAgents=agents)
        new_positions = {0: [3, 1], 1: [2, 2]}
        self.check_positions(episode, agents, new_positions, 6)
        new_positions_cars = {0: [5, 4]}
        self.check_positions_car(episode, car_agents.trainableCars, new_positions_cars, 6)

        rewards_target = {0: 1, 1: 10}
        self.check_reward(episode, rewards_target, 4)
        self.check_initializer_reward(episode, rewards_target, 4)
        rewards_target_car = {0: 0}
        self.check_car_reward(episode, rewards_target_car, 4)
        self.check_car_initializer_reward(episode, rewards_target_car, 4)

        measures_dict_positive = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 3,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 0,
                                  PEDESTRIAN_MEASURES_INDX.agent_dead: 0}
        measures_dict_negative = {PEDESTRIAN_MEASURES_INDX.dist_to_goal: 1,
                                  PEDESTRIAN_MEASURES_INDX.goal_reached: 1,
                                  PEDESTRIAN_MEASURES_INDX.agent_dead: 0}
        pedestrian_dict = {0: measures_dict_positive, 1: measures_dict_negative}
        self.evaluate_measures(4, episode, pedestrian_dict)
        self.evaluate_measures_initializer(4, episode, pedestrian_dict)

        measures_dict_negative = {CAR_MEASURES_INDX.dist_to_goal: 0, CAR_MEASURES_INDX.goal_reached: 1,
                                  CAR_MEASURES_INDX.agent_dead: 1}
        car_dict = {0: measures_dict_negative}
        self.evaluate_car_measures(4, episode, car_dict)
        self.evaluate_car_initializer_measures(4, episode, car_dict)



