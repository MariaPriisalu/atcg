import unittest
import numpy as np
import sys

from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE,NBR_MEASURES,PEDESTRIAN_MEASURES_INDX
from RL.environment_interaction import EntitiesRecordedDataSource, EnvironmentInteraction
from commonUtils.ReconstructionUtils import LAST_CITYSCAPES_SEMLABEL, cityscapes_labels_dict, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, SIDEWALK_LABELS,CHANNELS,OBSTACLE_LABELS_NEW, OBSTACLE_LABELS,cityscapes_labels_dict
# Test methods in episode.

class TestEpisode(unittest.TestCase):

    def get_reward(self, all_zeros=False):

        rewards = np.zeros(NBR_REWARD_WEIGHTS)
        if all_zeros:
            return rewards

        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
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
    def get_episode(self, cars, gamma, people, pos_x, pos_y, tensor, seq_len=15, rewards=[], agent_size=(0, 0, 0), people_dict={}, init_frames={}, car_dict={}, init_frames_cars={}, new_carla=False):
        if len(rewards)==0:
            rewards = self.get_reward(False)
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1

        agent, episode, environmentInteraction = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                         seq_len=seq_len, rewards=rewards,  agent_size=agent_size,
                                                                         people_dict=people_dict, init_frames=init_frames, car_dict=car_dict, init_frames_cars=init_frames_cars, new_carla=new_carla)
        episode.agent_size = [0, 0, 0]
        return  agent, episode, environmentInteraction
    # Help function. Setup for tests.
    def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor, seq_len=30, rewards=[],
                           agent_size=(0, 0, 0), people_dict={}, init_frames={}, car_dict={}, init_frames_cars={},new_carla=False):
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
                                    seq_len, rewards, rewards,agent_size=agent_size,
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

    def initialize_tensor(self, seq_len=30, size=3):
        tensor = np.zeros((size, size, size, 6))
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
        pos, i, vel = episode.initial_position(0, None, initialization=PEDESTRIAN_INITIALIZATION_CODE.randomly)

        agent.initial_position(pos, episode.pedestrian_data[0].goal[0,:])

    def update_agent_and_episode(self, action, agent, environmentInteraction, episode, frame, breadth=0, action_nbr=4):
        if breadth==0:
            breadth=episode.agent_size[1:]
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
#----
    # Test correct empty initialization.
    def test_initialize_randomly_in_empty_tensor(self):
        cars, gamma, people, pos_x, pos_y, tensor=self.initialize_tensor()
        agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
        pos, i, vel=episode.initial_position(0,None,PEDESTRIAN_INITIALIZATION_CODE.randomly)
        # print( pos)
        np.testing.assert_array_equal(episode.valid_positions[pos[1],pos[2] ], 1)
        np.testing.assert_array_equal(i, -1)


    # Test correct initialization when one voxel of sidewalk exists.
    def test_initialize_on_pavement_in_tensor_with_only_one_pavement_voxel(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        for i in SIDEWALK_LABELS:#[6,7,8,9,10]:
            tensor[1,1,1,CHANNELS.semantic]=i/NUM_SEM_CLASSES
            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
            Simple=episode.find_sidewalk(False)
            np.testing.assert_array_equal(Simple[0], [1])
            np.testing.assert_array_equal(Simple[1], [1])
            np.testing.assert_array_equal(Simple[2], [1])
            pos, k, vel = episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
            np.testing.assert_array_equal(pos, [1, 1, 1])
            np.testing.assert_array_equal(k, -1)



    # Test correct initialization when two voxels of road exists.
    def test_initialize_on_pavement_in_tensor_with_only_one_pavement_voxel_and_one_ground_voxel(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        tensor[:,:,:,CHANNELS.semantic] = cityscapes_labels_dict['ground']/ NUM_SEM_CLASSES*np.ones((3,3,3))
        tensor[1,1,1,CHANNELS.semantic]=cityscapes_labels_dict['sidewalk']/NUM_SEM_CLASSES
        agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
        Simple=episode.find_sidewalk(False) # test Simple function.
        np.testing.assert_array_equal(Simple[0], [1])
        np.testing.assert_array_equal(Simple[1], [1])
        np.testing.assert_array_equal(Simple[2], [1])
        pos, i , vel= episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
        np.testing.assert_array_equal(pos, [1, 1, 1])
        #self.assertEqual(i, -1)

    # Test initialization when any block is sidewalk.
    def test_initialization_on_pavement_when_all_voxels_are_pavement(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        for i in SIDEWALK_LABELS:
            tensor[:,:,:,CHANNELS.semantic] = i / NUM_SEM_CLASSES*np.ones((3,3,3))
            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
            pos, i, vel = episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
            np.testing.assert_array_less(np.sort(pos), 3)
            #np.testing.assert_array_equal(i, -1)

    # Test initialization when a two dimensional block is sidewalk.
    def test_initialization_on_pavement_when_all_ground_voxels_are_pavement(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        for i in SIDEWALK_LABELS:
            tensor[:, :, 0, CHANNELS.semantic] = i / NUM_SEM_CLASSES * np.ones((3, 3))
            agent, episode, environmentInteraction= self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
            pos, i, vel = episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
            self.assertLess(pos[0], 3)
            self.assertLess(pos[1], 3)
            self.assertEqual(pos[2], 0)
            #self.assertEqual(i, -1)

    # Test initialization when there is two voxels of sidewalk.
    def test_initialize_on_pavement_in_tensor_with_two_pavement_voxel(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        for i in SIDEWALK_LABELS:
            tensor[0, 2, 0, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            tensor[0, 0, 0, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            agent, episode, environmentInteraction= self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
            pos, i, vel = episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
            self.assertEqual(pos[0], 0)
            self.assertTrue(pos[1]== 2 or pos[1]==0)
            self.assertEqual(pos[2], 0)
            #self.assertEqual(i, -1)

    # Test initialization when there is a pedestrian.
    def test_walk_into_person(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        pedestrian_pos=np.random.randint(2,size=(3))
        random_ped=np.array([[pedestrian_pos[0], pedestrian_pos[0]],[pedestrian_pos[1], pedestrian_pos[1]],[pedestrian_pos[2], pedestrian_pos[2]]])
        trajectory=[]
        for frame in range(0,30):
            people[frame].append(random_ped)
            trajectory.append(random_ped)
        people_dict={0:trajectory}
        init_frames={0:0}

        mean=np.mean(random_ped, axis=1)


        # pos_x = 0
        # pos_y = 0
        # #reward=(-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0)
        # rewards = self.get_reward()
        # episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,rewards, rewards, agent_size=(0, 0, 0), defaultSettings=run_settings())
        agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor , people_dict=people_dict, init_frames=init_frames)

        pos, i , vel= episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory)
        np.testing.assert_array_equal(pos, [])
        pos, i, vel = episode.initial_position(0, None,initialization=PEDESTRIAN_INITIALIZATION_CODE.randomly)
        # np.testing.assert_array_equal(pos, [])
        agent.initial_position(pos, episode.pedestrian_data[0].goal[0,:])

        # self.assertEqual(i, 29)
        self.assertEqual(len(episode.collide_with_pedestrians(0,episode.pedestrian_data[0].agent[0], 0)),0)
        episode.pedestrian_data[0].agent[1]=mean
        action=mean-pos
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 1)
        self.assertEqual(len(episode.collide_with_pedestrians(0,episode.pedestrian_data[0].agent[1], 1)), 1)
        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[1], mean)

        action = pos- mean
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 2)
        self.assertEqual(len(episode.collide_with_pedestrians(0, episode.pedestrian_data[0].agent[2], 2)), 0)

        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 3)
        np.testing.assert_array_equal(episode.calculate_reward(0, episode_done=True)[0], -1)
        self.evaluate_measures(0, episode, {PEDESTRIAN_MEASURES_INDX.hit_pedestrians: 1,PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory: 1,PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:1})

        np.testing.assert_array_equal(episode.pedestrian_data[0].agent[2], pos)
        np.testing.assert_array_equal(episode.calculate_reward(1,episode_done=True)[0], 0)
        self.evaluate_measures(1, episode, {PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap:0})

    # Test initialization when there are two pedestrians.
    def test_walking_into_two_person(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        ped1 = np.array([0,0,0,0,0,0]).reshape((3,2))
        ped2 = np.array([0, 1, 1,0,1,1]).reshape((3,2))
        trajectory1 = []
        trajectory2 = []
        for frame in range(0, 30):
            people[frame].append(ped1)
            trajectory1.append(ped1)
            people[frame].append(ped2)
            trajectory2.append(ped2)
        people_dict = {0: trajectory1, 1:trajectory2}
        init_frames = {0: 0, 1:0}

        #mean = np.mean(random_ped, axis=1)
        # print(mean)
        # people[29].append(np.array([0,0,0,0,0,0]).reshape((3,2)))
        # people[10].append(np.array([0, 1, 1,0,1,1]).reshape((3,2)))

        # pos_x =0
        # pos_y = 0
        # #reward=(-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0)
        # rewards = self.get_reward()
        # episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,rewards,rewards, agent_size=(0, 0, 0), defaultSettings=run_settings())
        agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor ,people_dict=people_dict, init_frames=init_frames)
        episode.agent_size = [0,0,0]
        pos, i , vel= episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory)
        episode.pedestrian_data[0].agent[0]=[0,0,0]
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        self.assertEqual(len(episode.collide_with_pedestrians(0,episode.pedestrian_data[0].agent[0],0)), 1)
        episode.pedestrian_data[0].agent[1] = [2, 2, 2]
        self.update_agent_and_episode([2, 2, 2], agent, environmentInteraction, episode, 1)
        self.assertEqual(len(episode.collide_with_pedestrians(0,episode.pedestrian_data[0].agent[1],1)), 0)
        episode.pedestrian_data[0].agent[2] = [0, 1, 1]
        self.update_agent_and_episode([-2, -1, -1], agent, environmentInteraction, episode, 2)
        self.assertEqual(len(episode.collide_with_pedestrians(0,episode.pedestrian_data[0].agent[2],2)), 1)

    

    # Test method intercept_car with one car.
    def test_intercept_car(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        car_list=[]
        for frame in range(1,30):
            cars[frame].append([1,1+ 1,1,1+ 1,1,1+ 1])
            car_list.append([1,1+ 1,1,1+ 1,1,1+ 1])
        cars_dict={0:car_list}
        car_init={0:0}
        agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor, car_dict=cars_dict, init_frames_cars=car_init )
        episode.agent_size=[0,0,0]
        episode.pedestrian_data[0].agent[0]=np.array([0,0,0])
        #episode.pedestrian_data[0].agent[0] = np.array([1,0.8,1.1])
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[0], 0), 0)
        episode.pedestrian_data[0].agent[1] = np.array([1, 0, 0])
        action=episode.pedestrian_data[0].agent[1]-episode.pedestrian_data[0].agent[0]
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 1)
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[1], 1), 0)
        episode.pedestrian_data[0].agent[2] = np.array([1, .9, 0])
        action = episode.pedestrian_data[0].agent[2] - episode.pedestrian_data[0].agent[1]
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 2)
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[2],2), 0)
        episode.pedestrian_data[0].agent[3] = np.array([1, 0, .7])
        action = episode.pedestrian_data[0].agent[3] - episode.pedestrian_data[0].agent[2]
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 3)
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[3], 3), 0)
        episode.pedestrian_data[0].agent[4] = np.array([0, 0, 1.2])
        action = episode.pedestrian_data[0].agent[4] - episode.pedestrian_data[0].agent[3]
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 4)
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[4], 4), 0)
        episode.pedestrian_data[0].agent[5] = np.array([0, 1.3, 0])
        action = episode.pedestrian_data[0].agent[5] - episode.pedestrian_data[0].agent[4]
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 5)
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[5], 5), 0)
        episode.pedestrian_data[0].agent[6] = np.array([1, 1.1, .9])  # True
        action = episode.pedestrian_data[0].agent[6] - episode.pedestrian_data[0].agent[5]
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 6)
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[6], 6), 1)
        episode.pedestrian_data[0].agent[7] = np.array([1, 1.2, 2.1])
        action = episode.pedestrian_data[0].agent[7] - episode.pedestrian_data[0].agent[6]
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 7)
        episode.pedestrian_data[0].agent[7] = np.array([1, 1.2, 2.1])
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[7], 7), 0)
        episode.pedestrian_data[0].agent[8] = np.array([0, .9, 1.8])
        action = episode.pedestrian_data[0].agent[8] - episode.pedestrian_data[0].agent[7]
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 8)
        episode.pedestrian_data[0].agent[8] = np.array([0, .9, 1.8])
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[8], 8), 0)
        episode.pedestrian_data[0].agent[9] = np.array([2, 1.1, 2.1])
        action = episode.pedestrian_data[0].agent[9] - episode.pedestrian_data[0].agent[8]
        self.update_agent_and_episode(action, agent, environmentInteraction, episode, 9)
        episode.pedestrian_data[0].agent[9] = np.array([2, 1.1, 2.1])
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[9], 9), 0)

    # Test method intercept_car with two cars.
    def test_intercept_two_car(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        car_list1=[]
        car_list2=[]
        for frame in range(30):
            cars[frame].append([0,0+ 1,0,1+ 1,0,0+ 1])
            car_list1.append([0,0+ 1,0,1+ 1,0,0+ 1])
            cars[frame].append([2, 2+ 1, 0, 0+ 1, 0, 0+ 1])
            car_list2.append([2, 2+ 1, 0, 0+ 1, 0, 0+ 1])
        cars_dict = {0: car_list1, 1:car_list2}
        car_init = {0: 0, 1:0}
        agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor ,car_dict=cars_dict, init_frames_cars=car_init )
        episode.agent_size=[0,0,0]
        episode.pedestrian_data[0].agent[0]=[0,0,0]
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        #self.update_agent_and_episode([0,0,0], agent, environmentInteraction, episode, 1)
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[0], 0), 2)
        episode.pedestrian_data[0].agent[0] = [0, .9, 0]

        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[0], 0),1)
        episode.pedestrian_data[0].agent[0] = [0, 0, .9]
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[0], 0), 0)
        episode.pedestrian_data[0].agent[0] = [1, 0, 0.1]
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[0], 0), 2)
        episode.pedestrian_data[0].agent[0] = [2, 0.2, 0.1]
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[0], 0), 2)
        episode.pedestrian_data[0].agent[0] = [1, .9, 1.1]
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[0], 0), 0)

        for frame in range(30):
            cars[frame]=[]
        cars[1] = []
        cars[1].append([0, 1+ 1, 0, 1+ 1, 0, 1+ 1])
        cars_dict = {0: [[0, 1+ 1, 0, 1+ 1, 0, 1+ 1]]}
        car_init = {0: 1}
        agent, episode, environmentInteraction= self.get_episode( cars, gamma, people, pos_x, pos_y, tensor ,car_dict=cars_dict, init_frames_cars=car_init )
        episode.agent_size = [1,1,1]
        episode.pedestrian_data[0].agent[1] = [0, 0, 0]
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        self.update_agent_and_episode([0, 1.1, 0], agent, environmentInteraction, episode, 1)
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[1], 1), 1)
        episode.pedestrian_data[0].agent[1] = [0, 1.1, 0]
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[1], 1), 1)
        episode.pedestrian_data[0].agent[1] = [2, 2.1, 2.1]
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[1], 1), 1)
        episode.pedestrian_data[0].agent[0] = [2, 1.9, 1.8]
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[0], 0), 0)

        episode.pedestrian_data[0].agent[1] = [3, 2.9, 3.1]
        self.assertEqual(episode.intercept_car(episode.pedestrian_data[0].agent[0], 0), 0)

    # Test method intercep_obj with one object.
    def test_intercept_obj(self):
        #objs=range(11,21)#[1,4,5,11,12,13,14,15,16,17,18,19,20,21,24,25,26,27,28,29,30,31,32,33,34,35]
        for new_carla in [False, True]:
            cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
            if new_carla:
                obstacles = OBSTACLE_LABELS_NEW
            else:
                obstacles = OBSTACLE_LABELS
            for val in obstacles:
                tensor[0,0,0,CHANNELS.semantic]=val/NUM_SEM_CLASSES
                agent, episode, environmentInteraction= self.get_episode( cars, gamma, people, pos_x, pos_y, tensor ,agent_size=[1,1,1],new_carla=new_carla)
                episode.agent_size = [1,1,1]
                episode.pedestrian_data[0].agent[0] = [0, 0.1, 0.2]
                self.assertEqual(episode.intercept_objects(episode.pedestrian_data[0].agent[0]), 1/9.0)

    # Test method intercep_obj with two objects.
    def test_intercept_2_obj(self):
        #objs = range(11,21)#[1, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,34,35]
        for new_carla in [False, True]:
            cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
            if new_carla:
                obstacles = OBSTACLE_LABELS_NEW
            else:
                obstacles = OBSTACLE_LABELS
            prev_val=obstacles[-1]
            for val in obstacles:
                tensor[0, 0, 0, CHANNELS.semantic] = val / NUM_SEM_CLASSES
                tensor[0,1,0,CHANNELS.semantic]=prev_val/NUM_SEM_CLASSES
                prev_val=val
                agent, episode, environmentInteraction= self.get_episode( cars, gamma, people, pos_x, pos_y, tensor,agent_size=[1,1,1] , new_carla=new_carla)
                episode.agent_size = [1,1,1]
                episode.pedestrian_data[0].agent[0] = [0, 0.2, 0.1]
                self.assertEqual(episode.intercept_objects(episode.pedestrian_data[0].agent[0]), 2 / 9.0)

    # Test method intercep_obj with no object.
    def test_intercept_no_obj(self):

        #not_objs = [2,3,6,7,8,9,10,22,23]
        labels=['rectification border','out of roi','ground','road','sidewalk','parking','rail track','terrain', 'sky']
        not_objs = []
        for label in labels:
            not_objs.append(cityscapes_labels_dict[label])
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        prev_val=not_objs[-1]
        for val in not_objs:
            tensor[0, 0, 0, CHANNELS.semantic] = val / NUM_SEM_CLASSES
            tensor[0,1,0,CHANNELS.semantic]=prev_val/NUM_SEM_CLASSES
            prev_val=val
            agent, episode, environmentInteraction= self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
            episode.agent_size = [1,1,1]
            episode.pedestrian_data[0].agent[0] = [0, 0.2, 0.3]
            self.assertEqual(episode.intercept_objects(episode.pedestrian_data[0].agent[0]), 0 / 9.0)

    # Test method intercep_obj when not intercepring object.
    def test_intercept_no_inter(self):
        objs = [1, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        labels = ['ego vehicle','static','dynamic','building', 'wall', 'fence','guard rail',  'bridge', 'tunnel','pole',
                  'polegroup','traffic light', 'traffic sign', 'vegetation', 'person','rider','car','truck', 'bus',
                  'caravan','trailer','train',  'motorcycle','bicycle']
        objs = []
        for label in labels:
            objs.append(cityscapes_labels_dict[label])
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        prev_val = objs[-1]
        for val in objs:
            tensor[2, 2,0, CHANNELS.semantic] = val / NUM_SEM_CLASSES
            tensor[2,2, 0, CHANNELS.semantic] = prev_val / NUM_SEM_CLASSES
            prev_val = val
            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
            episode.agent_size = [1,1,1]
            episode.pedestrian_data[0].agent[0] = [0, -0.1, -0.2]
            self.assertEqual(episode.intercept_objects(episode.pedestrian_data[0].agent[0]), 0 / 9.0)

    # Test method on Simple
    def test_on_pavement(self):
        Simple=SIDEWALK_LABELS#[6,7,8,9,10]
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        for val in Simple:
            tensor[0,1,1,CHANNELS.semantic]=val/NUM_SEM_CLASSES
            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
            episode.agent_size = [0,0,0]
            episode.pedestrian_data[0].agent[0] = [0, 0.1, -0.1]
            self.assertFalse(episode.on_pavement(episode.pedestrian_data[0].agent[0]))
            episode.pedestrian_data[0].agent[1] = [0, 1.1, 0.1]
            self.assertFalse(episode.on_pavement(episode.pedestrian_data[0].agent[1]))
            episode.pedestrian_data[0].agent[2] = [0, .9, .8]
            self.assertTrue(episode.on_pavement(episode.pedestrian_data[0].agent[2]))
            episode.pedestrian_data[0].agent[3] = [1, 1.1, .9]
            self.assertTrue(episode.on_pavement(episode.pedestrian_data[0].agent[3]))

    def test_iou_pavement(self):
        Simple = SIDEWALK_LABELS#[6, 7, 8, 9, 10]
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        for val in Simple:
            tensor[0, 1, 1,CHANNELS.semantic] = val / NUM_SEM_CLASSES
            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
            episode.agent_size = [0, 0, 0]
            episode.pedestrian_data[0].agent[0] = [0, 0, 0.1]
            self.assertEqual(episode.iou_sidewalk(episode.pedestrian_data[0].agent[0]), 0)
            episode.pedestrian_data[0].agent[1] = [0, 1.2, 0.1]
            self.assertEqual(episode.iou_sidewalk(episode.pedestrian_data[0].agent[1]), 0)
            episode.pedestrian_data[0].agent[2] = [0, .9, 1.2]
            self.assertEqual(episode.iou_sidewalk(episode.pedestrian_data[0].agent[2]), 1)
            episode.pedestrian_data[0].agent[3] = [1, .9, 1.1]
            self.assertEqual(episode.iou_sidewalk(episode.pedestrian_data[0].agent[3]), 1)

    def test_reward_pavement_and_obstacle(self):
        seq_len=6
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        for i in SIDEWALK_LABELS:
            tensor[1, 1, 1, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            tensor[2, 2, 2, CHANNELS.semantic]=cityscapes_labels_dict['building']/NUM_SEM_CLASSES
            pos_x = 0
            pos_y = 0
            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor ,seq_len=seq_len)
            sidewalk = episode.find_sidewalk(False)
            episode.agent_size = [0, 0, 0]
            np.testing.assert_array_equal(sidewalk[0], [1])
            np.testing.assert_array_equal(sidewalk[1], [1])
            np.testing.assert_array_equal(sidewalk[2], [1])
            pos, i , vel= episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)

            np.testing.assert_array_equal(pos, [1, 1, 1])
            agent.initial_position(pos, episode.pedestrian_data[0].goal[0,:])
            self.assertFalse(episode.out_of_axis(episode.pedestrian_data[0].agent[0]))

            episode.pedestrian_data[0].agent[1] = np.array([1,1.2,1.3])
            episode.pedestrian_data[0].action[0]=4
            action= episode.pedestrian_data[0].agent[1] - episode.pedestrian_data[0].agent[0]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 1)
            episode.pedestrian_data[0].measures[0,:]=np.zeros_like(episode.pedestrian_data[0].measures[0,:])
            episode.pedestrian_data[0].agent[1] = np.array([1, 1.2, 1.3])
            np.testing.assert_array_equal(episode.calculate_reward(0)[0], 1)

            episode.pedestrian_data[0].action[1] = 4
            episode.pedestrian_data[0].agent[2] = np.array([0, 0.1, -0.3])
            action = episode.pedestrian_data[0].agent[2] - episode.pedestrian_data[0].agent[1]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 2)
            episode.pedestrian_data[0].agent[2] = np.array([0, 0.1, -0.3])
            episode.pedestrian_data[0].measures[1, :] = np.zeros_like(episode.pedestrian_data[0].measures[1, :])
            np.testing.assert_array_equal(episode.calculate_reward(1)[0], 0)

            episode.pedestrian_data[0].agent[3] = np.array([2, 1.8, 2.2])
            action = episode.pedestrian_data[0].agent[3] - episode.pedestrian_data[0].agent[2]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 3)
            episode.pedestrian_data[0].agent[3] = np.array([2, 1.8, 2.2])
            episode.pedestrian_data[0].measures[2, :] = np.zeros_like(episode.pedestrian_data[0].measures[2, :])
            np.testing.assert_array_equal(episode.calculate_reward(2)[0], -1/4.0)

            episode.pedestrian_data[0].action[4] = 4
            episode.pedestrian_data[0].agent[4] = np.array([0, 0.1, -0.1])
            action = episode.pedestrian_data[0].agent[4] - episode.pedestrian_data[0].agent[3]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 4)
            episode.pedestrian_data[0].agent[4] = np.array([0, 0.1, -0.1])
            episode.pedestrian_data[0].measures[3, :] = np.zeros_like(episode.pedestrian_data[0].measures[3, :])
            np.testing.assert_array_equal(episode.calculate_reward(3)[0], 0)

            episode.pedestrian_data[0].agent[5] = np.array([1, .9, 1.2])
            action = episode.pedestrian_data[0].agent[5] - episode.pedestrian_data[0].agent[4]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 5)
            episode.pedestrian_data[0].agent[5] = np.array([1, .9, 1.2])
            episode.pedestrian_data[0].measures[4, :] = np.zeros_like(episode.pedestrian_data[0].measures[4, :])
            np.testing.assert_almost_equal(episode.calculate_reward(4)[0], 1+(0.1*np.sqrt(5)))

    def test_two_pavement_and_obstacle(self):
        seq_len=6
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        for i in SIDEWALK_LABELS:
            tensor[1, 1, 1, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            tensor[1, 0, 1, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            tensor[2, 2, 2, CHANNELS.semantic] = cityscapes_labels_dict['building'] / NUM_SEM_CLASSES
            pos_x = 0
            pos_y = 0
            # rewards=[-1, 1, -1, 1, 0, 0,0,0,0,0,0,0,0,0,0]
            # rewards[2]=1
            # rewards[3] =-1
            # rewards[4] = 1

            agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                      seq_len=seq_len)


            pos, i , vel= episode.initial_position(0,None)
            episode.pedestrian_data[0].agent[0] =np.array( [1, 1.1, 1])
            agent.initial_position([1, 1.1, 1], episode.pedestrian_data[0].goal[0,:])

            episode.pedestrian_data[0].agent[1] = np.array([1, 1, 1.2])
            action = episode.pedestrian_data[0].agent[1] - episode.pedestrian_data[0].agent[0]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 1)
            episode.pedestrian_data[0].agent[1] = np.array([1, 1, 1.2])
            episode.pedestrian_data[0].measures[0, :] = np.zeros_like(episode.pedestrian_data[0].measures[0, :])
            np.testing.assert_array_equal(episode.calculate_reward(0)[0], 1)

            episode.pedestrian_data[0].agent[2] = np.array( [0, 0, 0.1])
            action = episode.pedestrian_data[0].agent[2] - episode.pedestrian_data[0].agent[1]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 2)
            episode.pedestrian_data[0].agent[2] = np.array([0, 0, 0.1])
            episode.pedestrian_data[0].measures[1, :] = np.zeros_like(episode.pedestrian_data[0].measures[1, :])
            np.testing.assert_array_equal(episode.calculate_reward(1)[0], 0)

            episode.pedestrian_data[0].agent[3] =np.array( [2, 2.2, 2])
            action = episode.pedestrian_data[0].agent[3] - episode.pedestrian_data[0].agent[2]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 3)
            episode.pedestrian_data[0].agent[3] = np.array([2, 2.2, 2])
            episode.pedestrian_data[0].measures[2, :] = np.zeros_like(episode.pedestrian_data[0].measures[2, :])
            np.testing.assert_array_equal(episode.calculate_reward(2)[0], -1/4.0)

            episode.pedestrian_data[0].agent[4] =np.array(  [0, 0.1, -.4])
            action = episode.pedestrian_data[0].agent[4] - episode.pedestrian_data[0].agent[3]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 4)
            episode.pedestrian_data[0].agent[4] = np.array([0, 0.1, -.4])
            episode.pedestrian_data[0].measures[3, :] = np.zeros_like(episode.pedestrian_data[0].measures[3, :])
            np.testing.assert_array_equal(episode.calculate_reward(3)[0], 0)

            episode.pedestrian_data[0].agent[5] = np.array([1, 0, 1.2])
            action = episode.pedestrian_data[0].agent[5] - episode.pedestrian_data[0].agent[4]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 5)
            episode.pedestrian_data[0].agent[5] = np.array([1, 0, 1.2])
            episode.pedestrian_data[0].measures[4, :] = np.zeros_like(episode.pedestrian_data[0].measures[4, :])
            np.testing.assert_array_equal(episode.calculate_reward(4)[0], 1+np.sqrt(1.25))


    def test_reward(self):
        seq_len = 30
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        for i in SIDEWALK_LABELS:
            tensor[1, 1, 1, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            pos_x = 0
            pos_y = 0
            rewards = np.zeros(15)
            # rewards[2] = 1
            # rewards[3] = -1
            # rewards[4] = 1
            # rewards[5] = -1

            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor ,seq_len=seq_len)
            Simple = episode.find_sidewalk(False)
            episode.agent_size=[0,0,0]
            np.testing.assert_array_equal(Simple[0], [1])
            np.testing.assert_array_equal(Simple[1], [1])
            np.testing.assert_array_equal(Simple[2], [1])
            pos, i , vel= episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
            np.testing.assert_array_equal(pos, [1, 1, 1])
            self.assertFalse(episode.out_of_axis(episode.pedestrian_data[0].agent[0]))

            episode.pedestrian_data[0].agent[1] = np.array([0, 0, -.9])
            episode.pedestrian_data[0].action[0] = 4
            action = episode.pedestrian_data[0].agent[1] - episode.pedestrian_data[0].agent[0]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 1)
            np.testing.assert_array_equal( episode.calculate_reward(0)[0], -1)


            episode.pedestrian_data[0].agent[2] = np.array([0, -.9, 0])
            episode.pedestrian_data[0].action[1] = 4
            action = episode.pedestrian_data[0].agent[2] - episode.pedestrian_data[0].agent[1]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 2)
            self.assertTrue(episode.out_of_axis(episode.pedestrian_data[0].agent[1]))
            np.testing.assert_array_equal(episode.calculate_reward( 1)[0], -1)

            episode.pedestrian_data[0].agent[3] = np.array([0, -1.1, 0])
            episode.pedestrian_data[0].action[2] = 4
            action = episode.pedestrian_data[0].agent[3] - episode.pedestrian_data[0].agent[2]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 3)
            self.assertTrue(episode.out_of_axis(episode.pedestrian_data[0].agent[2]))
            np.testing.assert_array_equal(episode.calculate_reward(2)[0],-1)


            for j in range(4,29):
                episode.pedestrian_data[0].agent[j]=np.array([0,0.1,0])
                episode.pedestrian_data[0].action[j-1] = 4
                action = episode.pedestrian_data[0].agent[j] - episode.pedestrian_data[0].agent[j-1]
                self.update_agent_and_episode(action, agent, environmentInteraction, episode, j)
                self.assertFalse(episode.out_of_axis(episode.pedestrian_data[0].agent[j]))
                np.testing.assert_array_equal(episode.calculate_reward(j-1)[0], 0)
            import math
            episode.pedestrian_data[0].agent[29] = np.array([0, 0, 0.2])
            episode.pedestrian_data[0].action[28] = 4
            action = episode.pedestrian_data[0].agent[29] - episode.pedestrian_data[0].agent[28]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, 29)
            self.assertFalse(episode.out_of_axis(episode.pedestrian_data[0].agent[29]))
            np.testing.assert_array_equal(episode.calculate_reward(28)[0],0)
            episode.discounted_reward(28)
            for i in range(26):
                np.testing.assert_array_almost_equal_nulp(episode.pedestrian_data[0].reward[28-i], 0, nulp=2 )
                np.testing.assert_array_almost_equal_nulp(episode.pedestrian_data[0].reward_d[28-i], 0, nulp=2)

            np.testing.assert_array_almost_equal_nulp(episode.pedestrian_data[0].reward_d[2], -1, nulp=2)
            np.testing.assert_array_equal(episode.pedestrian_data[0].reward[2], -1)
            np.testing.assert_array_almost_equal_nulp(episode.pedestrian_data[0].reward_d[1], -1.99, nulp=2)
            np.testing.assert_array_equal(episode.pedestrian_data[0].reward[1], -1)
            np.testing.assert_array_almost_equal_nulp(episode.pedestrian_data[0].reward_d[0], -1.99*.99-1, nulp=2)
            np.testing.assert_array_equal(episode.pedestrian_data[0].reward[0], -1)
            # np.testing.assert_array_almost_equal_nulp(episode.pedestrian_data[0].reward_d[0], (-1.99 * .99 - 1)*.99+1, nulp=2)
            # np.testing.assert_array_equal(episode.pedestrian_data[0].reward[0], 1)

    # Test the neigbourhood that the agent sees when inside allowed axis.
    def test_neigbourhood_in_axis_car(self):
        seq_len = 18
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        tensor = np.zeros((8, 4, 4, 6))
        car_list = []
        cars = []
        cars.append([])
        cars[0].append(np.array([0, 8, 0, 1, 0, 1]))
        car_list.append(np.array([0, 8, 0, 1, 0, 1]))
        # people.append([])
        # people[1].append(np.array([[0, 8], [0, 0], [0, 0]]))
        # people_list.append(np.array([[0, 8], [0, 0], [0, 0]]))
        for j in range(4):
            for i in range(4):
                car_list.append(np.array([0, 8, j, j + 1, i, i + 1]))
                cars.append([])
                cars[-1].append(np.array([0, 8, j, j + 1, i, i + 1]))
        car_list.append(np.array([0, 8, 3, 3 + 1, 3, 3 + 1]))
        cars.append([])
        cars[-1].append(np.array([0, 8, 3, 3 + 1, 3, 3 + 1]))
        init = {0: 0}
        car_dict = {0: car_list}

        tensor[0, :, :, CHANNELS.cars_trajectory] = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        pos_x = 0
        pos_y = 0
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  car_dict=car_dict, init_frames_cars=init,
                                                                  seq_len=seq_len)
        episode.pedestrian_data[0].agent[0] = np.array([0, 2.1, 2.1])
        episode.pedestrian_data[0].goal[0,:] = np.zeros(3)
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        for j in range(1, 17):
            episode.pedestrian_data[0].agent[j] = np.array([0, 2.1, 2.1])
            episode.pedestrian_data[0].action[j - 1] = 4
            action = episode.pedestrian_data[0].agent[j] - episode.pedestrian_data[0].agent[j - 1]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, j, breadth=[0, 1, 1])
            episode.pedestrian_data[0].measures[j, :] = np.zeros_like(episode.pedestrian_data[0].measures[j, :])
            self.assertFalse(episode.out_of_axis(episode.pedestrian_data[0].agent[j]))
            if j==11:
                episode.pedestrian_data[0].agent[11] = [0, 2.1, 2.1]
                neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, np.array(
                    [0, 2.1, 2.1]), [0, 1, 1], 11, temporal_scaling=1)
                expected = np.array([[6, 7, 8], [10, 11+1, 12+1], [0, 0, 0]])
                mask = expected > 0
                expected[mask] = expected[mask] - 11
                np.testing.assert_array_equal(tmp_cars_cv[:, :,0,0], expected)

        episode.pedestrian_data[0].agent[0] = [0, 2.1, 2.1]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, np.array([0, 2.1, 2.1]), [0, 1, 1], 16, temporal_scaling=1)
        expected=np.array([[6,7,8],[10,11,12],[14,15,16]])
        np.testing.assert_array_equal(neigh[:,:,CHANNELS.cars_trajectory],expected-16 )
        expected = np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16+1]])
        np.testing.assert_array_equal(tmp_cars_cv[:, :,0,0], expected - 16)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(tmp_cars[ :, :,0], expected )

    # Test the neigbourhood that the agent sees when inside allowed axis.
    def test_neigbourhood_in_axis(self):
        seq_len = 18
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        tensor = np.zeros((8, 4, 4, 6))
        people_list = []
        people = []
        people.append([])
        people[0].append(np.array([[0, 8], [0, 0], [0, 0]]))
        people_list.append(np.array([[0, 8], [0, 0], [0, 0]]))
        # people.append([])
        # people[1].append(np.array([[0, 8], [0, 0], [0, 0]]))
        # people_list.append(np.array([[0, 8], [0, 0], [0, 0]]))
        for j in range(4):
            for i in range(4):
                people_list.append(np.array([[0, 8], [j, j], [i, i]]))
                people.append([])
                people[-1].append(np.array([[0, 8], [j, j], [i, i]]))
        people_list.append(np.array([[0, 8], [3, 3], [3, 3]]))
        people.append([])
        people[-1].append(np.array([[0, 8], [3, 3], [3, 3]]))
        init = {0: 0}
        people_dict = {0: people_list}

        tensor[0, :, :, CHANNELS.pedestrian_trajectory] = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        pos_x = 0
        pos_y = 0
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  people_dict=people_dict, init_frames=init,
                                                                  seq_len=seq_len)
        episode.pedestrian_data[0].agent[0] = np.array([0, 2.1, 2.1])
        episode.pedestrian_data[0].goal[0,:] = np.zeros(3)
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        for j in range(1, 17):

            episode.pedestrian_data[0].agent[j] = np.array([0, 2.1, 2.1])
            episode.pedestrian_data[0].action[j - 1] = 4
            action = episode.pedestrian_data[0].agent[j] - episode.pedestrian_data[0].agent[j - 1]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, j, breadth=[0, 1, 1])
            self.assertFalse(episode.out_of_axis(episode.pedestrian_data[0].agent[j]))
            if j == 11:
                episode.pedestrian_data[0].agent[11] = [0, 2.1, 2.1]
                neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0,
                                                                                                          np.array(
                                                                                                              [0,
                                                                                                               2.1,
                                                                                                               2.1]),
                                                                                                          [0, 1, 1],
                                                                                                          11,
                                                                                                          temporal_scaling=1)
                expected = np.array([[6, 7, 8], [10, 11 + 1, 12 + 1], [0, 0, 0]])
                mask = expected > 0
                expected[mask] = expected[mask] - 11
                np.testing.assert_array_equal(tmp_people_cv[:, :, 0, 0], expected)

        episode.pedestrian_data[0].agent[0] = [0, 2.1, 2.1]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, np.array(
            [0, 2.1, 2.1]), [0, 1, 1], 16, temporal_scaling=1)
        expected = np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16]])
        np.testing.assert_array_equal(neigh[:, :, CHANNELS.pedestrian_trajectory], expected - 16)
        expected = np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16 + 1]])
        np.testing.assert_array_equal(tmp_people_cv[:, :, 0, 0], expected - 16)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(tmp_people[:, :, 0], expected)

    # Test the neigbourhood that the agent sees when inside allowed axis.
    def test_neigbourhood_out_of_axis(self):
        seq_len = 18
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        tensor = np.zeros((8, 4, 4, 6))
        people_list = []
        people = []
        people.append([])
        people[0].append(np.array([[0, 8], [0, 0], [0, 0]]))
        people_list.append(np.array([[0, 8], [0, 0], [0, 0]]))
        # people.append([])
        # people[1].append(np.array([[0, 8], [0, 0], [0, 0]]))
        # people_list.append(np.array([[0, 8], [0, 0], [0, 0]]))
        for j in range(4):
            for i in range(4):
                people_list.append(np.array([[0, 8], [j, j], [i, i]]))
                people.append([])
                people[-1].append(np.array([[0, 8], [j, j], [i, i]]))
        people_list.append(np.array([[0, 8], [3, 3], [3, 3]]))
        people.append([])
        people[-1].append(np.array([[0, 8], [3, 3], [3, 3]]))
        init = {0: 0}
        people_dict = {0: people_list}

        tensor[0, :, :, CHANNELS.pedestrian_trajectory] = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        pos_x = 0
        pos_y = 0
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  people_dict=people_dict, init_frames=init,
                                                                  seq_len=seq_len)
        episode.pedestrian_data[0].agent[0] = np.array([0, 2.1, 2.1])
        episode.pedestrian_data[0].goal[0,:] = np.zeros(3)
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        for j in range(1, 17):
            episode.pedestrian_data[0].agent[j] = np.array([0, 2.1, 2.1])
            episode.pedestrian_data[0].action[j - 1] = 4
            action = episode.pedestrian_data[0].agent[j] - episode.pedestrian_data[0].agent[j - 1]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, j, breadth=[0, 1, 1])
            self.assertFalse(episode.out_of_axis(episode.pedestrian_data[0].agent[j]))

        episode.pedestrian_data[0].agent[0] = [0, 2.1, 2.1]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, np.array(
            [0, 2.1, 2.1]), [0, 1, 1], 16, temporal_scaling=1)
        expected = np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16]])
        np.testing.assert_array_equal(neigh[:, :, CHANNELS.pedestrian_trajectory], expected - 16)
        expected = np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16+1]])
        np.testing.assert_array_equal(tmp_people_cv[:, :, 0, 0], expected - 16)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(tmp_people[:, :, 0], expected)
        # # Test the neigbourhood that the agent sees when outside allowed axis.
        # def test_neigbourhood_out_of_axis(self):
        #     cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        #     tensor = np.zeros((8, 4, 4, 6))
        #     tensor[0,:,:,CHANNELS.pedestrian_trajectory]=np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12],[13,14,15,16]])
        #     pos_x = 0
        #     pos_y = 0
        #     agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
        episode.pedestrian_data[0].agent[16]=[0,2.1,3.1]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv =episode.get_agent_neighbourhood(0,np.array([0,2.1,3.1]), [0, 1, 1],16, temporal_scaling=1)
        expected=np.array([[7,8,0],[11,12,0],[15,16,0]])
        mask=expected>0
        expected[mask]= expected[mask]-16
        np.testing.assert_array_equal(neigh[:,:,CHANNELS.pedestrian_trajectory],expected )
        expected = np.array([[7, 8, 0], [11, 12, 0], [15, 16+1, 0]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(tmp_people_cv[:,:,0,0], expected)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(tmp_people[:, :, 0], expected)

        episode.pedestrian_data[0].agent[16]= [0, 2.7, 3.1]
        expected = np.array([[11, 12, 0], [15, 16, 0], [0, 0, 0]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv= episode.get_agent_neighbourhood(0,[0, 2.7, 3.1],[ 0, 1, 1], 16, temporal_scaling=1)

        np.testing.assert_array_equal(neigh[ :, :, CHANNELS.pedestrian_trajectory], expected)
        expected = np.array([[11, 12, 0], [15, 16+1, 0], [0, 0, 0]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(tmp_people_cv[:,:,0,0], expected)
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_people[:, :, 0], expected)

        episode.pedestrian_data[0].agent[16]=[0, 4.1, 3.2]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0,[0, 4.1, 3.2], [0, 1, 1],16, temporal_scaling=1)
        expected = np.array([ [15-16, 16-16, 0], [0, 0, 0],[0,0,0]])
        np.testing.assert_array_equal(neigh[ :, :, CHANNELS.pedestrian_trajectory], expected)
        expected = np.array([[15 - 16, 16 - 16+1, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_people_cv[:,:,0,0], expected)
        expected = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_people[:, :, 0], expected)

        episode.pedestrian_data[0].agent[16] =[0,1.1,0]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv=episode.get_agent_neighbourhood(0,[0,1.1,0], [0, 1, 1],16, temporal_scaling=1)
        expected=np.array([[0,1,2],[0,5,6],[0,9,10]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(neigh[:,:,CHANNELS.pedestrian_trajectory],expected )

        np.testing.assert_array_equal(tmp_people_cv[:,:,0,0], expected)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_people[:, :, 0], expected)

        episode.pedestrian_data[0].agent[1] = [0, 0, 1.1]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0,[0, 0, 1.1], [0, 1, 1],16, temporal_scaling=1)
        expected = np.array([[0,0,0],[1, 2, 3], [5, 6, 7]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(neigh[ :, :, CHANNELS.pedestrian_trajectory], expected)
        np.testing.assert_array_equal(tmp_people_cv[:, :, 0, 0], expected)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_people[:, :, 0], expected)


        episode.pedestrian_data[0].agent[2] = [0, 0.1, 0]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv= episode.get_agent_neighbourhood(0,[0, 0.1, 0], [0, 1, 1],16, temporal_scaling=1)
        expected = np.array([[0, 0, 0], [0,1, 2], [0,5, 6]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(neigh[ :, :, CHANNELS.pedestrian_trajectory], expected)
        np.testing.assert_array_equal(tmp_people_cv[:, :, 0, 0], expected)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_people[:, :, 0], expected)

        # Test the neigbourhood that the agent sees when inside allowed axis.

    def test_neigbourhood_out_of_axis_car(self):
        seq_len = 18
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len)
        tensor = np.zeros((8, 4, 4, 6))
        car_list = []
        cars = []
        cars.append([])
        cars[0].append(np.array([0, 8,0, 1,0, 1]))
        car_list.append(np.array([0, 8,0, 1,0, 1]))
        # people.append([])
        # people[1].append(np.array([[0, 8], [0, 0], [0, 0]]))
        # people_list.append(np.array([[0, 8], [0, 0], [0, 0]]))
        for j in range(4):
            for i in range(4):
                car_list.append(np.array([0, 8,j, j+1,i, i+1]))
                cars.append([])
                cars[-1].append(np.array([0, 8,j, j+1,i, i+1]))
        car_list.append(np.array([0, 8,3, 3+1,3, 3+1]))
        cars.append([])
        cars[-1].append(np.array([0, 8,3, 3+1,3, 3+1]))
        init = {0: 0}
        car_dict = {0: car_list}

        tensor[0, :, :, CHANNELS.cars_trajectory] = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        pos_x = 0
        pos_y = 0
        agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                  car_dict=car_dict, init_frames_cars=init,
                                                                  seq_len=seq_len)
        episode.pedestrian_data[0].agent[0] = np.array([0, 2.1, 2.1])
        episode.pedestrian_data[0].goal[0,:] = np.zeros(3)
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        for j in range(1, 17):
            episode.pedestrian_data[0].agent[j] = np.array([0, 2.1, 2.1])
            episode.pedestrian_data[0].action[j - 1] = 4
            action = episode.pedestrian_data[0].agent[j] - episode.pedestrian_data[0].agent[j - 1]
            self.update_agent_and_episode(action, agent, environmentInteraction, episode, j, breadth=[0, 1, 1])
            episode.pedestrian_data[0].measures[j, :] = np.zeros_like(episode.pedestrian_data[0].measures[j, :])
            self.assertFalse(episode.out_of_axis(episode.pedestrian_data[0].agent[j]))

        episode.pedestrian_data[0].agent[0] = [0, 2.1, 2.1]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, np.array(
            [0, 2.1, 2.1]), [0, 1, 1], 16, temporal_scaling=1)
        expected = np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16]])
        np.testing.assert_array_equal(neigh[:, :, CHANNELS.cars_trajectory], expected - 16)
        expected = np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16 + 1]])
        np.testing.assert_array_equal(tmp_cars_cv[:, :, 0, 0], expected - 16)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(tmp_cars[:, :, 0], expected)
        # # Test the neigbourhood that the agent sees when outside allowed axis.
        # def test_neigbourhood_out_of_axis(self):
        #     cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        #     tensor = np.zeros((8, 4, 4, 6))
        #     tensor[0,:,:,CHANNELS.pedestrian_trajectory]=np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12],[13,14,15,16]])
        #     pos_x = 0
        #     pos_y = 0
        #     agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
        episode.pedestrian_data[0].agent[16] = [0, 2.1, 3.1]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, np.array(
            [0, 2.1, 3.1]), [0, 1, 1], 16, temporal_scaling=1)
        expected = np.array([[7, 8, 0], [11, 12, 0], [15, 16, 0]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(neigh[:, :, CHANNELS.cars_trajectory], expected)
        expected = np.array([[7, 8, 0], [11, 12, 0], [15, 16 + 1, 0]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(tmp_cars_cv[:, :, 0, 0], expected)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(tmp_cars[:, :, 0], expected)

        episode.pedestrian_data[0].agent[16] = [0, 2.7, 3.1]
        expected = np.array([[11, 12, 0], [15, 16, 0], [0, 0, 0]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, [0, 2.7, 3.1],
                                                                                                  [0, 1, 1], 16,
                                                                                                  temporal_scaling=1)

        np.testing.assert_array_equal(neigh[:, :, CHANNELS.cars_trajectory], expected)
        expected = np.array([[11, 12, 0], [15, 16 + 1, 0], [0, 0, 0]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(tmp_cars_cv[:, :, 0, 0], expected)
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_cars[:, :, 0], expected)

        episode.pedestrian_data[0].agent[16] = [0, 4.1, 3.2]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, [0, 4.1, 3.2],
                                                                                                  [0, 1, 1], 16,
                                                                                                  temporal_scaling=1)
        expected = np.array([[15 - 16, 16 - 16, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(neigh[:, :, CHANNELS.cars_trajectory], expected)
        expected = np.array([[15 - 16, 16 - 16 + 1, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_cars_cv[:, :, 0, 0], expected)
        expected = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_cars[:, :, 0], expected)

        episode.pedestrian_data[0].agent[16] = [0, 1.1, 0]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, [0, 1.1, 0],
                                                                                                  [0, 1, 1], 16,
                                                                                                  temporal_scaling=1)
        expected = np.array([[0, 1, 2], [0, 5, 6], [0, 9, 10]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(neigh[:, :, CHANNELS.cars_trajectory], expected)

        np.testing.assert_array_equal(tmp_cars_cv[:, :, 0, 0], expected)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_cars[:, :, 0], expected)

        episode.pedestrian_data[0].agent[1] = [0, 0, 1.1]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, [0, 0, 1.1],
                                                                                                  [0, 1, 1], 16,
                                                                                                  temporal_scaling=1)
        expected = np.array([[0, 0, 0], [1, 2, 3], [5, 6, 7]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(neigh[:, :, CHANNELS.cars_trajectory], expected)
        np.testing.assert_array_equal(tmp_cars_cv[:, :, 0, 0], expected)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_cars[:, :, 0], expected)

        episode.pedestrian_data[0].agent[2] = [0, 0.1, 0]
        neigh, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv = episode.get_agent_neighbourhood(0, [0, 0.1, 0],
                                                                                                  [0, 1, 1], 16,
                                                                                                  temporal_scaling=1)
        expected = np.array([[0, 0, 0], [0, 1, 2], [0, 5, 6]])
        mask = expected > 0
        expected[mask] = expected[mask] - 16
        np.testing.assert_array_equal(neigh[:, :, CHANNELS.cars_trajectory], expected)
        np.testing.assert_array_equal(tmp_cars_cv[:, :, 0, 0], expected)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(tmp_cars[:, :, 0], expected)

    # Test initialization when there is a pedestrian.
    def test_one_person_NN_overlap(self):
        seq_len=30
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        people[28].append(np.array([[2.1,2.1],[2.1,2.1],[2.1,2.1]]))
        mean = np.mean(people[28][0], axis=1)
        people_dict={0:[people[28][0], people[28][0], people[28][0]]}
        init_frames={0:28}

        for i in SIDEWALK_LABELS:
            tensor[0, 2, 0, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            tensor[0, 0, 0, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            # pos_x = 0
            # pos_y = 0
            # # rewards= (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0,0,0)
            # rewards = self.get_reward()
            # episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,rewards,rewards, agent_size=(0, 0, 0), defaultSettings=run_settings())
            agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                      people_dict=people_dict, init_frames=init_frames,
                                                                      seq_len=seq_len, agent_size=[0, 1, 1])
            episode.agent_size = [0, 0, 0]
            for nbr in range(29):
                episode.pedestrian_data[0].agent[nbr] = [0, 0, 0]
            #pos, i , vel= episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory)
            episode.pedestrian_data[0].agent[1] = [2.1, 2.1, 2.1]
            episode.pedestrian_data[0].agent[0] = np.array([0, 2.1, 2.1])
            episode.pedestrian_data[0].goal[0,:] = np.zeros(3)
            agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])



            for j in range(1, seq_len):
                episode.pedestrian_data[0].agent[j] = np.array([0, 2.1, 2.1])
                episode.pedestrian_data[0].action[j - 1] = 4
                action = episode.pedestrian_data[0].agent[j] - episode.pedestrian_data[0].agent[j - 1]
                self.update_agent_and_episode(action, agent, environmentInteraction, episode, j)


            self.assertEqual(episode.intercept_pedestrian_trajectory(0, episode.pedestrian_data[0].agent[0], 0 + 1, no_height=True), 0)
            self.assertEqual(len(episode.collide_with_pedestrians(0, episode.pedestrian_data[0].agent[0], 1)), 0)
            self.assertEqual(len(episode.collide_with_pedestrians(0,episode.pedestrian_data[0].agent[1] ,1)), 0)
            self.assertEqual(len(episode.collide_with_pedestrians(0, episode.pedestrian_data[0].agent[28], 28)), 1)




    def test_one_person_NN_overlap2(self):
        seq_len = 30
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        people[10].append(np.array([[2.1, 2.1],[2.1, 2.1],[2.1, 2.1]]))
        people[11].append(np.array([[2.1, 2.1],[2.1, 2.1],[2.1, 2.1]]))
        people[12].append(np.array([[2.1, 2.1],[2.1, 2.1],[2.1, 2.1]]))
        mean = np.mean(people[10][0], axis=1)
        people_dict = {0: [people[10][0], people[10][0], people[10][0]]}
        init_frames = {0: 10}

        for i in SIDEWALK_LABELS:
            tensor[0, 2, 0, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            tensor[0, 0, 0, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            # pos_x = 0
            # pos_y = 0
            # # rewards= (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0,0,0)
            # rewards = self.get_reward()
            # episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,rewards,rewards, agent_size=(0, 0, 0), defaultSettings=run_settings())
            agent, episode, environmentInteraction = self.get_episode(cars, gamma, people, pos_x, pos_y, tensor,
                                                                      people_dict=people_dict, init_frames=init_frames,
                                                                      seq_len=seq_len, agent_size=[0, 1, 1])
            episode.agent_size = [0, 0, 0]
            for nbr in range(29):
                episode.pedestrian_data[0].agent[nbr] = [0, 0, 0]
            # pos, i , vel= episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory)
            episode.pedestrian_data[0].agent[1] = [2.1, 2.1, 2.1]
            episode.pedestrian_data[0].agent[0] = np.array([0, 2.1, 2.1])
            episode.pedestrian_data[0].goal[0,:] = np.zeros(3)
            agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])

            for j in range(1, seq_len):
                episode.pedestrian_data[0].agent[j] = np.array([0, 2.1, 2.1])
                episode.pedestrian_data[0].action[j - 1] = 4
                action = episode.pedestrian_data[0].agent[j] - episode.pedestrian_data[0].agent[j - 1]
                self.update_agent_and_episode(action, agent, environmentInteraction, episode, j, breadth=[0, 1, 1])

            self.assertEqual(episode.intercept_pedestrian_trajectory(0,episode.pedestrian_data[0].agent[9] ,9), 1)
            self.assertEqual(episode.intercept_pedestrian_trajectory(0, episode.pedestrian_data[0].agent[9], 11), 1)
            episode.pedestrian_data[0].agent[1] = [1, .9, 1.2]

            self.assertEqual(episode.intercept_pedestrian_trajectory(0,episode.pedestrian_data[0].agent[28] ,27), 1)



    def test_one_car_input(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        cars_list=[]
        for i in range (3):
            for j in range(3):
                cars[i*3+j].append([0,8+ 1,2-i,2-i+ 1,j,j+ 1])

                cars_list.append([0,8+ 1,i,i+ 1,j,j+ 1])
        cars_dict={0:cars_list}
        cars_init = {0: 0}

        for i in SIDEWALK_LABELS:
            tensor[0, 1, 1, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            # tensor[0, 1, 1, 3] = i / NUM_SEM_CLASSES
            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor ,car_dict=cars_dict,init_frames_cars=cars_init )
            episode.agent_size = [0, 0, 0]

            pos, i, vel = episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
            agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
            for i in range(3):
                for j in range(3):
                    k=i*3+j
                    episode.pedestrian_data[0].agent[k]=np.array([0,i,j])
                    if k>0:
                        action = episode.pedestrian_data[0].agent[k] - episode.pedestrian_data[0].agent[k - 1]
                        self.update_agent_and_episode(action, agent, environmentInteraction, episode, k, breadth=[0, 1, 1])

            #Agent standing still!

            for l in range(1):
                k = 0
                for i in range(3):
                    for j in range(3):

                        pos = episode.pedestrian_data[0].agent[k]
                        #print ("Position "+str(pos)+" "+str(cars[k])+" "+str(k))
                        if i-pos[1]!=0:
                            self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0],1.0/(i-pos[1]) )
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 2 )
                        if j - pos[2] != 0:

                            self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 1.0/(j - pos[2]))
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 2 )
                        k=k+1

    def test_two_cars_input(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        cars_list = []
        cars_list_2 = []
        for i in range (3):
            for j in range(3):
                cars[i*3+j].append([0,8+ 1,i,i+ 1,j,j+ 1])
                cars_list.append([0,8+ 1,i,i+ 1,j,j+ 1])
                cars[i * 3 + j].append([0, 8+ 1, 2, 2+ 1, 2, 2+ 1])
                cars_list_2.append([0, 8+ 1, 2, 2+ 1, 2, 2+ 1])
        cars_dict = {0: cars_list, 1:cars_list_2}
        cars_init = {0: 0,1:0}
        #print("Cars "+str(cars))
        for i in SIDEWALK_LABELS:
            tensor[0, 1, 1, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            # tensor[0, 1, 1, 3] = i / NUM_SEM_CLASSES
            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor ,car_dict=cars_dict,init_frames_cars=cars_init )
            episode.agent_size = [0, 0, 0]

            pos, i, vel = episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
            agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
            for i in range(3):
                for j in range(3):
                    k = i * 3 + j
                    episode.pedestrian_data[0].agent[k] = np.array([0, i, j])
                    if k > 0:
                        action = episode.pedestrian_data[0].agent[k] - episode.pedestrian_data[0].agent[k - 1]
                        self.update_agent_and_episode(action, agent, environmentInteraction, episode, k,breadth=[0, 1, 1])
                        episode.pedestrian_data[0].agent[k]= np.array([0, i, j])
            self.update_agent_and_episode(np.zeros(3), agent, environmentInteraction, episode, 9, breadth=[0, 1, 1])
            #print("Agent " + str(episode.agent))
            #Agent standing still!

            for l in range(2):
                k = 0

                for i in range(3):
                    for j in range(3):
                        pos = episode.pedestrian_data[0].agent[k]
                        if i-pos[1]!=0 :
                            self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0],1.0/(i-pos[1]) )
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0],2)
                        if j - pos[2] != 0 :
                            self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 1.0/(j - pos[2]))
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 2)
                        k=k+1
            k = 0
            pos = episode.pedestrian_data[0].agent[8]
            print(" Position "+str(pos))
            for i in range(3):
                for j in range(3):
                    #print episode.get_input_cars_cont(pos, k)
                    print(" Position " + str(pos)+" frame "+str(k)+" cars "+str(cars[k]))
                    feature=episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)
                    self.assertEqual(feature[0,0], 2.0)
                    self.assertEqual(feature[0,1], 2)
                    k = k + 1

            # k = 0 # equal distance
            # pos = [0,1,0]
            # for i in range(3):
            #     for j in range(3):
            #         # print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
            #         if i - pos[1] != 0:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 1.0 / (i - pos[1]))
            #         else:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 2)
            #         if j - pos[2] != 0:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 1.0 / (j - pos[2]))
            #         else:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 2)
            #         k = k + 1
            # k = 0  # equal distance
            # pos = [0, 0, 1]
            # for i in range(3):
            #     for j in range(3):
            #         # print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
            #         if i - pos[1] != 0:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 1.0 / (i - pos[1]))
            #         else:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 2)
            #         if j - pos[2] != 0:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 1.0 / (j - pos[2]))
            #         else:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 2)
            #         k = k + 1
            # k=0 # closest car as close
            # pos = [0, 1, 1]
            # for i in range(3):
            #     for j in range(3):
            #         # print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
            #         if i - pos[1] != 0:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 1.0 / (i - pos[1]))
            #         else:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 2)
            #         if j - pos[2] != 0:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 1.0 / (j - pos[2]))
            #         else:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 2)
            #         k = k + 1
            #
            # k = 0
            # pos = [0, 0, 2]
            # for i in range(3):
            #     for j in range(3):
            #         #print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
            #         if k==6 or k==3 or k==7:
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 1/2.0)
            #         else:
            #             if i - pos[1] != 0:
            #                 self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 1.0 / (i - pos[1]))
            #             else:
            #                 self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 2)
            #             if j - pos[2] != 0:
            #                 self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 1.0 / (j - pos[2]))
            #             else:
            #                 self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 2)
            #         k = k + 1
            # k = 0
            # pos = [0, 1, 2]
            # for i in range(3):
            #     for j in range(3):
            #         # print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
            #         if k == 5 or k ==  4or k == 8 or k == 2:
            #             if i - pos[1] != 0:
            #                 self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 1.0 / (i - pos[1]))
            #             else:
            #                 self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0], 2)
            #             if j - pos[2] != 0:
            #                 self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 1.0 / (j - pos[2]))
            #             else:
            #                 self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1], 2)
            #         else:
            #
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,0],1 )
            #             self.assertEqual(episode.get_input_cars_cont(pos, k, np.array([0,0,0]), 2*np.pi)[0,1],2)
            #
            #         k = k + 1

    def test_goal_input(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor()
        goals=[]
        for i in range(3):
            for j in range(3):
                goals.append([0,i, j])

        for i in SIDEWALK_LABELS:
            tensor[0, 1, 1, CHANNELS.semantic] = i / NUM_SEM_CLASSES
            # tensor[0, 1, 1, 3] = i / NUM_SEM_CLASSES
            agent, episode, environmentInteraction = self.get_episode( cars, gamma, people, pos_x, pos_y, tensor )
            episode.agent_size = [0, 0, 0]

            pos, i, vel = episode.initial_position(0,None, initialization=PEDESTRIAN_INITIALIZATION_CODE.on_pavement)
            for i in range(3):
                for j in range(3):
                    episode.pedestrian_data[0].agent[i * 3 + j] = [0, i, j]

            # Agent standing still!

            for l in range(9):
                k = 0
                episode.goal=goals[l]


                for i in range(3):
                    for j in range(3):
                        pos = episode.pedestrian_data[0].agent[k]
                        #print "Test Goal "+str(goals[l])+" pos "+str(pos)+" dist "+str(goals[l][2]-pos[2])

                        np.testing.assert_approx_equal(episode.get_goal_dir_cont(pos,goals[l])[0,0],(goals[l][1]-pos[1])/np.sqrt(episode.reconstruction.shape[1]**2+episode.reconstruction.shape[2]**2) )
                        np.testing.assert_approx_equal(episode.get_goal_dir_cont(pos, goals[l])[0,1],(goals[l][2]-pos[2])/np.sqrt(episode.reconstruction.shape[1]**2+episode.reconstruction.shape[2]**2))
                        k=k+1


if __name__ == '__main__':
    unittest.main()