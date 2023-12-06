import unittest
import numpy as np
import sys

from RL.extract_tensor import frame_reconstruction
from commonUtils.ReconstructionUtils import objects_in_range, objects_in_range_map
from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX,PEDESTRIAN_INITIALIZATION_CODE,NBR_MEASURES,PEDESTRIAN_MEASURES_INDX
from RL.environment_interaction import EntitiesRecordedDataSource, EnvironmentInteraction
from commonUtils.ReconstructionUtils import LAST_CITYSCAPES_SEMLABEL, cityscapes_labels_dict, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, SIDEWALK_LABELS,CHANNELS,OBSTACLE_LABELS_NEW, OBSTACLE_LABELS,cityscapes_labels_dict
# Test methods in episode.

class TestExtractTensor(unittest.TestCase):
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

    def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor, seq_len=30, rewards=[],
                           agent_size=(0, 0, 0), people_dict={}, init_frames={}, car_dict={}, init_frames_cars={},
                           new_carla=False):
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
        episode.pedestrian_data[0].action[frame - 1] = 5

    def test_empty(self):
        cars_a, people_a, tensor = self.initialize()
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, tensor)

    def test_empty_online(self):
        cars_a, people_a, tensor = self.initialize()
        agent, episode, environmentInteraction = self.get_episode(cars_a, 1, people_a, 0, 0, tensor)
        np.testing.assert_array_equal(tensor, episode.reconstruction)

    def test_one_car(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        expected[1,1,1,5]=0.1
        cars_a[0].append((1,1,1,1,1,1))
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)

        np.testing.assert_array_equal(out, expected)

    def test_one_car_online(self):
        cars_a, people_a, tensor = self.initialize()
        expected = tensor.copy()
        expected[1, 1, 1, 5] = 1
        cars_a[0].append((1, 2, 1, 2, 1, 2))
        cars_dict={0:[np.array([1, 2, 1, 2, 1, 2]),np.array([1, 2, 1, 2, 1, 2])]}
        car_init={0:0}
        agent, episode, environmentInteraction = self.get_episode(cars_a, 1, people_a, 0, 0, tensor,car_dict=cars_dict, init_frames_cars=car_init)
        print(episode.reconstruction[1, 1, 1, 5])
        np.testing.assert_array_equal(expected, episode.reconstruction)

    def test_one_person(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        expected[1,1,1,4]=0.1

        people_a[0].append(np.ones((3,2), dtype=np.int))
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, expected)

    def test_one_person_online(self):
        cars_a, people_a, tensor = self.initialize()
        expected = tensor.copy()
        expected[1, 1, 1, 4] = 1
        people_a[0].append(np.ones((3,2), dtype=np.int))
        people_dict = {0: [np.ones((3,2)), np.ones((3,2))]}
        init = {0: 0}
        agent, episode, environmentInteraction = self.get_episode(cars_a, 1, people_a, 0, 0, tensor,
                                                                  people_dict=people_dict, init_frames=init)
        print(episode.reconstruction[1, 1, 1, 4])
        np.testing.assert_array_equal(expected, episode.reconstruction)

    def test_two_cars(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        cars_a[0].append((0,0,0,0,0,0))
        cars_a[0].append((0,0, 0, 0, 2,2))
        cars_a[2].append((0,0, 0, 0, 0,1))
        expected[0, 0, 0, 5] = 0.2
        expected[0, 0, 1, 5] = 0.1
        expected[0, 0, 2, 5] = 0.1
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, expected)

    def test_two_cars_online(self):
        cars_a, people_a, tensor = self.initialize()
        expected = tensor.copy()
        expected[0, 0, 0, 5] = 3
        expected[0, 0, 1, 5] =3
        expected[0, 0, 2, 5] = 2
        cars_a[0].append((0, 0+ 1, 0, 0+ 1, 0, 0+ 1))
        cars_a[0].append((0, 0+ 1, 0, 0+ 1, 2, 2+ 1))
        cars_a[1].append((0, 0+ 1, 0, 0+ 1, 0, 0+ 1))
        cars_a[1].append((0, 0+ 1, 0, 0+ 1, 2, 2+ 1))
        cars_a[2].append((0, 0+ 1, 0, 0+ 1, 0, 1+ 1))
        cars_dict = {0: [np.array([0, 1, 0, 1, 0, 1]), np.array([0, 1, 0, 1, 0,1]),np.array([0, 1, 0, 1, 0, 2])], 1:[(0, 1, 0, 1, 2, 3),(0, 1, 0, 1, 2, 3)]}
        car_init = {0: 0, 1:0}
        agent, episode, environmentInteraction = self.get_episode(cars_a, 1, people_a, 0, 0, tensor, car_dict=cars_dict,
                                                                  init_frames_cars=car_init)
        episode.pedestrian_data[0].agent[0]=[0,2,2]
        episode.pedestrian_data[0].goal[0,:]=np.array([0,0,0])
        # np.testing.assert_array_equal(pos, [])
        agent.initial_position(episode.pedestrian_data[0].agent[0], episode.pedestrian_data[0].goal[0,:])
        self.update_agent_and_episode(np.array([0,0,0]), agent, environmentInteraction, episode, 1)
        self.update_agent_and_episode(np.array([0,0,0]), agent, environmentInteraction, episode, 2)
        print(episode.reconstruction[1, 1, 1, 5])
        np.testing.assert_array_equal(expected, episode.reconstruction)

    def test_many_cars_online(self):
        cars_a, people_a, tensor = self.initialize()
        expected = tensor.copy()
        indx = 0
        cars_dict={}
        car_init={}
        for x in range(0, 3):
            for y in range(0, 3):
                cars_a[0].append((0, 1, x, x+1, y, y+1))
                cars_a[1].append((0, 1, x, x+1, y, y+1))
                cars_dict[indx]=np.array([[0, 1, x, x+1, y, y+1],[0, 1, x, x+1, y, y+1] ])
                car_init[indx]=0
                indx += 1

        agent, episode, environmentInteraction = self.get_episode(cars_a, 1, people_a, 0, 0, tensor, car_dict=cars_dict,
                                                                  init_frames_cars=car_init)
        episode.pedestrian_data[0].agent[0] = [0, 2, 2]
        expected[0, :, :, 5] =  np.ones(expected.shape[1:3])

        #out, cars_predicted, people_predicted, reconstruction_2D = frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(episode.reconstruction, expected)
    def test_many_cars(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        indx=0
        for x in range(0,3):
            for y in range(0,3):
                cars_a[indx].append((0,0, x, x, y,y))
                indx+=1
        expected[0,:,:,5]=0.1*np.ones(expected.shape[1:3])
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, expected)

    def test_many_people_online(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        indx=0
        people_dict={}
        people_init = {}
        for x in range(0,3):
            for y in range(0,3):
                people_a[0].append(np.array([[0,0], [y,y],[x, x]], np.int32))
                people_a[1].append(np.array([[0, 0], [y, y], [x, x]], np.int32))
                people_dict[indx]=[np.array([[0,0], [y,y],[x, x]], np.int32),np.array([[0,0], [y,y],[x, x]], np.int32)]
                people_init[indx]=0
                indx+=1
        agent, episode, environmentInteraction = self.get_episode(cars_a, 1, people_a, 0, 0, tensor, people_dict=people_dict,
                                                                  init_frames=people_init)
        expected[0, :, :, 4] = np.ones(expected.shape[1:3])
        #out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(expected, episode.reconstruction)
    #         expected[0, :, :, 5] = 0.1 * np.ones(expected.shape[1:3]) #
    #tensor, cars_a, people_a, no_dict=True, temporal=False, predict_future=False, run_2D=False, reconstruction_2D=[]
    # def test_many_car_dict(self):
    #     cars_a, people_a, tensor = self.initialize()
    #     expected = tensor.copy()
    #     indx = 0
    #     for x in range(0, 3):
    #         for y in range(0, 3):
    #             cars_a[indx].append((0, 0, x, x, y, y))
    #             indx += 1
    #     expected[0, :, :, 5] = 0.1 * np.ones(expected.shape[1:3])
    #     # reconstruction, cars_predicted,people_predicted, reconstruction_2D
    #     #frame_reconstruction(tensor, cars_a, people_a, no_dict=True, temporal=False, predict_future=False, run_2D=False, reconstruction_2D=[])
    #     out, cars_predicted, people_predicted, reconstruction_2D = frame_reconstruction(tensor, cars_a, people_a, no_dict=False,  temporal=True,  predict_future=True, run_2D=True, reconstruction_2D=[])
    #     np.testing.assert_array_equal(out, expected)

    def test_many_people(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        indx=0
        #vals=[[0,1],[0,2],[1,3],[2,3]]
        for x in range(0,3):
            for y in range(0,3):
                people_a[indx].append(np.array([[0,0], [y,y],[x, x]], np.int32))
                indx+=1
        expected[0, :, :, 4] = 0.1 * np.ones(expected.shape[1:3])
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, expected)

    def initialize(self, size=3):
        tensor = np.zeros((size, size, size, 6))
        people_a = []
        cars_a = []
        for i in range(30):
            people_a.append([])
            cars_a.append([])

        return cars_a, people_a, tensor

    def test_obj_in_range_corner(self):
        cars_a, people_a, tensor = self.initialize()
        cars_a[0].append((2,3,2,3,2,3))
        out=objects_in_range(cars_a, 0, 0, 3,3)
        people_a[0].append((2,3,2,3,2,3))
        np.testing.assert_array_equal(out, people_a)
        people_a[0]=[]
        out = objects_in_range(cars_a, 3, 0, 3, 3)
        people_a[0].append((2,3, 2, 3, -1, 0))
        np.testing.assert_array_equal(out, people_a)
        people_a[0] = []
        out = objects_in_range(cars_a, 0, 3, 3, 3)
        people_a[0].append((2, 3,-1, 0, 2, 3))
        np.testing.assert_array_equal(out, people_a)
        people_a[0] = []
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        people_a[0].append((2, 3, -1, 0, -1, 0))
        np.testing.assert_array_equal(out, people_a)

    def test_obj_in_range(self):
        cars_a, people_a, tensor = self.initialize()
        cars_a[0].append((2, 3, 3, 3, 2, 3))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        out = objects_in_range(cars_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        out = objects_in_range(cars_a, 0, 3, 3, 3)
        people_a[0].append((2, 3,0, 0, 2, 3))
        np.testing.assert_array_equal(out, people_a)

        people_a[0] = []
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        people_a[0].append((2, 3, 0, 0, -1, 0))
        np.testing.assert_array_equal(out, people_a)

        cars_a[0]=[]
        people_a[0] = []
        cars_a[0].append((3, 5, 3, 5, 3, 5))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        out = objects_in_range(cars_a, 0, 3, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        out = objects_in_range(cars_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        people_a[0].append((3,5,0,2,0,2))
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        np.testing.assert_array_equal(out, people_a)

    def test_people_in_range_corner(self):
        cars_a, people_a, tensor = self.initialize()
        cars_a[0].append(np.array([[2, 3],[2, 3] ,[2, 3]]))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        people_a[0].append(np.array([[2, 3],[2, 3] ,[2, 3]]))
        np.testing.assert_array_equal(out[0], people_a[0])
        people_a[0] = []
        out = objects_in_range(cars_a, 3, 0, 3, 3)
        people_a[0].append(np.array([[2, 3],[2, 3] ,[-1, 0]]))
        np.testing.assert_array_equal(out[0], people_a[0])
        people_a[0] = []
        out = objects_in_range(cars_a, 0, 3, 3, 3)
        people_a[0].append(np.array([[2, 3],[-1, 0] ,[2, 3]]))
        np.testing.assert_array_equal(out[0], people_a[0])
        people_a[0] = []
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        people_a[0].append(np.array([[2, 3],[-1, 0] ,[-1, 0]]))
        np.testing.assert_array_equal(out[0], people_a[0])

    def test_obj_in_range1(self):
        cars_a, people_a, tensor = self.initialize()
        cars_a[0].append(np.array([[2, 3],[3, 3] ,[2, 3]]))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        out = objects_in_range(cars_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        out = objects_in_range(cars_a, 0, 3, 3, 3)
        people_a[0].append(np.array([[2, 3],[0, 0] ,[2, 3]]))
        np.testing.assert_array_equal(out[0], people_a[0])

        people_a[0] = []
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        people_a[0].append(np.array([[2, 3],[0, 0] ,[-1, 0]]))
        np.testing.assert_array_equal(out[0], people_a[0])

        cars_a[0] = []
        people_a[0] = []
        cars_a[0].append(np.array([[3, 5],[3, 5] ,[3, 5]]))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        out = objects_in_range(cars_a, 0, 3, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        out = objects_in_range(cars_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        people_a[0].append(np.array([[3, 5],[0, 2] ,[0, 2]]))
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

    def test_random_people_in_range(self):
        cars_a, people_a, tensor = self.initialize()
        people_a[0].append(np.random.randint(3, size=(3,14)))
        people_a[0].append(np.random.randint(3, size=(3, 14))+3*np.ones((3,14)))
        out = objects_in_range(people_a, 3, 3, 3, 3)
        np.testing.assert_array_equal(1, len(out[0]))
        out = objects_in_range(people_a, 0,0, 3, 3)
        np.testing.assert_array_equal(1, len(out[0]))
        out = objects_in_range(people_a, 0, 3, 3, 3)
        np.testing.assert_array_equal(0, len(out[0]))
        out = objects_in_range(people_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(0, len(out[0]))


if __name__ == '__main__':
    unittest.main()