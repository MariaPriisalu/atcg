
from environment_abstract import AbstractEnvironment
from settings import PEDESTRIAN_INITIALIZATION_CODE

from commonUtils.ReconstructionUtils import CreateDefaultDatasetOptions_CarlaRealTime, CreateDefaultEmptyDatasetOptions,CreateDefaultDatasetOptions_CarlaOffline, CreateDefaultDatasetOptions_CarlaRealTimeOffline
import numpy as np
import os
import pickle
import time
if True: # Set this to False! or comment it out
    from visualization import make_movie, make_movie_eval

class CARLAEnvironment(AbstractEnvironment):

    def __init__(self, path,  sess, writer, gradBuffer, log, settings, net=None):
        super(CARLAEnvironment, self).__init__(path, sess, writer, gradBuffer, log, settings, net=net)
        self.viz_counter=0
        self.frame_counter=0
        self.scene_count=0
        self.viz_counter_test = 0
        self.scene_count_test = 0
        if settings.supervised_and_rl_car or settings.useRLToyCar  or settings.useHeroCar:
            self.init_methods = [PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                 PEDESTRIAN_INITIALIZATION_CODE.randomly]  # [1,5]#[1, 3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train = [PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                 PEDESTRIAN_INITIALIZATION_CODE.randomly]   # [1,5]#[1,  3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[9, 8,6,2,4]#[8, 6, 2, 4,] # [1, 8,1, 6,1, 2,1, 4, 1] [1]

        elif settings.continous:
            self.init_methods = [PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                 PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian,
                                 PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car,
                                 PEDESTRIAN_INITIALIZATION_CODE.near_obstacle]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train = [PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                       PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian,
                                       PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car,
                                       PEDESTRIAN_INITIALIZATION_CODE.near_obstacle]

        elif settings.useRealTimeEnv:
            self.init_methods = [PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                 PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                 PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                 PEDESTRIAN_INITIALIZATION_CODE.randomly]  # [1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train = [PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                       PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                       PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                       PEDESTRIAN_INITIALIZATION_CODE.randomly,
                                       PEDESTRIAN_INITIALIZATION_CODE.near_obstacle]  # [1,

        elif settings.goal_dir:
            self.init_methods = [PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                 PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                 PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                 PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                 PEDESTRIAN_INITIALIZATION_CODE.randomly]  # [1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train = [ PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                        PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                        PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                        PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                        PEDESTRIAN_INITIALIZATION_CODE.randomly,
                                        PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                        PEDESTRIAN_INITIALIZATION_CODE.near_obstacle]  # [1,
        else:
            self.init_methods =[PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                PEDESTRIAN_INITIALIZATION_CODE.randomly]#[1,5]#[1, 3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train =[PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                      PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                      PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                      PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                      PEDESTRIAN_INITIALIZATION_CODE.randomly,
                                      PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                      PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian,
                                      PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car,
                                      PEDESTRIAN_INITIALIZATION_CODE.near_obstacle]#[1,5]#[1,  3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[9, 8,6,2,4]#[8, 6, 2, 4,] # [1, 8,1, 6,1, 2,1, 4, 1] [1]

    # Cleanup here all the stored data between iteration which should be reset
    def cleanupPersistentDataBetweenIterations(self):
        if self.environmentInteraction is not None:
            self.environmentInteraction.destroyEnvironment(betweenIterations=True)
            self.environmentInteraction = None

    # Take steps, calculate reward and visualize actions of agent

    def work(self, cachePrefix, filepath, trainablePedestrians, poses_db, epoch, saved_files_counter, trainableCars=None, training=True, road_width=5, conv_once=False, time_file=None, act=True, save_stats=True, iterative_training=None, useRealTimeEnv=False, realtime_carla=False):
        # Setup some options for the episode
        print (" Cahce prefix "+str(cachePrefix))
        print(" File path " + str(filepath))

        file_agent = self.get_file_agent_path(filepath)
        pos_x, pos_y = self.get_camera_pos()
        seq_len, seq_len_pfnn = self.default_seq_lengths(training=training, evaluate=False)
        if iterative_training!=None:
            print("In work train car:" + str(iterative_training.train_car) + "  train initializer " + str(
                iterative_training.train_initializer))
        #Setup the episode

        start_setup = time.time()
        if self.settings.new_carla or useRealTimeEnv or realtime_carla:
            if self.settings.new_carla:
                file_name=filepath[0]
            else:
                file_name = filepath
            if realtime_carla:
                datasetOptions = CreateDefaultDatasetOptions_CarlaRealTimeOffline(self.settings)
            elif self.settings.realTimeEnvOnline or self.settings.test_online_dataset :
                datasetOptions=  CreateDefaultDatasetOptions_CarlaRealTime(self.settings)
            else:
                datasetOptions = CreateDefaultDatasetOptions_CarlaOffline(self.settings)

        else:
            file_name = filepath
            datasetOptions = CreateDefaultDatasetOptions_CarlaOffline(self.settings)

        datasetOptions.debugFullSceneRaycastRun = self.settings.debugFullSceneRaycastRun
        # print ("Dataset options---------------------------------------")
        # print(datasetOptions)
        self.cleanupPersistentDataBetweenIterations()

        episode = self.set_up_episode(cachePrefix, file_name, pos_x, pos_y, training, useCaching=self.settings.useCaching,
                                      time_file=time_file, seq_len_pfnn=seq_len_pfnn, trainableCars=trainableCars,
                                      datasetOptions=datasetOptions)
        setup_time = time.time() - start_setup
        # print(("Setup time in s {:0.2f}s".format(setup_time)))

        # Act and learn
        number_of_runs_per_scene = 1
        if self.settings.learn_init or self.settings.useRLToyCar or self.settings.useHeroCar:
            enough_car_inits=len(episode.init_cars) or self.settings.realTimeEnvOnline
            enough_valid_inits=False
            if self.settings.learn_init_car :
                if not self.settings.ignore_external_cars_and_pedestrians:
                    print("Remove extrenal cars")
                    self.all_car_agents.remove_cars_from_valid_pos()
                number_or_valid_pos=np.sum(self.all_car_agents.valid_init.valid_positions_cars)
                min_number_of_valid_cars=((max(self.settings.car_dim[1:])*2+4)**2+1)*self.settings.number_of_car_agents
                enough_valid_inits=number_or_valid_pos>min_number_of_valid_cars
                print("Number of valid pos "+str(min_number_of_valid_cars)+" need at least "+str(min_number_of_valid_cars)+ " proceed? "+str(enough_valid_inits))
            if  enough_valid_inits or enough_car_inits:
                saved_files_counter,initializer_stats = self.default_doActAndGetStats(training, number_of_runs_per_scene, trainablePedestrians, episode,
                                                                                      poses_db, file_agent, file_name,
                                                                                      saved_files_counter, save_stats=save_stats, iterative_training=iterative_training)
            else:
                print ("Not valid initialization")
                return self.counter, saved_files_counter, None

        else:
            run_episode=True
            test_init, succedeed, validRun = self.default_initialize_agent(training, trainablePedestrians, episode, None)
            if not succedeed:
                return self.counter, saved_files_counter



            if act and (not test_init or run_episode) or self.settings.realTimeEnvOnline:
                saved_files_counter,initializer_stats = self.default_doActAndGetStats(training, number_of_runs_per_scene, trainablePedestrians, episode, poses_db, file_agent, file_name, saved_files_counter, save_stats=save_stats)
            else:
                print ("Not valid initialization")

                return self.counter, saved_files_counter, None

        return self.counter, saved_files_counter, initializer_stats


    def evaluate(self, cachePrefix, filepath, trainablePedestrians, file_path, saved_files_counter, viz=False, folder="",useRealTimeEnv=False, realtime_carla=False,trainableCars=None):
        training = False
        # print(" Car in env.evaluate() " + str(trainableCars))
        # Setup some options for the episode
        file_agent = self.get_file_agent_path(filepath, eval_path=file_path)
        pos_x, pos_y = self.get_camera_pos()
        seq_len, seq_len_pfnn = self.default_seq_lengths(training=training, evaluate=True)
        self.settings.seq_len_test= seq_len # Override the test seq length because this is the one actually used, there is no different variable for evaluate in the core code


        # Setup the episode
        start_setup = time.time()
        if self.settings.new_carla or useRealTimeEnv or realtime_carla:
            if self.settings.new_carla:
                file_name = filepath[0]
            else:
                file_name = filepath
            datasetOptions = CreateDefaultDatasetOptions_CarlaRealTime(self.settings)
            # print ("New dataset otions ")

        else:
            file_name = filepath
            datasetOptions = CreateDefaultDatasetOptions_CarlaOffline(self.settings)


        episode = self.set_up_episode(cachePrefix, file_name, pos_x, pos_y, training,evaluate=True, useCaching=self.settings.useCaching, seq_len_pfnn=seq_len_pfnn,datasetOptions=datasetOptions,trainableCars=trainableCars)
        setup_time = time.time() - start_setup
        print(("Setup time in s {:0.2f}s".format(setup_time)))
        # Act and get stats
        stats = []
        initializer_stats=[]
        number_of_runs_per_scene = 1
        use_car = self.settings.learn_init or self.settings.useRLToyCar or self.settings.useHeroCar
        if use_car:
            enough_car_inits = len(episode.init_cars) or self.settings.realTimeEnvOnline
            enough_valid_inits = False
            if self.settings.learn_init_car:
                enough_car_inits= len(self.all_car_agents.valid_init.car_keys)>0
                if not self.settings.ignore_external_cars_and_pedestrians:
                    print("Remove extrenal cars")
                    self.all_car_agents.remove_cars_from_valid_pos()
                number_or_valid_pos = np.sum(self.all_car_agents.valid_init.valid_positions_cars)
                min_number_of_valid_cars = ((max(self.settings.car_dim[
                                                 1:]) * 2 + 4) ** 2 + 1) * self.settings.number_of_car_agents
                enough_valid_inits = number_or_valid_pos > min_number_of_valid_cars
                print("Number of valid pos " + str(min_number_of_valid_cars) + " need at least " + str(
                    min_number_of_valid_cars) + " proceed? " + str(enough_valid_inits))

            if enough_valid_inits or enough_car_inits:
                saved_files_counter, initializer_stats = self.default_doActAndGetStats(training, number_of_runs_per_scene,
                                                                                       trainablePedestrians, episode, None, file_agent,
                                                                                       file_name, saved_files_counter,
                                                                                       outStatsGather=stats,
                                                                                       evaluate=True, save_stats=True, viz=viz)
            else:
                assert ("No valid cars")
                return stats, saved_files_counter,initializer_stats
        else:
            run_episode=True
            test_init, succedeed, validRun = self.default_initialize_agent(training, trainablePedestrians, episode, None)
            if not succedeed:
                return stats, saved_files_counter,initializer_stats
            run_episode |= validRun

            if (not test_init or run_episode) or self.settings.realTimeEnvOnline:
                use_car=self.settings.learn_init or self.settings.useRLToyCar or self.settings.useHeroCar
                if (use_car and len(episode.init_cars)) or not use_car:
                    saved_files_counter,initializer_stats = self.default_doActAndGetStats(training,number_of_runs_per_scene, trainablePedestrians, episode, None, file_agent, file_name, saved_files_counter,  outStatsGather=stats, evaluate=True,save_stats=True, viz=viz)
            else:
                print ("Not valid initialization")
                return stats, saved_files_counter,initializer_stats

        return stats, saved_files_counter,initializer_stats

