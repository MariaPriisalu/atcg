import os
import sys

import numpy as np



if True: # Set this to False! or comment it out
    from visualization import make_movie

from episode import SimpleEpisode, AgentFrameData, EnvironmentInteraction, OfflineDataSource
from extract_tensor import objects_in_range, extract_tensor, objects_in_range_map
from RL.environment_interaction import EntitiesRecordedDataSource, AgentFrameData, EnvironmentInteraction
from settings import RLCarlaOnlineEnv,  NBR_MEASURES, NBR_MEASURES_CAR, NBR_STATS, NBR_POSES, NBR_MAPS,NBR_MAPS_CAR, \
    NBR_STATS_CAR, NBR_CAR_MAP,STATISTICS_INDX_MAP, NBR_MAP_STATS, STATISTICS_INDX_MAP_STAT,CAR_MEASURES_INDX,\
    PEDESTRIAN_MEASURES_INDX,STATISTICS_INDX_CAR, STATISTICS_INDX,  CAR_REWARD_INDX, run_settings

import time
import pickle
import joblib
import scipy
from dotmap import DotMap
import colmap.reconstruct
from RL.supervised_episode import SupervisedEpisode
import copy
from typing import List

from scipy import ndimage
from commonUtils import ReconstructionUtils
from commonUtils.ReconstructionUtils import  SIDEWALK_LABELS, ROAD_LABELS, NUM_SEM_CLASSES, CHANNELS
from settings import run_settings

import psutil
process = psutil.Process(os.getpid())

from memory_profiler import profile as profilemem

memoryLogFP_tick = None
if run_settings.memoryProfile:
    memoryLogFP_tick = open("report_mem_usage_tick.log", "w+")

memoryLogFP_actAndLearn = None
if run_settings.memoryProfile:
    memoryLogFP_actAndLearn = open("report_mem_usage_actandLearn.log", "w+")

memoryLogFP_takeDecision = None
if run_settings.memoryProfile:
    memoryLogFP_takeDecision = open("report_mem_usage_takeDecision.log", "w+")

class AbstractEnvironment(object):


    def visualize(self, episode, file_name, statistics, training, poses,initialization_map,initialization_car, agent, statistics_car ,initialization_goal, initialization_car_map):

        seq_len=self.settings.seq_len_train
        if not training:
            seq_len =self.settings.seq_len_test
        self.jointsParents = None
        if self.settings.pfnn:
            self.jointsParents=[]
            for agent_local in agent:
                self.jointsParents.append(agent_local.PFNN.getJointParents())
        if  self.settings.useRealTimeEnv :
            if self.settings.ignore_external_cars_and_pedestrians:
                tensor = episode.reconstruction
                people=[]
                cars=[]
                for frame in range(seq_len):
                    people.append([])
                    cars.append([])
            else:
                tensor=self.entitiesRecordedDataSource.reconstruction
                people=self.entitiesRecordedDataSource.people
                cars=self.entitiesRecordedDataSource.cars
        else:
            tensor = episode.reconstruction
            people = episode.people
            cars = episode.cars
        self.frame_counter = make_movie(tensor,
                                        people,
                                        cars,
                                        [statistics],
                                        self.settings.width,
                                        self.settings.depth,
                                        seq_len,
                                        self.frame_counter,
                                        self.settings.agent_shape,
                                        self.settings.agent_shape_s,
                                        poses,
                                        initialization_map,
                                        initialization_car,
                                        statistics_car, initialization_goal, initialization_car_map,
                                        training=training,
                                        number_of_agents=self.settings.number_of_agents,
                                        number_of_car_agents=self.settings.number_of_car_agents,
                                        name_movie=self.settings.name_movie,
                                        path_results=self.settings.statistics_dir + "/agent/",
                                        episode_name=os.path.basename(file_name),
                                        action_reorder=self.settings.reorder_actions,
                                        velocity_show=self.settings.velocity,
                                        velocity_sigma=self.settings.velocity_sigmoid,
                                        jointsParents=self.jointsParents,
                                        continous=self.settings.continous,
                                        goal=self.settings.learn_goal,
                                        gaussian_goal=self.settings.goal_gaussian)

    def save_people_cars(self,ep_nbr, episode, people_list, car_list):
        pass


    def __del__(self):
        # Release resources used manually to call connection triggers
        del self.environmentInteraction


    # gets the default camera x,y pos in the world
    def get_camera_pos(self):
        return self.settings.camera_pos_x, self.settings.camera_pos_y

    def default_seq_lengths(self, training, evaluate):
        # Set the seq length
        if evaluate:
            seq_len = self.settings.seq_len_evaluate
        else:
            if training:
                seq_len = self.settings.seq_len_train
            else:
                seq_len = self.settings.seq_len_test

        frameRate, frameTime = self.settings.getFrameRateAndTime()
        seq_len_pfnn = -1
        if self.settings.pfnn:
            seq_len_pfnn = seq_len * 60 // frameRate


        print(("Environment seq Len {}. seq Len pfnn {}".format(seq_len, seq_len_pfnn)))
        return seq_len, seq_len_pfnn

    def default_initialize_agent(self, training, agent, episode, poses_db):
        succedeed = False

        # Initialize the agent at a position by the given parameters
        if training:
            test_init = 1 in self.init_methods_train or 3 in self.init_methods_train
        else:
            test_init = 1 in self.init_methods_train or 3 in self.init_methods_train

        isEpisodeRunValid = (episode.people_dict is not None) and len(episode.valid_keys) > 0
        #print len(episode.valid_keys)

        if (len(self.init_methods) == 1 and not training and self.init_methods[0] > 1) or (
                len(self.init_methods_train) == 1 and training and self.init_methods[0] > 1):
            initialization = self.get_next_initialization(training)
            if self.settings.learn_init:
                return test_init, True, isEpisodeRunValid
            pos, indx, vel = self.agent_initialization(agent, episode, 0, poses_db, training,
                                                       init_m=initialization)
            if len(pos) == 0:
                succedeed = False
                return test_init, False, isEpisodeRunValid

        succedeed = True
        return test_init, succedeed, isEpisodeRunValid

    def default_doActAndGetStats(self, training, number_of_runs_per_scene, trainablePedestrians, episode, poses_db, file_agent, file_name, saved_files_counter,
                                 outStatsGather = None, evaluate = False,  save_stats=True, iterative_training=None,viz=False):

        if iterative_training:
            print("In default_doActAndGetStats train car:" + str(iterative_training.train_car) + "  train initializer " + str(
                iterative_training.train_initializer))
        for number_of_times in range(number_of_runs_per_scene):
            statistics, saved_files_counter, _, _, poses,initialization_map,initialization_car,statistics_car,initialization_goal, initialization_car_map = self.act_and_learn(trainablePedestrians, file_agent, episode,
                                                                                  poses_db, training,
                                                                                  saved_files_counter,  save_stats=save_stats, iterative_training=iterative_training, evaluate=evaluate,viz=viz)


            if len(statistics) > 0:
                if training:
                    self.scene_count = self.scene_count + 1
                else:
                    self.scene_count_test = self.scene_count_test + 1
                # if density >0.001* self.settings.height * self.settings.width * self.settings.depth:
                if self.scene_count == 1 or (
                            self.scene_count // self.settings.vizualization_freq > self.viz_counter and training) or (
                            self.scene_count_test // self.settings.vizualization_freq_test > self.viz_counter_test and not training):
                    # print "Make movie "+str(training) # Debug
                    self.visualize(episode, file_name, statistics, training, poses,initialization_map, initialization_car, trainablePedestrians, statistics_car,initialization_goal, initialization_car_map)
                    if training:
                        self.viz_counter = self.scene_count // self.settings.vizualization_freq
                    else:
                        self.viz_counter_test = self.scene_count_test // self.settings.vizualization_freq_test

        if not training and self.settings.useRLToyCar or self.settings.useHeroCar: # Should we save the statistics when an externally trained car is used?
            initializer_stats=DotMap()
            successes=np.zeros_like(statistics_car[:,:,0,STATISTICS_INDX_CAR.measures[0]+CAR_MEASURES_INDX.goal_reached])
            successes[np.any(statistics_car[:,:,:,STATISTICS_INDX_CAR.measures[0]+CAR_MEASURES_INDX.goal_reached]>0, axis=2)]=1
            initializer_stats.success_rate_car=np.mean(successes)
            collisions = np.zeros_like(statistics[:,:, 0, STATISTICS_INDX.measures[0]+PEDESTRIAN_MEASURES_INDX.hit_by_car])
            collisions[np.any(statistics[:,:, :, STATISTICS_INDX.measures[0]+PEDESTRIAN_MEASURES_INDX.hit_by_car] > 0, axis=2)] = 1
            initializer_stats.collision_rate_initializer = np.mean(collisions)
        else:
            initializer_stats=None

        return saved_files_counter, initializer_stats

    def  parseRealTimeEnvObservation(self, observation,observation_dict, episode):
        # Update things inside episode
        episode.update_pedestrians_and_cars(observation.frame,
                                            observation_dict,
                                            observation.people_dict,
                                            observation.cars_dict,
                                            observation.pedestrian_vel_dict,
                                            observation.car_vel_dict)

    # For each trainable agent inside the environment take a decision
    # This will return a map from each of these agent to the decision taken
    #profilemem(stream=memoryLogFP_takeDecision)
    def takeDecisions(self, episode, frame, training, iterative_training):
        trainableAgents = {}

        # Still hardcoding for a single pedestrian agent and car, But slowly preparing for multi-agent
        allPedestrianAgents = self.all_pedestrian_agents
        if self.all_car_agents:
            allCarAgents = self.all_car_agents.trainableCars
        else:
            allCarAgents =[]

        # episode.agent is the pedestrian agent...
        for pedestrianAgent in allPedestrianAgents :
            trainableAgents[pedestrianAgent] = None

        if allCarAgents is not None:
            for carAgent in allCarAgents:
                trainableAgents[carAgent] = None

        # Take and fill in decisions for pedestrian and cars agents
        for pedestrianAgent in allPedestrianAgents:
            pedestrianAgent_decision = pedestrianAgent.next_action(episode, training)
            trainableAgents[pedestrianAgent] = pedestrianAgent_decision

        # print ("After agent's action " + str(episode.cars))
        if self.settings.useHeroCar or self.settings.useRLToyCar:
            carObservation = self.getObservationForCar(episode, frame,allPedestrianAgents)
            #trainableAgents[self.all_car_agents]= {}

            agent_car_decision_dict = self.all_car_agents.next_action(carObservation, episode, training=(iterative_training == None or iterative_training.train_car), manual=self.settings.manual_car)

            for carAgent, decision in agent_car_decision_dict.items():
                trainableAgents[carAgent] =decision


        return trainableAgents

    # Updates the trainable agents positions on the last simulated tick
    # In the case of a real time online environment we consider that simulation means the decisions where applied and we got feedback from the environment at this point,
    #   it is like a restricted decision made in this case.
    def updateAgentsPositions(self, trainableAgentsData, episode, updated_frame):
        # Commit the actions observed in the environment (note: if the online realtime environment is not used, they will be applied according to the local rules)
        for agent, decisionTaken in trainableAgentsData.items():
            agent.perform_action(decisionTaken, episode)


    def update_metrics(self, episode, trainableAgentsData):
        for agent, decisionTaken in trainableAgentsData.items():
            if agent.getIsPedestrian():
                agent.update_metrics(episode)
        if self.all_car_agents:
            self.all_car_agents.update_metrics(episode)

    def debugStuff(self, episode, trainableAgentsData):
        for agent in self.all_pedestrian_agents:
            updatedFrame = agent.getFrame()
            self.print_agent_location(episode, updatedFrame, trainableAgentsData[agent], agent.id,car=False)
        if self.all_car_agents:
            for car in self.all_car_agents.trainableCars:
                updatedFrame = car.getFrame()
                self.print_agent_location(episode, updatedFrame, trainableAgentsData[car], car.id,car=True)
        self.print_reward_for_manual_inspection(episode, updatedFrame-1, self.img_from_above, self.jointsParents)
        self.plot_for_manual_inspection(episode,  updatedFrame-1, self.img_from_above, self.jointsParents)

    # Does a complete tick of the environment for each trainable agent
    #@profilemem(stream=memoryLogFP_tick)
    def tick(self, episode, trainableAgentsData, frame):
        # If real time env is used then tick it
        realTimeEnvObservation = None
        next_frame = frame + 1 # This is the frame we want to update next, we are at frame, we take a decision then update for next one

        if episode.useRealTimeEnv:
            # Raise signal to perform the decision made action to the environment
            self.environmentInteraction.signal_action(trainableAgentsData, updated_frame=next_frame)

            # Do the simulation for next tick using decisions taken on this tick
            # If an online realtime env is used this call will fill in the data from the simulator.
            # If offline, it will take needed data from recorded/offline data.
            self.environmentInteraction.tick(next_frame)

        # Update the agent positions by committing to the observed interaction from the last tick
        self.updateAgentsPositions(trainableAgentsData, episode, updated_frame=next_frame)

        self.onPostTick(episode, trainableAgentsData, next_frame)

        # Update the metrics - this must be done before the environment observation below call since that will use the metrics computed
        self.update_metrics(episode, trainableAgentsData)

        # print some debug stats
        self.debugStuff(episode, trainableAgentsData)


    # Update things after ticking and updating the positions
    def onPostTick(self, episode, trainableAgentsData, next_frame):
        # Bring new environment data for pedestrians and cars (all others)
        if episode.useRealTimeEnv:
            # Get the environment observation
            observation, observation_dict = self.environmentInteraction.getObservation(frameToUse=next_frame)
            self.parseRealTimeEnvObservation(observation,observation_dict, episode)


        # Advance the frame
        # Updates the resulted agent positions inside episode - NOTE: THIS CANNOT BE DONE BEFOR EPISODE HAS BEEN UPDATED WITH EXTERNAL CARS AND PEOPLE'S POSITIONS- otherwise we miss collisions!
        for agent, decisionTaken in trainableAgentsData.items():
            agent.update_agent_pos_in_episode(episode,next_frame)  # To DO: Why is this separate from  agent.perform_action?



        for agent, decisionTaken in trainableAgentsData.items():
            if agent.getIsPedestrian():
                agent.on_post_tick(episode)
                assert agent.getFrame() == next_frame, "Sanity check failed, we are not on the same frame"
        if self.all_car_agents:
            self.all_car_agents.on_post_tick(episode, next_frame)

    def onStepEnded(self, stepIndex):

        # Do some memory profiling
        if run_settings.memoryProfile and stepIndex % 50 == 0:
            memInfo = process.memory_info()
            memInfo_full = process.memory_full_info()
            mempercent = process.memory_percent()
            memMaps = process.memory_maps()
            print(f"frame iteration {stepIndex} - allocated in ram: {memInfo.rss / (1024.0 * 1024.0 * 1024.0) : .2f} GB")
            #print(f"DEtailed: \n mem: {memInfo} \n full {memInfo_full} \n mempercent {mempercent} \n {memMaps}")

    # Main loop is here! Perform actions in environment
    #@profilemem(stream=memoryLogFP_actAndLearn)
    def act_and_learn(self, trainablePedestrians, agent_file, episode, poses_db, training,saved_files_counter,  road_width=0, curriculum_prob=0, time_file=None, pos_init=[], viz=False, set_goal="", viz_frame=-1, save_stats=True, iterative_training=None,evaluate=False):
        if iterative_training!=None:
            print("In act_and_learn train car:" + str( iterative_training.train_car) + "  train initializer " + str(iterative_training.train_initializer))
        # initialize folders
        num_episode = 0
        repeat_rep = self.get_repeat_reps_and_init(training)
        statistics = np.zeros((repeat_rep,self.settings.number_of_agents, (episode.seq_len - 1),NBR_STATS), dtype=np.float64)

        poses = np.zeros((repeat_rep,self.settings.number_of_agents, episode.seq_len * 60 // episode.frame_rate + episode.seq_len, NBR_POSES),
                             dtype=np.float64)
        initialization_map=[]
        initialization_car=[]
        initialization_car_map = []
        initialization_goal = []

        if self.settings.learn_init:
            initialization_map=np.zeros((repeat_rep,self.settings.number_of_agents, self.settings.env_shape[1]*self.settings.env_shape[2] ,NBR_MAPS), dtype=np.float64)
            initialization_car = np.zeros((repeat_rep,self.settings.number_of_agents,  NBR_CAR_MAP),dtype=np.float64)
        if self.settings.learn_goal:
            for rep in range(repeat_rep):
                initialization_goal.append([])
                for agent_id in range(self.settings.number_of_agents):
                    initialization_goal[rep].append([])
        if self.settings.learn_init_car:
            initialization_car_map = np.zeros((repeat_rep, self.settings.number_of_car_agents,self.settings.env_shape[1] * self.settings.env_shape[2], NBR_MAPS_CAR),
                                          dtype=np.float64)

        if self.settings.useRealTimeEnv or self.settings.useRLToyCar:
            statistics_car = np.zeros((repeat_rep,self.settings.number_of_car_agents, (episode.seq_len - 1), NBR_STATS_CAR), dtype=np.float64)

        else:
            statistics_car=[]
        people_list=np.zeros((repeat_rep, episode.seq_len ,6), dtype=np.float64)
        car_list = np.zeros((repeat_rep, episode.seq_len,6), dtype=np.int)
        self.counter=0
        # Go through all episodes in a gradient batch.
        self.jointsParents, self.labels_indx = self.get_joints_and_labels(trainablePedestrians)

        self.all_pedestrian_agents = trainablePedestrians # The pedestrian agent

        print (" Episode seq len "+str(episode.seq_len))
        for ep_itr in range(repeat_rep):
            onlineEnvActorsContext = None if self.environmentInteraction is None or self.environmentInteraction.onlineEnvironment is None \
                else self.environmentInteraction.onlineEnvironment.envManagement.actorsContext

            # Reset episode internally
            #----------------------------------------
            initParams = DotMap()
            initParams.on_car = False

            # Init car if any used
            #----------------------------------------------------
            if onlineEnvActorsContext is not None:
                onlineEnvActorsContext.shuffleTrainableCars() # Shuffle the trianable cars positions to get a new trainable car at each episode

            if self.settings.useRLToyCar:
                if iterative_training ==None or iterative_training.train_car:
                    if ep_itr%2==0 and training:
                        initParams.on_car=self.settings.supervised_and_rl_car
                print (" Train on car? Environment ")

                # Reassign and reset all cars
                alreadyAssignedCarKeys = set()

                self.all_car_agents.valid_car_keys_trainableagents = list(onlineEnvActorsContext.trainableVehiclesIds) if self.settings.realTimeEnvOnline is True else []
                self.all_car_agents.reset(alreadyAssignedCarKeys, initParams)
                for car in self.all_car_agents.trainableCars:
                    assert onlineEnvActorsContext is None or (car.onlinerealtime_agentId in onlineEnvActorsContext.trainableVehiclesIds), "The car selected is not marked as trainable"
                    assert onlineEnvActorsContext is None or len(alreadyAssignedCarKeys) == len(car.valid_car_keys_trainableagents), "Incorrect assigned cars vs trainable agents. Please check"

            # Init the real time environment if used
            if self.settings.useRealTimeEnv:
                heroAgentCars = [] if self.all_car_agents is None else [agentCar for agentCar in self.all_car_agents.trainableCars]
                heroAgentPedestrians = self.all_pedestrian_agents


                # Associate online environment actor ids (that physically exist in the world) with the logical agent in our environment
                if self.settings.realTimeEnvOnline:
                    # Note that vehicles are assigned above in agentCar.reset
                    assignedWalkerIds = []
                    #heroCarIndex = 0
                    heroAgentPedestrianIndex = 0
                    assert len(onlineEnvActorsContext.trainableWalkerIds) == len(heroAgentPedestrians), "Incorrect size"
                    #assert len(actorsContext.trainableVehiclesIds) == len(heroAgentCars), "Incorrect size"

                    for trainableWalkerId in onlineEnvActorsContext.trainableWalkerIds:
                        heroAgentPedestrians[heroAgentPedestrianIndex].onlinerealtime_agentId = trainableWalkerId
                        heroAgentPedestrianIndex=heroAgentPedestrianIndex+1

                    #for trainableVehicleId in actorsContext.trainableVehiclesIds:
                    #    heroAgentCars[heroCarIndex].realtime_agentId = trainableVehicleId



                realTimeEnvObservation,observation_dict = self.environmentInteraction.reset(heroAgentCars=self.all_car_agents, heroAgentPedestrians=heroAgentPedestrians, episode=episode)
                self.parseRealTimeEnvObservation(realTimeEnvObservation,observation_dict, episode)

            print(("##### Episode "+str(ep_itr)))
            if self.settings.realTimeEnvOnline:
                print("Car agents are assigned to online env ids: ", "".join([str(agent.onlinerealtime_agentId) for agent in heroAgentCars]))
                print("Pedestrian agents are assigned to online env ids: ", "".join([str(agent.onlinerealtime_agentId) for agent in heroAgentPedestrians]))

            # Initialization of the pedestrian agents
            #----------------------------------------------------
            cum_r = 0
            cum_r_car = 0
            pos=[]
            itr_counter=0
            if self.settings.learn_init_car:
                self.all_car_agents.agent_initialization(episode,training=training)



            for agent in self.all_pedestrian_agents:
                episode.pedestrian_data[agent.id].measures = np.zeros(episode.pedestrian_data[agent.id].measures.shape)
                while len(pos)==0:
                    initialization = self.get_next_initialization(training)

                    pos, indx, vel_init = self.agent_initialization(agent, episode, ep_itr, poses_db, training,init_m=initialization, set_goal=set_goal, on_car=initParams.on_car)

                    itr_counter=itr_counter+1

                # Initialize agent on the chosen initial position
                agent.initial_position(pos, episode.pedestrian_data[agent.id].goal[0,:], vel=vel_init, episode=episode)

                if onlineEnvActorsContext is not None:
                    onlineEnvActorsContext.setPedestrianAgentSpawnData(agent, pos, vel_init, episode.goal, transformToWorldReference=True)

                if agent.net:
                    agent.net.reset_mem()
                pos = []
            self.img_from_above = self.initialize_img_above() # To DO: why is this saved?

            # print ("Before loop over frames " + str(episode.cars))
            # This is the main loop of what happens at each frame!
            # print(("sequence lenth episode: "+str(episode.seq_len)))

            # Everything is configured now, both car and pedestrian, tell the environment that we are starting a new episode in this epoch !
            if self.environmentInteraction is not None:
                self.environmentInteraction.onEpisodeStartup()
            frame=0

            all_agents_alive=True
            while frame< episode.seq_len-1 and all_agents_alive:
                for agent in self.all_pedestrian_agents:
                    self.print_input_for_manual_inspection(agent, episode, frame, self.img_from_above, self.jointsParents,
                                                           self.labels_indx,
                                                           training)
                # print ("Before agent's action " + str(episode.cars))

                # Step A: Take decisions for the trained agents
                trainableAgentsData = self.takeDecisions(episode, frame, training, iterative_training)

                # Step B: Perform the decisions for the trained agents and store information in episode about things that happened during the last transition
                self.tick(episode, trainableAgentsData, frame)

                self.onStepEnded(stepIndex=frame)

                all_agents_alive = self.are_all_agents_alive(all_agents_alive, episode, frame)
                frame = frame + 1

            #print("Episode done: all agents alive ? "+str(all_agents_alive)+" frame "+str(frame))
            # Calculate rewards that can only be calculated at the end of the episode.
            # print("Computing some rewards at the end of episode")
            for frame_l in range(0, frame):
                #print("Reward for Frame " + str(frame))
                reward= episode.calculate_reward(frame_l, episode_done=True, last_frame=frame)  # Return to this!


            # Calculate discounted reward
            episode.discounted_reward(frame)
            #print("Discounted rewards "+str(episode.reward_d))
            # Save all of the gathered statistics.

            episode.save(statistics, num_episode, poses, initialization_map,initialization_car,statistics_car,initialization_car_map, initialization_goal, ped_seq_len=frame)
            self.print_reward_after_discounting(cum_r, episode)

           # Print some statistics
            for id in range(len(episode.pedestrian_data)):
                if np.sum(episode.pedestrian_data[id].measures[:,13])>0:
                    self.successes =self.successes+1.0
            self.tries+=1
            # print(("Success rate "+str(self.successes/self.tries)+" nbr of tries: "+str(self.tries)))
            num_episode+=1
            self.num_sucessful_episodes += 1

            # Train the agent or evaluate it.
            if not viz:
                # print (" Save weights file")
                if training:
                    filename=self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training,saved_files_counter)
                    if self.settings.learn_init:
                        if (not self.settings.keep_init_net_constant  and iterative_training==None) or (iterative_training and iterative_training.train_initializer) or not self.settings.useRLToyCar:
                            print(" Train initializer")
                            # To Do: Need to think about what this implies for the initializer. Is each pedestrian a new sample from the initial distribution?
                            for agent in self.all_pedestrian_agents:
                                statistics = agent.init_net_train(ep_itr, statistics, episode,filename+ '_init_net.pkl',filename + '_weights_init_net.pkl' , poses, initialization_map, initialization_car,frame)
                            if self.settings.learn_init_car:
                                for agentCar in self.all_car_agents.trainableCars:
                                    agentCar.init_net_train(ep_itr, statistics, episode, filename + '_car_init_net.pkl',
                                                   filename + '_weights_car_init_net.pkl', poses, initialization_car_map,
                                                   initialization_car, frame, statistics_car=statistics_car)

                            if self.settings.train_init_and_pedestrian and self.settings.train_only_initializer:
                                print(" Train pedestrian")
                                for agent in self.all_pedestrian_agents:
                                    statistics = agent.train(ep_itr, statistics, episode, filename + '.pkl',
                                                             filename + '_weights.pkl', poses, frame)
                        # statistics, episode, filename, filename_weights, poses, priors, initialization_car
                        if (iterative_training ==None or iterative_training.train_car) and self.settings.useRLToyCar:
                            print(" Train cars")
                            for agentCar in self.all_car_agents.trainableCars:
                                agentCar.train(ep_itr, statistics,episode, filename + '_car_net.pkl',filename+ '_weights_car_net.pkl' , poses, initialization_map, statistics_car, frame)
                        if self.settings.train_init_and_pedestrian:
                            print(" Train pedestrian")
                            for agent in self.all_pedestrian_agents:
                                statistics = agent.train(ep_itr, statistics, episode, filename + '.pkl',
                                                         filename + '_weights.pkl', poses, frame)
                    elif self.settings.useRLToyCar:
                        print(" Train cars")
                        for agentCar in self.all_car_agents.trainableCars:
                            agentCar.train(ep_itr, statistics, episode, filename + '_car_net.pkl',
                                                   filename + '_weights_car_net.pkl', poses, initialization_map, statistics_car,frame)
                        if self.settings.train_init_and_pedestrian:
                            print(" Train pedestrian")
                            for agent in self.all_pedestrian_agents:
                                statistics = agent.train(ep_itr, statistics, episode, filename + '.pkl',
                                                         filename + '_weights.pkl', poses, frame)

                    else:
                        print(" Train pedestrian")
                        # ep_itr, statistics, episode, filename, filename_weights, poses
                        for agent in self.all_pedestrian_agents:
                            statistics=agent.train(ep_itr, statistics, episode,filename + '.pkl',filename + '_weights.pkl' , poses, frame)

                else:
                    if self.settings.learn_init:
                        for agent in self.all_pedestrian_agents:

                            statistics = agent.init_net_evaluate(ep_itr, statistics, episode, poses, initialization_map,
                                                              initialization_car, frame)

                        if self.settings.learn_init_car:
                            # (ep_itr, statistics, episode, filename + '_car_init_net.pkl',
                            #  filename + '_weights_car_init_net.pkl', poses, initialization_car_map,
                            #  initialization_car, frame, statistics_car=statistics_car)
                            for agentCar in self.all_car_agents.trainableCars:
                                statistics = agentCar.init_net_evaluate(ep_itr, statistics, episode,poses,  initialization_car_map,
                                                        initialization_car, frame, statistics_car=statistics_car)
                        if self.settings.useRLToyCar:
                            print(" Evaluate cars")
                            for agentCar in self.all_car_agents.trainableCars:
                                agentCar.evaluate(ep_itr, statistics, episode, poses, initialization_map,
                                                                                       statistics_car, frame)
                        if self.settings.train_init_and_pedestrian:
                            for agent in self.all_pedestrian_agents:
                                statistics = agent.evaluate(ep_itr, statistics, episode, poses, initialization_map,
                                                            frame)
                    else:
                        for agent in self.all_pedestrian_agents:
                            statistics= agent.evaluate(ep_itr, statistics, episode, poses, initialization_map, frame)
            if self.environmentInteraction:
                self.environmentInteraction.onEpisodeEnd()


        # Save the loss of the agent.
        #episode.save_loss(statistics)
        # Save all statistics to a file.
        if save_stats:
            stat_file_name=self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter)
            print(("Statistics file: "+stat_file_name+".npy"))

            np.save(stat_file_name+".npy", statistics)
            if self.settings.pfnn and not training:
                np.save(stat_file_name+"_poses.npy",poses[:,:,:96])
            if self.settings.learn_init and (not self.settings.keep_init_net_constant or evaluate):
                if self.settings.learn_init_car:
                    self.save_init_stats(stat_file_name + "_init_stat_map_car.npy",initialization_car_map)
                if self.settings.learn_goal:
                    self.save_init_stats_goal(stat_file_name + "_init_stat_map_goal.npy",initialization_goal, statistics)
                if  viz :#or not self.settings.save_init_stats:
                    print ("Save init maps " +str(viz)+" "+str(evaluate)+" "+str(self.settings.save_init_stats))
                    np.save(stat_file_name+"_init_map.npy", initialization_map)
                else:
                    print("Save init stats " +str(viz)+" "+str(evaluate)+" "+str(self.settings.save_init_stats))
                    self.save_init_stats(stat_file_name + "_init_stat_map.npy", initialization_map)
                np.save(stat_file_name+ "_init_car.npy",initialization_car)
            if episode.useRealTimeEnv:
                np.save(stat_file_name+ "_learn_car.npy",statistics_car)


            # if np.sum(episode.reconstruction[0,:,:,3])>0:
            #     np.save(self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter)+"reconstruction", episode.reconstruction[0,:,:,3:5])
            saved_files_counter = saved_files_counter + 1
        return statistics, saved_files_counter, people_list, car_list, poses, initialization_map, initialization_car,statistics_car,initialization_goal, initialization_car_map

    def are_all_agents_alive(self, all_agents_alive, episode, frame):
        id_not_alive=[]
        if self.settings.end_episode_on_collision:
            for pedestrian in episode.pedestrian_data:
                if pedestrian.measures[frame, PEDESTRIAN_MEASURES_INDX.agent_dead]:
                    return False

        return all_agents_alive

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

    def initialize_img_above(self):
        return None

    def get_joints_and_labels(self, agent):
        return None, None

    def print_reward_for_manual_inspection(self, episode, frame, img_from_above, jointsParents):
        pass
    def plot_for_manual_inspection(self, episode, frame, img_from_above, jointsParents):
        pass

    def print_agent_location(self, episode, frame, value, id, car):
        pass

    def print_input_for_manual_inspection(self, agent, episode, frame, img_from_above, jointsParents, labels_indx,
                                          training):
        pass

    def print_reward_after_discounting(self, cum_r, episode):
        pass

    def get_repeat_reps_and_init(self, training):
        if training:
            repeat_rep = self.settings.update_frequency
            init_methods = self.init_methods  # self.init_methods_train
        else:
            repeat_rep = self.settings.update_frequency_test
            init_methods = self.init_methods  # self.init_methods_train
        return repeat_rep

    def save_init_stats_goal(self, init_file_name,initialization_goal, statistics):


        initialization_map_stats_goal = []
        for ep in range(initialization_map.shape[0]):
            initialization_map_stats_goal.append([])
            for ped_id in range(initialization_map.shape[1]):
                initialization_map_stats_goal[ep].append([])
                goal_frames = statistics[ep, ped_id, 1:, STATISTICS_INDX.frames_of_goal_change]
                goal_frames = goal_frames[goal_frames > 0]
                initialization_map_stats_goal[ep][ped_id].append(np.zeros((len(goal_frames)+1, NBR_MAP_STATS_GOAL), dtype=np.float64))

                for frame in range(len(goal_frames)+1):
                    if frame == 0:
                        initialization_map_stats_goal[ep][ped_id][frame, STATISTICS_INDX_MAP_STAT_GOAL.frame] = 0
                    else:
                        initialization_map_stats_goal[ep][ped_id][frame, STATISTICS_INDX_MAP_STAT_GOAL.frame] = \
                        goal_frames[frame - 1]

                    distr = initialization_goal[ep_itr][ id][frame][ :, STATISTICS_INDX_MAP.init_distribution]
                    prior = initialization_map[ep_itr][ id][frame][ :, STATISTICS_INDX_MAP.prior]
                    prior_non_zero = prior > 0
                    product = distr * prior
                    initialization_map_stats_goal[ep][ped_id][frame, STATISTICS_INDX_MAP_STAT.entropy] = scipy.stats.entropy(distr)
                    initialization_map_stats_goal[ep][ped_id][frame,STATISTICS_INDX_MAP_STAT.entropy_prior] = scipy.stats.entropy(prior)
                    initialization_map_stats_goal[ep][ped_id][frame,STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior] = scipy.stats.entropy(
                        distr * prior_non_zero, qk=prior)
                    initialization_map_stats_goal[ep][ped_id][frame,STATISTICS_INDX_MAP_STAT.init_position_mode[0]:STATISTICS_INDX_MAP_STAT.init_position_mode[
                        1]] = np.unravel_index(np.argmax(product), self.settings.env_shape[1:])
                    initialization_map_stats_goal[ep][ped_id][frame,STATISTICS_INDX_MAP_STAT.init_prior_mode[0]:STATISTICS_INDX_MAP_STAT.init_prior_mode[
                        1]] = np.unravel_index(np.argmax(prior), self.settings.env_shape[1:])
                    initialization_map_stats_goal[ep][ped_id][frame,STATISTICS_INDX_MAP_STAT.prior_init_difference] = np.sum(np.abs(distr - prior), axis=1)

        np.save(init_file_name, initialization_map_stats)

    def save_init_stats(self, init_file_name, initialization_map):
        initialization_map_stats = np.zeros((initialization_map.shape[0],initialization_map.shape[1], NBR_MAP_STATS), dtype=np.float64)
        for ep_itr in range(initialization_map.shape[0]):
            for id in range(initialization_map.shape[1]):
                distr = initialization_map[ep_itr, id, :, STATISTICS_INDX_MAP.init_distribution]
                prior = initialization_map[ep_itr, id, :, STATISTICS_INDX_MAP.prior]
                prior_non_zero = prior > 0
                product = distr * prior
                diff=np.abs(distr - prior)

                initialization_map_stats[ep_itr,id,STATISTICS_INDX_MAP_STAT.entropy]=scipy.stats.entropy(distr)
                initialization_map_stats[ep_itr, id, STATISTICS_INDX_MAP_STAT.entropy_prior]=scipy.stats.entropy(prior)
                initialization_map_stats[ep_itr,id, STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior] =scipy.stats.entropy(distr*prior_non_zero, qk=prior)
                initialization_map_stats[ep_itr,id,STATISTICS_INDX_MAP_STAT.init_position_mode[0]:STATISTICS_INDX_MAP_STAT.init_position_mode[1]]=np.unravel_index(np.argmax(product), self.settings.env_shape[1:])
                initialization_map_stats[ep_itr,id, STATISTICS_INDX_MAP_STAT.init_prior_mode[0]:STATISTICS_INDX_MAP_STAT.init_prior_mode[1]]=np.unravel_index(np.argmax(prior), self.settings.env_shape[1:])
                initialization_map_stats[ep_itr, id, STATISTICS_INDX_MAP_STAT.prior_init_difference] = np.sum(diff)
        np.save(init_file_name,initialization_map_stats)


    def get_next_initialization(self, training):
        if training:
            if not self.settings.learn_init :
                print (" init methods "+str(self.init_methods_train)+" counter "+str(self.counter))
            self.counter=self.counter+1
            self.counter=self.counter%len(self.init_methods_train)
            if len(self.init_methods_train)==1:
                return self.init_methods[0]
            return self.init_methods_train[self.counter]
        else:

            self.counter = self.counter + 1
            self.counter = self.counter% len(self.init_methods)
        self.sem_counter = self.sem_counter + 1

        return self.init_methods[self.counter]



    # Get file name of statistics file.
    def statistics_file_name(self, file_agent, pos_x, pos_y, training,saved_files_counter, init_meth=-1):
        if not training:
            if init_meth>0:
                return file_agent + "_test"+str(init_meth)+"_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
            return file_agent + "_test_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
        return  file_agent + "_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)

    def getEpisodeCachePathByParams(self, episodeIndex, cameraPosX, cameraPosY , realtime):
        if not realtime:
            targetCacheFile = os.path.join(self.target_episodesCache_Path, "{0}_x{1:0.1f}_y{2:0.1f}.pkl".format(episodeIndex, cameraPosX, cameraPosY))
        else:
            targetCacheFile = os.path.join(self.target_realCache_Path,
                                           "{0}_x{1:0.1f}_y{2:0.1f}.pkl".format(episodeIndex, cameraPosX, cameraPosY))
        return targetCacheFile

    def tryReloadFromCache(self, episodeIndex, cameraPosX, cameraPosY, realtime=False):
        episode_unpickled = None
        targetCacheFile = self.getEpisodeCachePathByParams(episodeIndex, cameraPosX, cameraPosY, realtime)
        print(targetCacheFile)
        try:
            if os.path.exists(targetCacheFile):
                print(("Loading {} from cache".format(targetCacheFile)))
                with open(targetCacheFile, "rb") as testFile:
                    start_load = time.time()
                    #episode_unpickled = pickle.load(testFile)
                    episode_unpickled = joblib.load(testFile)
                    load_time = time.time() - start_load
                    print(("Load time from cache (FILE) in s {:0.2f}s".format(load_time)))
            else:
                print(("File {} not found in cache...going to take a while to process".format(targetCacheFile)))
        except ImportError:
            print("import error ")

        return episode_unpickled

    def trySaveToCache(self, episode, episodeIndex, cameraPosX, cameraPosY, realtime=False):
        targetCacheFile = self.getEpisodeCachePathByParams(episodeIndex, cameraPosX, cameraPosY, realtime)
        with open(targetCacheFile, "wb") as testFile:
            print(("Storing {} to cache".format(targetCacheFile)))
            serial_start = time.time()
            #episode_serialized = pickle.dump(episode, testFile, protocol=pickle.HIGHEST_PROTOCOL)
            joblib.dump(episode, testFile, protocol=pickle.HIGHEST_PROTOCOL)
            serial_time = time.time() - serial_start
            print(("Serialization time (FILE) in s {:0.2f}s".format(serial_time)))

    # This functions creates an episode based on previously computed / filled data (especially the entitiesRecordedDataSource structure)
    # It avoids code duplication between caching vs non-caching


    # If you need different data/behavior, put it as parameter OR cache inside self.entitiesRecordedDataSource
    def createEpisodeInstanceWithCar(self,cachePrefix, training, pos_x, pos_y, seq_len_pfnn,evaluate, trainableCars=None, centering=None,precalculated_init_data=None, useCaching=False):
        seq_len = self.settings.seq_len_train
        self.all_car_agents =None
        if not training:
            seq_len = self.settings.seq_len_test


        if self.settings.useRLToyCar:
            from RealTimeRLCarEnvInteraction import RLCarRealTimeEnv

            alreadyAssignedCarKeys = set()


            # TODO: expand this further to store the list of all agent cars
            getRealTimeEnvWaypointPosFunctor=None
            if  self.settings.realTimeEnvOnline:
                getRealTimeEnvWaypointPosFunctor=self.environmentInteraction.getRealTimeEnvWaypointPosFunctor


            newAgentCar = RLCarRealTimeEnv(cachePrefix,pos_x, pos_y, settings=self.settings,
                                           isOnline=self.settings.realTimeEnvOnline,
                                           offlineData=self.entitiesRecordedDataSource,
                                           trainableCars=trainableCars, reconstruction=self.entitiesRecordedDataSource.reconstruction,
                                           seq_len=seq_len, car_dim=self.settings.car_dim,
                                           max_speed=self.settings.car_max_speed_voxelperframe,
                                           min_speed=self.settings.car_min_speed_voxelperframe,
                                           car_goal_closer=self.settings.reward_weights_car[CAR_REWARD_INDX.reached_goal],
                                           physicalCarsDict=self.entitiesRecordedDataSource.env_physical_cars,
                                           physicalWalkersDict=self.entitiesRecordedDataSource.env_physical_pedestrian,
                                           getRealTimeEnvWaypointPosFunctor=getRealTimeEnvWaypointPosFunctor,
                                           seq_len_pfnn=seq_len_pfnn
                                           )
            if precalculated_init_data==None:
                precalculated_init_data=newAgentCar.valid_init

            self.all_car_agents=newAgentCar


        if self.settings.useRealTimeEnv:
            self.environmentInteraction.reset(self.all_car_agents, [], episode=None)
            observation,observation_dict= self.environmentInteraction.getObservation(frameToUse=None)
            
            if useCaching:
                res = self.tryReloadFromCache(cachePrefix, pos_x, pos_y, realtime=True)
                if res is not None:
                    self.finish_set_up_of_episode(evaluate, res, seq_len, seq_len_pfnn)
                    res.environmentInteraction = self.environmentInteraction
                    return res

            cars_dict_sample1 = observation.cars_dict
            people_dict_sample1 = observation.people_dict

            cars_sample1 = [list(cars_dict_sample1.values())]
            people_sample1 = [list(people_dict_sample1.values())]

            for car_key in cars_dict_sample1.keys():
                cars_dict_sample1[car_key] = [cars_dict_sample1[car_key]]

            for person_key in people_dict_sample1.keys():
                people_dict_sample1[person_key] = [people_dict_sample1[person_key]]


            episode = self.init_episode(cars_dict_sample1, cars_sample1,
                                        self.entitiesRecordedDataSource.init_frames, self.entitiesRecordedDataSource.init_frames_cars,
                                        people_dict_sample1,
                                        people_sample1, pos_x, pos_y, seq_len_pfnn, self.entitiesRecordedDataSource.reconstruction, training,
                                        useRealTimeEnv = self.settings.useRealTimeEnv or self.settings.useHeroCar,
                                        car_vel_dict=observation.car_vel_dict,
                                        people_vel_dict=observation.pedestrian_vel_dict,
                                        centering=centering,precalculated_init_data=precalculated_init_data)
            if useCaching:
                self.trySaveToCache(episode, cachePrefix, pos_x, pos_y, realtime=True)
            episode.environmentInteraction = self.environmentInteraction

            return episode
        else:
            return None

    # Sets up the real time interaction environment (either offline or online)
    def setupEnvironmentInteraction(self):
        # Big optimization - do not destroy the environment between different env setups and epochs !

        if self.environmentInteraction is None:

            self.environmentInteraction = EnvironmentInteraction(self.settings.realTimeEnvOnline,ignore_external_cars_and_pedestrians=self.settings.ignore_external_cars_and_pedestrians,
                                                                 entitiesRecordedDataSource=self.entitiesRecordedDataSource,
                                                                 parentEnvironment=self,
                                                                 args=self.settings)

            # Spawn the world using existing API
            self.environmentInteraction.spawnEnvironment(self.settings)

            # Put the agents ids in the dataset options
            self.currentDatasetOptionsUsed.trainableVehiclesIds = self.environmentInteraction.getTrainableVehiclesIds()
            self.currentDatasetOptionsUsed.trainableWalkerIds = self.environmentInteraction.getTrainableWalkerIds()

        # Get in motion data for a few frames and init everything
        for frameInit in range(1):
            print(f"Init frame {frameInit}")
            self.environmentInteraction.tick(0, isInitializationFrame=True)



        # Attach the physical actors correspondences
        if self.environmentInteraction.isOnline:
            self.entitiesRecordedDataSource.env_physical_cars = self.environmentInteraction.getPhysicalCarsDict()
            self.entitiesRecordedDataSource.env_physical_pedestrian = self.environmentInteraction.getPhysicalWalkersDict()


    # Set up episode given the ply file, camera position where the sample is taken from and few other parameters that need to be documented well :)
    def set_up_episode(self, cachePrefix, envPly, pos_x, pos_y, training, useCaching,evaluate=False,  time_file=None, seq_len_pfnn=-1, datasetOptions=None, supervised=False, trainableCars=None):
        # Setup some dataset used params
        assert datasetOptions is not None, "You must always create this object now. If its a dummy, set it up as an empty using the Create API"
        datasetOptions.setEnvironmentParams(dataPath=envPly,
                                                pos_x=pos_x, pos_y=pos_y,
                                                width=self.settings.width, depth=self.settings.depth, height=self.settings.height)
        self.currentDatasetOptionsUsed = datasetOptions

        print(" Car in set up episode " + str(trainableCars))
        cachePrefix += os.path.basename(envPly)

        # Put some extra environ

        # Check to see if caching is enabled and if it exists
        seq_len=self.settings.seq_len_train
        if not training:
            seq_len=self.settings.seq_len_test

        if useCaching:
            res = self.tryReloadFromCache(cachePrefix, pos_x, pos_y)
            if res is not None:
                if res is SupervisedEpisode and not supervised:
                    episode=SimpleEpisode([], # or res?
                                          res.people_e,
                                          res.cars_e,
                                          pos_x,
                                          pos_y,
                                          res.gamma,
                                          res.seq_len,
                                          res.reward_weights_pedestrian,res.reward_weights_initializer,
                                          res.agent_size,
                                          people_dict=res.people_dict,
                                          cars_dict=res.cars_dict,
                                          people_vel=res.people_vel,
                                          cars_vel=res.cars_vel,
                                          init_frames=res.init_frames,
                                          agent_height=self.settings.height_agent,
                                          multiplicative_reward_pedestrian=self.settings.multiplicative_reward_pedestrian,
                                          multiplicative_reward_initializer=self.settings.multiplicative_reward_initializer,
                                          learn_goal=self.settings.learn_goal or self.settings.separate_goal_net,
                                          use_occlusion=self.settings.use_occlusion,
                                          useRealTimeEnv=self.settings.useRealTimeEnv or self.settings.useHeroCar,
                                          new_carla=self.settings.new_carla,
                                          lidar_occlusion=self.settings.lidar_occlusion,
                                          centering=res.centering,
                                          people_dict_trainable=res.people_dict_trainable,
                                          cars_dict_trainable=res.cars_dict_trainable,
                                          use_car_agent = self.settings.useRLToyCar or self.settings.useHeroCar,
                                          use_pfnn_agent = self.settings.pfnn ,
                                          number_of_agents=self.settings.number_of_agents,
                                          number_of_car_agents=self.settings.number_of_car_agents,
                                          initializer_gamma=self.settings.initializer_gamma,
                                          prior_smoothing=self.settings.prior_smoothing,
                                          prior_smoothing_sigma=self.settings.prior_smoothing_sigma,
                                          occlude_some_pedestrians=self.settings.occlude_some_pedestrians,
                                          add_pavement_to_prior=self.settings.add_pavement_to_prior,
                                          assume_known_ped_stats=self.settings.assume_known_ped_stats,
                                          learn_init_car = self.settings.learn_init_car,
                                          assume_known_ped_stats_in_prior=self.settings.assume_known_ped_stats_in_prior,
                                          car_occlusion_prior=self.settings.car_occlusion_prior
                                          )
                    print ("Wrong Episode class!")



                self.finish_set_up_of_episode(evaluate, res, seq_len, seq_len_pfnn)
                if self.settings.realTimeEnvOnline:
                    self.currentDatasetOptionsUsed.centering = res.centering
                # else:
                #     self.currentDatasetOptionsUsed.centering =

                if self.settings.useRealTimeEnv:
                    # The data source is already stored inside, and usefull ONLY in the case of non-online environments

                    if self.settings.realTimeEnvOnline:
                        assert "entitiesRecordedDataSource" not in res.__dict__, "You should not serialize the data source in this online case !"
                        # Initialize an empty offline data source and let the enviornment feedback come in and fill data on each tick
                        self.entitiesRecordedDataSource = EntitiesRecordedDataSource(init_frames={},
                                                                                     init_frames_cars={},
                                                                                     cars_sample=[],
                                                                                     people_sample=[],
                                                                                     cars_dict_sample={},
                                                                                     people_dict_sample={},
                                                                                     cars_vel={},
                                                                                     ped_vel={},
                                                                                     reconstruction=res.reconstruction,
                                                                                     forced_num_frames=0,
                                                                                    )
                    else:
                        if "entitiesRecordedDataSource" not in res.__dict__:
                            print("You SHOULD serialize the data source in real time not online case for performance reasons!. OR DELETE YOUR PREVIOUS CACHED EPISODES !!")
                            assert False

                        set_up_done = self.is_setup_done(cachePrefix,  res.entitiesRecordedDataSource)
                        if not set_up_done:
                            res.entitiesRecordedDataSource.remove_floating_frames()
                        self.entitiesRecordedDataSource = res.entitiesRecordedDataSource


                    self.setupEnvironmentInteraction()

                if self.settings.useRLToyCar:
                    try:
                        centering = res.centering
                    except AttributeError:
                        centering =self.currentDatasetOptionsUsed.centering
                    try:
                        if np.sum(res.heatmap)>0 and np.sum(res.valid_positions_cars)>0:
                            precalculated_init_data=DotMap()
                            precalculated_init_data.heatmap=res.heatmap
                            precalculated_init_data.valid_positions_cars = res.valid_positions_cars
                            precalculated_init_data.valid_directions_cars = res.valid_directions_cars
                            precalculated_init_data.road= precalculated_init_data.road
                        else:
                            precalculated_init_data = None
                    except AttributeError:
                        precalculated_init_data = None

                    res = self.createEpisodeInstanceWithCar(cachePrefix, training=training, pos_x=pos_x, pos_y=pos_y,
                                                            seq_len_pfnn=seq_len_pfnn, evaluate= evaluate,trainableCars=trainableCars,
                                                            centering=centering, precalculated_init_data=precalculated_init_data)#, useCaching=useCaching)

                return res

        heroCarPos = None
        if datasetOptions.isCustom is False or datasetOptions.isColmap == False:
            if datasetOptions.debugFullSceneRaycastRun == 0:
                # Read the reconstruction from the ply episode file
                reconstruction, tensor, people, cars, scale, people_dict, cars_2D, people_2D, \
                    valid_ids, car_dict, init_frames, init_frames_cars, centering = ReconstructionUtils.reconstruct3D_ply(envPly, run_settings.scale_x,
                    read_3D=True, datasetOptions=datasetOptions)
            else:
                debugPath = "/DatasetCustom/Scene1_ep0"
                if not os.path.exists(debugPath):
                    debugPath=envPly
                # Read the reconstruction from the ply episode file
                reconstruction, tensor, people, cars, scale, people_dict, cars_2D, people_2D, \
                    valid_ids, car_dict, init_frames, init_frames_cars, centering = ReconstructionUtils.reconstruct3D_ply(debugPath, run_settings.scale_x,
                    read_3D=True, datasetOptions=datasetOptions)

                sys.exit()

        else:
            # Returns reconstruction in cityscapes coordinate system.

            centering = {}
            reconstruction, people, cars, scale, camera_locations_colmap, middle = colmap.reconstruct.reconstruct3D_ply(envPly, datasetOptions.envSettings, training)
            centering['scale'] = scale
            centering['middle'] = middle

            while len(people) < datasetOptions.LIMIT_FRAME_NUMBER:
                people.append([])
                cars.append([])

            init_frames = {}
            init_frames_cars = {}
            people_dict = {}
            car_dict = {}

        self.currentDatasetOptionsUsed.centering = centering

        # objects = [self.reconstruction, people_rec, cars_rec, scale, ped_dict, cars_2D, people_2D, valid_ids, car_dict, init_frames, init_frames_cars]
        # mem_occupancy = deep_getsizeof(objects, set())

        # TODO: delete me
        # Create tensors of input from the total 3D reconstruction.
        #tensor, density = extract_tensor(pos_x, pos_y, reconstruction, self.settings.height, self.settings.width, self.settings.depth)

        # print("From data people "+str(people[0]))
        # Setup the offline data source data structure to be used later
        # At this point data is expected to be in voxelized space , axis inverted, scaled and sampled according to the enviornment center, depth/width
        rlMotionDataSource: EntitiesRecordedDataSource = EntitiesRecordedDataSource(init_frames=init_frames,
                                                                                    init_frames_cars=init_frames_cars,
                                                                                    cars_sample=cars,
                                                                                    people_sample=people,
                                                                                    cars_dict_sample=car_dict,
                                                                                    people_dict_sample=people_dict,
                                                                                    cars_vel={},
                                                                                    ped_vel={},
                                                                                    reconstruction=tensor,
                                                                                    forced_num_frames=None)
        set_up_done = self.is_setup_done(cachePrefix, rlMotionDataSource)
        if not set_up_done:
            rlMotionDataSource.remove_floating_frames()
        self.entitiesRecordedDataSource = rlMotionDataSource
        # print("In enities recorded people "+str(self.entitiesRecordedDataSource.people[0]) )
        if self.settings.useRealTimeEnv:
            self.setupEnvironmentInteraction()
        # print("After Use Real time Env : " + str(len(rlMotionDataSource.people_dict)))
        if self.settings.useRLToyCar:
            episode = self.createEpisodeInstanceWithCar(cachePrefix, training=training, pos_x=pos_x, pos_y=pos_y, seq_len_pfnn=seq_len_pfnn, evaluate=evaluate, trainableCars=trainableCars, centering=centering)#, useCaching=useCaching)
            # print("After RL car  : " + str(len(episode.people_dict)))
        else:
            # print("Init peopel dict sampple " + str(len(self.entitiesRecordedDataSource.people_dict)))
            if self.settings.useRealTimeEnv:
                self.environmentInteraction.reset([], [], episode=None)
                observation, observation_dict = self.environmentInteraction.getObservation(frameToUse=None)
                car_vel_dict = observation.car_vel_dict
                people_vel_dict = observation.pedestrian_vel_dict
                people_dict={}
                for key, value in observation.people_dict.items():
                    people_dict[key]=[value]
                cars_dict= {}
                for key, value in observation.cars_dict.items():
                    cars_dict[key] = [value]
                cars=self.entitiesRecordedDataSource.cars
                people=self.entitiesRecordedDataSource.people
                init_frames=self.entitiesRecordedDataSource.init_frames
                init_frames_cars=self.entitiesRecordedDataSource.init_frames_cars
            else:
                car_vel_dict = self.entitiesRecordedDataSource.cars_vel
                people_vel_dict = self.entitiesRecordedDataSource.ped_vel
                people_dict = self.entitiesRecordedDataSource.people_dict
                cars_dict = self.entitiesRecordedDataSource.cars_dict
                cars=self.entitiesRecordedDataSource.cars
                people=self.entitiesRecordedDataSource.people
                init_frames=self.entitiesRecordedDataSource.init_frames
                init_frames_cars=self.entitiesRecordedDataSource.init_frames_cars


            episode = self.init_episode(cars_dict,cars,
                                        init_frames,
                                        init_frames_cars,
                                        people_dict,
                                        people,
                                        pos_x, pos_y,
                                        seq_len_pfnn, tensor, training,
                                        useRealTimeEnv = self.settings.useRealTimeEnv,
                                        car_vel_dict=car_vel_dict,
                                        people_vel_dict=people_vel_dict,
                                        centering=centering)

        episode.set_entities_recorded(self.entitiesRecordedDataSource)
        # print("After entitiesRecordedDataSource  : " + str(len(episode.people_dict)))

        if useCaching:

            self.trySaveToCache(episode, cachePrefix, pos_x, pos_y)


        return episode

    def is_setup_done(self, cachePrefix, res):
        set_up_done = self.settings.realTimeEnvOnline or self.settings.realtime_carla_only or not self.settings.carla
        if self.settings.carla and "realtime" not in cachePrefix:
            if len(res.people) == 500 and len(
                   res.cars) == 500:
                set_up_done = False
            else:
                set_up_done = True
        else:
            set_up_done = True
        return set_up_done

    def finish_set_up_of_episode(self, evaluate, res, seq_len, seq_len_pfnn):
        if res.seq_len >= seq_len and (
                (seq_len <= 450 and self.settings.carla) or (seq_len <= 450 and self.settings.waymo) or (
                seq_len <= 30 and not self.settings.waymo and not self.settings.carla)):

            # print("set settings!")
            res.set_correct_run_settings(self.settings.run_2D, seq_len,
                                         self.settings.stop_on_goal,
                                         self.settings.goal_dir,
                                         self.settings.threshold_dist,
                                         self.settings.gamma,
                                         self.settings.reward_weights_pedestrian, self.settings.reward_weights_initializer,
                                         self.settings.end_on_bit_by_pedestrians,
                                         self.settings.speed_input,
                                         self.settings.waymo,
                                         self.settings.reorder_actions,
                                         seq_len_pfnn,
                                         self.settings.velocity,
                                         self.settings.height_agent,
                                         evaluation=evaluate,
                                         defaultSettings=self.settings,
                                         multiplicative_reward_pedestrian=self.settings.multiplicative_reward_pedestrian,
                                         multiplicative_reward_initializer=self.settings.multiplicative_reward_initializer,
                                         learn_goal=self.settings.learn_goal or self.settings.separate_goal_net,
                                         use_occlusion=self.settings.use_occlusion,
                                         useRealTimeEnv=self.settings.useRealTimeEnv or self.settings.useHeroCar,
                                         new_carla=self.settings.new_carla,
                                         lidar_occlusion=self.settings.lidar_occlusion,
                                         car_dim=self.settings.car_dim,
                                         use_car_agent=self.settings.useRLToyCar or self.settings.useHeroCar,
                                         use_pfnn_agent=self.settings.pfnn,
                                         number_of_agents=self.settings.number_of_agents,
                                         number_of_car_agents=self.settings.number_of_car_agents,
                                         initializer_gamma=self.settings.initializer_gamma,
                                         prior_smoothing=self.settings.prior_smoothing,
                                         prior_smoothing_sigma=self.settings.prior_smoothing_sigma,
                                         occlude_some_pedestrians=self.settings.occlude_some_pedestrians,
                                         add_pavement_to_prior=self.settings.add_pavement_to_prior,
                                         assume_known_ped_stats=self.settings.assume_known_ped_stats,
                                         learn_init_car=self.settings.learn_init_car,
                                         assume_known_ped_stats_in_prior=self.settings.assume_known_ped_stats_in_prior,
                                         car_occlusion_prior=self.settings.car_occlusion_prior)

    def init_episode(self, cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample,
                     people_sample, pos_x, pos_y, seq_len_pfnn, tensor, training, useRealTimeEnv, car_vel_dict=None, people_vel_dict=None, centering=None,precalculated_init_data=None):
        print("Init episode use RealTime "+str(useRealTimeEnv))
        if training:
            seq_len=self.settings.seq_len_train
        else:
            while len(cars_sample) < self.settings.seq_len_test:
                if self.settings.carla  or self.settings.waymo:
                    cars_sample.append([])
                    people_sample.append([])
                else:
                    cars_sample.append(cars_sample[-1])
                    people_sample.append(people_sample[-1])
            seq_len=self.settings.seq_len_test
        episode = SimpleEpisode(tensor, people_sample, cars_sample, pos_x, pos_y, self.settings.gamma,
                                seq_len, self.settings.reward_weights_pedestrian,self.settings.reward_weights_initializer,
                                agent_size=self.settings.agent_shape,
                                people_dict=people_dict_sample,
                                cars_dict=cars_dict_sample,
                                people_vel=people_vel_dict,
                                cars_vel=car_vel_dict,
                                init_frames=init_frames, follow_goal=self.settings.goal_dir,
                                action_reorder=self.settings.reorder_actions,
                                threshold_dist=self.settings.threshold_dist, init_frames_cars=init_frames_cars,
                                temporal=self.settings.temporal, predict_future=self.settings.predict_future,
                                run_2D=self.settings.run_2D, agent_init_velocity=self.settings.speed_input,
                                velocity_actions=self.settings.velocity or self.settings.continous,
                                seq_len_pfnn=seq_len_pfnn, end_collide_ped=self.settings.end_on_bit_by_pedestrians,
                                stop_on_goal=self.settings.stop_on_goal, waymo=self.settings.waymo,
                                defaultSettings=self.settings,
                                multiplicative_reward_pedestrian=self.settings.multiplicative_reward_pedestrian,
                                multiplicative_reward_initializer=self.settings.multiplicative_reward_initializer,
                                learn_goal=self.settings.learn_goal or self.settings.separate_goal_net,
                                use_occlusion=self.settings.use_occlusion,
                                useRealTimeEnv=useRealTimeEnv, car_vel_dict=car_vel_dict,
                                people_vel_dict=people_vel_dict, car_dim=self.settings.car_dim,
                                new_carla=self.settings.new_carla, lidar_occlusion=self.settings.lidar_occlusion,
                                centering=centering, use_car_agent= self.settings.useRLToyCar or self.settings.useHeroCar,
                                use_pfnn_agent=self.settings.pfnn, number_of_agents=self.settings.number_of_agents,
                                number_of_car_agents=self.settings.number_of_car_agents,
                                prior_smoothing=self.settings.prior_smoothing,
                                prior_smoothing_sigma=self.settings.prior_smoothing_sigma,
                                occlude_some_pedestrians=self.settings.occlude_some_pedestrians,
                                add_pavement_to_prior=self.settings.add_pavement_to_prior,
                                assume_known_ped_stats=self.settings.assume_known_ped_stats,precalculated_init_data=precalculated_init_data,
                                learn_init_car=self.settings.learn_init_car,
                                assume_known_ped_stats_in_prior=self.settings.assume_known_ped_stats_in_prior,
                                car_occlusion_prior=self.settings.car_occlusion_prior) #To DO:  Check if needed heroCarDetails = heroCarDetails
        episode.environmentInteraction = self.environmentInteraction
        return episode

    # Initiaization.
    def agent_initialization(self, agent, episode, itr, poses_db, training, init_m=-1, set_goal="", on_car=False):

        if self.settings.learn_init and not on_car:
            # print (" Learn init")
            return agent.init_agent(episode, training)
        init_key = itr

        if itr >= len(episode.valid_keys) and len(episode.valid_keys) > 0:
            init_key = itr % len(episode.valid_keys)
        if init_m > 0:
            print(("init_m- initialization method " + str(init_m) + " " + str(init_key) + " " + str(training)))
            pos, indx, vel_init = episode.initial_position(agent.id ,poses_db, training=training, init_key=init_key, initialization=init_m)
        elif len(episode.valid_keys) > 0:
            print(("valid_keys" + " " + str(init_key) + " " + str(training)))
            pos, indx, vel_init = episode.initial_position(agent.id ,poses_db, training=training, init_key=init_key)
        else:
            print(("training " + str(training)))
            pos, indx, vel_init = episode.initial_position(agent.id ,poses_db, training=training)
        return pos, indx, vel_init


    def get_file_agent_path(self, file_name, eval_path=""):
        if self.settings.new_carla:
            basename=os.path.basename(file_name[0])
            city = basename
            seq_nbr = os.path.dirname(file_name[0])
        else:
            basename = os.path.basename(file_name)
            # print (basename)
            parts = basename.split('_')
            city = parts[0]
            seq_nbr = parts[1]
        directory=self.settings.statistics_dir
        if eval_path:
            file_agent = os.path.join(eval_path, self.settings.timestamp + "agent_" + city + "_" + seq_nbr)
        else:

            file_agent = os.path.join(directory, self.settings.mode_name,
                                      self.settings.timestamp + "agent_" + city + "_" + seq_nbr)

        if not os.path.exists(file_agent):
            os.makedirs(file_agent)

        return file_agent



    def __init__(self, path, sess, writer, gradBuffer, log,settings, net=None):
        self.num_sucessful_episodes = 0
        self.accummulated_r = 0
        self.counter=0
        self.frame_counter_eval=0
        self.img_path = path  # images
        self.files = []
        self.episode_buffer = []
        self.gradBuffer = gradBuffer
        self.reconstruction={}
        self.writer=writer
        self.sess=sess
        self.log=log
        self.settings=settings
        # self.net=net
        self.frame_counter=0
        self.sem_counter=0

        self.scene_count = 0
        self.scene_count_test = 0
        self.viz_counter_test = 0
        # self.goal_dist_reached=1
        self.init_methods = [1, 8, 6, 2, 4]
        self.init_methods_train =[1, 8, 6, 2, 4]
        self.successes = 0.0
        self.tries = 0
        self.environmentInteraction = None
        self.currentDatasetOptionsUsed = None
        self.all_car_agents = None

        self.target_episodesCache_Path = settings.target_episodesCache_Path
        self.target_realCache_Path= settings.target_realCache_Path


        if not os.path.exists(self.target_episodesCache_Path):
            os.makedirs(self.target_episodesCache_Path)

        for path, dirs, files in os.walk(self.img_path):
            self.files = self.files + dirs  # + files

        self.scene_count_test=0
        self.viz_counter_test=0
        #self.goal_dist_reached=1

        self.successes = 0.0
        self.tries = 0




