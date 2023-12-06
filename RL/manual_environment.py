import os
from datetime import datetime
external=False

import numpy as np

from carla_environment import CARLAEnvironment
from visualization import make_movie, make_movie_manual
from environment_waymo import WaymoEnvironment

from PFNNPythonLiveVisualizer import PFNNLiveVisualizer as PFNNViz
from environment_abstract import AbstractEnvironment

from RL.settings import PEDESTRIAN_MEASURES_INDX

def getBonePos(poseData, boneId):
    x = poseData[boneId*3 + 0]
    y = poseData[boneId*3 + 1]
    z = poseData[boneId*3 + 2]
    return x,y,z

def distance2D(x0, z0, x1, z1):
    return math.sqrt((x0-x1)**2 + (z0-z1)**2)

def prettyPrintBonePos(poseData, boneID):
    print(("Bone {0} has position: ({1:.2f},{2:.2f},{3:.2f})".format(boneID, *getBonePos(poseData, boneID))))


np.set_printoptions(precision=2)



class ManualAbstractEnvironment(AbstractEnvironment):

    def return_highest_object(self, tens):
        z = 0
        while z < tens.shape[0] - 1 and np.linalg.norm(tens[z])==0  :
            z = z + 1
        return z

    def return_highest_col(self, tens):
        z = 0
        while z < tens.shape[0] - 1 and np.linalg.norm(tens[z,:])==0  :
            z = z + 1
        return z



    def print_reward_after_discounting(self, cum_r, episode):
        for id in range(episode.number_of_agents):
            print(("Agent "+str(id)+" Reward after discounting: " + str(episode.pedestrian_data[id].reward_d)))
            #print(("  Reward: " + str(cum_r / self.settings.update_frequency)))
        for id in range(episode.number_of_agents):
            print(("Agent "+str(id)+" Reward after discounting: " + str(episode.initializer_data[id].reward_d)))
    def initialize_img_above(self):
        img_from_above = np.zeros(1)
        return img_from_above

    def get_joints_and_labels(self, trainablePedestrians):
        #    label_mapping[0] = 11 # Building
        #    label_mapping[1] = 13 # Fence
        #    label_mapping[2] = 4 # Other/Static, 'guard rail' , dynamic, 23-'sky' ,'bridge', tunnel
        #    label_mapping[3] = 17 # Pole, polegroup
        #    label_mapping[4] = 7 # Road, 'road', 'parking', 6-'ground', 'railtrack'
        #    label_mapping[5] = 8 # Sidewalk
        #    label_mapping[6] = 21 # Vegetation, 'terrain'
        #    label_mapping[7] = 12 # Wall
        #    label_mapping[8] = 20 # Traffic sign, traffic light
        # Go through all episodes in a gradient batch.
        labels_indx = {11: 0, 13: 1, 14: 2, 4: 2, 5: 2, 15: 2, 16: 2, 17: 3, 18: 3, 7: 4, 9: 4, 6: 4, 10: 4,
                       8: 5, 21: 6, 22: 6, 12: 7, 20: 8, 19: 8,34 :9, 35:10}
        jointsParents=[]
        poseData=[]
        for agent in trainablePedestrians:
            if agent.PFNN:
                poseData.append(agent.PFNN.getCurrentPose())  # Returns a numpy representing the pose
                jointsParents.append(agent.PFNN.getJointParents() ) # Returns the parent of each of the numjoints bones (same numpy)
            else:
                jointsParents.append(None)
        return jointsParents, labels_indx

    def print_reward_for_manual_inspection(self, episode, frame, img_from_above, jointsParents):
        print(("Reward for frame " + str(frame)) )
        reward = episode.calculate_reward(frame, episode_done=False, print_reward_components=True)  # Return to this!
        for id in range(len(episode.pedestrian_data)):
            print(("Reward \t" + " " + str(reward[id])) + " goal " + str(episode.pedestrian_data[id].goal))

        if episode.use_car_agent:
            for id in range(len(episode.car_data)):
                print (" Car reward  "+str(episode.car_data[id].reward_car[frame]))

    def plot_for_manual_inspection(self, episode, frame, img_from_above, jointsParents):
        if frame % self.settings.action_freq == 0:
            agent_dead=True
            for id in range(len(episode.pedestrian_data)):
                agent_dead= episode.pedestrian_data[id].measures[frame, PEDESTRIAN_MEASURES_INDX.agent_dead] and agent_dead
            for id in range(len(episode.car_data)):
                agent_dead = episode.car_data[id].measures_car[ frame, PEDESTRIAN_MEASURES_INDX.agent_dead] and agent_dead
            if not agent_dead:
                make_movie_manual(episode.people,
                                  episode.reconstruction,
                                  self.settings.width,
                                  self.settings.depth,
                                  episode.seq_len,
                                  self.settings.agent_shape,
                                  self.settings.agent_shape_s,
                                  episode,
                                  frame + 1,
                                  img_from_above,
                                  name_movie='manual_'+str(id)+'_agent',
                                  path_results=self.settings.LocalResultsPath,
                                  jointsParents=jointsParents)

    def print_agent_location(self, episode, frame, value, id, car):
        if car:
            agent_name=" car "
            pos_next = episode.car_data[id].car[frame]
            pos = episode.car_data[id].car[frame - 1]
        else:
            agent_name = " agent "
            pos_next = episode.pedestrian_data[id].agent[frame]
            pos = episode.pedestrian_data[id].agent[frame - 1]


        diff=pos_next-pos


        print("Frame " + str(frame)+agent_name+str(id)+" took action: " + str(value) + "true diff "+str(diff) +" Old position: " + str(pos)+ " New position: " + str(pos_next))

    def print_input_for_manual_inspection(self, agent, episode, frame, img_from_above, jointsParents, labels_indx, training):

        if True:
            print("Make movie frame : "+str(frame)+" -------------------------------------------------------")

            make_movie_manual( episode.people,
                              episode.reconstruction,
                              self.settings.width,
                              self.settings.depth,
                              episode.seq_len,
                              self.settings.agent_shape,
                              self.settings.agent_shape_s,
                              episode,
                              frame,
                              img_from_above,
                              name_movie='manual_',
                              path_results=self.settings.LocalResultsPath,
                              jointsParents=jointsParents)
            if True:
                for agent in self.all_pedestrian_agents:
                    print("Agent training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -------------------------------->")

                    training_tmp = True
                    input, people, cars, people_cv, cars_cv = episode.get_agent_neighbourhood(agent.position,
                                                                                              self.settings.agent_s,
                                                                                              frame,
                                                                                              training_tmp, pedestrian_view_occluded=self.settings.pedestrian_view_occluded, field_of_view=self.settings.field_of_view)
                    if self.settings.run_2D:
                        sem = np.zeros((people.shape[0], people.shape[1], 2))
                    else:
                        sem = np.zeros((people.shape[1], people.shape[2], 2))
                    if self.settings.old:

                        sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], 9 + 7))

                        segmentation = (input[:, :, :, 3] * 33.0).astype(np.int)
                        for x in range(sem.shape[0]):
                            for y in range(sem.shape[1]):
                                sem[x, y, 0] = np.max(input[:, x, y, 0])
                                sem[x, y, 1] = np.max(input[:, x, y, 1])
                                sem[x, y, 2] = np.max(input[:, x, y, 2])
                                if training:
                                    sem[x, y, 3] = self.settings.people_traj_gamma * np.max(input[:, x, y, 4])
                                    sem[x, y, 4] = self.settings.people_traj_gamma * np.max(input[:, x, y, 5])

                                sem[x, y, 5] = np.max(people[:, x, y, 0])
                                sem[x, y, 6] = np.max(cars[:, x, y, 0])
                                for label in segmentation[:, x, y]:
                                    if label > 0 and label != 23:
                                        sem[x, y, labels_indx[label] + 7] = 1
                    else:
                        sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], 9 + 7))
                        segmentation = (input[:, :, 3] * 33.0).astype(np.int)
                        for x in range(sem.shape[0]):
                            for y in range(sem.shape[1]):

                                sem[x, y, 0] = input[x, y, 0]
                                sem[x, y, 1] = input[x, y, 1]
                                sem[x, y, 2] = input[x, y, 2]
                                if self.settings.predict_future:
                                    # if training:
                                    #     sem[x, y, 3] =tensor[self.return_highest_object(tensor[:,x,y,4]), x, y, 4]
                                    #     sem[x, y, 4] =tensor[self.return_highest_object(tensor[:,x,y,5]), x, y, 5]
                                    # else:
                                    sem[x, y, 3] = people_cv[x, y, 0]
                                    sem[x, y, 4] = cars_cv[x, y, 0]
                                elif training or self.settings.temporal:
                                    sem[x, y, 3] = self.settings.people_traj_gamma * input[x, y, 4]
                                    sem[x, y, 4] = self.settings.people_traj_gamma * input[x, y, 5]

                                sem[x, y, 5] = np.max(people[x, y, 0])
                                sem[x, y, 6] = np.max(cars[x, y, 0])

                                if segmentation[x, y] > 0 and segmentation[x, y] != 23 and segmentation[ x, y]in labels_indx:
                                    sem[x, y, labels_indx[segmentation[x, y]] + 7] = 1
                    print(("Input cars " + str(np.sum(sem[:, :, 5])) + " people " + str(
                        np.sum(sem[:, :, 6])) + " Input cars traj" + str(np.sum(sem[:, :, 4])) + " people traj " + str(
                        np.sum(sem[:, :, 3]))))
                    # print "Output shape: "+str(sem[:, :, 0].shape)

                    #    label_mapping[0] = 11 # Building
                    #    label_mapping[1] = 13 # Fence
                    #    label_mapping[2] = 4 # Other/Static, 'guard rail' , dynamic, 23-'sky' ,'bridge', tunnel
                    #    label_mapping[3] = 17 # Pole, polegroup
                    #    label_mapping[4] = 7 # Road, 'road', 'parking', 6-'ground', 'railtrack'
                    #    label_mapping[5] = 8 # Sidewalk
                    #    label_mapping[6] = 21 # Vegetation, 'terrain'
                    #    label_mapping[7] = 12 # Wall
                    #    label_mapping[8] = 20 # Traffic sign, traffic light

                    print(("Input building " + str(np.sum(sem[:, :, 7])) + " fence " + str(
                        np.sum(sem[:, :, 7 + 1])) + " static " + str(np.sum(sem[:, :, 7 + 2])) + " pole " + str(
                        np.sum(sem[:, :, 7 + 3]))))
                    print(("Input sidewalk " + str(np.sum(sem[:, :, 7 + 5])) + " road " + str(
                        np.sum(sem[:, :, 7 + 4])) + " veg. " + str(
                        np.sum(sem[:, :, 7 + 6])) + " wall " + str(np.sum(sem[:, :, 7 + 7])) + " sign " + str(
                        np.sum(sem[:, :, 7 + 8]))))

                    # for x in range(sem.shape[0]):
                    #     for y in range(sem.shape[1]):
                    #         if self.settings.predict_future:
                    #             if training_tmp:
                    #                 if not self.settings.run_2D:
                    #                     sem[x, y, 0] = input[self.return_highest_object(input[:, x, y, 4]), x, y, 4]
                    #                     sem[x, y, 1] = input[self.return_highest_object(input[:, x, y, 5]), x, y, 5]
                    #                 else:
                    #                     sem[x, y, 0] = input[ x, y, 4]
                    #                     sem[x, y, 1] = input[ x, y, 5]
                    #             else:
                    #                 if not self.settings.run_2D:
                    #                     sem[x, y, 0] = people_cv[self.return_highest_object(people_cv[:, x, y, 0]),x, y, 0]
                    #                     sem[x, y, 1] = cars_cv[self.return_highest_object(cars_cv[:, x, y, 0]),x, y, 0]
                    #                 else:
                    #                     sem[x, y, 0] = people_cv[x, y, 0]
                    #                     sem[x, y, 1] = cars_cv[ x, y, 0]
                    #         elif training_tmp:
                    #             if not self.settings.run_2D:
                    #                 sem[x, y, 0] = self.settings.people_traj_gamma * input[self.return_highest_object(input[:, x, y, 4]),x, y, 4]
                    #                 sem[x, y, 1] = self.settings.people_traj_gamma * input[self.return_highest_object(input[:, x, y, 4]),x, y, 5]
                    #             else:
                    #                 sem[x, y, 0] = self.settings.people_traj_gamma * input[ x, y, 4]
                    #                 sem[x, y, 1] = self.settings.people_traj_gamma * input[ x, y, 5]



                    # print "Input cars " + str(sem[:,:,1])

                    # print "Input people " + str(sem[:,:,5])

                    # print "Agent testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
                    # training_tmp=False
                    # input, people, cars, people_cv, cars_cv = episode.get_agent_neighbourhood(agent.position,
                    #                                                                           self.settings.agent_s,
                    #                                                                           frame,
                    #                                                                           training_tmp)
                    # #print "Output shape: " + str(sem[:, :, 0].shape)
                    # if self.settings.run_2D:
                    #     sem = np.zeros((people.shape[0], people.shape[1], 2))
                    # else:
                    #     sem = np.zeros((people.shape[1], people.shape[2], 2))
                    # for x in range(sem.shape[0]):
                    #     for y in range(sem.shape[1]):
                    #         if self.settings.predict_future:
                    #             if training_tmp:
                    #                 if not self.settings.run_2D:
                    #                     sem[x, y, 0] = input[self.return_highest_object(input[:, x, y, 4]), x, y, 4]
                    #                     sem[x, y, 1] = input[self.return_highest_object(input[:, x, y, 5]), x, y, 5]
                    #                 else:
                    #                     sem[x, y, 0] = input[ x, y, 4]
                    #                     sem[x, y, 1] = input[ x, y, 5]
                    #             else:
                    #                 if not self.settings.run_2D:
                    #                     sem[x, y, 0] = people_cv[self.return_highest_object(people_cv[:, x, y, 0]),x, y, 0]
                    #                     sem[x, y, 1] = cars_cv[self.return_highest_object(cars_cv[:, x, y, 0]),x, y, 0]
                    #                 else:
                    #                     sem[x, y, 0] = people_cv[x, y, 0]
                    #                     sem[x, y, 1] = cars_cv[ x, y, 0]
                    #         elif training_tmp:
                    #             if not self.settings.run_2D:
                    #                 sem[x, y, 0] = self.settings.people_traj_gamma * input[self.return_highest_object(input[:, x, y, 4]),x, y, 4]
                    #                 sem[x, y, 1] = self.settings.people_traj_gamma * input[self.return_highest_object(input[:, x, y, 5]),x, y, 5]
                    #             else:
                    #                 sem[x, y, 0] = self.settings.people_traj_gamma * input[ x, y, 4]
                    #                 sem[x, y, 1] = self.settings.people_traj_gamma * input[ x, y, 5]


                    goal_dir = episode.get_goal_dir(episode.agent[frame], episode.goal[0,:])

                    print(("Goal_dir " + str(goal_dir)))

                    if frame==0:
                        vel=episode.vel_init
                    else:
                        vel=episode.velocity[frame-1]

                    car_in = episode.get_input_cars(episode.agent[frame], frame, vel, self.settings.field_of_view)

                    print(("Car dir " + str(car_in)))

                    agent_frame = frame
                    if self.settings.action_mem:
                        values = np.zeros((9 + 1) * self.settings.action_mem)
                        if self.settings.old:
                            for past_frame in range(1, self.settings.action_mem + 1):
                                if agent_frame - past_frame >= 0:
                                    # print str(len(episode.action))+" "+str(max(agent_frame - past_frame, 0))
                                    # print (past_frame-1)*(9+1)+int(episode.action[max(agent_frame - past_frame, 0)])
                                    # print values.shape
                                    values[(past_frame - 1) * (9 + 1) + int(
                                        episode.action[max(agent_frame - past_frame, 0)])] = 1
                                    values[past_frame * past_frame - 1] = episode.measures[
                                        max(agent_frame - past_frame, 0), 3]
                            print(("Actions memory:" + str(np.sum(values))))
                            if self.settings.nbr_timesteps > 0:
                                values = np.zeros((9 + 1) * self.settings.nbr_timesteps)
                                for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 1):
                                    if agent_frame - past_frame >= 0:
                                        # print str(len(episode.action))+" "+str(max(agent_frame - past_frame, 0))
                                        # print (past_frame-1)*(9+1)+int(episode.action[max(agent_frame - past_frame, 0)])
                                        # print values.shape
                                        values[(past_frame - 1) * (9 + 1) + int(
                                            episode.action[max(agent_frame - past_frame, 0)])] = 1
                                        values[past_frame * past_frame - 1] = episode.measures[
                                            max(agent_frame - past_frame, 0), 3]
                                print(("Actions adiitional memory:" + str(np.sum(values))))
                        else:
                            for past_frame in range(1, self.settings.action_mem + 1):
                                if frame - past_frame >= 0:
                                    pos = (past_frame - 1) * (9 + 1) + int(
                                        episode.action[max(frame - past_frame, 0)])
                                    values[pos] = 1
                                    values[past_frame * (9 + 1) - 1] = episode.measures[
                                        max(frame - past_frame, 0), 3]

                            print(("Actions memory:" + str(np.sum(values))))
                            if self.settings.nbr_timesteps > 0:
                                values = np.zeros((9 + 1) * self.settings.nbr_timesteps)
                                for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 1):
                                    if frame - past_frame >= 0:
                                        values[(past_frame - 2) * (9 + 1) + int(
                                            episode.action[max(frame - past_frame, 0)])] = 1
                                        values[(past_frame - 1) * (9 + 1) - 1] = \
                                            episode.measures[max(frame - past_frame - 1, 0), 3]
                                print(("Actions adiitional memory:" + str(np.sum(values))))
                    if self.settings.pose:
                        itr = int(episode.agent_pose_frames[agent_frame])
                        # print "Get feature Frame "+str(agent_frame)+" "+str(itr)
                        pose_in = np.expand_dims(np.array(episode.agent_pose_hidden[itr, :]) * (1 / 100.0),
                                                              axis=0)
                        if np.sum(np.abs(pose_in)) < 1e-6:
                            pose_in = np.zeros_like(pose_in)
                        print("Hidden layer: " + str(pose_in[0, :5])+" abs "+str(np.sum(np.sum(pose_in))))



class ManualCARLAEnvironment(ManualAbstractEnvironment, CARLAEnvironment):
    def __init__(self, path, sess, writer, gradBuffer, log, settings, net=None):
        super(ManualCARLAEnvironment, self).__init__(path, sess, writer, gradBuffer, log,settings, net=net)
        if self.settings.learn_init:
            self.init_methods = [7]  # [1,5]#[1, 3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train = [7]


