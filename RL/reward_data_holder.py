from RL.settings import NBR_MEASURES, PEDESTRIAN_MEASURES_INDX, PEDESTRIAN_REWARD_INDX
import numpy as np
#from initializer_abstract_data_holder import DataHolder

class RewardDataHolder(object):
    def __init__(self,  seq_len, DTYPE, valid_positions=None):
        #super(RewardDataHolder, self).__init__(seq_len, DTYPE, valid_positions=valid_positions)
        vector_len = max(seq_len, 1)
        self.measures = np.zeros((vector_len, NBR_MEASURES), dtype=DTYPE)
        self.reward = np.zeros(vector_len, dtype=DTYPE)
        self.seq_len = seq_len
        self.goal= np.zeros((vector_len, 3), dtype=DTYPE)

        self.goal_time =  np.zeros((vector_len), dtype=DTYPE)
        self.goal_person_id = -1
        self.reward = np.zeros(vector_len, dtype=DTYPE)
        self.reward_d = np.zeros(vector_len, dtype=DTYPE)
        self.init_method = 0
        self.accummulated_r = 0
        self.goal_to_agent_init_dist = -1*np.ones(vector_len, dtype=DTYPE)

    def set_values(self, agent_data): # If the values get changed- make a copy here
        self.measures=agent_data.measures
        self.goal_to_agent_init_dist =agent_data.goal_to_agent_init_dist
        self.goal_time = agent_data.goal_time
        self.goal_person_id = agent_data.goal_person_id

    # def get_original_dist_to_goal(self, frame):
    #     raise NotImplementedError("Please Implement this method")

    def calculate_reward(self, frame, multiplicative_reward, prev_mul_reward, reward_weights, print_reward_components,
                         max_step, end_on_hit_by_pedestrians, stop_on_goal):
        if multiplicative_reward and not prev_mul_reward:
            self.reward[frame] = 1
        else:
            self.reward[frame] = 0
        reward_activated = False
        # if supervised and  self.goal_person_id>0 and frame+1<len(self.valid_people_tracks[self.goal_person_id_val]):
        #     self.measures[frame, 9]=np.linalg.norm(np.mean(self.valid_people_tracks[self.goal_person_id_val][frame+1]-self.valid_people_tracks[self.goal_person_id_val][frame], axis=1)-self.velocity[frame])
        #     self.reward[frame]=-self.measures[frame, 9]
        #     self.agent[frame+1]=np.mean(self.valid_people_tracks[self.goal_person_id_val][frame+1], axis=1)
        #     return self.reward[frame]

        # Penalty for large change in poses
        # itr = self.agent_pose_frames[frame]
        # itr_prev = self.agent_pose_frames[max(frame - 1, 0)]
        # previous_pose = self.agent_pose[itr_prev, :]
        # current_pose = self.agent_pose[itr, :]
        # diff = current_pose - previous_pose
        # self.reward[frame] += reward_weights[14]*max(0, np.max(diff)-25.0)

        if reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose] != 0:
            evaluated_term = min(max(0, abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_pose]) - 1.2), 2.0)
            if multiplicative_reward:
                if abs(evaluated_term) > 1e-8:
                    multiplicative_term = 0
                    if reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose] > 0 or not prev_mul_reward:
                        multiplicative_term = reward_weights[
                                                  PEDESTRIAN_REWARD_INDX.large_change_in_pose] * evaluated_term
                        reward_activated = True
                    elif reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose] < 0:
                        multiplicative_term = reward_weights[ PEDESTRIAN_REWARD_INDX.large_change_in_pose] / evaluated_term
                        if prev_mul_reward:
                            multiplicative_term =-multiplicative_term
                        reward_activated = True
                    if abs(multiplicative_term) > 0:
                        if print_reward_components:
                            print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                (multiplicative_term + 1)) + " times R " + str(
                                self.reward[frame] * (multiplicative_term + 1)))
                        if prev_mul_reward:
                            self.reward[frame] *= multiplicative_term
                        else:
                            self.reward[frame] *= (multiplicative_term + 1)
            else:

                self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose] * min(
                    max(0, abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_pose]) - 1.2), 2.0)

            if print_reward_components:
                print("Pose measure " + str(
                    self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_pose]) + "after first threshold " + str(
                    max(0, abs(self.measures[
                                   frame, PEDESTRIAN_MEASURES_INDX.change_in_pose]) - 1.2)) + " after thresholding " + str(
                    min(max(0, abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_pose]) - 1.2),
                        2.0)) + " times " + str(
                    reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose]) + " = reward " + str(
                    reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose] * min(
                        max(0, abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_pose]) - 1.2),
                        2.0)) + " reward " + str(self.reward[frame]))

        # Penalty for hitting pedestrians
        if reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] != 0 and abs(
                self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]) > 1e-8:
            if multiplicative_reward:
                reward_activated = True
                multiplicative_term = 0
                if reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] > 0 or not prev_mul_reward:
                    multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] * \
                                          self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]
                else:
                    multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] / \
                                          self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]
                    if prev_mul_reward:
                        multiplicative_term = -multiplicative_term
                if abs(multiplicative_term) > 0:
                    if print_reward_components:
                        print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                            (multiplicative_term + 1)) + " times R " + str(
                            self.reward[frame] * (multiplicative_term + 1)))

                    if prev_mul_reward:
                        self.reward[frame] *= multiplicative_term
                    else:
                        self.reward[frame] *= (multiplicative_term + 1)

            else:
                self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] * self.measures[
                    frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]
            if print_reward_components:
                print("Penalty for hitting pedestrians " + str(
                    reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] * self.measures[
                        frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]) + " reward " + str(self.reward[frame]))

        if self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] == 0:
            if print_reward_components:
                print(" Reward has not hit pedestrians")
            # Reward for on pedestrian trajectory
            if frame <0 or self.measures[frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init] > \
                    self.measures[frame-1, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]:
                if reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] != 0 and abs(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]) > 1e-8:

                    if multiplicative_reward:
                        reward_activated = True
                        multiplicative_term = 0
                        if reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] > 0 or not prev_mul_reward:
                            multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] * \
                                                  self.measures[
                                                      frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]
                        else:
                            multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] / \
                                                  self.measures[
                                                      frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]
                            if prev_mul_reward:
                                multiplicative_term = -multiplicative_term
                        if abs(multiplicative_term) > 0:
                            if print_reward_components:
                                print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                    (multiplicative_term + 1)) + " times R " + str(
                                    self.reward[frame] * (multiplicative_term + 1)))

                            if prev_mul_reward:
                                self.reward[frame] *= multiplicative_term
                            else:
                                self.reward[frame] *= (multiplicative_term + 1)
                    else:
                        self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] * \
                                              self.measures[
                                                  frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]
                    if print_reward_components:
                        print("Add reward for on traj " + str(
                            reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] * self.measures[
                                frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]) + " reward " + str(
                            self.reward[frame]))
                if print_reward_components:
                    print("Weight " + str(reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap]) + " size " + str(
                        abs(self.measures[
                                frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]) > 1e-8) + " value " + str(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]))
                if reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] > 0:
                    if reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] != 0 and abs(
                            self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]) > 1e-8:
                        if multiplicative_reward:
                            reward_activated = True
                            multiplicative_term = 0
                            if reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] > 0 or not prev_mul_reward:
                                multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] * \
                                                      self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]
                            else:
                                multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] / \
                                                      self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]
                                if prev_mul_reward:
                                    multiplicative_term = -multiplicative_term
                            if abs(multiplicative_term) > 0:
                                if print_reward_components:
                                    print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                        (multiplicative_term + 1)) + " times R " + str(
                                        self.reward[frame] * (multiplicative_term + 1)))

                                if prev_mul_reward:
                                    self.reward[frame] *= multiplicative_term
                                else:
                                    self.reward[frame] *= (multiplicative_term + 1)
                        else:
                            self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] * \
                                                  self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]
                    if print_reward_components:
                        print("Add reward for ped heatmap " + str(
                            reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] * self.measures[
                                frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]) + " reward " + str(
                            self.reward[frame]))

            # Reward for on sidewalk
            if reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] > 0:
                if multiplicative_reward:
                    if reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] != 0 and abs(
                            self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]) > 1e-8:
                        reward_activated = True
                        multiplicative_term = 0
                        if reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] > 0 or not prev_mul_reward:
                            multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] * self.measures[
                                frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]
                        else:
                            multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] / self.measures[
                                frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]
                            if prev_mul_reward:
                                multiplicative_term = -multiplicative_term
                        if abs(multiplicative_term) > 0:
                            if print_reward_components:
                                print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                    (multiplicative_term + 1)) + " times R " + str(
                                    self.reward[frame] * (multiplicative_term + 1)))

                            if prev_mul_reward:
                                self.reward[frame] *= multiplicative_term
                            else:
                                self.reward[frame] *= (multiplicative_term + 1)
                else:
                    self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] * self.measures[
                        frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]
                if print_reward_components:
                    print("Add reward for sidewalk " + str(
                        reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] * self.measures[
                            frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]) + " reward " + str(self.reward[frame]))
        # Penalty for hitting objects
        if self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles] > 0:
            if reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] != 0 and abs(
                    self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]) > 1e-8:
                if multiplicative_reward:
                    reward_activated = True
                    multiplicative_term = 0
                    if reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] > 0 or not prev_mul_reward:
                        multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] * \
                                              self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                    else:
                        multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] / \
                                              self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                        if prev_mul_reward:
                            multiplicative_term = -multiplicative_term
                        # print ("Add reward for hitting objs " + str(reward_weights[3] * self.measures[frame, 3]))
                    if abs(multiplicative_term) > 0:
                        if print_reward_components:
                            print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                (multiplicative_term + 1)) + " times R " + str(
                                self.reward[frame] * (multiplicative_term + 1)))

                        if prev_mul_reward:
                            self.reward[frame] *= multiplicative_term
                        else:
                            self.reward[frame] *= (multiplicative_term + 1)
                else:
                    self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] * self.measures[
                        frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                if print_reward_components:
                    print("Add reward for hitting objs " + str(
                        reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] * self.measures[
                            frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]) + " reward " + str(self.reward[frame]))

        # Distance travelled:

        if np.linalg.norm(reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled]) > 0:
            # Regular positive reward for distance travelled
            if print_reward_components:
                print("Frame " + str(frame) + " equal " + str(self.seq_len - 2))
            if frame == (self.seq_len - 2):
                if print_reward_components:
                    print("Pavement  " + str(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]) + " out of axis " + str(reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis]) + " hit by car " + str(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]))
                if self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement] > 0 and not reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis]>0 and self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car] == 0:
                    if multiplicative_reward:

                        if reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] != 0 and abs(
                                self.measures[frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]) > 1e-8:
                            reward_activated = True
                            multiplicative_term = 0
                            if reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] > 0 or not prev_mul_reward:
                                multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] * \
                                                      self.measures[
                                                          frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]  # /(self.seq_len*np.sqrt(2))
                            else:
                                multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] / \
                                                      self.measures[
                                                          frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]  # /(self.seq_len*np.sqrt(2))
                                if prev_mul_reward:
                                    multiplicative_term = -multiplicative_term
                            if abs(multiplicative_term) > 0:
                                if print_reward_components:
                                    print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                        (multiplicative_term + 1)) + " times R " + str(
                                        self.reward[frame] * (multiplicative_term + 1)))

                                if prev_mul_reward:
                                    self.reward[frame] *= multiplicative_term
                                else:
                                    self.reward[frame] *= (multiplicative_term + 1)
                    else:
                        self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] * self.measures[
                            frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]
                    if print_reward_components:
                        print("Reward for dist travelled" + str(
                            reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] * self.measures[
                                frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]) + " reward " + str(
                            self.reward[frame]))

        # Negative reward for relative distance not travelled in flight-distance.

        if np.linalg.norm(reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance]) > 0:
            local_measure = 0
            if self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_final_pos] > 0.5:
                local_measure = (self.measures[frame, PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init] /
                                 self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_final_pos] - 1)
                if print_reward_components:
                    print("Add dist travelled total " + str(self.measures[
                                                                frame, PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init]) + " distance on the fly " + str(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_final_pos]) + " ratio " + str(
                        1 - local_measure) + " 1-ratio " + str(local_measure) + " reward " + str(reward_weights[
                                                                                                     PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] * local_measure) + " reward " + str(
                        self.reward[frame]))

            else:
                local_measure = 2  # penalty for standing still
                if print_reward_components:
                    print("Add penalty for standing still " + str(reward_weights[
                                                                      PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] * local_measure) + " reward " + str(
                        self.reward[frame]))

            if multiplicative_reward:

                if reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] != 0 and abs(
                        local_measure) > 1e-8:
                    reward_activated = True
                    multiplicative_term = 0
                    if reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] < 0 or not prev_mul_reward:
                        multiplicative_term = reward_weights[
                                                  PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] * local_measure
                    else:
                        multiplicative_term = reward_weights[
                            PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] / local_measure
                        if prev_mul_reward:
                            multiplicative_term = -multiplicative_term
                    if abs(multiplicative_term) > 0:
                        if print_reward_components:
                            print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                (multiplicative_term + 1)) + " times R " + str(
                                self.reward[frame] * (multiplicative_term + 1)))

                        if prev_mul_reward:
                            self.reward[frame] *= multiplicative_term
                        else:
                            self.reward[frame] *= (multiplicative_term + 1)
            else:
                self.reward[frame] += reward_weights[
                                          PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] * local_measure
            if print_reward_components:
                print(" reward " + str(self.reward[frame]))
        # Agent out of axis?
        if reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] != 0 and abs(
                self.measures[frame, PEDESTRIAN_MEASURES_INDX.out_of_axis]) > 1e-8:
            if multiplicative_reward:
                reward_activated = True
                multiplicative_term = 0
                if reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] > 0 or not prev_mul_reward:
                    multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] * self.measures[
                        frame, PEDESTRIAN_MEASURES_INDX.out_of_axis]
                else:
                    multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] / self.measures[
                        frame, PEDESTRIAN_MEASURES_INDX.out_of_axis]
                    if prev_mul_reward:
                        multiplicative_term = -multiplicative_term
                if abs(multiplicative_term) > 0:
                    if print_reward_components:
                        print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                            (multiplicative_term + 1)) + " times R " + str(
                            self.reward[frame] * (multiplicative_term + 1)))

                    if prev_mul_reward:
                        self.reward[frame] *= multiplicative_term
                    else:
                        self.reward[frame] *= (multiplicative_term + 1)
            else:
                self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] * self.measures[
                    frame, PEDESTRIAN_MEASURES_INDX.out_of_axis]
            if print_reward_components:
                print("Add Out of axis " + str(reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] * self.measures[
                    frame, PEDESTRIAN_MEASURES_INDX.out_of_axis]) + " reward " + str(self.reward[frame]))

        # % Distance travelled towards goal
        # If pos reward and initial distance to goal is greater than 0.
        # print "Reward before goal terms : "+str( self.reward[frame])+" initial distance to goal: "+str(np.linalg.norm(self.measures[frame, 6]))
        if (reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] > 0 or reward_weights[
            PEDESTRIAN_REWARD_INDX.reached_goal] > 0) > 0:
            if self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] ==0:#= self.measures[max(frame-1, 0), PEDESTRIAN_MEASURES_INDX.goal_reached]:
                # print ("Frame in reward " + str(frame))
                local_measure = 0
                if frame == 0 or self.measures[frame-1, PEDESTRIAN_MEASURES_INDX.goal_reached]:
                    orig_dist = self.get_original_dist_to_goal(frame)
                    # print " Reward for one step: " + str(reward_weights[6] * (
                    # 1 - (self.measures[frame, 7] /orig_dist))) + " quotient: " + str(
                    #     (self.measures[frame, 7] / orig_dist)) + " dist cur" + str(
                    #     self.measures[frame, 7]) + " dist prev: " + str(orig_dist)
                    local_measure = (
                            (orig_dist - self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) / max_step)
                    if print_reward_components:
                        print(" Reward for one step to goal frame 0: " + str(reward_weights[
                                                                                 PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] * local_measure) + " dist prev: " + str(
                            orig_dist) + " cur dist  " + str(
                            self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) + " reward " + str(
                            self.reward[frame]))
                    # print " Difference "+str((orig_dist-self.measures[frame, 7]))+" max_step "+str(max_step)+" ration "+str((orig_dist-self.measures[frame, 7])/max_step)
                elif frame > 0 and self.measures[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] > 0 or not prev_mul_reward:
                    local_measure = ((self.measures[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] - self.measures[
                        frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) / max_step)
                    if print_reward_components:
                        print(" Reward for one step to goal : " + str(reward_weights[
                                                                          PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] * local_measure) + " diff: " + str(
                            (self.measures[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] - self.measures[
                                frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal])) + " dist cur" + str(
                            self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) + " dist prev: " + str(
                            self.measures[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) + " reward " + str(
                            self.reward[frame]))

                    # print " Difference " + str((self.measures[frame-1, 7] - self.measures[frame, 7])) + " max_step " + str(
                    #     max_step) + " ration " + str((self.measures[frame-1, 7] - self.measures[frame, 7]) / max_step)
                if multiplicative_reward:
                    if print_reward_components:
                        print(" Reward for one step to goal  enter if? weigt " + str(reward_weights[
                                                                                         PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal]) + " local measure " + str(
                            local_measure))
                    if reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] != 0 and abs(
                            local_measure) > 1e-8:
                        reward_activated = True
                        multiplicative_term = 0
                        if reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] > 0 or not prev_mul_reward:
                            multiplicative_term = reward_weights[
                                                      PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] * local_measure
                        else:
                            multiplicative_term = reward_weights[
                                PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] / local_measure
                            if prev_mul_reward:
                                multiplicative_term = -multiplicative_term
                        if abs(multiplicative_term) > 0:

                            if print_reward_components:
                                print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                    (multiplicative_term + 1)) + " times R " + str(
                                    self.reward[frame] * (multiplicative_term + 1)))
                            if prev_mul_reward:
                                self.reward[frame] *= multiplicative_term
                            else:
                                self.reward[frame] *= (multiplicative_term + 1)

                else:
                    self.reward[frame] += reward_weights[
                                              PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] * local_measure
                if print_reward_components:
                    print(" reward " + str(self.reward[frame]))
            else:

                if multiplicative_reward:
                    reward_activated = True

                    self.reward[frame] *= reward_weights[PEDESTRIAN_REWARD_INDX.reached_goal]
                    if self.goal_time[frame] > 0:
                        temp_diff = min(abs(self.goal_time[frame] - frame) / self.goal_time[frame], 1)
                        if reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] != 0 and abs(
                                self.measures[frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time]) > 1e-8:
                            multiplicative_term = 0
                            if reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] > 0 or not prev_mul_reward:
                                multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] * \
                                                      self.measures[
                                                          frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time]
                            else:
                                multiplicative_term = reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] / \
                                                      self.measures[
                                                          frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time]
                                if prev_mul_reward:
                                    multiplicative_term = -multiplicative_term
                            if abs(multiplicative_term) > 0:
                                if print_reward_components:
                                    print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                        (multiplicative_term + 1)) + " times R " + str(
                                        self.reward[frame] * (multiplicative_term + 1)))

                                if prev_mul_reward:
                                    self.reward[frame] *= multiplicative_term
                                else:
                                    self.reward[frame] *= (multiplicative_term + 1)
                        # print (" Reward for reaching goal "+str(reward_weights[8])+" minus "+str(reward_weights[13]*temp_diff)+" diff "+str(temp_diff))
                else:
                    self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.reached_goal]
                    if self.goal_time[frame] > 0:
                        temp_diff = min(abs(self.goal_time[frame] - frame) / self.goal_time[frame], 1)

                        self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] * \
                                              self.measures[frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time]
                if print_reward_components:
                    print("Reward for reaching goal " + str(
                        reward_weights[PEDESTRIAN_REWARD_INDX.reached_goal]) + " reward " + str(self.reward[frame]))
        # Following correct pedestrian -if initialized on a pedestrian. Negative reward when not follwoing.
        if self.goal_person_id >= 0 and abs(reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] )> 0:
            denominator = (frame + 1) * max_step # *2
            # print (" Following pedestrian: Denominator " + str(denominator)+" measure "+str(self.measures[frame, 9])+" frame "+str(frame))
            fraction = 1 - (self.measures[frame, PEDESTRIAN_MEASURES_INDX.one_step_prediction_error] / denominator)
            if multiplicative_reward:
                if reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] != 0 and abs(fraction) > 1e-8:
                    reward_activated = True
                    multiplicative_term = 0
                    if reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] > 0 or not prev_mul_reward:
                        multiplicative_term = reward_weights[
                                                  PEDESTRIAN_REWARD_INDX.one_step_prediction_error] * fraction
                    else:
                        multiplicative_term = reward_weights[
                            PEDESTRIAN_REWARD_INDX.one_step_prediction_error] / fraction
                        if prev_mul_reward:
                            multiplicative_term = -multiplicative_term
                    if abs(multiplicative_term) > 0:
                        if print_reward_components:
                            print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                (multiplicative_term + 1)) + " times R " + str(
                                self.reward[frame] * (multiplicative_term + 1)))

                        if prev_mul_reward:
                            self.reward[frame] *= multiplicative_term
                        else:
                            self.reward[frame] *= (multiplicative_term + 1)
            else:
                self.reward[frame] += reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] * fraction
            if print_reward_components:
                print("Reward " + str(reward_weights[11] * (1 - fraction)) + " " + str(1 - fraction) + " " + str(
                    fraction) + " " + str(self.measures[frame, 9]) + " " + str(denominator) + " reward " + str(
                    self.reward[frame]))

        # Penalty for changing moving directions
        if reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] < 0:
            if multiplicative_reward:
                if reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] != 0 and abs(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction]) > 1e-8:
                    reward_activated = True
                    multiplicative_term = 0
                    if reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] > 0 or not prev_mul_reward:
                        multiplicative_term = self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction] * \
                                              reward_weights[
                                                  PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently]
                    else:
                        multiplicative_term = reward_weights[
                                                  PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] / \
                                              self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction]
                        if prev_mul_reward:
                            multiplicative_term = -multiplicative_term
                    if abs(multiplicative_term) > 0:
                        if print_reward_components:
                            print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                (multiplicative_term + 1)) + " times R " + str(
                                self.reward[frame] * (multiplicative_term + 1)))

                        if prev_mul_reward:
                            self.reward[frame] *= multiplicative_term
                        else:
                            self.reward[frame] *= (multiplicative_term + 1)
            else:
                self.reward[frame] += self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction] * \
                                      reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently]
            if print_reward_components:
                print("Reward Changing direction " + str(
                    self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction] * reward_weights[
                        PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently]) + " reward " + str(
                    self.reward[frame]))

        # Hitting cars?
        if self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car] > 0:
            if abs(reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car_agent]) > 0 and self.measures[
                frame, PEDESTRIAN_MEASURES_INDX.hit_by_hero_car] > 0 :
                reward_activated = True
                self.reward[frame] = reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car_agent] * self.measures[
                    frame, PEDESTRIAN_MEASURES_INDX.hit_by_hero_car]
                if multiplicative_reward and not prev_mul_reward:
                    self.reward[frame] += 1
            else:
                reward_activated = True
                self.reward[frame] = reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car] * self.measures[
                    frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]
                if multiplicative_reward and not prev_mul_reward:
                    self.reward[frame] += 1
            if print_reward_components:
                print("Penalty collision with car " + str(
                    reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car] * self.measures[
                        frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]) + " reward " + str(self.reward[frame]))

        # reward reciprocal distance to car
        if reward_weights[PEDESTRIAN_REWARD_INDX.inverse_dist_to_car] > 0 or not prev_mul_reward:
            reward_activated = True
            self.reward[frame] = reward_weights[PEDESTRIAN_REWARD_INDX.inverse_dist_to_car] * self.measures[
                frame, PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car]
            if multiplicative_reward and not prev_mul_reward:
                self.reward[frame] += 1
            if print_reward_components:
                print("Reward reciprocal distance to car " + str(
                    reward_weights[PEDESTRIAN_REWARD_INDX.inverse_dist_to_car] * self.measures[
                        frame, PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car]) + " reward " + str(
                    self.reward[frame]))

        # Correct reward! when hitting a car!

        # print "Hit by car? "+str(self.measures[frame, 0])


        if  self.measures[frame,PEDESTRIAN_MEASURES_INDX.agent_dead]:#hit_by_car or reached_goal:
            if print_reward_components:
                print("Agent dead reward 0")
                print("Reached goal " + str(
                    self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached]) + " hit by car " + str(
                    self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached]))
                if frame > 0:
                    print("Reached goal prev " + str(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] - 1) + " hit by car prev " + str(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] - 1))

            self.reward[frame] = 0

        hit_pedestrian = (
                self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] > 0 and frame > 0 and self.measures[
            frame - 1, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] > 0)

        if end_on_hit_by_pedestrians and hit_pedestrian:
            if print_reward_components:
                print("Agent dead hit by pedestrians")
            self.reward[frame] = 0

        if multiplicative_reward and not prev_mul_reward:
            if not reward_activated or self.measures[frame,PEDESTRIAN_MEASURES_INDX.agent_dead] :
                self.reward[frame] = 0
            elif self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached]==0:#= self.measures[max(frame-1, 0), PEDESTRIAN_MEASURES_INDX.goal_reached]:
                self.reward[frame] = self.reward[frame] - 1
        # else:
        #     self.reward[frame] = self.reward[frame]-0.01
        # print "Intercept car "+str(self.intercept_car(frame , all_frames=False))

        # if frame % 10==0:
        #     print "Frame "+str(frame)+" Reward: "+str(self.reward[frame])+" hit by car: "+str(self.intercept_car(frame + 1, all_frames=False))+"  Reached goal "+str(self.measures[frame, 13])+" agent "+str(self.agent[frame+1])
        return self.reward[frame]

    def discounted_reward(self, gamma, frame):
        self.accummulated_r = 0
        for frame_n in range(frame-1, -1, -1):
            tmp = self.accummulated_r * gamma
            self.accummulated_r = tmp
            self.accummulated_r += self.reward[frame_n]
            self.reward_d[frame_n] = self.accummulated_r
