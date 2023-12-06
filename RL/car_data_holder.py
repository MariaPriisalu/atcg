
import numpy as np
from RL.settings import DEFAULT_SKIP_RATE_ON_EVALUATION_CARLA, DEFAULT_SKIP_RATE_ON_EVALUATION_WAYMO, PEDESTRIAN_MEASURES_INDX, CAR_MEASURES_INDX,PEDESTRIAN_INITIALIZATION_CODE,PEDESTRIAN_REWARD_INDX, NBR_MEASURES, NBR_MEASURES_CAR,STATISTICS_INDX, STATISTICS_INDX_POSE, STATISTICS_INDX_CAR,STATISTICS_INDX_CAR_INIT, STATISTICS_INDX_MAP, RANDOM_SEED_NP, RANDOM_SEED, CAR_REWARD_INDX


class CarDataHolder(object):

    def __init__(self,seq_len, DTYPE, valid_positions=None):
        # Car variables
        #super(CarDataHolder, self).__init__(seq_len, DTYPE, valid_positions=valid_positions)
        self.car_goal = []

        vector_len = max(seq_len, 1)
        self.measures_car = np.zeros((vector_len, NBR_MEASURES_CAR), dtype=DTYPE)
        self.car = [[] for _ in range(seq_len)]
        self.car[0] = np.zeros(1 * 3, dtype=DTYPE)
        self.velocity_car = [[]] * (vector_len)
        # self.velocity_car = np.zeros((vector_len,3), dtype=self.DTYPE)
        self.speed_car = np.zeros(vector_len, dtype=DTYPE)
        self.action_car = np.zeros(vector_len, dtype=DTYPE)
        self.reward_car = np.zeros(vector_len, dtype=DTYPE)
        self.reward_car_d = np.zeros(vector_len, dtype=DTYPE)
        self.accummulated_r_car = 0
        self.loss_car = np.zeros(vector_len, dtype=DTYPE)
        self.probabilities_car = np.zeros((vector_len, 2), dtype=DTYPE)
        self.car_dir = np.zeros(3, dtype=DTYPE)

        self.car_bbox = [None] * seq_len
        self.car_angle = np.zeros(vector_len, dtype= DTYPE)
        self.seq_len = seq_len
        self.supervised_car_vel=[]
        self.goal_to_agent_init_dist =-1
        self.on_car=False
        self.external_car_vel = [[]] * (vector_len)
        self.closest_car = [None] * seq_len
        self.angle = [None] * seq_len

    def get_original_dist_to_goal(self, frame):
        if self.goal_to_agent_init_dist<0:
            self.goal_to_agent_init_dist=np.lianlg.norm(self.car[0][1:]-self.car_goal[1:])
        return self.goal_to_agent_init_dist

    def discounted_reward(self, gamma, frame):
        self.accummulated_r_car = 0
        for frame_n in range(frame-1, -1, -1):
            tmp = self.accummulated_r_car * gamma
            self.accummulated_r_car = tmp
            self.accummulated_r_car += self.reward_car[frame_n]
            self.reward_car_d[frame_n] = self.accummulated_r_car
    
    def calculate_reward(self, frame, reward_weights, car_reference_speed,car_max_speed_voxelperframe , allow_car_to_live_through_collisions):
        #max_speed=70000/3600*5/17 # 70 km/h
        self.reward_car[frame]=0
        if self.measures_car[frame, CAR_MEASURES_INDX.agent_dead]:
            return
        self.measures_car[frame,CAR_MEASURES_INDX.distance_travelled_from_init] = np.linalg.norm(self.car[frame + 1] - self.car[0])
        if np.abs(reward_weights[CAR_REWARD_INDX.distance_travelled])>0:
            self.reward_car[frame]=reward_weights[CAR_REWARD_INDX.distance_travelled]*self.measures_car[frame, CAR_MEASURES_INDX.distance_travelled_from_init]/(car_max_speed_voxelperframe*(frame+1))

        if allow_car_to_live_through_collisions:
            # print (" Penalty for collision ?"+str(self.measures_car[frame, CAR_MEASURES_INDX.hit_by_agent]))
            self.reward_car[frame]+= reward_weights[CAR_REWARD_INDX.collision_pedestrian_with_car] *self.measures_car[frame, CAR_MEASURES_INDX.hit_by_agent]
        else:
            self.reward_car[frame] +=reward_weights[CAR_REWARD_INDX.collision_pedestrian_with_car]*self.measures_car[frame,CAR_MEASURES_INDX.hit_pedestrians]
            self.reward_car[frame] += reward_weights[CAR_REWARD_INDX.collision_car_with_car] * \
                                  self.measures_car[frame, CAR_MEASURES_INDX.hit_by_car]


            self.reward_car[frame] += reward_weights[CAR_REWARD_INDX.collision_car_with_objects] * \
                                  self.measures_car[frame, CAR_MEASURES_INDX.hit_obstacles]


            self.reward_car[frame] += reward_weights[CAR_REWARD_INDX.penalty_for_intersection_with_sidewalk] * \
                                  self.measures_car[frame, CAR_MEASURES_INDX.iou_pavement]
        self.reward_car[frame] += reward_weights[CAR_REWARD_INDX.penalty_for_speeding] * max(self.speed_car[frame]-car_reference_speed, 0)
        # print(" Car speed "+str(self.speed_car[frame])+" refernce "+str(car_reference_speed)+" diffrenece to reference "+str(self.speed_car[frame]-car_reference_speed)+" reward "+str(reward_weights[CAR_REWARD_INDX.penalty_for_speeding] * max(self.speed_car[frame]-car_reference_speed, 0)))
        if abs(reward_weights[CAR_REWARD_INDX.reached_goal] )>0:
            if self.measures_car[frame, CAR_MEASURES_INDX.goal_reached] == 0:
                local_measure = 0
                if frame == 0:
                    orig_dist = np.linalg.norm(np.array(self.car_goal[1:]) - self.car[0][1:])
                    local_measure = ((orig_dist - self.measures_car[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) / car_max_speed_voxelperframe)
                    # print (" Frame is zero. initial dist to goal "+str(orig_dist)+" current dist "+str(self.measures_car[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal])+" diff "+str((orig_dist - self.measures_car[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) )+" normalized by speed "+str(local_measure) +" normalized by "+str(max_speed))
                if frame > 0 and self.measures_car[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] > 0:
                    local_measure = ((self.measures_car[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] - self.measures_car[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) / car_max_speed_voxelperframe)
                    # print ("Previous dist to goal " + str(self.measures_car[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] ) + " current dist " + str(
                    #     self.measures_car[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) + " diff " + str((self.measures_car[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] - self.measures_car[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal])) + " normalized by speed " + str(
                    #     local_measure) +" normalized by "+str(max_speed))

                self.reward_car[frame] += reward_weights[ CAR_REWARD_INDX.distance_travelled_towards_goal] * local_measure

                # print (" Linear reward for getting closer to goal " + str(self.reward_car[frame]) + " local measure " + str(local_measure))

            else:
                self.reward_car[frame] += reward_weights[CAR_REWARD_INDX.reached_goal]
                # print (" Reached goal " + str(self.reward_car[frame]) + " car  " +str(self.car[frame + 1])+ "goal"+str(self.car_goal)+" frame "+str(frame))
        if frame>0:
            collision = (
            self.measures_car[frame, CAR_MEASURES_INDX.agent_dead] and self.measures_car[frame - 1, CAR_MEASURES_INDX.agent_dead])
            # if collision:
            #     print ("Car Frame "+str(frame)+" collisons. "+str( self.measures_car[frame, CAR_MEASURES_INDX.agent_dead])+" collisons. "+str( self.measures_car[frame-1, CAR_MEASURES_INDX.agent_dead]))
            reached_goal = self.measures_car[frame, CAR_MEASURES_INDX.goal_reached] and self.measures_car[
                frame - 1, CAR_MEASURES_INDX.goal_reached]
            #print("Car collided "+str(collision)+" goal reached "+str(reached_goal))
            if (collision or reached_goal):
                self.reward_car[frame] =0