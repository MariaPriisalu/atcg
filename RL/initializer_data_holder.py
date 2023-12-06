
import numpy as np
#from RL.settings import DEFAULT_SKIP_RATE_ON_EVALUATION_CARLA, DEFAULT_SKIP_RATE_ON_EVALUATION_WAYMO, PEDESTRIAN_MEASURES_INDX, CAR_MEASURES_INDX,PEDESTRIAN_INITIALIZATION_CODE,PEDESTRIAN_REWARD_INDX, NBR_MEASURES, NBR_MEASURES_CAR,STATISTICS_INDX, STATISTICS_INDX_POSE, STATISTICS_INDX_CAR,STATISTICS_INDX_CAR_INIT, STATISTICS_INDX_MAP, RANDOM_SEED_NP, RANDOM_SEED
from utils.utils_functions import overlap, find_border, mark_out_car_cv_trajectory,find_occlusions
from copy import copy
from RL.reward_data_holder import RewardDataHolder
from RL.initializer_abstract_data_holder import InitializerAbstractDataHolder

from scipy import ndimage

class InitializerDataHolder(InitializerAbstractDataHolder,RewardDataHolder):

    def __init__(self, valid_positions, seq_len, DTYPE):

        RewardDataHolder.__init__(self, seq_len, DTYPE, valid_positions=valid_positions)
        InitializerAbstractDataHolder.__init__(self,seq_len, DTYPE, valid_positions=valid_positions)

        self.goal_priors = []
        self.goal_distributions  = []
        self.frames_of_goal_change=[]

        self.init_car_id = -1
        self.init_car_vel = []
        self.init_car_pos = []
        self.car_max_dim=-1
        self.car_min_dim=-1
        self.init_car_bbox=[]
        self.car_goal=-1

    def initBorderOcclusionPrior(self, valid_positions):
        # Find borders of valid areas to initialize agent there, init the prio, goal, occlusion map

        self.prior = valid_positions.copy()
        self.goal_priors = []#*self.heatmap
        self.occluded=valid_positions.copy()


    def calculate_prior(self, agent_size,use_occlusions, lidar_occlusion, field_of_view_car, heatmap=[]):

        if use_occlusions:
            # print("Use occlusions")
            self.calculate_occlusions( agent_size, lidar_occlusion, field_of_view_car)
            self.prior = self.occlusions.copy() * self.valid_positions.copy()  # *self.heatmap
        else:
            self.prior = self.valid_positions.copy()
        if len(heatmap)>0 and np.sum(heatmap)>1e-5:
            self.prior=self.prior*heatmap


        self.prior =mark_out_car_cv_trajectory(self.prior, self.init_car_bbox,self.car_goal, self.init_car_vel, agent_size)

        # print("Prior before adding any values "+str(np.sum(np.abs(self.prior))))
        car_speed = np.linalg.norm(self.init_car_vel) * .2
        agent_dim = np.sqrt((agent_size[1] + 1) ** 2 + ((agent_size[2] + 1) ** 2)) * .2

        # print ("Car speed "+str(car_speed)+" agent dim "+str(agent_dim))
        # check if car is moving faster than 1km/h
        if car_speed > 0.081:
            alpha = np.arctan2(self.init_car_vel[1], self.init_car_vel[0])  # Car's direction of movement

            max_ped_speed = 3 * 5  # Pedestrian's maximal speep 3m/s in voxels
            beta = np.arctan(
                max_ped_speed / car_speed)  # The maximal difference in angle between the pedestrian and the car's direction of movement leading to a crash.

            car_vel_unit = self.init_car_vel * (1 / np.linalg.norm(self.init_car_vel))

            # Distance to the front and back ends of the car.
            car_pos_back = self.init_car_pos - (car_vel_unit * self.car_max_dim * .5)
            car_pos_front = self.init_car_pos + (car_vel_unit * self.car_max_dim * .5)

            # Find among the angles which results in an upper and which in a lower constraint.
            constraints = [np.tan(alpha + beta), np.tan(alpha - beta)]
            multiplier_of_upper_constraint = max(constraints)
            multiplier_of_lower_constraint = min(constraints)
            # print("Constraints: "+str(constraints))
            for x in range(self.prior.shape[0]):
                for y in range(self.prior.shape[1]):
                    # Displacement from current position [x,y] to car's back end
                    displacement_from_car = np.array([x, y]) - car_pos_back
                    # Displacement from current position [x,y] to car's front
                    displacement_to_car_front = np.array([x, y]) - car_pos_front
                    # Distance to car's front
                    distance_to_car = np.linalg.norm(displacement_to_car_front) * .2 - agent_dim
                    # Check if the point [x, y] is in front of the car.
                    point_in_front_of_car = np.dot(displacement_from_car, car_vel_unit)

                    # If the point is on front of the car and lies between the upper and lower constraint from above then it shoudl receive a higher weight.
                    if point_in_front_of_car > 0 and (
                            displacement_from_car[1] < multiplier_of_upper_constraint * displacement_from_car[0] + (
                            self.car_min_dim * .5) or displacement_from_car[1] > multiplier_of_lower_constraint *
                            displacement_from_car[0] - (self.car_min_dim * .5)):  # Inside cone of possible initializations
                        # If [x,y] is within braking distance on dry road, then don't initialize here.
                        # print(" distance to car "+ str(distance_to_car)+" car_speed "+str(car_speed)+" test "+str(distance_to_car<(car_speed**2/(250*0.8))))
                        if distance_to_car < (car_speed ** 2 / (250 * 0.8)):
                            self.prior[x, y] = 0
                        else:
                            self.prior[x, y] = self.prior[x, y] / distance_to_car
                    else:
                        if distance_to_car < agent_dim + (self.car_max_dim * .5):
                            self.prior[x, y] = 0
                        else:
                            self.prior[x, y] = self.prior[x, y] / (
                                        distance_to_car ** 2)

        else:
            # print (" otherwise ")
            for x in range(self.prior.shape[0]):
                for y in range(self.prior.shape[1]):
                    displacement_to_car_front = np.array([x, y]) - self.init_car_pos
                    distance_to_car = np.linalg.norm(displacement_to_car_front) * .2 - agent_dim
                    if distance_to_car < 1:
                        self.prior[x, y] = 0
                    else:
                        self.prior[x, y] = self.prior[x, y] / (
                            distance_to_car)
        self.prior = self.prior * (
                    1 / np.sum(self.prior[:]))
        return self.prior

    # Car vel is voxels per second
    # time is in seconds, distances in voxels

    # Note Goal prior is not normalized! ie does not sum to 1!!!!!
    def calculate_goal_prior(self, agent_init_pos, frame_time, agent_size, frame):
        self.frames_of_goal_change.append(frame)
        self.goal_priors.append(self.valid_positions.copy())
        assert(len(self.goal_priors)-1==frame)
        if frame>0:
            return self.calculate_goal_prior_after_crossing_road(frame, agent_init_pos, frame_time, agent_size, previous_dir)
        car_dim_x = self.car_max_dim
        car_dim_y = self.car_max_dim
        print("Calculate prior input " + str(self.init_car_pos) + " car vel " + str(self.init_car_vel) + " agent init pos " + str(
            agent_init_pos))
        car_vel = self.init_car_vel

        car_speed = np.linalg.norm(car_vel) * .2
        vel_to_car = self.init_car_pos - agent_init_pos  # pointing from pedestrian to car
        margin = [agent_size[1] + car_dim_x, agent_size[2] + car_dim_y]

        max_speed_ped = 15.0  # voxels per second
        max_time = self.seq_len * frame_time

        CONST_MINVEL_EPS = 0.0001  # A small epsilon value for epsilon used for avoiding zero divide on 0 values for velocity
        INV_CONST_MINVEL_EPS = 1 / CONST_MINVEL_EPS
        if car_speed > 0.081 ** 2:

            for x in range(self.prior.shape[0]):
                for y in range(self.prior.shape[1]):
                    pos = np.array([x, y])
                    t1 = (x - self.init_car_pos[0]) / car_vel[0]
                    t2 = (y - self.init_car_pos[1]) / car_vel[1]
                    if abs(t1 - t2) < CONST_MINVEL_EPS:
                        self.goal_priors[frame][x, y] = 1
                    else:
                        vel_to_goal = pos - agent_init_pos
                        dist_to_goal = np.linalg.norm(vel_to_goal)
                        if dist_to_goal > 2:
                            vel_to_goal = vel_to_goal * (1 / dist_to_goal)

                            numerator = (car_vel[0] * vel_to_car[1] - car_vel[1] * vel_to_car[0])
                            denominator = (vel_to_goal[0] * vel_to_car[1] - vel_to_goal[1] * vel_to_car[0])
                            min_s = numerator / denominator  # scaling of pedestrian velocity

                            max_s = min_s
                            max_candidates_variables, min_candidates_variables = self.get_lower_and_upper_bounds(
                                car_vel,
                                margin,
                                max_s,
                                vel_to_car,
                                vel_to_goal)

                            range_in_s = [[], []]
                            range_in_t = [[], []]

                            for coordinate_index in range(2):

                                lower_function_value_at_max_s, lower_function_value_at_min_s, upper_function_value_at_max_s, upper_function_value_at_min_s = self.evaluate_end_points_of_upper_and_lower_bound(
                                    coordinate_index, max_candidates_variables, min_candidates_variables, max_s, max_s)
                                range_in_t[0].append(copy(lower_function_value_at_max_s))
                                range_in_t[1].append(copy(upper_function_value_at_max_s))

                            max_t, min_t = self.get_extreme_s_t(range_in_s, range_in_t)

                            if min_t < max_t and min_s > 0 and min_s < max_speed_ped:  # and min_s<max_s:

                                time = max(min_t, 1e-3) * max_s
                                #print(" Time "+str(time)+" max s "+str(max_s)+" "+str(1/time))
                                if time > INV_CONST_MINVEL_EPS:
                                    self.goal_priors[frame][x, y] = CONST_MINVEL_EPS
                                elif CONST_MINVEL_EPS > time:
                                    self.goal_priors[frame][x, y] = self.goal_priors[frame][x, y] / CONST_MINVEL_EPS
                                else:
                                    self.goal_priors[frame][x, y] = self.goal_priors[frame][ x, y] / (time)  # dist to collision

                            else:

                                self.goal_priors[frame][x, y] = 0
                        else:
                            self.goal_priors[frame][x, y] = 0
        else:
            print(" Goal same as prior ")
            self.goal_priors[frame] = copy(self.prior)

        # If the entire map of goal priors is 0 then use prior.
        sumOfGoalPriors = np.sum(self.goal_priors[frame][:])
        if sumOfGoalPriors <1e-5:
            self.goal_priors[frame] = copy(self.prior)

        return self.goal_priors[frame]

    def calculate_goal_prior_after_crossing_road(self,frame, agent_init_pos, frame_time, agent_size, previous_dir):
        previous_distance_to_goal=np.linalg.norm(previous_dir)
        for x in range(self.prior.shape[0]):
            for y in range(self.prior.shape[1]):
                pos = np.array([x, y])
                if self.valid_positions[x,y]:
                    goal_dir=pos-agent_init_pos
                    dist_to_goal=np.linalg.norm(goal_dir)
                    scalar_product=np.dot(goal_dir, previous_dir)
                    if dist_to_goal>1e-5:
                        alpha=np.arccos(scalar_product/(dist_to_goal*previous_distance_to_goal))
                        if alpha<= np.pi/2:
                            self.goal_priors[frame][x, y] = dist_to_goal
                        else:
                            self.goal_priors[frame][x, y] = 0
                    else:
                        self.goal_priors[frame][x, y] = 0

        self.goal_priors[frame] = self.goal_priors[frame] * (1 / np.sum(self.goal_priors[frame][:]))
        return self.goal_priors[frame]




    def get_extreme_s_t(self, range_in_s, range_in_t):
        min_t = max(range_in_t[0])
        max_t = min(range_in_t[1])
        return max_t, min_t

    def get_range_for_each_dimension(self, break_point, lower_function_value_at_max_s,
                                     lower_function_value_at_min_s, range_in_s, range_in_t,
                                     upper_function_value_at_max_s, upper_function_value_at_min_s,
                                     upper_function_value_at_break_point,
                                     x, y, condition, min_t, max_t, min_s, max_s, frame):

        if condition:
            # Check on which side of the translation point is the upper constraint above the lower?
            if upper_function_value_at_max_s > lower_function_value_at_max_s:
                range_in_s[0].append(copy.copy(break_point))
                range_in_s[1].append(copy.copy(max_s))
                range_in_t[0].append(copy.copy(max(lower_function_value_at_max_s, min_t)))
                range_in_t[1].append(copy.copy(min(upper_function_value_at_break_point, max_t)))
            elif upper_function_value_at_min_s > lower_function_value_at_min_s:
                range_in_s[0].append(min_s)
                range_in_s[1].append(copy.copy(break_point))
                range_in_t[0].append(copy.copy(max(lower_function_value_at_min_s, min_t)))
                range_in_t[1].append(copy.copy(min(upper_function_value_at_break_point, max_t)))
            else:
                self.goal_priors[frame][x, y] = 0
                # print ("No solution!")
                range_in_s[0].append(1)
                range_in_s[1].append(-1)
                range_in_t[0].append(1)
                range_in_t[1].append(-1)
        else:
            if upper_function_value_at_max_s < lower_function_value_at_max_s:
                self.goal_priors[frame][x, y] = 0
                # print ("No solution!")
                range_in_s[0].append(1)
                range_in_s[1].append(-1)
                range_in_t[0].append(1)
                range_in_t[1].append(-1)
            else:
                range_in_s[0].append(copy.copy(min_s))
                range_in_s[1].append(copy.copy(max_s))
                range_in_t[0].append(max(lower_function_value_at_min_s, min_t))
                range_in_t[1].append(min(upper_function_value_at_max_s, max_t))

    def evaluate_end_points_of_upper_and_lower_bound(self, coordinate_index, max_candidates_variables,
                                                     min_candidates_variables, min_s, max_s):
        if coordinate_index >= 0:
            max_variables = max_candidates_variables[coordinate_index]
            min_variables = min_candidates_variables[coordinate_index]
        else:
            max_variables = max_candidates_variables
            min_variables = min_candidates_variables
        upper_function_value_at_max_s = max_variables[0] / max(max_s - max_variables[1], 1e-5)
        lower_function_value_at_max_s = min_variables[0] / max(max_s - min_variables[1], 1e-5)
        upper_function_value_at_min_s = max_variables[0] / max(min_s - max_variables[1], 1e-5)
        lower_function_value_at_min_s = min_variables[0] / max(min_s - min_variables[1], 1e-5)
        # if min_s >0 and min_s<15:
        #     print ("Upper bounds max: "+str(upper_function_value_at_max_s)+" lower bounds: "+str(lower_function_value_at_max_s) )
        #     print ("Upper bounds min: " + str(upper_function_value_at_min_s) + " lower bounds: " + str(
        #         lower_function_value_at_min_s))
        return lower_function_value_at_max_s, lower_function_value_at_min_s, upper_function_value_at_max_s, upper_function_value_at_min_s

    def get_lower_and_upper_bounds(self, car_vel, margin, max_speed_ped, vel_to_car, vel_to_goal):
        min_candidates_variables = []
        max_candidates_variables = []
        # decide on upper and lower bounding functions.
        for i in range(2):
            #if abs(vel_to_goal[i])>1e-5:
            condition_for_reverting_inequality = (vel_to_goal[i] * max_speed_ped) - car_vel[
                i]  # revert inequality this is negative. We divide by this!
            if condition_for_reverting_inequality > 0:
                min_candidates_variables.append([vel_to_car[i] - margin[i] / vel_to_goal[i], car_vel[i] / vel_to_goal[
                    i]])  # vertical scaling, horizontal scaling, translation
                max_candidates_variables.append(
                    [vel_to_car[i] + margin[i] / vel_to_goal[i], car_vel[i] / vel_to_goal[i]])
            else:
                min_candidates_variables.append(
                    [vel_to_car[i] + margin[i] / vel_to_goal[i], car_vel[i] / vel_to_goal[i]])
                max_candidates_variables.append(
                    [vel_to_car[i] - margin[i] / vel_to_goal[i], car_vel[i] / vel_to_goal[i]])
        return max_candidates_variables, min_candidates_variables



