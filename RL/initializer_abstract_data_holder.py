from utils.utils_functions import overlap, find_border, mark_out_car_cv_trajectory,find_occlusions
import numpy as np
from scipy import ndimage



class InitializerAbstractDataHolder(object):

    def __init__( self,seq_len, DTYPE, valid_positions=None):
        #super(InitializerAbstractDataHolder, self).__init__(seq_len, DTYPE, valid_positions=valid_positions)
        self.DTYPE=DTYPE
        self.valid_positions=valid_positions.copy()
        self.prior = valid_positions.copy()
        self.occluded = valid_positions.copy()
        self.init_distribution = None
        self.prior_normalizing_factor=0



    def get_original_dist_to_goal(self, frame):
        return self.goal_to_agent_init_dist

    def calculate_occlusions(self, agent_size,lidar_occlusion,field_of_view_car):

        self.occlusions = self.valid_positions.copy()
        # Mark out cv trajectory in from of car so cannot initialize in front of car.


        self.occlusions = mark_out_car_cv_trajectory(self.occlusions,self.init_car_bbox, self.car_goal, self.init_car_vel,agent_size)

        # print ("Car pos "+str(car_pos)+" car vel "+str(car_vel))
        # print ("OCCLUSION SUM BEFORE CHANGES "+str(np.sum(self.occlusions)))
        car_speed = np.linalg.norm(self.init_car_vel) * .2
        agent_dim = np.sqrt((agent_size[1] + 1) ** 2 + ((agent_size[2] + 1) ** 2))
        max_dim = self.car_max_dim + agent_dim

        # Valid positions includes only positions that the agent cannot be initialized in.
        # To get non-occupied locations I add points where there is nothing but the cannot fit due to its size.
        # This is used in border calculation.
        er_mask = np.ones((agent_size[1] * 2 + 1, agent_size[2] * 2 + 1))
        occupied_positions = ndimage.binary_dilation(self.valid_positions, er_mask)
        # print("Occupied solutions " + str(np.sum(occupied_positions)))
        self.occlusions = find_occlusions(self.occlusions,occupied_positions, self.init_car_pos, car_speed, self.init_car_vel,max_dim, self.prior.shape, lidar_occlusion, max_angle=field_of_view_car/2)
        self.occlusions = np.logical_and(self.occlusions,self.valid_positions)

        return self.occlusions
