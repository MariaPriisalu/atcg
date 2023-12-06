
import numpy as np
#from RL.settings import DEFAULT_SKIP_RATE_ON_EVALUATION_CARLA, DEFAULT_SKIP_RATE_ON_EVALUATION_WAYMO, PEDESTRIAN_MEASURES_INDX, CAR_MEASURES_INDX,PEDESTRIAN_INITIALIZATION_CODE,PEDESTRIAN_REWARD_INDX, NBR_MEASURES, NBR_MEASURES_CAR,STATISTICS_INDX, STATISTICS_INDX_POSE, STATISTICS_INDX_CAR,STATISTICS_INDX_CAR_INIT, STATISTICS_INDX_MAP, RANDOM_SEED_NP, RANDOM_SEED
from utils.utils_functions import overlap, find_border, mark_out_car_cv_trajectory,find_occlusions
from copy import copy
from RL.car_data_holder import CarDataHolder
from initializer_abstract_data_holder import InitializerAbstractDataHolder
from scipy import ndimage
class InitializerCarDataHolder(CarDataHolder, InitializerAbstractDataHolder):

    def __init__(self, valid_positions, valid_directions, seq_len, DTYPE):
        vector_len = max(seq_len - 1, 1)

        CarDataHolder.__init__(self,seq_len, DTYPE, valid_positions=valid_positions)
        InitializerAbstractDataHolder.__init__(self, seq_len, DTYPE, valid_positions=valid_positions)
        self.DTYPE=DTYPE
        self.valid_positions=valid_positions.copy()
        self.valid_directions = valid_directions.copy()
        self.prior = valid_positions.copy()
        self.occluded = valid_positions.copy()
        self.init_distribution = None

    def calculate_prior(self,car_pos, car_vel,field_of_view_car, occlusion_prior=False):#, agent_size, lidar_occlusion, field_of_view_car):

        self.prior=self.valid_positions.copy()
        if occlusion_prior and len(car_pos)>0:
            return self.calculate_occlusion_maximizing_prior(car_pos, car_vel, field_of_view_car)
        for x in range(self.prior.shape[0]):
            for y in range(self.prior.shape[1]):
                if self.prior[x,y]!=0:
                    t=[]
                    vel=self.valid_directions[x,y,1:]*1/np.linalg.norm(self.valid_directions[x,y,1:])
                    for i in range(2):
                        points=[0,self.prior.shape[i]]

                        if abs(vel[i]) > 1e-5:
                            for p in points:
                                if i == 0:
                                    dist = p - x
                                else:
                                    dist = p - y
                                if abs(dist)>0:
                                    time=dist*.01/vel[i]
                                    if time>0:
                                        t.append(time)
                                    else:
                                        t.append(0.5 * .01 / abs(vel[i]))
                                else:
                                    t.append(abs(0.5*.01/ vel[i]))
                    if len(t)>0:
                        self.prior[x,y]=min(t)
                    else:
                        self.prior[x, y] =1e-5
        nonzero_values=self.prior>0
        self.prior=self.prior/np.sum(self.prior)
        np.where(self.prior[nonzero_values]==0)
        return self.prior

    def calculate_occlusion_maximizing_prior(self,car_pos, car_vel, field_of_view_car):
        car_speed=np.linalg.norm(car_vel)
        for x in range(self.prior.shape[0]):
            for y in range(self.prior.shape[1]):
                if self.prior[x, y] != 0:
                    pos=np.array([x,y])
                    diff=pos-car_pos
                    diff_norm=np.linalg.norm(diff)
                    if car_speed> 1e-5:
                        scalar_product=np.dot(diff,car_vel)
                        alpha=np.arccos(scalar_product/(diff_norm*car_speed))
                        orthogonal_proj_of_diff_on_vel_len=abs(scalar_product)/car_speed

                        self.prior[x, y] =1/diff_norm
                        if alpha > field_of_view_car / 2:
                            self.prior[x, y] =self.prior[x, y]/2
                    else:
                        self.prior[x, y]=1/diff_norm
        nonzero_values = self.prior > 0
        self.prior = self.prior / np.sum(self.prior)
        np.where(self.prior[nonzero_values] == 0)
        return self.prior


