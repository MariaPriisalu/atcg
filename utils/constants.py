
import os

from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

def proc(v,u):

    _ ,_ , disparity =procrustes(u.reshape(-1,3), v.reshape(-1,3)) # np.norm(u.reshape(-1,3)- v.reshape(-1,3))
    return disparity


class Constants(object):

    def thresholdDist(v,u):

        dist=abs(u-v)
        dist_out=0
        for row in dist:
            if row>400:
                dist_out+=400
            else:
                dist_out+=row
        return dist_out

    def __init__(self):

        # Debug mode?
        self.debug=0

        # Display images?
        self.display=0

        ########################################################################################################################## CODE BEGINS HERE
        self.gt_human_label=24
        self.gt_biker_label=25
        self.gt_car_labels=[26,27,28,29,30,31,32,33,1]

        self.human_label=11
        self.biker_label=12
        self.bike_label=17
        self.motorcycle_label=18
        self.car_labels = [13,14,15,16,17,18]


        if self.model==2 or self.model==0:
            self.human_label=24
            self.biker_label=25
            self.bike_label=32
            self.motorcycle_label=33
            self.car_labels=[26,27,28,29,30,31,32,33,1]

        self.dynamic_labels=self.car_labels+[self.motorcycle_label]+[self.bike_label]+[self.biker_label]+[self.human_label]

