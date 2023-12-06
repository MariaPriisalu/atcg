import glob
import numpy as np

import random
import numpy as np

import sys
print(sys.path)
import json
from datetime import datetime
from RL.settings import *
import pickle
import os

def loadStatFiles(stats_path, timestamp):
    match_pattern=stats_path+'*'+timestamp+'*'
    files=glob.glob(match_pattern)
    #print(files)

    assert len(files) >= 3, "Wrong stats file !!"

    stat_file_path_poses = [x for x in files if "poses" in x][0]
    #print(stat_file_path_poses)
    stat_file_car = [x for x in files if "learn_car" in x][0]
    #print(stat_file_car)

    stat_file_agent = None
    for x in files:
        basename = os.path.basename(x)
        nbrs = basename.strip()[:-len('.npy')]
        vals = nbrs.split('_')
        if vals[-1].isnumeric():
            stat_file_agent = x
            break
    #print(stat_file_agent)

    #file_agent + "_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
    pos_stat=np.load(stat_file_path_poses)
    #print(f"pose shape{pos_stat.shape}")
    car_stat = np.load(stat_file_car)
    #print(f"car shape {car_stat.shape}")
    agent_stat = np.load(stat_file_agent)
    #print(f"agent shape {agent_stat.shape}")

    return agent_stat, car_stat, pos_stat

# Gets trajectories from recorded training of an agent (car or pedestrian) into CARLA space coordinate system.
# baseDataPath is required to know where the centering file is
# Any episodedex can be used
def parseAgentTrajectoryInCARLASpace(baseDataPath, agentStats, episodedex, isCar):
    # Load data
    centeringPath = os.path.join(baseDataPath, "centering.p")
    centering = pickle.load(open(centeringPath, "rb"), encoding='latin1', fix_imports=True)
    print(centering)

    stats_ep = agentStats[episodedex]

    if isCar == True:
        agent_pos = stats_ep[:, STATISTICS_INDX_CAR.agent_pos[0]:STATISTICS_INDX_CAR.agent_pos[1]].copy()
        agent_vel = stats_ep[:,STATISTICS_INDX_CAR.velocity[0]:STATISTICS_INDX_CAR.velocity[1]].copy()
    else:
        agent_pos = stats_ep[:, STATISTICS_INDX.agent_pos[0]:STATISTICS_INDX.agent_pos[1]].copy()
        agent_vel = stats_ep[:, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]].copy()

    # PFNN first frame matrix/orientation
    rotation_matrix = np.zeros((2, 2), np.float)
    inverse_rotation_matrix = np.zeros((2, 2), np.float)
    init_pos=agent_pos[0,:].copy()
    y = (agent_vel[0,1])
    z = (agent_vel[0,1])
    d = np.sqrt(y ** 2 + z ** 2)
    if d > 0:
        # Rotation matrix from my position to PFNN
        rotation_matrix[0, 0] = y / d
        rotation_matrix[0, 1] = z / d
        rotation_matrix[1, 1] = y / d
        rotation_matrix[1, 0] = -z / d
        # Rotation matrix from PFNN to my coordinate system
        '''
        inverse_rotation_matrix[0, 0] = y / d
        inverse_rotation_matrix[0, 1] = -z / d
        inverse_rotation_matrix[1, 1] = y / d
        inverse_rotation_matrix[1, 0] = z / d
        '''

        '''
        inverse_rotation_matrix[0, 0] = 1.0
        inverse_rotation_matrix[0, 1] = 0.0
        inverse_rotation_matrix[1, 1] = 1.0
        inverse_rotation_matrix[1, 0] = 0.0
        '''
        inverse_rotation_matrix = rotation_matrix


    print ("Velocity "+str(agent_vel[0,:]))
    print (rotation_matrix)

    print ("Agent init positions: "+str(init_pos)+" "+str(agent_pos[1,:]))

    pos_y=-128 / 2 # todo TAKE IT FROM FILENAME !
    R = np.eye(3) # WorldToCameraRotation = np.eye(3)
    #WorldToCameraTranslation = np.array(get_translation(filepath)).reshape((3,1))
    P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    middle = [0,0,0]

    agent_pos_carla = []
    agent_vel_carla = []
    for frame in range(agent_pos.shape[0]):
        agentPosThisFrame_modelSpace = agent_pos[frame, :]
        p = agentPosFromModelToCARLA(agentPos=agentPosThisFrame_modelSpace,
                                     pos_y=pos_y, centering=centering, R=R, P=P, middle=centering['middle'], frame=frame)
        agent_pos_carla.append(p.copy())

        # Not touching the vel
        agent_vel_carla.append(agent_vel[frame].copy())

    return agent_pos_carla, agent_vel_carla


def agentPosFromModelToCARLA(agentPos, pos_y, centering, R, P, middle, frame):
    FRAME_DEBUG = -123
    p = agentPos.copy()
    if frame == FRAME_DEBUG:
        print(("point " + str(p)))
    p[1] = p[1] + pos_y
    if frame == FRAME_DEBUG:
        print(("point+y " + str(p)))
    p[0] = p[0] + centering["height"]
    if frame == FRAME_DEBUG:
        print(("point+ height " + str(p)))

    p = p[[2,1,0]]
    p = np.reshape(p, (3,1))
    p *= 1.0 / centering['scale']
    if frame == FRAME_DEBUG:
        print(("point scaled " + str(p)))
    #p = np.matmul(P, p)
    #if frame == 0:
    #    print(("point P" + str(p)))
    #p = p * np.reshape([-1, -1, 1], (3, 1))
    #if frame == 0:
    #    print(("point p " + str(p)))
    p = p - np.reshape(middle, (3, 1))
    if frame == FRAME_DEBUG:
        print(("point middle " + str(p)))
    p = np.matmul(np.transpose(R), p)
    if frame == FRAME_DEBUG:
        print(("point R " + str(p)))
    return p

# Returns the cars and pedestrians trajectories by giving a base data path, a timestamp to use, and an episode
def readRecordedTrajectories(baseDataPath, timestamp, episodedex):
    pathToStatFiles = os.path.join(baseDataPath, "stats/")

    pedestrian_stat, car_stat, pos_stat = loadStatFiles(pathToStatFiles, timestamp)
    pedestrianTrajectory, pedestrianVelocities = parseAgentTrajectoryInCARLASpace(baseDataPath, pedestrian_stat, episodedex=episodedex, isCar=False)
    carTrajectory, carVelocities = parseAgentTrajectoryInCARLASpace(baseDataPath, car_stat, episodedex=episodedex, isCar=True)

    return carTrajectory, carVelocities, pedestrianTrajectory, pedestrianVelocities

if __name__ == "__main__":
    timestamp="2021-06-17-18-14-55.647761agent_test_174_test_0_-64_0"
    baseDataPath = "DatasetCustom/Data1/scene100/Town03/0/test_0"

    carTrajectory, carVelocities, pedestrianTrajectory, pedestrianVelocities = readRecordedTrajectories(baseDataPath=baseDataPath, timestamp=timestamp, episodedex=0)
    res = 1

