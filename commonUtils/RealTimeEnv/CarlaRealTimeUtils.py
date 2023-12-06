import glob
import os
import sys
import random
import logging
import time
import argparse
import re
import math
from enum import Enum
from typing import Dict, List, Tuple, Set
import pathlib
import json

import numpy
import transforms3d.euler
from matplotlib import cm
#from RL.settings import run_settings
from commonUtils.ReconstructionUtils import METER_TO_VOXEL, VOXEL_TO_METER
from RL.settings import realTimeEnvOnline

import carla


CARLA_INSTALL_PATH=""
CARLA_INSTALL_PATH = os.getenv('CARLA_INSTALL_PATH')



pathToCarlaAPI = os.path.join(CARLA_INSTALL_PATH, 'PythonAPI/carla')
sys.path.append(pathToCarlaAPI)

from agents.navigation.global_route_planner import GlobalRoutePlanner

# Relative import for local waymo package to be sure that we use it from everywhere we need
if os.path.exists("./waymo-open-dataset/tutorial"):
    sys.path.append("./waymo-open-dataset/tutorial")
    sys.path.append("./waymo-open-dataset")
    sys.path.append("./commonUtils")


print(sys.path)
# This is to make sure that waymo will get access back to ReconstructionUtils

from pipeline_pointCloudReconstruction import * #save_3d_pointcloud_asRGB, save_3d_pointcloud_asSegLabel, save_3d_pointcloud_asSegColored, Point3DInfoType
from commonUtils.ReconstructionUtils import FRAME_INDEX_FOR_SPAWN_DETAILS, reconstruct3D_ply, CreateDefaultDatasetOptions_CarlaRealTime

from commonUtils.RealTimeEnv.DemoParseStats import readRecordedTrajectories
from RL.settings import run_settings

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


# The default and minimum debugger 2D observer window coordinates
DEFAULT_OBSERVER_WINDOW_WIDTH = 102.4#51.2#70
DEFAULT_OBSERVER_WINDOW_DEPTH = 102.4#51.2#70

MIN_OBSERVER_WINDOW_DEPTH = 50
MIN_OBSERVER_WINDOW_WIDTH = 50

############# SOME USFEULL GLOBAL PARAMS ##########
ENABLE_LOCAL_SCENE_RENDERING = True # If true, a window will be created on the RLAgent simulator side to show graphical outputs real time
if ENABLE_LOCAL_SCENE_RENDERING:
    import pygame


# TODO put this in an env variable
if realTimeEnvOnline:
    CARLA_INSTALL_PATH=""
    CARLA_INSTALL_PATH = os.getenv('CARLA_INSTALL_PATH')

    assert CARLA_INSTALL_PATH, "Define CARLA_INSTALL_PATH. Inside it should contain the PythonAPI folder directly "

    common_placement_carla=-1
    try:
        carlaModuleFilter = os.path.join(CARLA_INSTALL_PATH, 'PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))# instead of 7 sys.version_info.minor,
        carlaModulePath = glob.glob(carlaModuleFilter)[0]

        assert os.path.exists(carlaModulePath), "Can't find carla egg"
        sys.path.insert(common_placement_carla,carlaModulePath)
    except IndexError:
        assert False, "Path to carla egg not found !"


    sys.path.insert(common_placement_carla,os.path.join(CARLA_INSTALL_PATH, 'PythonAPI/carla'))

    import carla


    # This is CARLA agent
    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.local_planner import LocalPlanner
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    #from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
    from agents.tools.misc import is_within_distance, compute_magnitude_angle

    from carla import Client as Cli


import copy
import numpy as np
import pickle
import traceback
import shutil
from enum import Enum
import math
from typing import Dict

try:
    import queue
except ImportError:
    import Queue as queue


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def voxelSpaceLocationToCarlaLocation(voxelLoc):
    posInCarlaSpace = carla.Location(x = voxelLoc[2] * VOXEL_TO_METER, y = voxelLoc[1] * VOXEL_TO_METER, z = voxelLoc[0] * VOXEL_TO_METER)
    return posInCarlaSpace

def voxelSpaceLocationToCarlaNumpy(voxelLoc):
    posInCarlaSpace = np.array([voxelLoc[2] * VOXEL_TO_METER, voxelLoc[1] * VOXEL_TO_METER, voxelLoc[0] * VOXEL_TO_METER])
    return posInCarlaSpace

def carlaLocationToVoxelSpaceLocation(carlaLocation):
    posInVoxelSpace = np.array([carlaLocation.z * METER_TO_VOXEL, carlaLocation.y * METER_TO_VOXEL, carlaLocation.x * METER_TO_VOXEL])
    return posInVoxelSpace



# length of a vector
def getVectorLength(v):
    """Returns the length of a vector"""
    if isinstance(v, numpy.ndarray):
        if v.shape == 3:
            return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        else:
            return math.sqrt(v[0]**2 + v[1]**2)
    else:
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

# Returns a new normalized vector
def getVectorNormalized(v, knownLength=None):
    vnew = v
    knownLength = knownLength if knownLength is not None else getVectorLength(vnew)
    return vnew/knownLength if knownLength != 0.0 else vnew

def set_fixed_seed(seedValue):
    # TODO maybe tensorflow, torch ? :)
    np.random.seed(seedValue)
    random.seed(seedValue)

def check_far(value):
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError(
            "{} must be a float between 0.0 and 1.0")
    return fvalue


def getCenterOfBBox(bboxAsNumpyArray):
    if bboxAsNumpyArray.shape == (1,6) or bboxAsNumpyArray.shape == (6,):
        return np.array([bboxAsNumpyArray[0] + (bboxAsNumpyArray[1] - bboxAsNumpyArray[0])*0.5,
                         bboxAsNumpyArray[2] + (bboxAsNumpyArray[3] - bboxAsNumpyArray[2])*0.5,
                         bboxAsNumpyArray[4] + (bboxAsNumpyArray[5] - bboxAsNumpyArray[4])*0.5])
    elif bboxAsNumpyArray.shape == (3,2):
        return np.array([bboxAsNumpyArray[0][0] + (bboxAsNumpyArray[0][1] - bboxAsNumpyArray[0][0])*0.5,
                        bboxAsNumpyArray[1][0] +  (bboxAsNumpyArray[1][1] - bboxAsNumpyArray[1][0])*0.5,
                        bboxAsNumpyArray[2][0] +  (bboxAsNumpyArray[2][1] - bboxAsNumpyArray[2][0])*0.5])
        raise NotImplementedError()


class BBox2D:
    def __init__(self):
        self.left = None
        self.top = None
        self.right = None
        self.bottom = None

    # Returns a bounding box given the middle x,y and width depth extents
    @staticmethod
    def getBboxWithCenterAndExtent(xmid, ymid, width, depth):
        width_half = width * 0.5
        depth_half = depth * 0.5
        res = BBox2D()
        res.loadFromList([xmid - width_half, ymid - depth_half, width, depth])
        return res


    # Loads the bbox from a list given by [left, top, width, height]
    def loadFromList(self, lst):
        self.left = lst[0]
        self.top = lst[1]
        self.right = lst[0] + lst[2]
        self.bottom = lst[1] + lst[3]

    def convertToList(self):
        return [self.left, self.top, self.right - self.left, self.bottom - self.top]

    # Extend by either a point 2D (as a list or from a bbox)
    def extend(self, other):
        if isinstance(other, BBox2D):
            self.left = min(self.left, other.left) if self.left is not None else other.left
            self.top = min(self.top, other.top) if self.top is not None else other.top
            self.right = max(self.right, other.right) if self.right is not None else other.right
            self.bottom = max(self.bottom, other.bottom) if self.bottom is not None else other.bottom

        elif isinstance(other, list) or isinstance(other, np.ndarray): # [x,y,z] coordinates
            tempBbox = BBox2D()
            tempBbox.left = tempBbox.right = other[0]
            tempBbox.top = tempBbox.bottom = other[1]
            self.extend(tempBbox)

    # Extend from left/right and top/bottom with given dims
    def extend_from_margins(self, on_width, on_height):
        self.left -= on_width
        self.right += on_width
        self.top -= on_height
        self.bottom += on_height

    # Return the x,y center
    def getCenter(self):
        return [(self.left + self.right) * 0.5, (self.top + self.bottom) * 0.5]

    # Returns the width and depth (x,y axis)
    def getDims(self):
        return [(self.right - self.left), (self.bottom - self.top)]
if realTimeEnvOnline:
    def carlaRotationToNumpy(rot : carla.Rotation):
        if rot is None:
            return None
        return np.array([rot.pitch, rot.yaw, rot.roll])

    def NumpyToCarlaRotation(rotNumpy) -> carla.Rotation:
        if rotNumpy is None:
            return None
        return carla.Rotation(pitch=rotNumpy[0], yaw=rotNumpy[1], roll=rotNumpy[2])

# This is to convert from a 3x3 rotation matrix to a carla.Rotation helper func
def rotationMatrixToCarlaRotation(carlaRot33):
    pitch, roll, yaw = transforms3d.euler.mat2euler(carlaRot33, axes='sxyz')
    pitch = np.rad2deg(pitch)
    roll = np.rad2deg(roll)
    yaw = np.rad2deg(yaw)
    rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
    return rotation

# Transforms from voxel space (pos and dir) to a full Carla.Transform. If transformToWorld is specified, then
# a conversion from World to reference to World will be done with the provided matrix
def getCarlaSpaceTransformFromVoxelSpace(posInVoxelSpace, dirInVoxelSpace,
                                         transformToWorld, worldRef_2_world_matrix):
    # Put the new transform on the spawn frame details
    carlaNewRot = Vector3DVoxelSpace_ToEulerCarlaSpace(dirInVoxelSpace)
    carlaNewPos = voxelSpaceLocationToCarlaLocation(posInVoxelSpace)

    carlaNew_Transform = carla.Transform(carlaNewPos, carlaNewRot)
    carlaNew_Transform_matrix = get_matrix(carlaNew_Transform)

    carlaNew_Rot_matrix = carlaNew_Transform_matrix[0:3, 0:3]

    if transformToWorld:
        carlaNewPos_asNumpy = carlaVector3DToNumpy(carlaNewPos)
        carlaNewPos = np.dot(worldRef_2_world_matrix, np.append(carlaNewPos_asNumpy, 1.0).T)
        carlaNewRot_asMatrix = np.dot(worldRef_2_world_matrix[0:3, 0:3], carlaNew_Rot_matrix)

        carlaRotation = rotationMatrixToCarlaRotation(carlaNewRot_asMatrix)
        carlaLocation = NumpyToCarlaVector3D(carlaNewPos)

        finalTransform = carla.Transform(location=carlaLocation, rotation=carlaRotation)
    else:
        finalTransform = carla.Transform(location=carlaNewPos, rotation=carlaNewRot)

    return finalTransform

# Given Carla.Transform (position and location pair) this function returns a new Carla.Transform,
# but transformed by the given matrix
if realTimeEnvOnline:
    def getCarlaTransformByMatrix(transform : carla.Transform, matrixToTransformWith):
        # Get the location as numpy array, multiply with the matrix, transform it back to a carla.Location
        transform_loc_asNumpy = carlaVector3DToNumpy(transform.location)
        transform_loc_asNumpy = np.dot(matrixToTransformWith, np.append(transform_loc_asNumpy, 1.0).T)
        transform_loc = NumpyToCarlaVector3D(transform_loc_asNumpy)

        # Get the rotation as numpy matrix, multiply with the transform rotation matrix, then move it back to carla.Rotation
        transform_asMatrix = get_matrix(transform)
        transform_rotation_asMatrix = transform_asMatrix[0:3, 0:3]
        transform_rotation_asMatrix = np.dot(matrixToTransformWith[0:3, 0:3], transform_rotation_asMatrix)
        transform_rotation = rotationMatrixToCarlaRotation(transform_rotation_asMatrix)

        #sanityCheck_finalTransform_asMatrix = np.dot(matrixToTransformWith, transform_asMatrix)

        finalTransform = carla.Transform(location=transform_loc, rotation=transform_rotation)
        return finalTransform

    def carlaVector3DToNumpy(vec3D : carla.Vector3D):
        if vec3D is None:
            return None
        return np.array([vec3D.x, vec3D.y, vec3D.z])

    def NumpyToCarlaVector3D(vecNumpy) -> carla.Vector3D:
        return carla.Location(x=vecNumpy[0], y=vecNumpy[1], z=vecNumpy[2])

    def carlaTransformToNumpy(transf : carla.Transform):
        if transf is None:
            return None
        loc : carla.Vector3D = transf.location
        rot : carla.Rotation = transf.rotation
        loc_np = carlaVector3DToNumpy(loc)
        rot_np = carlaRotationToNumpy(rot)
        res = np.stack([loc_np, rot_np])
        return res

    def NumpyToCarlaTransform(transf_numpy) -> carla.Transform:
        if transf_numpy is None:
            return None

        assert list(transf_numpy.shape) == [2, 3]
        loc_np = transf_numpy[0]
        rot_np = transf_numpy[1]
        loc = NumpyToCarlaVector3D(loc_np)
        rot = NumpyToCarlaRotation(rot_np)
        res = carla.Transform(location=loc, rotation=rot)
        return res


# Converts a 3D vector representing direction from voxel space to carla.Rotation
def Vector3DVoxelSpace_ToEulerCarlaSpace(vector3DVoxelSpace):
    yaw = math.atan2(vector3DVoxelSpace[1], vector3DVoxelSpace[2]) * 180.0 / math.pi
    pitch = 0
    roll = 0
    carlaRot = carla.Rotation(pitch=pitch, roll=roll, yaw=yaw)
    return carlaRot

# Returns the angle between two vectors
def getAngleBetweenVectors(v1, v2, alreadyNormlized=False):
    if alreadyNormlized is False:
        v1 = getVectorNormalized(v1)
        v2 = getVectorNormalized(v2)

    angle = math.degrees(math.acos(np.clip(np.dot(v1, v2), -1., 1.)))
    return angle

# Returns if the inputV is in clockwise direction to targetV
def isVectorInClockwiseDirectionTo(inputV, targetV):
    res = np.cross(inputV, targetV)
    return res <= 0

SYNC_TIME = 0.1
SYNC_TIME_PLUS = 3.0 # When destroying the environment

# Entity data (car, pedestrian, etc) on a single frame
class EntityReplayDataPerFrame:
    def __init__(self):
        self.transform  = None # np array 2x3
        self.velocity = None # np array 1x3 # carla.Vector3D = None

# Like a union for both cars and pedestrian spawn details...
# Warning ! The things below are in world coordinates, not world reference coordinates, used mostly for replay only not training !
class EntitySpawnDetails:
    def __init__(self, transform : carla.Transform, velocity : carla.Vector3D, blueprint : str,
                 replayId : any, color=None, driver_id=None, speed=None, targetPoint : carla.Transform=None,
                 world_2_worldRef_matrix = None): # The objects could be in world space but we want them in relative to the reference pos in the world
        self.replayDataPerFrame = EntityReplayDataPerFrame()
        self.replayDataPerFrame.transform = carlaTransformToNumpy(transform)
        self.replayDataPerFrame.velocity = carlaVector3DToNumpy(velocity)


        self.blueprint = blueprint
        self.replayId = replayId
        self.color = color
        self.driver_id = driver_id
        self.speed = speed
        self.targetPoint = carlaTransformToNumpy(targetPoint)
        self.world_2_worldRef_matrix = world_2_worldRef_matrix

    def convertToCarla(self):
        self.replayDataPerFrame.transform = NumpyToCarlaTransform(self.replayDataPerFrame.transform)
        self.replayDataPerFrame.velocity = NumpyToCarlaVector3D(self.replayDataPerFrame.velocity)
        self.targetPoint = NumpyToCarlaTransform(self.targetPoint)

    def convertToNumpy(self):
        self.replayDataPerFrame.transform = carlaTransformToNumpy(self.replayDataPerFrame.transform)
        self.replayDataPerFrame.velocity = carlaVector3DToNumpy(self.replayDataPerFrame.velocity)
        self.targetPoint = carlaTransformToNumpy(self.targetPoint)

class SimOptions:

    TRAINED_CAR_BLUEPRINT = "vehicle.tesla.model3" # TODO Online Ciprian - > these should be somewhere in the config file !!! Both for spawn and replay data
    TRAINED_PEDESTRIAN_BLUEPRINT = "walker.pedestrian.0004"

    def __init__(self, simulateReplay = False, pathToReplayData = None,
                 simulationReplayUseTrainedStats_car=False, simulationReplayUseTrainedStats_ped=False,
                 simulationReplayTimestamp=None,
                 simEpisodeIndex=None,
                 simOnlineEnv = None):
        self.simulateReplay = simulateReplay # If we simulate a scene with replay data (trajectories and trainable agents stuff)
        self.simOnlineEnv = simOnlineEnv #  Real time RL training !
        self.pathToReplayData = pathToReplayData
        self.carSpawnedIdToReplayId : Dict[any, any] = {} # Dict from an id from replay from id spawned in the world
        self.pedestrianSpawnedIdToReplayId : Dict[any, any] = {}

        self.carsData : Dict[any, Dict[any, EntityReplayDataPerFrame]] = {}
        self.pedestriansData : Dict[any, Dict[any, EntityReplayDataPerFrame]] = {}

        # Mapping from replay id of each entity to its original spawn details
        # We currently expect the ids to remain constant
        # TODO Replay: support dynamic creation/destruction of entities
        self.carReplayIdToSpawnData : Dict[any, EntitySpawnDetails] = {}
        self.pedestriansReplayIdToSpawnData : Dict[any, EntitySpawnDetails] = {}
        self.maxFramesIndexRecorded = None

        self.useRecordedTrainedCar = self.simulateReplay == True and simulationReplayUseTrainedStats_ped == True
        self.useRecordedTrainedPed = self.simulateReplay == True and simulationReplayUseTrainedStats_car == True
        self.useRecordedTrainedAgents = self.useRecordedTrainedCar or self.useRecordedTrainedPed
        self.useRecordedTrainedAgents_timestamp=simulationReplayTimestamp
        self.useRecordedTrainedAgents_baseDataPath = self.pathToReplayData
        self.simEpisodeIndex = simEpisodeIndex

# Returns the world 2 world ref and inverse of it
    def loadWorldToWorldRefMatrixConversion(self, outputCurrentEpisodePath):
        targetFilePath = os.path.join(outputCurrentEpisodePath, DataGatherParams.CAMERAREPLAY_DESC_FILE)
        assert os.path.exists(targetFilePath)
        with open(targetFilePath, "rb")  as cameraDescReplayFile:
            self.world_2_worldRef_matrix = pickle.load(cameraDescReplayFile, encoding="latin1", fix_imports=True)
            self.worldRef_2_world_matrix = np.linalg.inv(self.world_2_worldRef_matrix)

        return self.world_2_worldRef_matrix, self.worldRef_2_world_matrix

    # Will load the replay data for agents in the specified file and inject into some existing cars and people dictionaries
    def loadTrainedAgentsReplayData(self, injectCarsDict, injectPeopleDict):
        # Load from stats file, results are in world carla reference
        carTrajectory, carVelocities, pedestrianTrajectory, pedestrianVelocities \
            = readRecordedTrajectories(baseDataPath=self.useRecordedTrainedAgents_baseDataPath,
                                       timestamp=self.useRecordedTrainedAgents_timestamp,
                                       episodedex=self.simEpisodeIndex)

        # Then put them in the people and cars per frame. Later, will be put on spawn data too !
        episodeLength = len(injectPeopleDict) - (1 if FRAME_INDEX_FOR_SPAWN_DETAILS in injectPeopleDict else 0) # Sanity check

        # Maybe we need to complete/subtract a bit by hand the last frames from stats...
        diff = episodeLength - len(carTrajectory)
        if diff < 0:
            carTrajectory = carTrajectory[:episodeLength]
            carVelocities = carVelocities[:episodeLength]
            pedestrianTrajectory = pedestrianTrajectory[:episodeLength]
            pedestrianVelocities = pedestrianVelocities[:episodeLength]
        elif diff > 0:
            for i in range(diff):
                carTrajectory.append(carTrajectory[-1])
                carVelocities.append(carVelocities[-1])
                pedestrianTrajectory.append(pedestrianTrajectory[-1])
                pedestrianVelocities.append(pedestrianVelocities[-1])

        assert len(carTrajectory) == len(carVelocities) == len(pedestrianTrajectory) == len(pedestrianVelocities) == episodeLength

        # Get the initial car and ped yaw angles for a given frame
        lastKnownYaw_car = carVelocities[0]
        lastKnownYaw_ped = pedestrianVelocities[0]

        def getYaw(traj, vel, frame, lastKnownYaw):
            init_dir = traj[frame] - traj[frame-1]
            if init_dir[0] == 0.0: # y/x problem
                return lastKnownYaw
                #init_dir = vel[frame]
            yaw = np.arctan2(init_dir[1], init_dir[0])
            yaw = np.rad2deg(yaw)
            return yaw

        for frame in range(episodeLength):
            # Grab the required data for trained car and pedestrian agents
            trained_car_pos = carTrajectory[frame]

            yaw_car = getYaw(traj=carTrajectory, vel=carVelocities, frame=max(1,frame), lastKnownYaw=lastKnownYaw_car)
            lastKnownYaw_car = yaw_car
            trained_car_rot = {'yaw':yaw_car}
            trained_car_vel = carVelocities[frame]

            trained_ped_pos = pedestrianTrajectory[frame]
            yaw_ped = getYaw(traj=pedestrianTrajectory, vel=pedestrianVelocities, frame=max(1,frame), lastKnownYaw=lastKnownYaw_ped)
            lastKnownYaw_ped = yaw_ped
            trained_ped_rot = {'yaw':yaw_ped}
            trained_ped_vel = pedestrianVelocities[frame]

            dict_trained_car = {'WorldRefLocation' : trained_car_pos,
                                'WorldRefRotation' : trained_car_rot,
                                'VelocityRef' : trained_car_vel }

            dict_trained_ped = {'WorldRefLocation' : trained_ped_pos,
                                'WorldRefRotation' : trained_ped_rot,
                                'VelocityRef' : trained_ped_vel }

            injectPeopleDict[frame][DataGatherParams.MAGIC_TRAINED_PEDESTRIAN_ID] = dict_trained_ped
            injectCarsDict[frame][DataGatherParams.MAGIC_TRAINED_CAR_ID] = dict_trained_car

    if realTimeEnvOnline:
        def getTrainedPedestrianSpawnTransform(self) -> carla.Transform:
            assert DataGatherParams.MAGIC_TRAINED_PEDESTRIAN_ID in self.pedestriansReplayIdToSpawnData
            return self.pedestriansReplayIdToSpawnData[DataGatherParams.MAGIC_TRAINED_PEDESTRIAN_ID].replayDataPerFrame.transform

        def getTrainedCarSpawnTransform(self) -> carla.Transform:
            assert DataGatherParams.MAGIC_TRAINED_CAR_ID in self.carReplayIdToSpawnData
            return self.carReplayIdToSpawnData[DataGatherParams.MAGIC_TRAINED_CAR_ID].replayDataPerFrame.transform

    def loadEntitiesReplayData(self, entitiesDict, loadCars):
        self.maxFramesIndexRecorded = min(max(entitiesDict.keys()), (self.maxFramesIndexRecorded if self.maxFramesIndexRecorded != None else 99999999))

        # Load and transform the entities data on each frame
        for i, frame in enumerate(sorted(entitiesDict.keys())):
            if frame == FRAME_INDEX_FOR_SPAWN_DETAILS:
                continue

            entitiesOnFrame = entitiesDict[frame]

            destinationData = self.carsData if loadCars == True else self.pedestriansData
            destinationData[frame] = {}
            destinationData_frame = destinationData[frame]

            # Read all entities on this frame
            for entity_id, entity_data in entitiesOnFrame.items():
                entity_loc = entity_data['WorldRefLocation']
                entity_loc = np.dot(self.worldRef_2_world_matrix, np.append(entity_loc, 1.0).T)

                entity_rot = entity_data['WorldRefRotation']

                if isinstance(entity_rot, dict):
                    pitch = float(entity_rot['pitch']) if 'pitch' in entity_rot else 0.0
                    yaw = float(entity_rot['yaw']) if 'yaw' in entity_rot else 0.0
                    roll = float(entity_rot['roll']) if 'roll' in entity_rot else 0.0
                else:
                    entity_rot = np.dot(self.worldRef_2_world_matrix[0:3,0:3], entity_rot)
                    pitch, roll, yaw = transforms3d.euler.mat2euler(entity_rot, axes='sxyz')
                    pitch = np.rad2deg(pitch)
                    roll = np.rad2deg(roll)
                    yaw = np.rad2deg(yaw)


                entity_vel = entity_data['VelocityRef']
                entity_vel = np.dot(self.worldRef_2_world_matrix[0:3,0:3], entity_vel.T)

                entity_vel = carla.Vector3D(x=entity_vel[0], y=entity_vel[1], z=entity_vel[2])
                entity_data_transf = EntityReplayDataPerFrame()
                entity_data_transf.transform = carla.Transform(location=carla.Location(entity_loc[0], entity_loc[1], entity_loc[2]),
                                                               rotation=carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
                entity_data_transf.velocity = entity_vel
                destinationData_frame[entity_id] = entity_data_transf

        # Load spawn details
        destinationSpawnData = self.carReplayIdToSpawnData if loadCars is True else self.pedestriansReplayIdToSpawnData
        assert FRAME_INDEX_FOR_SPAWN_DETAILS in entitiesDict, "The spawn details magic frame doesn't exist "
        spawnFrameDetails = entitiesDict[FRAME_INDEX_FOR_SPAWN_DETAILS]

        # First, add the spawning point for our trained agents cars/peds.
        if loadCars:
            if self.useRecordedTrainedCar:
                initFrameData= self.carsData[0][DataGatherParams.MAGIC_TRAINED_CAR_ID]
                spawnFrameDetails[DataGatherParams.MAGIC_TRAINED_CAR_ID] = EntitySpawnDetails(transform=initFrameData.transform,
                                                                             velocity=initFrameData.velocity,
                                                                             blueprint=self.TRAINED_CAR_BLUEPRINT,
                                                                             replayId=DataGatherParams.MAGIC_TRAINED_CAR_ID,
                                                                             color="255, 0, 0",
                                                                             driver_id=None,
                                                                             speed=0.0,
                                                                             targetPoint=None)
        else:
            if self.useRecordedTrainedPed:
                initFrameData = self.pedestriansData[0][DataGatherParams.MAGIC_TRAINED_PEDESTRIAN_ID]
                spawnFrameDetails[DataGatherParams.MAGIC_TRAINED_PEDESTRIAN_ID] = EntitySpawnDetails(transform=initFrameData.transform,
                                                                                    velocity=initFrameData.velocity,
                                                                                    blueprint=self.TRAINED_PEDESTRIAN_BLUEPRINT,
                                                                                    replayId=DataGatherParams.MAGIC_TRAINED_PEDESTRIAN_ID,
                                                                                    color="none",
                                                                                    driver_id="none",
                                                                                    speed=2.0, # doesn't matter
                                                                                    targetPoint=initFrameData.transform) # doesn't matter

            # Then process them all and make connections
        for entity_id, entity_spawn_details in spawnFrameDetails.items():
            assert isinstance(entity_spawn_details, EntitySpawnDetails)

            # Transform to carla some stuff inside the structure (Deserialize)
            #entity_spawn_details.convertToWorldSpace(self.worldRef_2_world_matrix)
            entity_spawn_details.convertToCarla()
            destinationSpawnData[entity_id] = entity_spawn_details


    # Given a frame index, an entity by its spawn id and if's car or pedestrian, return the entity's data on the given frame
    # customReplayData is used when you don't want to use SELF data
    def getEntityDataPerFrame(self, frameId, spawnId, isCar, customReplayData=None) -> EntityReplayDataPerFrame:
        sourceData = customReplayData if customReplayData != None else (self.carsData if isCar == True else self.pedestriansData)
        sourceEntitySpawnIdToReplayId = self.carSpawnedIdToReplayId if isCar == True else self.pedestrianSpawnedIdToReplayId

        frameId = min(frameId, self.maxFramesIndexRecorded) if self.maxFramesIndexRecorded is not None else frameId
        assert frameId in sourceData
        sourceData_frame = sourceData[frameId]

        if spawnId not in sourceEntitySpawnIdToReplayId:
            print(f"WARNING entity spawn {spawnId} is not in the source mapping !!")
            return None

        replayId = sourceEntitySpawnIdToReplayId[spawnId]
        if replayId not in sourceData_frame:
            print(f"WARNING entity replay id {replayId} not in frame {frameId}. Probably destroyed ??")

        entityReplayDataPerFrame : EntityReplayDataPerFrame = sourceData_frame[replayId]
        if not isinstance(entityReplayDataPerFrame, EntityReplayDataPerFrame):
            entityReplayDataPerFrame = entityReplayDataPerFrame.replayDataPerFrame
            assert isinstance(entityReplayDataPerFrame, (EntityReplayDataPerFrame, np.ndarray, carla.Transform))
        return entityReplayDataPerFrame

# Powered by a map
class SimpleGridSpacePartition:
    def __init__(self, cellSize=1.0):
        self.cellSize = cellSize
        self.dataS : Dict[Tuple[int, int], int] = {} # dataS[(i,j)] = how many are in cell (i,j)

    def _worldLocationToGrid(self, loc:carla.Location) -> Tuple[int, int]:
        locInGridSpace = (int(loc.x/self.cellSize), int(loc.y/self.cellSize))
        return locInGridSpace

    # How many items are in grid around the given position ?
    def getItemsAround(self, loc: carla.Location) -> int:
        locInGridSpace = self._worldLocationToGrid(loc)
        return 0 if (locInGridSpace not in self.dataS) else self.dataS[locInGridSpace]

    # Add another item to the given location
    def occupy(self, loc: carla.Location):
        locInGridSpace = self._worldLocationToGrid(loc)
        if locInGridSpace not in self.dataS:
            self.dataS[locInGridSpace] = 0

        self.dataS[locInGridSpace] = 1

##################################### BEGIN BASIC FUNCTIONS ########################################
def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within a certain distance from a reference object.
    A vehicle in front would be something around 0 deg, while one behind around 180 deg.
        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :param d_angle_th_up: upper thereshold for angle
        :param d_angle_th_low: low thereshold for angle (optional, default is 0)
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


# Tests if a position is in front of an observer knowing its position and forward
def isPosInFaceOfObserverPos(observerForward, observerPos, targetPos):
    observeToPos = targetPos - observerPos
    return dot2D(observeToPos, observerForward) > 0

def compute_distance(location_1, location_2, on2DOnly = False):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = 0.0 if on2DOnly else (location_2.z - location_1.z)
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm

def dot3D(v1, v2):
    return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z)

def dot2D(v1, v2):
    return (v1.x * v2.x + v1.y * v2.y)

    # Given a carla transform, create a rotation matrix and return it
def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

# Get the bbox of an actor in world space
def getActorWorldBBox(actor: carla.Actor, actorCustomTransformMatrix=None):
    actorBBox = actor.bounding_box
    cords = np.zeros((8, 4))

    # Get the box extent
    extent = actorBBox.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])

    bb_transform = carla.Transform(actorBBox.location)
    bb_matrix = get_matrix(bb_transform)
    
    actor_world_matrix = get_matrix(actor.get_transform()) if actorCustomTransformMatrix is None else actorCustomTransformMatrix
    
    bb_world_matrix = np.dot(actor_world_matrix, bb_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords
if realTimeEnvOnline:
    def convertMToCM(loc : carla.Location):
        newLoc = carla.Location(loc.x * 100.0, loc.y * 100.0, loc.z * 100.0)
        return newLoc

##################################### END BASIC FUNCTIONS ########################################


##################################### BEGIN LOCAL RENDERING FUNCTIONS ########################################
class RenderUtils(object):
    class EventType(Enum):
        EV_NONE = 0
        EV_QUIT = 1
        EV_SWITCH_TO_DEPTH = 2
        EV_SWITCH_TO_RGBANDSEG = 3
        EV_SIMPLIFIED_TOPVIEW = 4

    @staticmethod
    def draw_image(surface, image, blend=False):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))

    @staticmethod
    def get_font():
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)

    @staticmethod
    def get_input_event():
        for event in pygame.event.get():
            # See QUIT first then swap
            if event.type == pygame.QUIT:
                return RenderUtils.EventType.EV_QUIT
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return RenderUtils.EventType.EV_QUIT
                if event.key == pygame.K_d:
                    return RenderUtils.EventType.EV_SWITCH_TO_DEPTH
                if event.key == pygame.K_s:
                    return RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG
                if event.key == pygame.K_t:
                    return RenderUtils.EventType.EV_SIMPLIFIED_TOPVIEW
        return RenderUtils.EventType.EV_NONE

# Manager to synchronize output from different sensors.

##################################### END LOCAL RENDERING FUNCTIONS ########################################


##################################### BEGIN Traffic Management FUNCTIONS ########################################
class CustomGlobalRoutePlanner(GlobalRoutePlanner):
    def __init__(self, dao):
        super(CustomGlobalRoutePlanner, self).__init__(dao=dao)

    def compute_direction_velocities(self, origin, velocity, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)

        origin_xy = np.array([origin.x, origin.y])
        velocity_xy = np.array([velocity.x, velocity.y])
        first_node_xy = self._graph.nodes[node_list[0]]['vertex']
        first_node_xy = np.array([first_node_xy[0], first_node_xy[1]])
        target_direction_vector = first_node_xy - origin_xy
        target_unit_vector = np.array(target_direction_vector) / np.linalg.norm(target_direction_vector)

        vel_s = np.dot(velocity_xy, target_unit_vector)

        unit_velocity = velocity_xy / (np.linalg.norm(velocity_xy) + 1e-8)
        angle = np.arccos(np.clip(np.dot(unit_velocity, target_unit_vector), -1.0, 1.0))
        vel_perp = np.linalg.norm(velocity_xy) * np.sin(angle)
        return vel_s, vel_perp

    def compute_distance(self, origin, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)
        # print('Node list:', node_list)
        first_node_xy = self._graph.nodes[node_list[1]]['vertex']
        # print('Diff:', origin, first_node_xy)

        # distance = 0.0
        distances = []
        distances.append(np.linalg.norm(np.array([origin.x, origin.y, 0.0]) - np.array(first_node_xy)))

        for idx in range(len(node_list) - 1):
            distances.append(super(CustomGlobalRoutePlanner, self)._distance_heuristic(node_list[idx], node_list[idx + 1]))
        # print('Distances:', distances)
        # import pdb; pdb.set_trace()
        return np.sum(distances)


######################### BEGIN: Weather and scene methods ####################
class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        min_alt, max_alt = [20, 90]
        self.altitude = 0.5 * (max_alt + min_alt) + 0.5 * (max_alt - min_alt) * math.cos(self._t)

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 60.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, world, changing_weather_speed):
        self.world = world
        self.reset()
        self.weather = world.get_weather()
        self.changing_weather_speed = changing_weather_speed
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

    def reset(self):
        weather_params = carla.WeatherParameters(sun_altitude_angle=90.)
        self.world.set_weather(weather_params)

    def tick(self):
        self._sun.tick(self.changing_weather_speed)
        self._storm.tick(self.changing_weather_speed)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude
        self.world.set_weather(self.weather)

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)
######################### END: Weather and scene methods ####################


######################### BEGIN: Sync methods ####################


class SensorsDataManagement(object):
    def __init__(self, world, fps, sensorsDict):
        self.world = world
        self.sensors = sensorsDict  # [name->obj instance]
        self.frame = None

        self.delta_seconds = 1.0 / fps
        self._queues = {}  # Data queues for each sensor given + update messages
        self.registeredEvents = {} # From sensorname to registration id
        self.RenderAsDepth = RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG

        # returns the id of the registered event
        def make_queue(register_event, sensorname):
            q = queue.Queue()
            registration_id = register_event(q.put)
            assert self._queues.get(sensorname) is None
            self._queues[sensorname] = q

            self.registeredEvents[sensorname] = registration_id

        make_queue(self.world.on_tick, "worldSnapshot") # The ontick register event
        for name, inst in self.sensors.items():
            make_queue(inst.listen, name)

    def cleanup(self):
        for name, inst in self.sensors.items():
            inst.stop()

        for sensorname, sensor_reg_id in self.registeredEvents.items():
            if sensor_reg_id is not None:
                self.world.remove_on_tick(sensor_reg_id)


    # Tick the manager using a timeout to get data and a targetFrame that the parent is looking to receive data for
    def tick(self, targetFrame, timeout):
        #logging.log(logging.INFO, ("Ticking manager to get data for targetFrame {0}").format(targetFrame))

        """
        print("--Debug print the queue status")
        for name,inst in self._queues.items():
             print(f"--queue name {name} has size {inst.qsize()}")
        """

        data = {name: self._retrieve_data(targetFrame, q, timeout) for name, q in self._queues.items()}
        assert all(inst.frame == targetFrame for key,inst in data.items())  # Need to get only data for the target frame requested
        #logging.log(logging.INFO, ("Got data for frame {0}").format(targetFrame))

        return data

    # Gets the target frame from a sensor queue
    def _retrieve_data(self, targetFrame,  sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            # assert data.frame <= targetFrame, ("You are requesting an old frame which was already processed !. data %d, target %d" % (data.frame, targetFrame))
            if data.frame == targetFrame:
                return data

######################### END: Sync methods ####################


# This is used to understand the parameters - input/output of an online or offline data gathering process
# Here we put static things about the environment, simulation , params etc.
class DataGatherParams:
    # First, some fixed parameters for data capture. Maybe some of them must be expxorted as well
    #---------------------------------------------------------------
    frame_step = 10  # Save one image at every frame_step frames. Will be overwritten if a setting is specified
    MIN_WAYPOINTS_TO_DESTINATION = 20

    CAMERAREPLAY_DESC_FILE = "camera_replay.p"
    MAGIC_TRAINED_PEDESTRIAN_ID = "mariaPed"
    MAGIC_TRAINED_CAR_ID = "mariaCar"

# How to find the episode LENGTH
    # Explanation: In one second you have fixedFPS. We record a measure at each frame_step frames. If fixedFPS=30
    # then in one second you have 30/frame_step captures.
    # If number_of_frames_to_capture = 200 => An episode length is 100/3 seconds

    # A relative position for Z of the observer location..to keep things simple
    RAYCAST_ACTOR_Z = 52

    # Camera specifications, size, local pos and rotation to car.
    # NOTE: THE Z values before of cameras are relative to the above RAYCAST_ACTOR_Z value
    image_size = [800, 600]
    # camera_local_pos = carla.Location(x=0.3, y=0.0, z=1.3)  # [X, Y, Z] of local camera
    rotationToGround_topview = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    camera_front_transform_normal = carla.Transform(location = carla.Location(x=0.0, y = 0.0, z=RAYCAST_ACTOR_Z - 10), rotation=rotationToGround_topview)
    # Lidar specifications
    lidar_front_transform_normal = carla.Transform(location = carla.Location(x=0.0, y = 0.0, z=RAYCAST_ACTOR_Z - 11), rotation=rotationToGround_topview)

    fov = 70
    isAutopilotForPlayerVehicle = True

    # Orientations for replay mode
    rotationToGround_replay = carla.Rotation(pitch=-30.0, yaw=0.0, roll=0.0)
    camera_front_transform_replay = carla.Transform(location = carla.Location(x=27.0, y = 0.0, z=-10), rotation=rotationToGround_topview)
    #carla.Transform(location = carla.Location(x=-15.0, y = 0.0, z=-5), rotation=rotationToGround_replay)

    lidar_front_transform_replay = carla.Transform(location = carla.Location(x=27.0, y = 0.0, z=-11), rotation=rotationToGround_topview)
    #carla.Transform(location = carla.Location(x=-15.0, y = 0.0, z=-5), rotation=rotationToGround_replay)

    # Some orientation for replay mode with statistics agents
    #-----
    # Top ones
    camera_front_transform_replay_stats_top = carla.Transform(location = carla.Location(x=-2.0, y = 0.0, z=-30), rotation=rotationToGround_topview)
    lidar_front_transform_replay_stats_top = carla.Transform(location = carla.Location(x=-2.0, y = 0.0, z=-31), rotation=rotationToGround_topview)
    # Perspective ones
    rotationToGround_perspective_stats_1 = carla.Rotation(pitch=-20.0, yaw=0.0, roll=0.0)
    camera_front_transform_replay_stats_perspective_1 = carla.Transform(location = carla.Location(x=-12.0, y = 0.0, z=-45),
                                                                    rotation=rotationToGround_perspective_stats_1)
    lidar_front_transform_replay_stats_perspective_1 = carla.Transform(location = carla.Location(x=-12.0, y = 0.0, z=-46),
                                                                    rotation=rotationToGround_perspective_stats_1)
    camera_front_transform_replay_stats = camera_front_transform_replay_stats_perspective_1
    lidar_front_transform_replay_stats = lidar_front_transform_replay_stats_perspective_1


    #----

    def moveTransformInForwardDirectionOfAnother(self, attachedTransform, baseActorTransform, distToMove):
        forwardVector = baseActorTransform.get_forward_vector()#.make_unit_vector()
        displacementVector = carla.Vector3D(1.0, 0.0, 0.0) * distToMove #forwardVector * distToMove
        newAttachedTransform = carla.Transform(location=attachedTransform.location + displacementVector, rotation=attachedTransform.rotation)
        return newAttachedTransform

    def getObserverFrontCameraTransform(self, observerActorTransform):
        if self.simulationOptions.simulateReplay:
            if self.simulationOptions.useRecordedTrainedAgents:
                return self.camera_front_transform_replay_stats
            else:
                return self.camera_front_transform_replay
        else:
            res = self.moveTransformInForwardDirectionOfAnother(attachedTransform=self.camera_front_transform_normal,
                                                                      baseActorTransform=observerActorTransform,
                                                                      distToMove=self.sensorsDisplacementDist)

            # Invert the yaw with pitch
            #res.rotation.roll = observerActorTransform.rotation.yaw

            return res

    def getObserverFrontLidarTransform(self, observerActorTransform):
        if self.simulationOptions.simulateReplay:
            if self.simulationOptions.useRecordedTrainedAgents:
                return self.lidar_front_transform_replay_stats
            else:
                return self.lidar_front_transform_replay
        else:
            res = self.moveTransformInForwardDirectionOfAnother(attachedTransform=self.lidar_front_transform_normal,
                                                                      baseActorTransform=observerActorTransform,
                                                                      distToMove=self.sensorsDisplacementDist)
            return res

    # This gets the prefered intiial point of observer camera when replaying stats
    def getPreferedReplyingStatsAttachedObserverTransform(self):
        if self.simulationIsReplayingStats is False:
            return None

        # Prefer the car first :)
        if self.simulationOptions.useRecordedTrainedCar:
            return self.simulationOptions.getTrainedCarSpawnTransform()
        elif self.simulationOptions.useRecordedTrainedAgents:
            return self.simulationOptions.getTrainedPedestrianSpawnTransform()
        return None

    # Environment settings
    OUTPUT_SEG = "CameraSeg"
    OUTPUT_SEGCITIES = "CameraSegCities"
    OUTPUT_DEPTH = "CameraDepth"
    OUTPUT_DEPTHLOG = "CameraDepthProc"
    OUTPUT_RGB = "CameraRGB"
    OUTPUT_LIDARINTENSITY = "LidarIntensity"

    OUTPUT_SEG_REPLAY = "CameraSeg_rep"
    OUTPUT_SEGCITIES_REPLAY = "CameraSegCities_rep"
    OUTPUT_DEPTH_REPLAY = "CameraDepth_rep"
    OUTPUT_DEPTHLOG_REPLAY = "CameraDepthProc_rep"
    OUTPUT_RGB_REPLAY = "CameraRGB_rep"
    OUTPUT_LIDARINTENSITY_REPLAY = "LidarIntensity_rep"
    OUTPUT_POINTCLOUD_REPLAY = "PlyFilesInReplay"

    ENABLED_SAVING = True  # If activated, output saving data is enabled
    ENABLED_SAVING_IN_REPLAY = True # If activated, output saving data is enabled in replay mode
    CLEAN_PREVIOUS_DATA = False  # If activated, previous data folder is deleted

    # If true, then car stays and waits for pedestrians and other cars to move around. If
    # If false, then on each frame it moves to a random proximity waypoint
    STATIC_CAR = False

    def prepareEpisodeOutputFolders(self, episodeDataPath, forcedObserverSpawnPointIndex,
                                    forcedObserverSpawnPointTransform):
        if self.CLEAN_PREVIOUS_DATA:
            if os.path.exists(episodeDataPath):
                shutil.rmtree(episodeDataPath)

        if True: #not os.path.exists(episodeDataPath):
            #os.makedirs(episodeDataPath, exist_ok=True)

            # Cache the folders for storing the outputs
            self.outputFolder_depth  = self.getOutputFolder_depth(episodeDataPath)
            self.outputFolder_depthLog = self.getOutputFolder_depthLog(episodeDataPath)
            self.outputFolder_rgb    = self.getOutputFolder_rgb(episodeDataPath)
            self.outputFolder_seg    = self.getOutputFolder_seg(episodeDataPath)
            self.output_segcities    = self.getOutputFolder_segcities(episodeDataPath)

            self.outputFolder_lidarIntensityRGB = self.getOutputFolder_lidarIntensityRGB(episodeDataPath)
            self.outputFolder_lidarPointCloud = self.getOutputFolder_lidarPointCloud(episodeDataPath)

            os.makedirs(self.outputFolder_depth, exist_ok=True)
            os.makedirs(self.outputFolder_depthLog, exist_ok=True)
            os.makedirs(self.outputFolder_rgb, exist_ok=True)
            os.makedirs(self.outputFolder_seg, exist_ok=True)
            os.makedirs(self.output_segcities, exist_ok=True)

            os.makedirs(self.outputFolder_lidarIntensityRGB, exist_ok=True)
            if not os.path.exists(self.outputFolder_lidarPointCloud):
                os.makedirs(self.outputFolder_lidarPointCloud, exist_ok=True)


            forcedObserverLocation = forcedObserverSpawnPointTransform.location
            forcedObserverRotation = forcedObserverSpawnPointTransform.rotation

            # Write the metadata JSON file
            dictionary = {
                "sceneName": self.sceneName,
                "observerSpawnIndex": forcedObserverSpawnPointIndex if forcedObserverSpawnPointIndex != None else -1,
                "observerSpawn_X": forcedObserverLocation.x,
                "observerSpawn_Y": forcedObserverLocation.y,
                "observerSpawn_Z": forcedObserverLocation.z,
                "observerSpawn_Pitch": forcedObserverRotation.pitch,
                "observerSpawn_Yaw": forcedObserverRotation.yaw,
                "observerSpawn_Roll": forcedObserverRotation.roll,
            }

            # Serializing json
            json_object = json.dumps(dictionary, indent=4)

            # Writing to sample.json
            with open(os.path.join(episodeDataPath, "metadata.json"), "w") as outfile:
                outfile.write(json_object)

            shutil.copy(self.args.pathToDatasetConfigFile, episodeDataPath)

    def isReplay(self):
        return self.simulationOptions.simulateReplay

    def prepareSceneOutputFolders(self, sceneDataPath):
        if self.CLEAN_PREVIOUS_DATA:
            if os.path.exists(sceneDataPath):
                shutil.rmtree(sceneDataPath)

        if True: #not os.path.exists(sceneDataPath):
            os.makedirs(sceneDataPath, exist_ok=True)
    #----------------------------------------------------------------
    def getOutputFolder_seg(self, baseFolder):
        return os.path.join(baseFolder, self.OUTPUT_SEG if not self.isReplay() else self.OUTPUT_SEG_REPLAY)

    def getOutputFolder_segcities(self, baseFolder):
        return os.path.join(baseFolder, self.OUTPUT_SEGCITIES if not self.isReplay() else self.OUTPUT_SEGCITIES_REPLAY)

    def getOutputFolder_rgb(self, baseFolder):
        return os.path.join(baseFolder, self.OUTPUT_RGB if not self.isReplay() else self.OUTPUT_RGB_REPLAY)

    def getOutputFolder_depth(self, baseFolder):
        return os.path.join(baseFolder, self.OUTPUT_DEPTH if not self.isReplay() else self.OUTPUT_DEPTH_REPLAY)

    def getOutputFolder_depthLog(self, baseFolder):
        return os.path.join(baseFolder, self.OUTPUT_DEPTHLOG if not self.isReplay() else self.OUTPUT_DEPTHLOG_REPLAY)

    def getOutputFolder_lidarIntensityRGB(self, baseFolder):
        return os.path.join(baseFolder, self.OUTPUT_LIDARINTENSITY if not self.isReplay() else self.OUTPUT_LIDARINTENSITY_REPLAY)

    def getOutputFolder_lidarPointCloud(self, baseFolder):
        return baseFolder if not self.isReplay() else os.path.join(baseFolder, self.OUTPUT_POINTCLOUD_REPLAY)

    @staticmethod
    # Configures a camera blueprint and returns a dictionary with intrisics of the camera
    def configureCameraBlueprint(cameraBp):
        cameraBp.set_attribute('image_size_x', str(DataGatherParams.image_size[0]))
        cameraBp.set_attribute('image_size_y', str(DataGatherParams.image_size[1]))
        cameraBp.set_attribute('fov', str(DataGatherParams.fov))
        # Set the time in seconds between sensor captures
        # cameraBp.set_attribute('sensor_tick', '1.0')

        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        image_w = cameraBp.get_attribute("image_size_x").as_int()
        image_h = cameraBp.get_attribute("image_size_y").as_int()
        fov = cameraBp.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        cameraBP_intrisics = {'image_w':image_w, 'image_h':image_h,
                              'fov':fov, 'focal':focal, 'K':K}
        return cameraBP_intrisics


    # outputEpisodeDataPath will be used to put the scene data + simulation data such that you can read that folder and load it in the RLAgent directly after as a dataset example
    # Num frames is for how many number of frames the simulation should happen. -1 means infinite
    # Use copyScenePaths to copy the ply files from base data path to each sampled scene
    def __init__(self,  outputEpisodeDataPath,
                        sceneName=None,
                        episodeIndex = 0,
                        numFrames=-1,
                        maxNumberOfEpisodes=1,
                        lidarData=None,
                        mapsToTest = ["Town03"],
                        copyScenePaths=False,
                        simulationOptions=None,
                        host="localhost",
                        port=2000,
                        sensorsDisplacementDist = 0,
                        args=None):  # If true, the RGB/semantic images will be gathered from a hero car perspective instead of top view

        self.outputEpisodesBasePath = os.path.join(outputEpisodeDataPath, sceneName) if sceneName != None else outputEpisodeDataPath
        self.sceneName = sceneName
        self.outputEpisodesBasePath_currentSceneData = self.outputEpisodesBasePath
        os.makedirs(self.outputEpisodesBasePath_currentSceneData, exist_ok=True)
        self.outputCurrentEpisodePath = None
        self.episodeIndex = episodeIndex
        self.numFrames = numFrames
        self.maxNumberOfEpisodes = maxNumberOfEpisodes
        self.mapsToTest = mapsToTest
        self.host = host
        self.port = port
        self.collectSceneData = False
        self.copyScenePaths = copyScenePaths
        self.use_hero_actor_forDataGathering = None # Will be filled later
        self.lidarData = lidarData
        self.simulationOptions : SimOptions = simulationOptions
        self.simulationIsReplayingStats = False
        self.args = args
        self.sensorsDisplacementDist = sensorsDisplacementDist

        # If simulation is enabled, modify the episodes output files folders depending on various options in simulation
        if self.simulationOptions.simulateReplay and self.simulationOptions.useRecordedTrainedAgents:
            self.simulationIsReplayingStats = True
            replayStatsEpIndex = str(int(self.simulationOptions.simEpisodeIndex))
            self.OUTPUT_SEG_REPLAY += replayStatsEpIndex
            self.OUTPUT_SEGCITIES_REPLAY += replayStatsEpIndex
            self.OUTPUT_DEPTH_REPLAY += replayStatsEpIndex
            self.OUTPUT_DEPTHLOG_REPLAY += replayStatsEpIndex
            self.OUTPUT_RGB_REPLAY += replayStatsEpIndex
            self.OUTPUT_LIDARINTENSITY_REPLAY += replayStatsEpIndex
            self.OUTPUT_POINTCLOUD_REPLAY += replayStatsEpIndex

        # Check if we need to rewrite the point cloud scene files
        if not self.simulationOptions.simulateReplay:
            needSceneReconstruct = False
            if args.onlineEnvSettings.forceSceneReconstruction:
                needSceneReconstruct = True
            else:
                # Check if the ply files are there
                plyFilesAreThere = False
                src_files = os.listdir(self.outputEpisodesBasePath_currentSceneData)
                for file_name in src_files:
                    if "ply" in pathlib.Path(file_name).suffix:
                        plyFilesAreThere = True
                        break
                needSceneReconstruct = plyFilesAreThere == False

            self.rewritePointCloud = needSceneReconstruct # TODO fix from params
        else:
            self.rewritePointCloud = False

    def isSavingEnabled(self) -> bool:
        if self.simulationOptions.simulateReplay:
            return self.ENABLED_SAVING_IN_REPLAY
        else:
            return self.ENABLED_SAVING

    # Controls what to save when replay....a quick way to control performance when replay
    def isDetailedSavedEnabled(self) -> bool:
        if self.simulationOptions.simulateReplay:
            return (self.ENABLED_SAVING_IN_REPLAY and False)
        else:
            return self.ENABLED_SAVING

    def getSavingFrameStep(self):
        if self.simulationOptions.simulateReplay:
            return self.frame_step
        else:
            return self.frame_step

    def saveHeroCarPerspectiveData(self, syncData, frameId, heroVehicleSensors, heroVehicleIntrisics, world_2_worldRef):
        if self.use_hero_actor_forDataGathering:
            worldSnapshot = syncData['worldSnapshot']
            image_seg_data = syncData["seg"]
            image_rgb_data = syncData["rgb"]
            image_depth_data = syncData["depth"]
            lidar_data = syncData["lidar"] if "lidar" in syncData else None
            assert (image_seg_data.frame == image_rgb_data.frame == image_depth_data.frame == worldSnapshot.frame) and (lidar_data is None or lidar_data.frame == worldSnapshot.frame), "sensors desync"

            # Get the sensor instances
            image_rgb_sensor = heroVehicleSensors['rgb']
            lidar_sensor = heroVehicleSensors['lidar']


            # Save output from sensors to disk if requested
            if self.isSavingEnabled() and (frameId % self.getSavingFrameStep() == 0):
                fileName = ("%06d.png" % frameId)
                # Save RGB
                image_rgb_data.save_to_disk(os.path.join(self.outputFolder_rgb, fileName))

                # Save seg, CARLA
                image_seg_data.save_to_disk(os.path.join(self.outputFolder_seg, fileName))

                # Save depth
                if self.isDetailedSavedEnabled():
                    image_depth_data.save_to_disk(os.path.join(self.outputFolder_depth, fileName))
                    image_depth_data.convert(carla.ColorConverter.Depth)
                    image_depth_data.save_to_disk(os.path.join(self.outputFolder_depthLog, fileName))

                if lidar_data and self.isDetailedSavedEnabled():
                    # Get the camera we want to project lidar data on, intrisics of it and transform
                    image_rgb_sensor_intrisics = heroVehicleIntrisics['rgb']
                    camera_K = image_rgb_sensor_intrisics['K']
                    camera_image_w = image_rgb_sensor_intrisics['image_w']
                    camera_image_h = image_rgb_sensor_intrisics['image_h']

                    # Get some transformation matrices that we need
                    # This (4, 4) matrix transforms the points from lidar space to world space.
                    lidar_2_world = lidar_sensor.get_transform().get_matrix()

                    # This (4, 4) matrix transforms the points from world to sensor coordinates.
                    world_2_camera = np.array(image_rgb_sensor.get_transform().get_inverse_matrix())


                    # Get the lidar data and convert it to a numpy array.
                    p_cloud_size = len(lidar_data)
                    p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
                    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

                    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's focus on the 3D points.
                    intensity = np.array(p_cloud[:, 3])

                    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
                    local_lidar_points = np.array(p_cloud[:, :3]).T

                    # Add an extra 1.0 at the end of each 3d point so it becomes of
                    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
                    local_lidar_points = np.r_[local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

                    # Transform the points from lidar space to world space.
                    world_points = np.dot(lidar_2_world, local_lidar_points)

                    # Transform the points from world space to camera space.
                    sensor_points = np.dot(world_2_camera, world_points)

                    # New we must change from UE4's coordinate system to an "standard"
                    # camera coordinate system (the same used by OpenCV):

                    # ^ z                       . z
                    # |                        /
                    # |              to:      +-------> x
                    # | . x                   |
                    # |/                      |
                    # +-------> y             v y

                    # This can be achieved by multiplying by the following matrix:
                    # [[ 0,  1,  0 ],
                    #  [ 0,  0, -1 ],
                    #  [ 1,  0,  0 ]]

                    # Or, in this case, is the same as swapping:
                    # (x, y ,z) -> (y, -z, x)
                    point_in_camera_coords = np.array([
                        sensor_points[1],
                        sensor_points[2] * -1,
                        sensor_points[0]])

                    # Finally we can use our K matrix to do the actual 3D -> 2D.
                    points_2d = np.dot(camera_K, point_in_camera_coords)

                    # Remember to normalize the x, y values by the 3rd value.
                    points_2d = np.array([
                        points_2d[0, :] / points_2d[2, :],
                        points_2d[1, :] / points_2d[2, :],
                        points_2d[2, :]])

                    # Select only the points in front of the camera
                    # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
                    # contains all the y values of our points. In order to properly
                    # visualize everything on a screen, the points that are out of the screen
                    # must be discarded, the same with points behind the camera projection plane.
                    points_2d = points_2d.T
                    intensity = intensity.T
                    points_in_canvas_mask = \
                        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < camera_image_w) & \
                        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < camera_image_h) & \
                        (points_2d[:, 2] > 0.0)
                    points_2d = points_2d[points_in_canvas_mask]
                    intensity = intensity[points_in_canvas_mask]
                    world_points = world_points.T
                    world_points = world_points[points_in_canvas_mask]

                    # Extract the screen coords (uv) as integers.
                    u_coord = points_2d[:, 0].astype(np.int)
                    v_coord = points_2d[:, 1].astype(np.int)


                    if True: # TODO, arg - save the lidar data too

                        # Step 1: Save the RGB, Seg and Seg colors
                        img_rgb_array = np.copy(np.frombuffer(image_rgb_data.raw_data, dtype=np.dtype("uint8")))
                        img_rgb_array = np.reshape(img_rgb_array, (image_rgb_data.height, image_rgb_data.width, 4))
                        img_rgb_array = img_rgb_array[:, :, :3][:, :, ::-1]

                        img_seg_array = np.frombuffer(image_seg_data.raw_data, dtype=np.dtype("uint8")) # No copy needed we are not modifying this
                        img_seg_array = np.reshape(img_seg_array, (image_seg_data.height, image_seg_data.width, 4))
                        img_seg_label = img_seg_array[:,:,2] # The image comes in BGRA format, the label is encoded in the R channel

                        # Now convert and save the real data...
                        plyDataPoints = {}
                        # We must have on each row:  [(x,y,z, camX, camY)]
                        world_points_in_refFrame = np.dot(world_2_worldRef, world_points.T).T # convert points to world reference
                        points3D_and_cp = [None] * len(world_points_in_refFrame)
                        assert (len(world_points_in_refFrame) == len(u_coord) == len(v_coord) == len(points_2d)), "Dimensions do not coincide..something is fishy for sure"
                        for pointsIter in range(world_points_in_refFrame.shape[0]):
                            x,y,z,w = world_points_in_refFrame[pointsIter]
                            camX = u_coord[pointsIter]
                            camY = v_coord[pointsIter]
                            points3D_and_cp[pointsIter] = (x,y,z, camX, camY)


                        # TODO: for visualization purposes, save motion files too later, use useEnvironmentPoints variable to control this, just reiterate over the same code again !
                        # Process the points see comments inside
                        useEnvironmentPoints = True
                        processPoints(points3D_and_cp, plyDataPoints, imageCameraIndex=0, rgbData = [img_rgb_array], segData = [img_seg_label],
                                      useEnvironmentPoints=useEnvironmentPoints, useRawCarlaSegLabels=True)

                        # Finally, save the output ply to files
                        outputFramePath_rgb =  os.path.join(self.outputFolder_lidarPointCloud,          ("{0:05d}.ply").format(frameId) if useEnvironmentPoints == True else ("{0:05d}_motion.ply").format(frameId))
                        outputFramePath_seg =  os.path.join(self.outputFolder_lidarPointCloud,          ("{0:05d}_seg.ply").format(frameId) if useEnvironmentPoints == True else ("{0:05d}_motionseg.ply").format(frameId))
                        outputFramePath_segColored = os.path.join(self.outputFolder_lidarPointCloud,    ("{0:05d}_segColor.ply").format(frameId) if useEnvironmentPoints == True else ("{0:05d}_motionsegColor.ply").format(frameId))

                        plyDataPointsFlattened = convertDictPointsToList(plyDataPoints)
                        save_3d_pointcloud_asRGB(plyDataPointsFlattened, outputFramePath_rgb)  # TODO: save one file with test_x.ply, using RGB values, another one test_x_seg.ply using segmented data.
                        save_3d_pointcloud_asSegLabel(plyDataPointsFlattened, outputFramePath_seg)
                        save_3d_pointcloud_asSegColored(plyDataPointsFlattened, outputFramePath_segColored)

                        # Step 2: Save the intensity functions. NOTE: the img_rgb_array is modified so don't rely on it anymore
                        # Since at the time of the creation of this script, the intensity function
                        # is returning high values, these are adjusted to be nicely visualized.
                        intensity = 4 * intensity - 3
                        color_map = np.array([
                            np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                            np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                            np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T

                        dot_extent = 2
                        # Draw the 2d points on the image as squares of extent args.dot_extent.
                        for i in range(len(points_2d)):
                            # I'm not a NumPy expert and I don't know how to set bigger dots
                            # without using this loop, so if anyone has a better solution,
                            # make sure to update this script. Meanwhile, it's fast enough :)
                            img_rgb_array[v_coord[i]-dot_extent : v_coord[i]+dot_extent,
                            u_coord[i]-dot_extent : u_coord[i]+dot_extent] = color_map[i]

                        # Save the image using Pillow module.
                        lidarIntensityImage = Image.fromarray(img_rgb_array)
                        lidarIntensityImage.save(os.path.join(self.outputFolder_lidarIntensityRGB, fileName))

                        # Step 3: Convert seg data to citiscape and save in that format too
                        image_seg_data.convert(carla.ColorConverter.CityScapesPalette)
                        image_seg_data.save_to_disk(os.path.join(self.output_segcities, fileName))
                else: # Not saving LIDAR< but save the seg citiscapes even in not detailed mode
                    image_seg_data.convert(carla.ColorConverter.CityScapesPalette)
                    image_seg_data.save_to_disk(os.path.join(self.output_segcities, fileName))
                 # TODO:
            else:
                image_seg_data.convert(carla.ColorConverter.CityScapesPalette)
                image_depth_data.convert(carla.ColorConverter.Depth)


    # Save some simulation data on the episode's output path
    def saveSimulatedData(self, vehiclesData, pedestriansData, world_2_worldRefMatrix):
        try:
            if True: #self.use_hero_actor and self.outputCurrentEpisodePath:
                filepathsAndDictionaries = {'pedestrians': (pedestriansData, os.path.join(self.outputCurrentEpisodePath, "people.p")),
                                            'cars': (vehiclesData, os.path.join(self.outputCurrentEpisodePath, "cars.p")),
                                            'pedestrians_old': (pedestriansData, os.path.join(self.outputCurrentEpisodePath, "people_old.p")),
                                            'cars_old': (vehiclesData, os.path.join(self.outputCurrentEpisodePath, "cars_old.p"))
                                            }

                for key, value in filepathsAndDictionaries.items():
                    dataObj = value[0]

                    if "old" in key:
                        del dataObj[FRAME_INDEX_FOR_SPAWN_DETAILS]

                    filePath = value[1]
                    with open(filePath, mode="wb") as fileObj:
                        pickle.dump(dataObj, fileObj, protocol=2)  # Protocol 2 because seems to be compatible between 2.x and 3.x !


                # Save the world to world reference point file matrix
                with open(os.path.join(self.outputCurrentEpisodePath, DataGatherParams.CAMERAREPLAY_DESC_FILE), "wb") as cameraDescReplayFile:
                    pickle.dump(world_2_worldRefMatrix, cameraDescReplayFile, protocol=2)

        except Exception:
            traceback.print_exc()

    def prepareSceneOutput(self):
        # Output reconstructed  plys, centering, and stuff
        datasetOptions = CreateDefaultDatasetOptions_CarlaRealTime(metadata=None, forcedNumFramesPerEpisodes=self.numFrames)
        datasetOptions.setEnvironmentParams(dataPath=self.outputCurrentEpisodePath,
                                                pos_x=self.args.camera_pos_x,
                                                pos_y=self.args.camera_pos_y,
                                                width=self.args.width,
                                                depth=self.args.depth,
                                                height=self.args.height)

        reconstruction, final_tensor, people, cars, scale, people_dict, cars_2D, people_2D, \
        valid_ids, car_dict, init_frames, init_frames_cars, centering = reconstruct3D_ply(self.outputCurrentEpisodePath,
                                                                                               recalculate=True if self.simulationOptions.simulateReplay is False else False,
                                                                                               read_3D=True,
                                                                                               datasetOptions=datasetOptions)

class RenderType(Enum):
    RENDER_NONE = 1 # No local rendering
    RENDER_COLORED=2 # Full RGB/ semantic rendering of the scene
    RENDER_SIMPLIFIED=3  # Simplified dots/lines rendering

# How is RLAgent controlling the pedestrian
class ControlledPedestrianType:
    CONTROLLED_PEDESTRIAN_AUTHORITIVE= 1, # RLAgents sets the position and orientation, no control from Engine
    CONTROLLED_PEDESTRIAN_AUTHORITIVE_WITH_POSE = 2 # Same as above, RLAgents also sets the POSE
    CONTROLLED_PEDESTRIAN_ENGINE_CONTROLLED = 3 # Engine controls the agent, RLAgent that sends the velocity

class RenderOptions:
    def __init__(self, renderType : RenderType, topViewResX, topViewResY): # Res of the output vis
        if ENABLE_LOCAL_SCENE_RENDERING == False:
            assert renderType != RenderType.RENDER_NONE

        self.sceneRenderType = renderType
        self.topViewResX = topViewResX
        self.topViewResY = topViewResY

class ControlledCarSpawnParams:
    def __init__(self, position : carla.Location, isPositionInVoxels, yaw : float):
        self.position = position
        self.yaw = yaw
        self.isPositionsInVoxels = isPositionInVoxels

class ControlledPedestrianSpawnParams:
    def __init__(self, position : carla.Location, isPositionInVoxels, yaw : float, type : ControlledPedestrianType):
        self.position = position
        self.yaw = yaw
        self.isPositionsInVoxels = isPositionInVoxels
        self.type = type

class EnvSetupParams:
    # Some fixed set of params first
    #-------------------------------
    # Sim params
    fixedFPS = 17  # FPS for recording and simulation
    tm_port = 8000
    speedLimitExceedPercent = -50 # Exceed the speed limit by 50%
    distanceBetweenVehiclesCenters = 2.0 # Normal distance to keep between one distance vehicle to the one in its front
    synchronousComm = True
    useOnlySpawnPointsNearCrossWalks = False


    vehicles_filter_str = "vehicle.*"
    walkers_filter_str = "walker.pedestrian.*"

    # To promote having agents around the player spawn position, we randomly select F * numPedestrians locations as start/destination points
    PedestriansSpawnPointsFactor = 1000
    PedestriansDistanceBetweenSpawnpoints = 1  # m
    #-------------------------------

    def __init__(self, controlledCarsParams : List[ControlledCarSpawnParams], controlledPedestriansParams : List[ControlledPedestrianSpawnParams], # Parameters for what cars and pedestrian to spawn in the environment
                 NumberOfCarlaVehicles, NumberOfCarlaPedestrians,
                 observerSpawnTransform : carla.Transform = None,
                 observerVoxelSize = 5.0,
                 observerNumVoxelsX = 2048,
                 observerNumVoxelsY = 1024,
                 observerNumVoxelsZ = 1024,
                 forceExistingRaycastActor = False,
                 mapToUse = ["Town3"],
                 numberOfTrainableVehicles=0,
                 numberOfTrainablePedestrians=0,
                 sensorsDisplacementDist = 0,
                 args=None):

        self.observerSpawnTransform = observerSpawnTransform
        self.observerVoxelSize = observerVoxelSize
        self.observerNumVoxelsX = observerNumVoxelsX
        self.observerNumVoxelsY = observerNumVoxelsY
        self.observerNumVoxelsZ = observerNumVoxelsZ
        self.forceExistingRaycastActor = forceExistingRaycastActor
        self.args = args

        self.NumberOfTrainableVehicles = numberOfTrainableVehicles # For real time online
        self.NumberOfTrainablePedestrians = numberOfTrainablePedestrians # For real time online

        settingsAllowsCarTraining = (self.args.useRLToyCar or self.args.useHeroCar)
        assert self.NumberOfTrainableVehicles == 0 or settingsAllowsCarTraining is True, "Either you chose in your scene to NOT train any cars, or you allow the settings to do so !"

        self.NumberOfCarlaVehicles = NumberOfCarlaVehicles + self.NumberOfTrainableVehicles
        self.NumberOfCarlaPedestrians = NumberOfCarlaPedestrians + self.NumberOfTrainablePedestrians
        self.controlledCarsParams = controlledCarsParams
        self.controlledPedestriansParams = controlledPedestriansParams
        self.mapToUse = mapToUse
        self.episodeIndex = 99999

        # Index and transform of the spawn point observer index if any
        self.forcedObserverSpawnPointIndex = None
        self.forcedObserverSpawnPointTransform = None

        self.sensorsDisplacementDist = sensorsDisplacementDist
