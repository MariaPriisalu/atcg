from RL.settings import RANDOM_SEED
from commonUtils.ReconstructionUtils import get_people_and_cars

import RL.settings

from RL.settings import RANDOM_SEED_NP, RANDOM_SEED


if RL.settings.run_settings.realTimeEnvOnline:
    from commonUtils.RealTimeEnv.CarlaRealTimeUtils import *
    from commonUtils.RealTimeEnv.RealTimeEnvInteraction import *
else:
    import numpy as np
    import random
    from typing import List

from dotmap import DotMap
import copy
np.random.seed( RANDOM_SEED_NP)
random.seed(RANDOM_SEED)
class EntitiesRecordedDataSource:
    def __init__(self, init_frames, init_frames_cars, cars_sample, people_sample, cars_dict_sample, people_dict_sample,
                 cars_vel, ped_vel, reconstruction, forced_num_frames=None):
        self.init_frames = init_frames
        self.init_frames_cars = init_frames_cars

        self.cars = cars_sample
        self.people = people_sample

        self.cars_dict = cars_dict_sample  # a list of size num_frames each containing the dictionary of items appearing on each frame in order
        self.people_dict = people_dict_sample # same as as above
        self.cars_vel = cars_vel
        self.ped_vel = ped_vel
        self.reconstruction = reconstruction
        self.num_frames = forced_num_frames


        # When using online real time env, Mapping from key IDs for cars and pedestrian to their corresponding Actor instances
        self.env_physical_cars = {}
        self.env_physical_pedestrian = {}

    def remove_floating_frames(self):

        first_frame = 50

        self.people = self.people[first_frame:]
        self.cars = self.cars[first_frame:]

        for pedestrian_key in self.people_dict.keys():
            diff = self.init_frames[pedestrian_key] - first_frame
            if diff < 0:
                if abs(diff)>=len(self.people_dict[pedestrian_key]):
                    self.people_dict[pedestrian_key] =[]
                else:
                    self.people_dict[pedestrian_key] = self.people_dict[pedestrian_key][-diff:]
                self.init_frames[pedestrian_key] = 0
            else:
                self.init_frames[pedestrian_key] = diff
        for car_key in self.cars_dict.keys():
            diff = self.init_frames_cars[car_key] - first_frame
            if diff < 0:
                if abs(diff) >= len(self.cars_dict[car_key]):
                    self.cars_dict[car_key] =[]
                else:
                    self.cars_dict[car_key] = self.cars_dict[car_key][-diff:]
                self.init_frames_cars[car_key] = 0
            else:
                self.init_frames_cars[car_key] = diff

    def clean(self, exceptReconstruction, forced_num_frames=None):
        self.num_frames = forced_num_frames
        self.cars_dict = { }  # Dictionary from {pedestrian id : frames in order }
        self.people_dict = { } # Same as above #i:{} for i in range(self.num_frames)}
        self.cars_vel = {}
        self.ped_vel = {}

        if self.num_frames:
            self.cars = [[] for i in range(self.num_frames)]
            self.people = [[] for i in range(self.num_frames)]
        else:
            self.cars = []
            self.people = []

        self.init_frames = {} # The init frames for pedestrians , from pedestrian id => first frame of appear
        self.init_frames_cars = {}

    def removeFrames(self, start = None, end = None):
        if start is None:
            start = 0
        if end is None:
            end = max(self.num_frames, len(self.cars)) if self.num_frames is not None else len(self.cars)

        if start >= end:
            return

        for i in range(start, end):
            self.cars[i] = []
            self.people[i] = []

        for key in self.init_frames:
            self.init_frames[key] = start - 1
        for key in self.init_frames_cars:
            self.init_frames_cars[key] = start - 1

        def removeFramesFromDict(inOutDict, start, end):
            for entKey, entEntries in inOutDict.items():
                for i in range(start, end):
                    entEntries.pop(i, None)

        removeFramesFromDict(self.cars_dict, start, end)
        removeFramesFromDict(self.people_dict, start, end)
        removeFramesFromDict(self.cars_vel, start, end)
        removeFramesFromDict(self.ped_vel, start, end)

    # Given a list of frames to be integrated and their corresponding datasource, merge int othe existing data
    def merge(self, framesList : List[int], otherDataSource, isInitializationFrame=False):
        self.reconstruction = self.reconstruction if self.reconstruction is not None else otherDataSource.reconstruction

        # Add entities data for the frame between source and destionation7
        def addEntityDictData(dest_entityDict, source_entityDict, frameId, frameIndex):
            for entityId in source_entityDict.keys():
                if entityId not in dest_entityDict:
                    dest_entityDict[entityId] = {}

                assert isInitializationFrame or (frameId not in dest_entityDict[entityId]), f"Seems like frame {frameId} is already in the entity {entityId} data !"
                dest_entityDict[entityId][frameId] = source_entityDict[entityId][frameIndex] # because frameIndex is relative !

        # Init frames are updated only if there is no data for that corresponding entity
        # WARNING: this must happen BEFORE doing the addition to the dictionary data !!!
        def updateEntitiesInitFrames(dest_initFrames, source_initFrames, dest_entityDict):
            for entityKey, entityInitFrame in source_initFrames.items():
                # If not even exist, put it there
                if entityKey not in dest_initFrames:
                    dest_initFrames[entityKey] = entityInitFrame
                    dest_entityDict[entityKey] = {}
                # If exist, check if there are really existing data previously recorded there. If there is none, we consider the new frame init
                else:
                    assert dest_initFrames[entityKey] <= source_initFrames[entityKey], "Are you giving the time back ??"
                    if len(dest_entityDict[entityKey]) == 0:
                        dest_initFrames[entityKey] = entityInitFrame

        # Update the init frames for entities first thing !
        updateEntitiesInitFrames(dest_initFrames=self.init_frames_cars,
                                 source_initFrames=otherDataSource.init_frames_cars,
                                 dest_entityDict = self.cars_dict)
        updateEntitiesInitFrames(dest_initFrames=self.init_frames,
                                 source_initFrames=otherDataSource.init_frames,
                                 dest_entityDict = self.people_dict)

        # Add lists for entities on both flatten and dicitonary data
        for frameIndex, frameId in enumerate(framesList):
            # Add the cars and people list on the corresponding frame
            assert isInitializationFrame or (frameId not in self.cars and frameId not in self.people) or (len(self.cars[frameId]) == 0 and len(self.people[frameId]) == 0), f"Seems like frame {frameId} already contains data !!!"
            if frameId not in self.cars:
                self.cars.append(None) # Normally a single frame should be added...
                self.people.append(None)
            self.cars[frameId]      = otherDataSource.cars[frameIndex]
            self.people[frameId]    = otherDataSource.people[frameIndex]

            # Add the cars and people dicts
            addEntityDictData(dest_entityDict=self.cars_dict, source_entityDict=otherDataSource.cars_dict, frameId=frameId, frameIndex=frameIndex)
            addEntityDictData(dest_entityDict=self.people_dict, source_entityDict=otherDataSource.people_dict, frameId=frameId, frameIndex=frameIndex)
            addEntityDictData(dest_entityDict=self.cars_vel, source_entityDict=otherDataSource.cars_vel, frameId=frameId, frameIndex=frameIndex)
            addEntityDictData(dest_entityDict=self.ped_vel, source_entityDict=otherDataSource.ped_vel, frameId=frameId, frameIndex=frameIndex)

    # These two functions are used to avoid serializing in the cached version some things that we don't need.
    # Probably need to add more !
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't include states that shouldn't be saved
        objsNotNeeded = ["env_physical_cars", "env_physical_pedestrian"]
        for objNotNeeded in objsNotNeeded:
            assert objNotNeeded in state
            del state[objNotNeeded]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # TODO: add the things back if needed !
        self.baz = 0
        self.env_physical_cars = {}
        self.env_physical_pedestrian = {}


class AgentFrameData:
    def __init__(self):
        self.reset()

    def reset(self):
        self.velInput                        = None
        self.velUsed                         = None
        self.not_hitting_object              = None
        self.not_hitting_object_and_alive    = None
        self.frame = None
        self.change_in_pose = None

    def __str__(self):
            str = f"Vel input {self.velInput} - used {self.velUsed}. New Pos {self.nextPos}. nothitting: {self.notHittingObject}"
            return str


class EnvironmentInteraction:
    def __init__(self, isOnline,ignore_external_cars_and_pedestrians=False, entitiesRecordedDataSource : EntitiesRecordedDataSource = None,
                 parentEnvironment =None, args=None):
        self.isOnline = isOnline
        self.entitiesRecordedDataSource = entitiesRecordedDataSource
        assert isOnline == True or entitiesRecordedDataSource is not None
        self.cleanup()

        self.onlineEnvironment = None
        self.parentEnvironment = parentEnvironment
        if self.parentEnvironment!=None:
            self.currentDatasetOptions = self.parentEnvironment.currentDatasetOptionsUsed

        self.args = args

        # Cached dictionaries from key to actors' instances in the environment
        self.cachedPhysicalWalkersDict : Dict[any,any] = None # putting any to avoid Carla dependencies and separate logic from environment usage
        self.cachedPhysicalCarsDict : Dict[any, any] = None

        # This is true immediately after the environment was spawned. When the episode gets played, it gets on True
        self.isFirstEpisodePending = False

        # Cached the transformation matrices
        self.world_2_worldRef_matrix = None
        self.worldRef_2_world_matrix = None
        self.ignore_ext_cars_and_pedestrians=ignore_external_cars_and_pedestrians

    def cleanup(self):
        self.actorsRegistry = {}
        self.frame = 0
        self.episode = None
        self.world_2_worldRef_matrix = None
        self.worldRef_2_world_matrix = None

        if "onlineEnvironment" in self.__dict__ and self.onlineEnvironment is not None:
            self.onlineEnvironment.onEpisodeEnd()

            # Remove all frames from the recorded data starting at frame 1 till the end
            # We keep only frame 0, the initialization one
            self.entitiesRecordedDataSource.removeFrames(start=1, end=None)
        else:
            self.isFirstEpisodePending = False


    # Resets the environment data and prepares for a new episode
    def reset(self, heroAgentCars, heroAgentPedestrians, episode):
        self.episode = episode
        self.actorsRegistry = {}

        self.agents_car = []
        self.agents = []
        self.carInteraction=heroAgentCars

        #assert len(heroAgentCars)  <= 1 and len(heroAgentPedestrians) <= 1, "In the future you will be allowed to use multiple agents too :)"
        self.frame = 0

        # TODO: in the CARLA real time it needs to reload world etc
        # Respawn ??? Where are the respawn condition stored ??
        # In the offline env it needs to grab the offline data from the dataset store
        if heroAgentCars:
            for car in heroAgentCars.trainableCars:

                self.registerActor(car)
                self.agents_car.append(car)


        for ped in heroAgentPedestrians:
            self.registerActor(ped)
            self.agents.append(ped)

        # Get the initial observation
        observation, observation_dict = self.getObservation(frameToUse=None)

        return observation, observation_dict

    def onEpisodeStartup(self):
        self.frame = 0
        self.world_2_worldRef_matrix = None
        self.worldRef_2_world_matrix = None

        if self.onlineEnvironment:
            self.onlineEnvironment.onEpisodeStartup(isFirstEpisode=self.isFirstEpisodePending)

            self.world_2_worldRef_matrix = self.onlineEnvironment.envManagement.actorsContext.world_2_worldRef_matrix
            self.worldRef_2_world_matrix = self.onlineEnvironment.envManagement.actorsContext.worldRef_2_world_matrix

            # When not the first episode, we are actually resetting the episode so we need to run the tick initialization frame again
            # for taking into account the new environment setup
            if True or not self.isFirstEpisodePending:
                # We tick the world to take into account our new stuff, and then we're ready to go !
                self.tick(0, isInitializationFrame=True)

            if not self.isFirstEpisodePending:
                # Remove all frames from the recorded data starting at frame 1 till the end
                # We keep only frame 0, the initialization one
                self.entitiesRecordedDataSource.removeFrames(start=1, end=None)
            else:
                self.isFirstEpisodePending = False

    def signal_action(self, trainableAgentsData, updated_frame):
        # Fill in commands in a batched way for optimization purposes
        # Works much faster than sending commands individually
        # Could forward these to the agent classes maybe to avoid all the code here ?
        # Send further the decisions taken as signals to the online environment
        # Decision taken is currently just a velocity...but can be extended further in the future
        batch_pedestrians = []
        batch_vehicles = []

        for agent, decisionTaken in trainableAgentsData.items():

            agentDataFrameToFill = self.getAgentLastFrameData(agent)
            agentDataFrameToFill.reset()
            agentDataFrameToFill.frame = updated_frame

            # Prepare batches of commands only if the enviroment is an online one
            if self.isOnline == True:
                if agent.getIsPedestrian() == False: # Supposing a vehicle class
                    for car, car_decisionTaken in decisionTaken:
                        # Car stuff
                        onlineEnvAgentId = car.getOnlineRealtimeAgentId()
                        onlineEnvAgentInstance = self.getPhysicalCarsDict()[onlineEnvAgentId]
                        # Get current vehicle transform from world space but transform it back to world 2 ref space since this is the space we are working and taking decisions in
                        current_transform_carla = onlineEnvAgentInstance.get_transform()
                        current_transform_carla = getCarlaTransformByMatrix(current_transform_carla, self.world_2_worldRef_matrix)
                        current_loc_carla = current_transform_carla.location
                        current_rot_carla = current_transform_carla.rotation
                        current_forward_vector = carlaVector3DToNumpy(current_transform_carla.get_forward_vector())



                        # In this case the decision Taken is the input velocity of the car only
                        agentInputVelocity_voxels = car_decisionTaken
                        agentDataFrameToFill.velInput = agentInputVelocity_voxels

                        # Compare angles and directions in carla space (normal one)  - look on the 2d only, xy plane
                        agentInputVelocity_carla = voxelSpaceLocationToCarlaNumpy(agentInputVelocity_voxels)
                        angleBetweenVectors = getAngleBetweenVectors(current_forward_vector[0:2], agentInputVelocity_carla[0:2])
                        isClocwiseTurn = isVectorInClockwiseDirectionTo(current_forward_vector[0:2], agentInputVelocity_carla[0:2])

                        # When 0 angle, no steering, when 90, max steering 1
                        steerAmount = np.interp(angleBetweenVectors, [0., 45., 90.], [0., 0.4, 1.])
                        if isClocwiseTurn == False:
                            steerAmount *= -1.

                        # Speed on xy plane carla
                        agentSpeed2D_carla = getVectorLength(agentInputVelocity_carla[0:2])
                        throttle_amount = min(np.interp(agentSpeed2D_carla, [0., 0.1], [0, 1.]), 1)
                        brake_amount = min((1.0 if (agentSpeed2D_carla <= 0.0 or angleBetweenVectors > 90.0) else 0.0), 1.0)
                        handbrake_amount = True if agentSpeed2D_carla else False

                        agentDataFrameToFill.velInput = agentInputVelocity_voxels

                        vehicleController = carla.VehicleControl()
                        vehicleController.throttle = throttle_amount
                        vehicleController.brake = brake_amount
                        vehicleController.steer = steerAmount
                        vehicleController.hand_brake = handbrake_amount

                        batch_vehicles.append(carla.command.ApplyVehicleControl(actor_id=onlineEnvAgentId, control=vehicleController))


                        if self.args.DEBUG_LOG_AGENT_INPUTS:
                            print(f"F:{updated_frame }Car agent:{onlineEnvAgentId}, pos:{current_loc_carla} fwvec:{current_forward_vector}"
                                  f" was signaled to move with: T={throttle_amount}, S={steerAmount}, B={brake_amount}. Original input decision velocity: {agentInputVelocity_carla}-carla space")



                else: # Supposing that it is a pedestrian class
                    # Pedestrian stuff
                    onlineEnvAgentId = agent.getOnlineRealtimeAgentId()
                    walkersPhysicalActorsDict = self.getPhysicalWalkersDict()
                    onlineEnvAgentInstance = walkersPhysicalActorsDict[onlineEnvAgentId]

                    # Get the real agent transform, get the target position in carla space then send the walker there
                    current_transform_carla = onlineEnvAgentInstance.get_transform()
                    current_transform_carla = getCarlaTransformByMatrix(current_transform_carla, self.world_2_worldRef_matrix)
                    current_loc_carla = current_transform_carla.location
                    current_rot_carla = current_transform_carla.rotation
                    current_forward_vector_carla = carlaVector3DToNumpy(current_rot_carla.get_forward_vector())

                    agentInputVelocity_voxels = decisionTaken
                    agentDataFrameToFill.velInput = agentInputVelocity_voxels

                    target_loc_invoxels, step_in_voxels = agent.get_planned_position(True, agentInputVelocity_voxels)
                    target_loc_incarla = voxelSpaceLocationToCarlaLocation(target_loc_invoxels)
                    walkerMoveVel_carla = carla.Vector3D(target_loc_incarla - current_loc_carla)
                    walkerSpeed_carla = getVectorLength(walkerMoveVel_carla) # This is meters per frame!
                    walkerSpeed_carla *= self.args.dataset_frameRate # Now it is in meters per second as it should be for carla !
                    walkerDir_carla = getVectorNormalized(walkerMoveVel_carla)

                    # At this point walkerDir_carla is in world 2 ref space, but for carla it must be given in the world space
                    walkerDir_carla_world_space = np.dot(self.worldRef_2_world_matrix[0:3,0:3], carlaVector3DToNumpy(walkerDir_carla).T)
                    walkerDir_carla_world_space = NumpyToCarlaVector3D(walkerDir_carla_world_space)

                    # Now make the input in carla API and add it to the commands' batch
                    agentWalkerController = carla.WalkerControl()
                    agentWalkerController.jump = False
                    agentWalkerController.speed = walkerSpeed_carla
                    agentWalkerController.direction = walkerDir_carla_world_space
                    batch_pedestrians.append(carla.command.ApplyWalkerControl(actor_id=onlineEnvAgentId, control=agentWalkerController))

                    if self.args.DEBUG_LOG_AGENT_INPUTS:
                        #print(f"Pedestrian agent {onlineEnvAgentId} was signaled to walk with velocity: {walkerMoveVel_carla}-carla space, {agentInputVelocity_voxels}-voxels")
                        print(f"F:{updated_frame}Pedestrian agent:{onlineEnvAgentId}, pos:{current_loc_carla} fwvec:{current_forward_vector_carla}"
                            f" was signaled to move with velocity: {walkerMoveVel_carla}-carla space")

        # Apply the batch now
        if self.isOnline == True:
            full_batch = batch_vehicles + batch_pedestrians
            if len(full_batch) > 0:
                assert len(full_batch) == len(self.getTrainableWalkerIds()) + len(self.getTrainableVehiclesIds()), "Sanity check failed to get commands for all trainable characters"
                results = self.onlineEnvironment.carlaConnection.client.apply_batch_sync(full_batch, True)
                for res_index, res_data in enumerate(results):
                    if res_data.error:
                        print(
                            f"Can't solve request for actor index {res_index} while #vehicles={len(batch_vehicles)}, #walkers={len(batch_pedestrians)}")

    # Called when the episode ends
    def onEpisodeEnd(self):
        self.cleanup()

    def registerActor(self, actor):

        self.actorsRegistry[actor] = AgentFrameData()

    # Gets the last frame data (actions, observations, any other data stuff) for a given actor
    def getAgentLastFrameData(self, actor):
        return self.actorsRegistry[actor]

    def getTrainableVehiclesIds(self):
        if self.onlineEnvironment:
            return self.onlineEnvironment.envManagement.actorsContext.trainableVehiclesIds
        return []

    def getTrainableWalkerIds(self):
        if self.onlineEnvironment:
            return self.onlineEnvironment.envManagement.actorsContext.trainableWalkerIds
        return []

    # Or step according to openai terminology..
    def tick(self, frame, isInitializationFrame=False):
        if self.isOnline:
            # Sim one frame and put the first data in the dataset
            #print("Simulating a frame")
            lastFrameIndex, lastFrameVehiclesData, lastFramePedestriansData = self.onlineEnvironment.SimulateFrame(isInitializationFrame)
            assert  frame == lastFrameIndex or isInitializationFrame


            #print("Processing the new frame")
            # Process the new data into a data source data structure
            framesList : List[int] = [frame]
            cars, people, ped_dict, cars_2D, people_2D, poses, valid_ids, car_dict, init_frames, init_frames_cars = get_people_and_cars(
                self.currentDatasetOptions.WorldToCameraRotation,
                [], # cars=0
                None, #filepath
                framesList,
                self.currentDatasetOptions.centering['middle'], #or WorldToCameraTranslation
                self.currentDatasetOptions.centering['height'],
                [], # people
                self.currentDatasetOptions.centering['scale'],
                find_poses = False,
                datasetOptions = self.currentDatasetOptions,
                carsData_external = { frame : lastFrameVehiclesData},
                peopleData_external = { frame : lastFramePedestriansData})


            # Extract velocities - we put them in the same format as cars and people - frame indexed from 0, but the offset is given in frame (frameList)
            cars_vel = {key : { 0 : value['VelocityRef'] } for key,value in lastFrameVehiclesData.items()}
            ped_vel = {key : { 0 : value['VelocityRef'] } for key,value in lastFramePedestriansData.items()}

            framesDataSource: EntitiesRecordedDataSource = EntitiesRecordedDataSource(init_frames=init_frames,
                                                                                      init_frames_cars=init_frames_cars,
                                                                                      cars_sample=cars,
                                                                                      people_sample=people,
                                                                                      cars_dict_sample=car_dict,
                                                                                      people_dict_sample=ped_dict,
                                                                                      cars_vel =cars_vel,
                                                                                      ped_vel=ped_vel,
                                                                                      reconstruction=None,  # No need for it here
                                                                                      forced_num_frames=None)

            # Append the data to the existing things
            self.entitiesRecordedDataSource.merge(framesList, framesDataSource, isInitializationFrame)

            #print("Filling in data")

            # Now fill in the last frame records (real data of what happened in the last frame) to the trainable agents
            for trainableActor in self.actorsRegistry: #TODO Does this need to be changed?

                actorOnlineEnvId = trainableActor.getOnlineRealtimeAgentId()
                dataSource = self.entitiesRecordedDataSource.cars_dict[actorOnlineEnvId] if trainableActor.isPedestrian is False else self.entitiesRecordedDataSource.people_dict[actorOnlineEnvId]

                agentDataFrameToFill = self.getAgentLastFrameData(trainableActor)
                if frame not in dataSource:
                    #print(f"Trainable actor {trainableActor} with online id {actorOnlineEnvId} is no longer available ! in frame {frame}")
                    agentDataFrameToFill.notHittingObject = False # This makes the different between agent dead vs collision
                    agentDataFrameToFill.notHittingObjectAndAlive = True
                    agentDataFrameToFill.velUsed = np.array([0,0,0]) # Consider no movement
                    agentDataFrameToFill.nextPos = getCenterOfBBox(np.array(dataSource[max(dataSource.keys())])) if len(dataSource) > 0 else np.array([0,0,0]) # last known position
                    continue

                currFrameAgentDataBBox_voxelsSpace = np.array(dataSource[frame])
                prevFrameAgentDataBBox_voxelsSpace = np.array(dataSource[frame-1]) if (frame > 0 and frame-1 in dataSource) else None
                currFrameAgentDataPos_voxelsSpace = getCenterOfBBox(currFrameAgentDataBBox_voxelsSpace)
                prevFrameAgentDataPos_voxelsSpace = getCenterOfBBox(prevFrameAgentDataBBox_voxelsSpace) if prevFrameAgentDataBBox_voxelsSpace is not None else currFrameAgentDataPos_voxelsSpace


                if trainableActor.isPedestrian is False:
                    # TODO: vehicle stuff processing
                    assert trainableActor.isPedestrian == False, "Sanity check failed"

                    agentDataFrameToFill.velUsed = currFrameAgentDataPos_voxelsSpace - prevFrameAgentDataPos_voxelsSpace
                    agentDataFrameToFill.nextPos = currFrameAgentDataPos_voxelsSpace
                    agentDataFrameToFill.notHittingObject = True  # TODO : Check the B1 TODO
                    agentDataFrameToFill.notHittingObjectAndAlive = True  # TODO : Check the B1 TODO

                else:
                    # TODO: pedestrian stuff processing
                    assert trainableActor.isPedestrian == True, "Sanity check failed"

                    agentDataFrameToFill.velUsed = currFrameAgentDataPos_voxelsSpace - prevFrameAgentDataPos_voxelsSpace
                    agentDataFrameToFill.nextPos = currFrameAgentDataPos_voxelsSpace
                    agentDataFrameToFill.notHittingObject = True  # TODO : Check the B1 TODO
                    agentDataFrameToFill.notHittingObjectAndAlive = True # TODO : Check the B1 TODO


                    pass

                #print(f"Agent {trainableActor.getOnlineRealtimeAgentId()} output from environment {agentDataFrameToFill}")




        assert self.frame == 0 or self.frame == frame - 1, "Possibly incorrect frame index updated ? Double check please !"
        self.frame = frame

    # This gets the observation from the environment

    # This function converts a real velocity from environment to something that is optimal for our purpose of estimation the occupancy map / trajectory over time
    def transformVelocityForOnlineEnvEstimation(self, inVelocity, isCar):
        inVelocity = inVelocity.copy()

        # Convert to voxel space normalized
        inVelocity = np.array([inVelocity[2], inVelocity[1], inVelocity[0]]) #* self.currentDatasetOptions.scale
        inVelocity = getVectorNormalized(inVelocity)

        refVel = self.args.ped_reference_speed if isCar == False else self.args.car_reference_speed
        res = inVelocity*refVel
        return res


    def getObservation(self, frameToUse):
        #print("Is online? ------------------------------------"+str(self.isOnline))
        originalFrameSent = frameToUse
        observation_dict={}

        if False and self.isOnline == True:
            # TODO OnlineEnv:  might take some of the data below (most of it) from the carla environment instead?
            raise NotImplementedError
        else:

            # print (" Get cars and people frame " + str(frame))
            for car in self.agents_car:
                realTimeEnvObservation = DotMap()
                # Fill in trained hero car details
                if frameToUse is None or frameToUse <= 0:
                    frameToUse = 0
                    realTimeEnvObservation.car_init_dir = car.init_dir
                else:
                    realTimeEnvObservation.car_init_dir =[]
                    realTimeEnvObservation.heroCarVel       = copy.copy(car.velocities[frameToUse-1])
                    realTimeEnvObservation.heroCarAction    =copy.copy(car.action[frameToUse-1])
                    realTimeEnvObservation.heroCarAngle     = copy.copy(car.angle[frameToUse-1])
                    #print(" Transfer " + str(self.car.action[frame - 1]))
                # print (" Get cars and people frame "+str(frame))
                # print (" Car init dir "+str(realTimeEnvObservation.car_init_dir))
                #assert car.frame ==self.frame-1, "Frames not in sync. Car frame "+str(car.frame)+ " frame here "+str(self.frame)

                # Check if frames are in sync (will not be in the initialization/reset frame)
                if originalFrameSent != None and car.frame != frameToUse-1:
                    print ("Frames not in sync. Car frame "+str(car.frame)+ " next frame here "+str(frameToUse))
                    assert False

                realTimeEnvObservation.frame = copy.copy(frameToUse) # TODO : should be + 1 or frameToUse ???
                realTimeEnvObservation.heroCarPos = copy.copy(car.car[frameToUse])  # We have update the next car frame position, so we use that one
                realTimeEnvObservation.heroCarGoal = copy.copy(car.goal)

                realTimeEnvObservation.heroCarBBox =copy.copy(car.bbox[frameToUse])   # Same reason as above
                realTimeEnvObservation.measures = copy.copy(car.measures[max(frameToUse-1,0), :])


                # if self.agent_car != None and car.reward[max(frameToUse-1,0)]==None:
                #     realTimeEnvObservation.reward =0
                # else:
                realTimeEnvObservation.reward =copy.copy(car.reward[max(frameToUse-1,0)])
                # print(" Copy reward to RLRealTimeEnv "+str(realTimeEnvObservation.reward)+" frame "+str(max(frameToUse,0)))
                realTimeEnvObservation.probabilities = copy.copy(car.probabilities[max(frameToUse-1,0)])
                # print (" Get probabilities "+str(realTimeEnvObservation.probabilities)+" frame "+str(max(frameToUse,0)))
                # print ("Saved initial position ? " + str(self.car.car[0]) +" saved "+str(realTimeEnvObservation.heroCarPos))
                # print(" Goal car " + str(self.car.goal) +" saved "+str(realTimeEnvObservation.heroCarGoal))
                # print (" Car dir " + str(self.car.init_dir)+ " saved "+str(realTimeEnvObservation.heroCarVel))
                observation_dict[car]=realTimeEnvObservation

            realTimeEnvObservation = DotMap()
            if frameToUse is None or frameToUse <= 0:
                frameToUse = 0
            realTimeEnvObservation.frame=frameToUse
            # Get cars and people data into observation. Note: we use the next frame, as we did for pedestrian and car agent
            pedestrian_dict = {}
            pedestrian_vel_dict = {}
            if self.ignore_ext_cars_and_pedestrians==False:
                for pedestrian_key in self.entitiesRecordedDataSource.people_dict.keys():
                    local_frame = frameToUse - (self.entitiesRecordedDataSource.init_frames[pedestrian_key])
                    if local_frame>=0 and local_frame < len(self.entitiesRecordedDataSource.people_dict[pedestrian_key]):

                        if self.entitiesRecordedDataSource.init_frames[pedestrian_key] <= frameToUse and local_frame< len(self.entitiesRecordedDataSource.people_dict[pedestrian_key]):

                            value=self.entitiesRecordedDataSource.people_dict[pedestrian_key][local_frame]
                            pedestrian_dict[pedestrian_key] = value

                            if not self.isOnline:
                                # print(self.entitiesRecordedDataSource.people_dict[pedestrian_key][min(local_frame + 1, len(self.entitiesRecordedDataSource.people_dict[pedestrian_key]) - 1)] -
                                #     self.entitiesRecordedDataSource.people_dict[pedestrian_key][local_frame])
                                pedestrian_vel_dict[pedestrian_key] = np.mean(
                                    self.entitiesRecordedDataSource.people_dict[pedestrian_key][min(local_frame + 1, len(self.entitiesRecordedDataSource.people_dict[pedestrian_key]) - 1)] -
                                    self.entitiesRecordedDataSource.people_dict[pedestrian_key][local_frame], axis=1)

                            else:
                                pedestrian_vel_dict[pedestrian_key] = self.transformVelocityForOnlineEnvEstimation(self.entitiesRecordedDataSource.ped_vel[pedestrian_key][local_frame], isCar=False)

            realTimeEnvObservation.people_dict = pedestrian_dict

            realTimeEnvObservation.pedestrian_vel_dict = pedestrian_vel_dict
            # print (" pedestrian_dict " + str(realTimeEnvObservation.pedestrian_vel_dict ))

            car_dict = {}
            car_vel_dict = {}
            if self.ignore_ext_cars_and_pedestrians == False:
                for car_key in self.entitiesRecordedDataSource.cars_dict.keys():
                    local_frame = frameToUse - self.entitiesRecordedDataSource.init_frames_cars[car_key]
                    if local_frame>=0 and local_frame < len(self.entitiesRecordedDataSource.cars_dict[car_key]):

                        if self.entitiesRecordedDataSource.init_frames_cars[car_key] <= self.frame and local_frame< len(self.entitiesRecordedDataSource.cars_dict[car_key]):

                            car_current = self.entitiesRecordedDataSource.cars_dict[car_key][local_frame]

                            car_next = self.entitiesRecordedDataSource.cars_dict[car_key][min(local_frame + 1, len(self.entitiesRecordedDataSource.cars_dict[car_key]) - 1)]
                            diff = [[], [], []]
                            for i in range(len(car_current)):
                                diff[int(i / 2)].append(car_next[i] - car_current[i])

                            car_dict[car_key] = car_current

                            if not self.isOnline:
                                car_vel_dict[car_key] = np.mean(np.array(diff), axis=1)
                            else:
                                car_vel_dict[car_key] = self.transformVelocityForOnlineEnvEstimation(self.entitiesRecordedDataSource.cars_vel[car_key][local_frame], isCar=True)

            realTimeEnvObservation.cars_dict = car_dict
            realTimeEnvObservation.car_vel_dict = car_vel_dict

        return realTimeEnvObservation,observation_dict

    # Spawns the environment
    def spawnEnvironment(self, args):
        # Nothing to do if an external side (Carla or whatever simulator) is not used
        if self.isOnline is False or self.isOnline is None:
            return

        self.onlineEnvironment = CarlaRealTimeEnv(args)
        self.onlineEnvironment.spawnNewEnvironment(args, self.currentDatasetOptions.dataPath)

        self.isFirstEpisodePending = True

    def destroyEnvironment(self, betweenIterations=False):
        if self.onlineEnvironment is not None:
            self.onlineEnvironment.destroyCurrentEnv(withError=False, betweenIterations=betweenIterations)
            self.onlineEnvironment = None

    def getPhysicalCarsDict(self):
        if self.cachedPhysicalCarsDict is None or len(self.cachedPhysicalCarsDict) == 0:
            self.cachedPhysicalCarsDict = {}
            if not self.isOnline:
                return

            for vehicleActor in self.onlineEnvironment.envManagement.actorsContext.s_vehicles_list:
                self.cachedPhysicalCarsDict[vehicleActor.id] = vehicleActor

        return self.cachedPhysicalCarsDict

    def getPhysicalWalkersDict(self):
        if self.cachedPhysicalWalkersDict is None or len(self.cachedPhysicalWalkersDict) == 0:
            self.cachedPhysicalWalkersDict = {}
            if not self.isOnline:
                return


            for walkerActor in self.onlineEnvironment.envManagement.actorsContext.all_pedestrian_actors:
                if walkerActor is None:
                    continue

                self.cachedPhysicalWalkersDict[walkerActor.id] = walkerActor

        return self.cachedPhysicalWalkersDict


    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Given a position in voxels in the world, gives back the closest vehicle waypoint pos to that.
    def getRealTimeEnvWaypointPosFunctor(self, inputLocationInVoxel):
        if not self.isOnline:
            return False

        inputlocation_InUnrealSpace = voxelSpaceLocationToCarlaLocation(inputLocationInVoxel)

        waypointClosestToLocation = self.onlineEnvironment.carlaConnection.map.get_waypoint(inputlocation_InUnrealSpace)
        location_InUnrealSpace = waypointClosestToLocation.transform.location
        outputLocationInVoxel = carlaLocationToVoxelSpaceLocation(location_InUnrealSpace)

        return outputLocationInVoxel

    """ Very dangeros to activate this because of unplickling.It will basically disconnect everything
     # I'm living it commented here to let you know that you SHOULD NOT DO THIS
    def __del__(self):
        self.destroyEnvironment()
    """


