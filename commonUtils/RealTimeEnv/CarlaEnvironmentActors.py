import logging
import math

import os, sys
CARLA_INSTALL_PATH=""
CARLA_INSTALL_PATH = os.getenv('CARLA_INSTALL_PATH')

sys.path.append(os.path.join(CARLA_INSTALL_PATH, 'PythonAPI/carla'))

import numpy as np
from . CarlaRealTimeUtils import *
from . CarlaServerConnection import *
import carla
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from RL.settings import RLCarlaOnlineEnv
from commonUtils.ReconstructionUtils import FILENAME_CARS_TRAJECTORIES, FILENAME_PEOPLE_TRAJECTORIES


INVALID_ACTOR_TAG = None #99999999

class WalkerSpawnDetails:
    def __init__(self, spawn_point, target_point, distance, usable):
        self.spawn_point = spawn_point
        self.target_point = target_point
        self.distance = distance
        self.usable = usable
        self.speed = None
        self.blueprint = None
        self.replayId = None # The id of the reply in the simulation data for this walker


# This handles the management of actors, blueprints etc in the scene
class EnvironmentActors:
    def __init__(self, parent, carlaConnectionManager : CarlaServerConnection, args):
        self.parent = parent
        self.args = args
        self.s_heroCaptureCarPerspective = []  # The list of all actors currently spawned for hero perspective car capture (his car, sensor cameras ,etc)
        self.s_vehicles_list = []  # The list of all vehicle
        self.s_all_pedestrian_ids = []  # controller,walker pairs
        self.all_pedestrian_actors = []  # controller, walker pairs
        self.pedestrian_actors = []  # As above, but contains only the walkers
        self.s_heroCaptureCarPerspective_sensors = dict()  # THe dict of sensors (part of s_players_actor_list), name to sensor instance
        self.s_heroCaptureCarPerspective_intrisics = dict() # Mapping for name to intrisics or other details about the sensor
        self.world_2_worldRef_matrix = None # World to World reference location as matrix
        self.worldRef_2_world_matrix = None # World reference location to World as matrix
        self.carlaConnectionManager = carlaConnectionManager
        self.dataGatherParams : DataGatherParams = None
        self.simulationOptions : SimOptions = None

        # These are temporary list created during the spawning process to record detailed data
        self.tempSpawnedVehicles : Dict[any, EntitySpawnDetails] = {} # spawn id to spawn data
        self.tempSpawnedPedestrians : Dict[any, EntitySpawnDetails] = {}

        # Mapping from the index of an walker id to its attached controller actor
        self.walkerIdToAIController : Dict[any, carla.WalkerAIController] = {}
        self.walkerIdToWalkerController : Dict[any, carla.WalkerControl] = {}

        # Mapping from the index of a trainable agent id to its attached controller and id
        #self.trainableWalkerIndexToActorData : Dict[any, {}] = {} # walker trainable index = > { "controller" : carla.WalkerController , "id" : carla.ACtorId}
        #self.trainableVehicleIndexToActorData : Dict[any, {}]  = {}# vehicle trainable index => same as above but for car

        self.trainableVehiclesIds = set() # The set of trainable car ids
        self.trainableWalkerIds = set() # The set of pedestrian ids to train

        # Vehicles and pedestrians data as dicts of [FrameId][EntityId]['BBoxMinMax'], each with a 3x2 describing the bounding box as min value on column 0 and max on column 1
        # And 'velocity'
        self.vehicles_data = {}
        self.pedestrians_data = {}



    # Promote uniform sampling around map + spawn in front of walkers
    @staticmethod
    def uniformSampleSpawnPoints(allSpawnpoints, numToSelect):
        availablePoints = [(index, transform) for index, transform in allSpawnpoints]
        selectedPointsAndIndices = [] #[None]*numToSelect

        for selIndex in range(numToSelect):
            # Select the one available that is furthest from existing ones
            bestPoint = None
            bestDist = -1
            for x in availablePoints:
                target_index = x[0]
                target_transform = x[1].location

                # Find the closest selected point to x
                closestDist = math.inf
                closestSelPoint = None
                for y in selectedPointsAndIndices:
                    selPointLocation = y[1].location
                    d = compute_distance(target_transform, selPointLocation)
                    if d < closestDist:
                        closestDist     = d
                        closestSelPoint = y

                if closestSelPoint == None or bestDist < closestDist:
                    bestDist = closestDist
                    bestPoint = x

            if  bestPoint != None:
                availablePoints.remove(bestPoint)
                selectedPointsAndIndices.append(bestPoint)

        return selectedPointsAndIndices

    def createWalkersBlueprintLibrary(self):
        blueprints = self.blueprint_library.filter(EnvSetupParams.walkers_filter_str)
        return blueprints

    def createVehiclesBlueprintLibrary(self):
        # Filter some vehicles library
        blueprints = self.blueprint_library.filter(EnvSetupParams.vehicles_filter_str)
        #blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        return blueprints

    # Despawns the actors create in the current enviornment
    def despawnActors(self, client):
        if len(self.s_vehicles_list) == 0 and len(self.s_heroCaptureCarPerspective) == 0 and len(self.s_all_pedestrian_ids) == 0:
            logging.log(logging.INFO, 'Environment already distroyed')
            return

        logging.log(logging.INFO, 'Destroying %d vehicles' % len(self.s_vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in self.s_vehicles_list])
        self.s_vehicles_list = []

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        logging.log(logging.INFO,"Stopping the walker controllers")
        for i in range(0, len(self.all_pedestrian_actors), 2):
            if not self.all_pedestrian_actors[i]==None:
                self.all_pedestrian_actors[i].stop()

        logging.log(logging.INFO, f'Destroying all {len(self.s_all_pedestrian_ids)/2} walkers actors spawned')
        client.apply_batch([carla.command.DestroyActor(x) for x in self.s_all_pedestrian_ids if x != None])
        self.s_all_pedestrian_ids = []

        # Clean the registed stuff for the SensorManager
        self.dataManager.cleanup()

        if len(self.s_heroCaptureCarPerspective) > 0:
            logging.log(logging.INFO, f"Destroying all {len(self.s_heroCaptureCarPerspective) / 2} player\'s actors")
            client.apply_batch([carla.command.DestroyActor(x) for x in self.s_heroCaptureCarPerspective if x != None])
            self.s_heroCaptureCarPerspective = []

        time.sleep(SYNC_TIME_PLUS)
        self.world.tick()

        logging.log(logging.INFO, "===End destroying the environment...")

        self.s_all_pedestrian_ids = []
        self.vehicles_data = []
        self.pedestrians_data = []
        self.s_vehicles_list = []
        self.simulationOptions.carsData = {}
        self.simulationOptions.pedestriansData = {}


        # Creates the point cloud or copies it if needed. Check the comment above DataGatheringParams class

    def solveSceneData(self):
        # We do raycasting only when not using the hero car mode. On that mode, data will be written as output on each episode
        if (not self.dataGatherParams.use_hero_actor_forDataGathering and not self.simulationOptions.simulateReplay) and self.args.onlineEnvSettings.isDataGatherAndStoreEnabled == True:
            if self.dataGatherParams.rewritePointCloud:
                print("WRITING the point cloud files...please wait")
                self.capturePointCloud(self.dataGatherParams.outputEpisodesBasePath_currentSceneData)
                #time.sleep(SYNC_TIME)
                #self.world.tick()
                #time.sleep(SYNC_TIME)
                self.dataGatherParams.rewritePointCloud = False

            # Copy all files from outputEpisodesBasePath to the outputCurrentEpisodePath to have data for the corresponding episode
            if self.dataGatherParams.copyScenePaths:
                os.makedirs(self.dataGatherParams.outputEpisodesBasePath_currentSceneData, exist_ok=True)
                src_files = os.listdir(self.dataGatherParams.outputEpisodesBasePath_currentSceneData)
                for file_name in src_files:
                    fullFileName = os.path.join(self.dataGatherParams.outputEpisodesBasePath_currentSceneData, file_name)
                    if os.path.isfile(fullFileName):
                        if not os.path.exists(self.dataGatherParams.outputCurrentEpisodePath):
                            os.makedirs(self.dataGatherParams.outputCurrentEpisodePath)
                        shutil.copy(fullFileName, self.dataGatherParams.outputCurrentEpisodePath)

    def capturePointCloud(self, outScenePath):
        self.world.capture_raycastActor(outpath=outScenePath, synchronous=True)

    def loadReplayData(self):
        # Load matrices conversions...
        self.simulationOptions.loadWorldToWorldRefMatrixConversion(self.dataGatherParams.outputCurrentEpisodePath)

        # Load dicts of people and cars
        assert self.simulationOptions.simulateReplay , "call this only when you need sim data !"
        cars_path = os.path.join(self.simulationOptions.pathToReplayData, FILENAME_CARS_TRAJECTORIES)
        with open(cars_path, 'rb') as handle:
            cars_dict = pickle.load(handle, encoding="latin1", fix_imports=True)

        people_path = os.path.join(self.simulationOptions.pathToReplayData, FILENAME_PEOPLE_TRAJECTORIES)
        with open(people_path, 'rb') as handle:
            people_dict = pickle.load(handle, encoding="latin1", fix_imports=True)

        # Load stats file and inject them to the cars and people dict
        if self.simulationOptions.useRecordedTrainedAgents:
            self.simulationOptions.loadTrainedAgentsReplayData(cars_dict, people_dict)

        # Now load internally in an understandable language for CARLA simulation
        self.simulationOptions.loadEntitiesReplayData(people_dict, loadCars=False)
        self.simulationOptions.loadEntitiesReplayData(cars_dict, loadCars=True)

    def spawnWalkers(self):
        # -------------
        # Spawn Walkers
        # -------------
        # some pedestrians settings
        percentagePedestriansRunning = 0.3  # how many pedestrians will run
        percentagePedestriansCrossing = 0.6  # how many pedestrians will walk through the road
        blueprints_walkers  = self.createWalkersBlueprintLibrary()

        SpawnActorFunctor = carla.command.SpawnActor

        # Step 0: Find the spawn points for walkers and their destination points.
        spawnAndDestinationPoints : List[WalkerSpawnDetails] = []
        spawnAndDestinationPoints_replayIds : List[any] = None # Used only on replay mode...

        # When not replaying, setup the algorithm below to add some good points around
        if not self.simulationOptions.simulateReplay:
            playerSpawnForward = self.envSetup.observerSpawnTransform.rotation.get_forward_vector()

            logging.log(logging.DEBUG, 'Spawning walkers...')
            # 1. take all the random locations to spawn
            spawnAndDestinationPoints_extended : List[WalkerSpawnDetails]= []
            # To promote having agents around the player spawn position, we randomly select F * numPedestrians locations,
            # on the navmesh, then select the closest ones to the spawn position
            numSpawnPointsToGenerate = self.envSetup.PedestriansSpawnPointsFactor * self.envSetup.NumberOfCarlaPedestrians
            for i in range(numSpawnPointsToGenerate):
                loc1 = self.world.get_random_location_from_navigation()
                loc2 = self.world.get_random_location_from_navigation()
                if (loc1 != None and loc2 != None):
                    # Check if both of them are in front of the car
                    isLoc1InFront = isPosInFaceOfObserverPos(playerSpawnForward, self.observerSpawnLocation, loc1)
                    isLoc2InFront = isPosInFaceOfObserverPos(playerSpawnForward, self.observerSpawnLocation, loc2)
                    if isLoc1InFront and isLoc2InFront:
                        # Swap spawn with destination maybe position
                        #if isLoc1InFront == False:
                        #    loc2, loc1 = loc1, loc2
                        spawn_point = carla.Transform()
                        spawn_point.location = loc1
                        spawn_point.rotation = carla.Rotation(yaw=np.random.randint(low=0, high=360))
                        destination_point = carla.Transform()
                        destination_point.location = loc2
                        destination_point.rotation = carla.Rotation(yaw=np.random.randint(low=0, high=360))
                        distance = compute_distance(spawn_point.location, self.envSetup.observerSpawnTransform.location)
                        spawnAndDestinationPoints_extended.append(WalkerSpawnDetails(spawn_point, destination_point, distance, True))

            # Sort the points depending on their distance to playerSpawnTransform
            spawnAndDestinationPoints_extended = sorted(spawnAndDestinationPoints_extended, key = lambda walkerSpawnDetails : walkerSpawnDetails.distance)

            if len(spawnAndDestinationPoints_extended) > 0:
                # Now select points that are Xm depart from each other
                spacePartitonTemp = SimpleGridSpacePartition(cellSize=(self.envSetup.PedestriansDistanceBetweenSpawnpoints + 1.0))

                spawnAndDestinationPoints = [spawnAndDestinationPoints_extended[0]]
                unselected_points = []
                for pIndex in range(1, len(spawnAndDestinationPoints_extended)):
                    potential_point = spawnAndDestinationPoints_extended[pIndex]
                    shortedDistToAnySelected = math.inf

                    """ # Brute force commented
                    for selectedPoint in spawnAndDestinationPoints:
                        distToThisSelPoint = compute_distance(potential_point.spawn_point.location, selectedPoint.spawn_point.location)
                        if distToThisSelPoint < shortedDistToAnySelected:
                            shortedDistToAnySelected = distToThisSelPoint
                    if shortedDistToAnySelected > self.envSetup.PedestriansDistanceBetweenSpawnpoints:
                        spawnAndDestinationPoints.append(potential_point)
                    else:
                        unselected_points.append(potential_point)
                    """
                    potentialSpawnPointLocation : carla.Location = potential_point.spawn_point.location
                    numPointInGridAlreadyThere = spacePartitonTemp.getItemsAround(potentialSpawnPointLocation)
                    if numPointInGridAlreadyThere <= 0:
                        spawnAndDestinationPoints.append(potential_point)
                        spacePartitonTemp.occupy(potentialSpawnPointLocation)
                    else:
                        if len(unselected_points) < self.envSetup.NumberOfCarlaPedestrians * 50:
                            unselected_points.append(potential_point)

                    # Selecting enough, so leaving
                    # Commented because we need more spawn points...what if we didn't succeed to spawn because of collisoons ?
                    #if len(spawnAndDestinationPoints) >= self.envSetup.NumberOfCarlaPedestrians:
                    #    break

                # Didn't complete the list with the filter above ? just chose some random points
                diffNeeded = self.envSetup.NumberOfCarlaPedestrians - len(spawnAndDestinationPoints)
                if diffNeeded > 0:
                    random.shuffle(unselected_points)
                    numPointsToAppendExtra = min(diffNeeded, len(unselected_points))
                    if numPointsToAppendExtra > 0:
                        spawnAndDestinationPoints.extend(unselected_points[:numPointsToAppendExtra])

                #spawnAndDestinationPoints = spawnAndDestinationPoints[:self.envSetup.NumberOfCarlaPedestrians]

                # Destination points are from the same set, but we shuffle them
                #destination_points = spawnAndDestinationPoints
                #random.shuffle(destination_points)
        else:
            # Take walker spawn poitns details out from replay data
            spawnAndDestinationPoints : List[WalkerSpawnDetails] = []
            spawnAndDestinationPoints_replayIds : List[any]  = [] # The corresponding replay ids for the walkers above
            for replayId, spawnData in self.simulationOptions.pedestriansReplayIdToSpawnData.items():

                # Convert from EntitySpawnDetails to WalkerSpawnDetails
                walkerSpawnDetails = WalkerSpawnDetails(spawn_point=spawnData.replayDataPerFrame.transform,
                                                        target_point=spawnData.targetPoint,
                                                        distance=0.0, usable=True)
                walkerSpawnDetails.speed = spawnData.speed
                walkerSpawnDetails.blueprint=spawnData.blueprint
                walkerSpawnDetails.replayId=spawnData.replayId

                # Append to the lists
                spawnAndDestinationPoints.append(walkerSpawnDetails)
                spawnAndDestinationPoints_replayIds.append(spawnData.replayId)

        # 2. we spawn the walker objects
        # Try for a given number of times to spawn all needed pedestrians
        MAX_SPAWN_RETRIES = 200
        numWalkersSuccesfullySpawned = 0
        numTries = 0
        prevInitWalkers = 0 # How many walkers did we already initialized (target positon, speed, etc)s
        localTrainableAgentIds = set() # This is a local set of agents ids that have been assigned to controlled pedestrians below
        localTrainableAgentInstances = [] # the list containing above actors

        #print("Spawning pedestrians , phase 0")
        phase_index = 0 # Phase_index = 0 => spawn the hero (trainable) agents, phase_index = 1 => controlled auto pilot, phase_index = 2 => finished
        while numTries < MAX_SPAWN_RETRIES and phase_index < 2:
            #print(f"attempt :{numTries}")
            if numWalkersSuccesfullySpawned >= self.envSetup.NumberOfCarlaPedestrians:
                break
            numTries += 1

            # Prepare a chunck of pending pedestrians
            numWalkersPendingSpawn = 0
            numRemainingPedestriansToSpawn = self.envSetup.NumberOfCarlaPedestrians - numWalkersSuccesfullySpawned

            # Check the number of hero walkers to be spawned and phase index
            #---
            if phase_index == 0:
                if numWalkersSuccesfullySpawned >= self.envSetup.NumberOfTrainablePedestrians:
                    phase_index = 1
                    #print("Spawning pedestrians , phase 1")
                else:
                    numRemainingPedestriansToSpawn = min(numRemainingPedestriansToSpawn, self.envSetup.NumberOfTrainablePedestrians)
            else:
                pass
            #---

            logging.log(logging.DEBUG, f"Starting new iter of walkers spawning need {numRemainingPedestriansToSpawn} more")

            walkers_list = [] # The list of successfully spawned walkers
            batch = [] # The pending batch to be spawned
            batch_parameters = [] # The parameters of the batch used above
            walker_speed = [] # each walker speed, target and spawn index used
            target_points = []
            pending_walkerSpawnIndices = []

            for walkerSpawnPointIndex, walkerSpawnPointDetails in enumerate(spawnAndDestinationPoints):
                if walkerSpawnPointDetails.usable == False:
                    continue

                # If not simulating from replay, we don't know the target speed and the
                if not self.simulationOptions.simulateReplay:
                    # First select a blueprint
                    walker_bp = random.choice(blueprints_walkers)
                    walkerSpawnPointDetails.blueprint = walker_bp.id

                    if walker_bp.has_attribute('role_name') and phase_index == 0:
                        if phase_index == 0:
                            walker_bp.set_attribute('role_name', 'hero')
                        else:
                            walker_bp.set_attribute('role_name', 'autopilot')

                    if walker_bp.has_attribute('speed'):
                        maxRunningSpeed = float(walker_bp.get_attribute('speed').recommended_values[2])
                        maxWalkingSpeed = float(walker_bp.get_attribute('speed').recommended_values[1])
                        minRunningSpeed = maxWalkingSpeed
                        minWalkingSpeed = max(1.2, maxWalkingSpeed * 0.5)

                        outSpeed = maxWalkingSpeed
                        if random.random() > percentagePedestriansRunning:
                            # walking
                            outSpeed = minWalkingSpeed + np.random.rand() * (maxWalkingSpeed - minWalkingSpeed)
                        else:
                            # running
                            outSpeed = minRunningSpeed + np.random.rand() * (maxRunningSpeed - minRunningSpeed)

                        walkerSpawnPointDetails.speed = outSpeed
                    else:
                        print("Walker has no speed")
                        walkerSpawnPointDetails.speed = 0.0

                # set as not invincible
                walker_bp = self.blueprint_library.find(walkerSpawnPointDetails.blueprint)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'true')


                boundingSize = self.carlaConnectionManager.client.get_blueprint_meshBoundingSize(walker_bp) * 0.01
                walkerSpawnPointDetails.spawn_point.location.z = max(boundingSize.z + 0.5,
                                                                     walkerSpawnPointDetails.spawn_point.location.z)

                walker_speed.append(walkerSpawnPointDetails.speed)
                target_points.append(walkerSpawnPointDetails.target_point)

                batch.append(SpawnActorFunctor(walker_bp, walkerSpawnPointDetails.spawn_point))
                batch_parameters.append(walkerSpawnPointDetails)

                pending_walkerSpawnIndices.append(walkerSpawnPointIndex)
                numWalkersPendingSpawn += 1
                if numWalkersPendingSpawn >= numRemainingPedestriansToSpawn:
                    break

            if numWalkersPendingSpawn <= 0:
                print("Can't get more walkers pending to spawn....")
                break

            # Reduce the number of next future walker spawning data to try
            spawnAndDestinationPoints = spawnAndDestinationPoints[numWalkersPendingSpawn:]
            if spawnAndDestinationPoints_replayIds is not None:
                spawnAndDestinationPoints_replayIds = spawnAndDestinationPoints_replayIds[numWalkersPendingSpawn:]

            assert len(batch) == len(batch_parameters)
            results = self.carlaConnectionManager.client.apply_batch_sync(batch, True)

            # Store from walker speeds and target points only those that succeeded
            walker_speed2 = []
            target_points2 = []
            assert (len(results) == len(batch) == len(batch_parameters))
            for i in range(len(results)):
                if results[i].error:
                    #logging.error(results[i].error)
                    # Allowed to fail for example in case of collisions
                    pass
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
                    target_points2.append(target_points[i])

                    entityId = results[i].actor_id
                    walkerSpawnDetails : WalkerSpawnDetails = batch_parameters[i]
                    self.tempSpawnedPedestrians[entityId] = EntitySpawnDetails(transform=walkerSpawnDetails.spawn_point,
                                                                          velocity=carla.Vector3D(x=0.0, y=0.0, z=0.0),
                                                                          blueprint=walkerSpawnDetails.blueprint,
                                                                          replayId=entityId,
                                                                          color="none",
                                                                          driver_id="none",
                                                                          speed=walkerSpawnDetails.speed,
                                                                          targetPoint=walkerSpawnDetails.target_point)

                    # Associate this pedestrian id with the id in the replay data
                    if self.simulationOptions.simulateReplay:
                        self.simulationOptions.pedestrianSpawnedIdToReplayId[entityId] = batch_parameters[i].replayId

            walker_speed = walker_speed2
            targets_point = target_points2

            pendingTrainableActors = 0

            # 3. we spawn the walker controller only when not driving the agent manually (replay simulation or trainable agents controlled by RLAgent)
            isAgentControlledManually = self.simulationOptions.simulateReplay == True or phase_index == 0
            if not isAgentControlledManually:
                batch = []
                # Try to spawn the walker ai controllers
                walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                for i in range(len(walkers_list)):
                    batch.append(SpawnActorFunctor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))

                results = self.carlaConnectionManager.client.apply_batch_sync(batch, True)

                for i in range(len(results)):
                    if results[i].error:
                        logging.error(results[i].error)
                        assert False, " This should not fail."
                        pass
                    else:
                        walkers_list[i]["con"] = results[i].actor_id
            else:
                for i in range(len(walkers_list)):
                    walkers_list[i]["con"] = None

            # 4. we put altogether the walkers and controllers id to get the objects from their id
            for i in range(len(walkers_list)):
                controllerId = walkers_list[i]["con"]
                walkerId = walkers_list[i]["id"]
                self.s_all_pedestrian_ids.append(controllerId if controllerId is not None else INVALID_ACTOR_TAG)
                self.s_all_pedestrian_ids.append(walkerId)

            numWalkersSuccesfullySpawned += len(walkers_list)
            assert numWalkersSuccesfullySpawned == len(self.s_all_pedestrian_ids)/2
            logging.log(logging.INFO, f"Spawning walkers round {numTries}. Spawned {len(walkers_list)}, Total {len(self.s_all_pedestrian_ids)/2}")

            # 5. initialize each new controller and set target to walk to (list is [controler, actor, controller, actor ...])
            if not isAgentControlledManually:
                logging.log(logging.INFO, "Start initialize controllers !")

                #self.all_pedestrian_actors = self.world.get_actors(self.s_all_pedestrian_ids)
                self.all_pedestrian_actors = []
                for pedId in self.s_all_pedestrian_ids:
                    if pedId is None or pedId == INVALID_ACTOR_TAG:
                        self.all_pedestrian_actors.append(None)

                    else:
                        self.all_pedestrian_actors.append(self.world.get_actor(pedId))
                        logging.log(logging.INFO,"Added actor "+str(self.all_pedestrian_actors[-1])+" len "+str(len(self.all_pedestrian_actors)))
                # Select only pedestrian actors, not including tbe walker controllers in this internal list
                self.pedestrian_actors = [self.all_pedestrian_actors[i] for i in range(1, len(self.s_all_pedestrian_ids), 2)]

                logging.log(logging.INFO, f"Spawned the following actor ids (carla space): {self.all_pedestrian_actors}")

                # wait for a tick to ensure client receives the last transform of the walkers we have just created
                #self.world.tick()

                # set how many pedestrians can cross the road
                logging.log(logging.INFO, f"Setting crossing factors at {percentagePedestriansCrossing}!")
                self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)

                logging.log(logging.INFO, f"Setting target destinations for the actors!")
                # Here we are iterating over the walkers' AI behavior
                for i in range(prevInitWalkers, len(self.s_all_pedestrian_ids), 2):
                    relativeIndexToThisAttempt = i - prevInitWalkers
                    # start walker
                    self.all_pedestrian_actors[i].start()

                    # set walk to random point
                    if True or self.simulationOptions.simulateReplay == False:
                        self.all_pedestrian_actors[i].go_to_location(targets_point[int(relativeIndexToThisAttempt/2)].location)


                    # max speed setting
                    maxSpeed = float(walker_speed[int(relativeIndexToThisAttempt / 2)])
                    #print(f"$$ For pedestrian {i} i will set max speed of {maxSpeed}")
                    self.all_pedestrian_actors[i].set_max_speed(maxSpeed)


                    correspondingPedestrianActor = self.all_pedestrian_actors[i+1]
                    self.walkerIdToAIController[correspondingPedestrianActor.id]        = self.all_pedestrian_actors[i]
                    self.walkerIdToWalkerController[correspondingPedestrianActor.id]    = carla.WalkerControl()



            # 5.1 - fill in the mapping between walker ids and their corresponding walker controller instance
            # This is needed for optimizations, to avoid allocations on each frame
            for i in range(prevInitWalkers + 1, len(self.s_all_pedestrian_ids), 2): # + 1 here because [i = index of AI controller of walker, i + 1 = index of walker actor]
                walkerId = self.s_all_pedestrian_ids[i]
                self.walkerIdToWalkerController[walkerId]    = carla.WalkerControl()

                if phase_index == 0:
                    localTrainableAgentIds.add(walkerId)


            prevInitWalkers = len(self.s_all_pedestrian_ids)
            logging.log(logging.INFO, f"Finished configured this spawned batch !")

        logging.log(logging.INFO, "FInished all batches ! everything is spawned !")
        # Check the trainable walker agents. they should have no walker ai controller
        if not self.simulationOptions.simulateReplay:
            assert prevInitWalkers >= self.envSetup.NumberOfTrainablePedestrians, "Couldnt' spawn enough pedestrians as the requested number of trainable units !!"

            batch_ai_controllersToDestroy = []
            numTrainableWalkerAssigned = 0
            for trainableWalkerIndex in range(0, self.envSetup.NumberOfTrainablePedestrians*2, 2):
                # Add the ai controller to the destroying batch, we don't need it
                walker_ai_controller = self.all_pedestrian_actors[trainableWalkerIndex]
                walker_actor = self.all_pedestrian_actors[trainableWalkerIndex + 1]

                if walker_ai_controller is not None:
                    assert walker_ai_controller.id in localTrainableAgentIds, "Something is wrong with the actors. Seems like the order is not stored"
                    localTrainableAgentIds.remove(walker_ai_controller.id)
                localTrainableAgentInstances.append(walker_actor)

                if walker_ai_controller is not None:
                    batch_ai_controllersToDestroy.append(carla.command.DestroyActor(walker_ai_controller))
                    # Make it Null
                    #self.all_pedestrian_actors[trainableWalkerIndex] = None
                    self.walkerIdToWalkerController[walker_actor.id] = None

                #self.trainableWalkerIndexToActorData[numTrainableWalkerAssigned] = {"controller": carla.WalkerControl(),
                #                                                                    "id": walker_actor.id}
                self.trainableWalkerIds.add(walker_actor.id)
                numTrainableWalkerAssigned += 1

                if len(batch_ai_controllersToDestroy) > 0:
                    self.carlaConnectionManager.client.apply_batch(batch_ai_controllersToDestroy, True)

        #assert len(localTrainableAgentIds) == 0, "Not all trainable agents assigned have been configured properly"
        print(f'Spawned {len(self.s_vehicles_list)} vehicles and {len(self.s_all_pedestrian_ids)/2} walkers')

        if self.args.onlineEnvSettings.shouldTrainableAgentsIgnoreTrafficLights:
            self.carlaConnectionManager.disableTrafficLightsForHeroAgents(localTrainableAgentInstances)

        #time.sleep(SYNC_TIME)
        #self.world.tick()

    # Spawn observer and its sensors. Either a hero or a static agent
    def spawnObserver(self):
        # Spawn the player's vehicle (either hero or global observer point of view like a raycast actor)
        raycastActorFinalTransform : carla.Transform = None
        if self.dataGatherParams.use_hero_actor_forDataGathering:
            self.envSetup.observerSpawnTransform = self.closestSpawnPointTransform
            self.observerSpawnLocation = self.envSetup.observerSpawnTransform.location
            self.currWaypoint = self.map.get_waypoint(self.envSetup.observerSpawnTransform.location)  # This is its first waypoint

            idOfVehicleInReplayData= None # Not needed when not replaying
            if not self.simulationOptions.simulateReplay:
                vehiclesLib = self.blueprint_library.filter('vehicle.audi.a*')
                vehicleToSpawnBp = random.choice(vehiclesLib)
                color = random.choice(vehicleToSpawnBp.get_attribute('color').recommended_values) if vehicleToSpawnBp.has_attribute('color') else "none"
                driver_id = random.choice(vehicleToSpawnBp.get_attribute('driver_id').recommended_values) if vehicleToSpawnBp.has_attribute('driver_id') else "none"

            else:
                idOfVehicleInReplayData = self.vehicles_spawn_points_replayIds[self.closestSpawnPointIndex]
                spawnDataInReplay : EntitySpawnDetails = self.simulationOptions.carReplayIdToSpawnData[idOfVehicleInReplayData]
                assert spawnDataInReplay.replayId == idOfVehicleInReplayData
                vehicleToSpawnBp = self.blueprint_library.find(spawnDataInReplay.blueprint)
                assert vehicleToSpawnBp
                color = spawnDataInReplay.color
                driver_id = spawnDataInReplay.driver_id

            self.setupVehicleBlueprint(vehicleToSpawnBp, color=color, driver_id=driver_id, isHeroTrainableVehicleAttempt=True)
            self.observerVehicle = self.world.spawn_actor(vehicleToSpawnBp, self.envSetup.observerSpawnTransform)
            self.s_heroCaptureCarPerspective.append(self.observerVehicle)
            entityId = self.observerVehicle.id

            assert entityId not in self.tempSpawnedVehicles, f"Entity id {entityId} already spawned !!"
            self.tempSpawnedVehicles[entityId] = EntitySpawnDetails(transform=self.envSetup.observerSpawnTransform,
                                                               velocity=carla.Vector3D(x=0.0, y=0.0, z=0.0),
                                                               blueprint=vehicleToSpawnBp.id,
                                                               replayId=entityId,
                                                               color=color,
                                                               driver_id=driver_id,
                                                               speed=0.0,
                                                               targetPoint=None)

            if self.simulationOptions.simulateReplay:
                self.simulationOptions.carSpawnedIdToReplayId[entityId] = idOfVehicleInReplayData


            #self.observerVehicle.set_simulate_physics(False)
            #self.observerVehicle.set_autopilot(True)

            # Instantiate a behavior model for this vehicle
            if not DataGatherParams.STATIC_CAR:
                self.observerVehicle_model = BehaviorAgent(self.observerVehicle)

                # Set the fixed destination if one given
                observerVehicleDestination = self.dataGatherParams.destination if self.dataGatherParams.destination != None else self.getRandomObserverVehicleDestination()
                self.observerVehicle_model.set_destination(self.currWaypoint.transform.location, observerVehicleDestination)
            else:
                self.observerVehicle_model = None

            #assert observerSpawnTransform == envSetup.observerSpawnTransform, "They must coincide otherwise world2_worldref will have a bug!"

            # Set the raycast actor at the hero car location by default
            # BUT if i'm sim replaying an agent stats file then place the camera above that one to see how it does
            self.raycastActor = self.carlaConnectionManager.world.get_raycastActor()
            obsPos : carla.Transform = self.envSetup.observerSpawnTransform

            transformToUseForRaycastActor : carla.Transform = self.envSetup.observerSpawnTransform \
                                                                if self.dataGatherParams.simulationIsReplayingStats is False else \
                                                                self.dataGatherParams.getPreferedReplyingStatsAttachedObserverTransform()
            locationTouseForRaycastActor = transformToUseForRaycastActor.location

            # Keep the z of the initial raycast actor..otherwise it will collid with objects :)
            initialRaycastTransform :carla.Transform = self.raycastActor.get_transform()
            transformToUseForRaycastActor = carla.Transform(location=carla.Location(locationTouseForRaycastActor.x,
                                                                                    locationTouseForRaycastActor.y,
                                                                                    self.dataGatherParams.RAYCAST_ACTOR_Z), #initialRaycastTransform.location.z),
                                                            rotation=transformToUseForRaycastActor.rotation)

            raycastActorFinalTransform = transformToUseForRaycastActor
            self.raycastActor.set_transform(transformToUseForRaycastActor)
        else:
            # Handle the raycasting actor scene things
            self.raycastActor = self.carlaConnectionManager.world.get_raycastActor()

            isThereAnExistingRaycastActor = self.envSetup.forceExistingRaycastActor == True and self.raycastActor != None and self.raycastActor.id != 0
            assert isThereAnExistingRaycastActor or self.envSetup.observerSpawnTransform is not None, "Either you provide a raycasting actor or you specify the transform to spawn one at !"
            if isThereAnExistingRaycastActor is False or self.envSetup.observerSpawnTransform is not None:
                obsPosList : carla.Transform = self.envSetup.observerSpawnTransform
                loc = obsPosList.location  #carla.Location(-2920.0, 13740.0, 1140.0)
                dir = obsPosList.rotation
                dir = carla.Vector3D(dir.pitch, dir.yaw, dir.roll) #obsPosList.rotation
                loc_forUnreal = convertMToCM(loc)
                self.carlaConnectionManager.world.spawn_raycastActor(location=loc_forUnreal, direction=dir,
                                                                     voxelsize=self.envSetup.observerVoxelSize, numvoxelsX=self.envSetup.observerNumVoxelsX,
                                                                     numvoxelsY=self.envSetup.observerNumVoxelsY, numvoxelsZ=self.envSetup.observerNumVoxelsZ)

                self.raycastActor = self.carlaConnectionManager.world.get_raycastActor()
                raycastActorTransform = carla.Transform(loc, carla.Rotation(dir.x, dir.y, dir.z))
                self.raycastActor.set_transform(raycastActorTransform)

                raycastActorFinalTransform = raycastActorTransform

                #time.sleep(SYNC_TIME)
                #self.world.tick()  # Be sure that player's vehicle is spawned

                assert self.raycastActor, "Can't create raycast actor !!!!"

            # Capture some scene data related stuff when using raycasting
            self.solveSceneData()

            # Altought the observer is up in the sky we want its Z to be on the closest to ground as possible for
            # further reference calculations. This will be used from now on, NOTE that the scene was solved from sky
            # using the solveSceneData call above
            # Check where observerSpawLocation is used locally in this function further
            self.envSetup.observerSpawnTransform = raycastActorFinalTransform #self.raycastActor.get_transform()
            self.envSetup.observerSpawnTransform.location.z = self.closestSpawnPointTransform.location.z
            self.observerSpawnLocation = self.envSetup.observerSpawnTransform.location

            if not self.dataGatherParams.use_hero_actor_forDataGathering:
                self.observerVehicle = self.raycastActor

        # Establish the attachement of cameras depending on replay mode on/off
        self.observerCameraAttachment = self.observerVehicle if self.simulationOptions.simulateReplay == False else self.raycastActor


        # Spawn the camera sensors/actors for the car perspective, lidar if needed
        #------------------------------------------------------
        if self.dataGatherParams.use_hero_actor_forDataGathering:
            logging.log(logging.INFO, 'Spawning sensors...')
            # Create sensors blueprints
            cameraRgbBlueprint = self.blueprint_library.find('sensor.camera.rgb')
            cameraDepthBlueprint = self.blueprint_library.find('sensor.camera.depth')
            cameraSegBlueprint = self.blueprint_library.find('sensor.camera.semantic_segmentation')

            # Create the camera sensors
            cameraBlueprints = [('rgb',cameraRgbBlueprint), ('depth',cameraDepthBlueprint), ('seg',cameraSegBlueprint)]
            for camName, camBlueprint in cameraBlueprints:
                cameraIntrisics = self.dataGatherParams.configureCameraBlueprint(camBlueprint)


                observerCameraTransform=self.dataGatherParams.getObserverFrontCameraTransform(observerActorTransform=self.envSetup.observerSpawnTransform)
                camInstance = self.world.spawn_actor(camBlueprint,
                                                     observerCameraTransform,
                                                     attach_to=self.observerCameraAttachment)
                self.s_heroCaptureCarPerspective.append(camInstance)
                self.s_heroCaptureCarPerspective_sensors[camName] = camInstance
                self.s_heroCaptureCarPerspective_intrisics[camName] = cameraIntrisics

            # Create and configure the lidar blueprints
            lidarSetupData = self.dataGatherParams.lidarData
            if lidarSetupData is not None and lidarSetupData['useLidar'] == 1:
                lidar_bp = self.blueprint_library.filter("sensor.lidar.ray_cast")[0]
                lidar_bp.set_attribute('dropoff_general_rate', str(lidarSetupData['noise_dropoff_general_rate']))
                lidar_bp.set_attribute('dropoff_intensity_limit', str(lidarSetupData['noise_dropoff_intensity_limit']))
                lidar_bp.set_attribute('dropoff_zero_intensity', str(lidarSetupData['noise_dropoff_intensity_limit']))
                lidar_bp.set_attribute('upper_fov', str(lidarSetupData['upperFOV']))
                lidar_bp.set_attribute('lower_fov', str(lidarSetupData['lowerFOV']))
                lidar_bp.set_attribute('channels', str(lidarSetupData['channels']))
                lidar_bp.set_attribute('range', str(lidarSetupData['range']))
                lidar_bp.set_attribute('points_per_second', str(lidarSetupData['pointsPerSecond']))

                # Set the rotation frequency same as FPS to be sure that we get the full scene in each single frame
                # You can increase the points_per_second thus to control the density of the points
                lidar_bp.set_attribute('rotation_frequency', str(self.envSetup.fixedFPS))

                # Create the lidar sensors
                lidar = self.world.spawn_actor(blueprint=lidar_bp, transform=self.dataGatherParams.getObserverFrontLidarTransform(observerActorTransform=self.envSetup.observerSpawnTransform), attach_to=self.observerCameraAttachment)
                self.s_heroCaptureCarPerspective.append(lidar)
                self.s_heroCaptureCarPerspective_sensors['lidar'] = lidar

        # Allow above actors spawning
        #self.world.tick()

        # Set the world to world reference matrix transform, such that from now one everything will related to this (if needed)
        # Note: will be set to lidar position for capturing purposes...

        # DEBUG THING
        DEBUG_THING = False
        if DEBUG_THING is True:
            testT = carla.Transform(rotation=carla.Rotation(yaw=-30.0, pitch=-45, roll =45))
            testT_matrix = testT.get_matrix()
            test_T_inv_matrix = testT.get_inverse_matrix()

            testT_inv2 = carla.Transform(rotation=carla.Rotation(yaw=30.0))#, pitch=45, roll=-45))
            testT_inv2_matrix = testT_inv2.get_matrix()

            res = np.dot(testT_matrix, testT_inv2_matrix)

            baseTransformAsWorldRef = carla.Transform(rotation=carla.Rotation(yaw=-45.0), location=carla.Location(x=1.0, y=1.0, z=0.0))
        else:
            baseTransformAsWorldRef = self.envSetup.observerSpawnTransform
            if self.dataGatherParams.use_hero_actor_forDataGathering:
                lidarInstance = self.s_heroCaptureCarPerspective_sensors['lidar']
                #baseTransformAsWorldRef = lidarInstance.get_transform()
            baseTransformAsWorldRef = self.envSetup.observerSpawnTransform

        self.world_2_worldRef_matrix = np.array(baseTransformAsWorldRef.get_inverse_matrix())
        self.worldRef_2_world_matrix = np.linalg.inv(self.world_2_worldRef_matrix) # hope for the best

        if DEBUG_THING:
            actorVelocity_w = np.array([0.0, 0.0, 0.0, 1.0])
            actorVelocity_ref = np.dot(self.world_2_worldRef_matrix, actorVelocity_w.T)

            worldRef_to_World = np.linalg.inv(self.world_2_worldRef_matrix)
            actorVelocity_w_2 = np.dot(worldRef_to_World, actorVelocity_ref)

        #------------------------------------------------------

        # Register the observers sensors
        self.dataManager = SensorsDataManagement(self.world, self.envSetup.fixedFPS, self.s_heroCaptureCarPerspective_sensors)

    def setupVehicleBlueprint(self, blueprint, color, driver_id, isHeroTrainableVehicleAttempt):
            if blueprint.has_attribute('color') and color != None and color != "none":
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id') and driver_id != None and driver_id != "none":
                blueprint.set_attribute('driver_id', driver_id)

            if isHeroTrainableVehicleAttempt:
                blueprint.set_attribute('role_name', 'hero')
            else:
                blueprint.set_attribute('role_name', 'autopilot')

    def spawnVehicles(self,):
        # --------------
        # Spawn vehicles
        # --------------
        # TODO Ciprian: SHould we spawn this in a batch later ? currently we have only a few cars usually, not as many as walkers....
        blueprints_vehicles = self.createVehiclesBlueprintLibrary()

        prevTrainableVehiclesInited = 0
        localTrainableAgentInstances = []

        logging.log(logging.INFO, 'Spawning vehicles')
        observerTransform = self.envSetup.observerSpawnTransform
        self.vehicles_spawn_points = sorted(self.vehicles_spawn_points, key = lambda transform : compute_distance(transform.location, observerTransform.location))
        for n, transform in enumerate(self.vehicles_spawn_points):
            blueprint = None

            # When simulating from replay do not filter the positions
            if not self.simulationOptions.simulateReplay:
                potentialSpawnPointLoc = transform.location
                # Check if in front of the car
                isLocInFront = isPosInFaceOfObserverPos(observerTransform.rotation.get_forward_vector(),
                                                        self.envSetup.observerSpawnTransform.location,
                                                        potentialSpawnPointLoc)

                if isLocInFront == False:
                    continue

                # Choose a blueprint and configure it
                blueprint = random.choice(blueprints_vehicles)
                color = random.choice(blueprint.get_attribute('color').recommended_values) if blueprint.has_attribute('color') else "none"
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values) if blueprint.has_attribute('driver_id') else "none"
            else:
                idOfVehicleInReplayData = self.vehicles_spawn_points_replayIds[n]
                spawnDataInReplay : EntitySpawnDetails = self.simulationOptions.carReplayIdToSpawnData[idOfVehicleInReplayData]
                assert spawnDataInReplay.replayId == idOfVehicleInReplayData
                blueprint = self.blueprint_library.find(spawnDataInReplay.blueprint)
                assert blueprint
                color = spawnDataInReplay.color
                driver_id = spawnDataInReplay.driver_id

            # Spawn the vehicle
            numTrainableVehiclesNeeded = self.envSetup.NumberOfTrainableVehicles - prevTrainableVehiclesInited
            isHeroTrainableVehicleAttempt = (numTrainableVehiclesNeeded > 0)

            self.setupVehicleBlueprint(blueprint=blueprint, color=color, driver_id=driver_id, isHeroTrainableVehicleAttempt=isHeroTrainableVehicleAttempt)
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                self.s_vehicles_list.append(vehicle)

                entityId = vehicle.id
                assert entityId not in self.tempSpawnedVehicles, f"Entity id {entityId} already spawned !!"
                self.tempSpawnedVehicles[entityId] = EntitySpawnDetails(transform=transform,
                                                                   velocity=carla.Vector3D(x=0.0, y=0.0, z=0.0),
                                                                   blueprint=blueprint.id,
                                                                   replayId=entityId,
                                                                   color=color,
                                                                   driver_id=driver_id,
                                                                   speed=None,
                                                                   targetPoint=None)

                # Associate this vehicle id with the id in the replay data
                if self.simulationOptions.simulateReplay:
                    self.simulationOptions.carSpawnedIdToReplayId[entityId] = self.vehicles_spawn_points_replayIds[n]

                if isHeroTrainableVehicleAttempt:
                    #self.trainableVehicleIndexToActorData[prevTrainableVehiclesInited] = {
                    #        "controller": carla.VehicleControl(),
                    #        "id": entityId}
                    self.trainableVehiclesIds.add(entityId)
                    prevTrainableVehiclesInited += 1

                    localTrainableAgentInstances.append(vehicle)

                if len(self.s_vehicles_list) >= self.envSetup.NumberOfCarlaVehicles:
                    break

        if self.args.onlineEnvSettings.shouldTrainableAgentsIgnoreTrafficLights:
            self.carlaConnectionManager.disableTrafficLightsForHeroAgents(localTrainableAgentInstances)

        # Wait to have all things spawned on server side
        #time.sleep(SYNC_TIME)
        #self.world.tick()


    def processSpawnPoints(self):
        crosswalks = self.map.get_crosswalks()
        landmarks = self.map.get_all_landmarks()
        self.blueprint_library = self.carlaConnectionManager.world.get_blueprint_library()

        # Consider all points avaible on the map when not simulating
        if not self.simulationOptions.simulateReplay:
            self.vehicles_spawn_points = self.map.get_spawn_points()

            self.player_spawn_pointsAndIndices = None
            if EnvSetupParams.useOnlySpawnPointsNearCrossWalks:
                assert False, "Should not be used anymore"
                # These are the spawnpoints for the player vehicle
                # Get one for each episode indeed, sorted by importance
                # We try to spawn the player close and with view to crosswalks
                self.spawn_points_nearcrosswalks = self.map.get_spawn_points_nearcrosswalks()
                self.player_spawn_pointsAndIndices = self.uniformSampleSpawnPoints(self.spawn_points_nearcrosswalks,
                                                                                   self.dataGatherParams.maxNumberOfEpisodes)
            else:
                self.player_spawn_pointsAndIndices = [(i, transform) for i, transform in enumerate(self.vehicles_spawn_points)]
        else:
            # Put in only the used spawn points used by the vehicles in the replay
            self.vehicles_spawn_points : List[carla.Transform] = []
            self.vehicles_spawn_points_replayIds : List[any] = []
            for replayId, spawnData in self.simulationOptions.carReplayIdToSpawnData.items():
                self.vehicles_spawn_points.append(spawnData.replayDataPerFrame.transform)
                self.vehicles_spawn_points_replayIds.append(spawnData.replayId)

            self.player_spawn_pointsAndIndices = [(i, transform) for i, transform in enumerate(self.vehicles_spawn_points)]

        if len(self.player_spawn_pointsAndIndices) <= 0:
            "There are no interesting spawn points on this map. Remove map or lower requirements from the server side"
            self.carlaConnectionManager.releaseServerConnection()
            raise Exception()

        self.dataGatherParams.maxNumberOfEpisodes = min(len(self.player_spawn_pointsAndIndices), self.dataGatherParams.maxNumberOfEpisodes)
        logging.log(logging.INFO, "There are %d interesting spawn points on the map" % len(self.player_spawn_pointsAndIndices))


        # Save the spawn point index that is closest to our observer
        self.closestSpawnPointIndex = None
        self.closestSpawnPointDist = None
        self.closestSpawnPointTransform = None
        for spawnPointIndex, spawnPointTransform in self.player_spawn_pointsAndIndices:
            if self.envSetup.forcedObserverSpawnPointIndex != None and self.envSetup.forcedObserverSpawnPointIndex != spawnPointIndex:
                continue

            distToThisSpawnPoint = compute_distance(spawnPointTransform.location, self.envSetup.observerSpawnTransform.location)
            if self.closestSpawnPointIndex is None or distToThisSpawnPoint < self.closestSpawnPointDist:
                self.closestSpawnPointDist = distToThisSpawnPoint
                self.closestSpawnPointIndex = spawnPointIndex
                self.closestSpawnPointTransform = spawnPointTransform

        assert self.closestSpawnPointIndex != None and self.closestSpawnPointTransform != None, "No closeset spawn point was detected ! Something is WRONG you need to investigate !"

        if self.envSetup.forcedObserverSpawnPointTransform != None:
            forcedLocation_AsVector = carlaVector3DToNumpy(self.envSetup.forcedObserverSpawnPointTransform.location)
            forcedRotation_AsVector = carlaRotationToNumpy(self.envSetup.forcedObserverSpawnPointTransform.rotation)

            closestLocation_AsVector = carlaVector3DToNumpy(self.closestSpawnPointTransform.location)
            closestRotation_AsVector = carlaRotationToNumpy(self.closestSpawnPointTransform.rotation)
            assert numpy.allclose(closestLocation_AsVector, forcedLocation_AsVector) and numpy.allclose(forcedRotation_AsVector, closestRotation_AsVector), "THE FORCED TRANSFORM DOESN'T CORRESPOND ANMORE !!!! PLEASE CHECK THIS , RECREATE DATASET MAYBE SOMETHING HAS CHANGED in THE MAP !!"

        print(f"$$$ For scene name {self.dataGatherParams.sceneName} we found that the closest spawn point index for OBSERVER is {self.closestSpawnPointIndex}")

    def spawnActors(self, envSetup : EnvSetupParams, dataGatherParams : DataGatherParams, simulationOptions : SimOptions):
        self.dataGatherParams = dataGatherParams
        self.simulationOptions = simulationOptions
        self.envSetup = envSetup

        # Load the replay data (each frame in the replay) inside the simulationOptions DS

        if self.simulationOptions.simulateReplay:
            # When simulating, point the replayFrom data to the folder where you previously run and evaluated episode
            self.dataGatherParams.outputCurrentEpisodePath = self.simulationOptions.pathToReplayData
            self.loadReplayData()

        # reset some stats - see constructor to understand these
        self.vehicles_data = {}
        self.pedestrians_data = {}
        self.simulationOptions.carsData = {}
        self.simulationOptions.pedestriansData = {}

        # Step 0: First reload the world
        self.carlaConnectionManager.reloadWorld(envSetup)
        self.world = self.carlaConnectionManager.world
        self.map = self.carlaConnectionManager.world.get_map()

        # Step 1: Get feedback from the created world
        #print(dir(self.carlaConnectionManager.world))
        self.spectator = self.carlaConnectionManager.world.get_spectator()

        self.processSpawnPoints()

        # Prepare output paths and folders
        # When simulating, point the replayFrom data to the folder where you previously run and evaluated episode
        self.dataGatherParams.outputCurrentEpisodePath = os.path.join(dataGatherParams.outputEpisodesBasePath, str(self.envSetup.mapToUse), str(self.envSetup.episodeIndex), str(self.closestSpawnPointIndex)) \
                                                            if self.simulationOptions.simulateReplay is False and self.args.onlineEnvSettings.isDataGatherAndStoreEnabled == True   \
                                                            else self.simulationOptions.pathToReplayData

        if self.args.onlineEnvSettings.isDataGatherAndStoreEnabled:
            logging.log(logging.INFO, "Preparing episode output to %s", self.dataGatherParams.outputCurrentEpisodePath)
            self.dataGatherParams.prepareEpisodeOutputFolders(self.dataGatherParams.outputCurrentEpisodePath,
                                                              self.envSetup.forcedObserverSpawnPointIndex,
                                                              self.envSetup.forcedObserverSpawnPointTransform)

        logging.log(logging.INFO, "Starting to create the environment...")

        # Step 2: Spawn the requested entities
        #---------------------------------------------------------------------
        self.tempSpawnedVehicles.clear()
        self.tempSpawnedPedestrians.clear()
        self.walkerIdToAIController.clear()

        self.spawnObserver()


        #time.sleep(SYNC_TIME)
        #self.world.tick() # Be sure that player's vehicle is spawned

        # Set the spectator pos and rot
        self.updateSpectator()

        self.spawnVehicles()

        #time.sleep(SYNC_TIME)
        #self.world.tick()

        self.spawnWalkers()

        #time.sleep(SYNC_TIME)
        #self.world.tick()

        # Some post spawn processes

        # Set auto pilot for vehicles spawned
        autopilotForAllVehicles = True if self.simulationOptions.simulateReplay == False else False
        for v in self.s_vehicles_list:
            if v.id not in self.trainableVehiclesIds:
                v.set_autopilot(autopilotForAllVehicles)

        # Save the spawned data records
        self.recordSpawnData(carsSpawnedData=self.tempSpawnedVehicles, pedestriansSpawnedData=self.tempSpawnedPedestrians)

    # This function swaps the current trainable agents(veh or peds) with others existing ONLY by their spawn transform,
    # we keep same actors because they could were configured to be heros, to stop lightnings etc
    def swapTrainableEntitiesSpawnData(inCurrentTrainableIds, allActorsData, internalSpawnDataDict):
        for oldTrainable_id in inCurrentTrainableIds:
            # Select a random actor to swap for each existing trainable one
            newTrainableActor = random.choice(allActorsData)
            newTrainableActor_id = newTrainableActor.id

            # Swap now their spawn transforms
            tempTransform = internalSpawnDataDict[newTrainableActor_id].replayDataPerFrame
            internalSpawnDataDict[newTrainableActor_id].replayDataPerFrame = internalSpawnDataDict[oldTrainable_id].replayDataPerFrame
            internalSpawnDataDict[oldTrainable_id].replayDataPerFrame = tempTransform

    # Swap around the trainable cars such that they keep the same IDs (see the reason in function swapTrainable comment) but only change positions
    def shuffleTrainableCars(self):
        EnvironmentActors.swapTrainableEntitiesSpawnData(self.trainableVehiclesIds, self.s_vehicles_list, self.vehicles_data[FRAME_INDEX_FOR_SPAWN_DETAILS])

    # Swap around the pedestrians positions such that they keep the same IDS (see the reason in function swapTrainable comment) but only change positions and vel with the one given
    # If transformToWorldReference = True, then it means that the position you are sending here is actually a global World position,
    #, and you fi you are working with a dataset remember that every dataset sample has a reference position,
    #, so the rotation and location must be converted to the World =>> World Reference. World Reference is the location/orientation where the datasample was taken from
    def setPedestrianAgentSpawnData(self, netAgent, pos, vel_init, goal, transformToWorldReference=False):
        # Take the trainable pedestrian and set its new transform to the one given
        # Currently not using the goal since CARLA doesnt have to do anything about it....the RLAgent will control step by step to the goal
        onlineEnvAgentId = netAgent.onlinerealtime_agentId # Expecting a SimplifiedAgent base class

        # Put the new transform on the spawn frame details
        carlaNewRot = Vector3DVoxelSpace_ToEulerCarlaSpace(vel_init)
        carlaNewPos = voxelSpaceLocationToCarlaLocation(pos)

        carlaNew_Transform = carla.Transform(carlaNewPos, carlaNewRot)
        carlaNew_Transform_matrix = get_matrix(carlaNew_Transform)

        carlaNew_Rot_matrix = carlaNew_Transform_matrix[0:3,0:3]

        if transformToWorldReference:
            carlaNewPos_asNumpy = carlaVector3DToNumpy(carlaNewPos)
            carlaNewPos = np.dot(self.worldRef_2_world_matrix, np.append(carlaNewPos_asNumpy, 1.0).T)
            carlaNewRot_asMatrix = np.dot(self.worldRef_2_world_matrix[0:3,0:3], carlaNew_Rot_matrix)

            carlaRotation = rotationMatrixToCarlaRotation(carlaNewRot_asMatrix)
            carlaLocation = NumpyToCarlaVector3D(carlaNewPos)

            finalTransform = carla.Transform(location=carlaLocation, rotation=carlaRotation)
        else:
            finalTransform = carla.Transform(location=carlaNewPos, rotation=carlaNewRot)

        self.pedestrians_data[FRAME_INDEX_FOR_SPAWN_DETAILS][onlineEnvAgentId].replayDataPerFrame = finalTransform

    # Idea: put everyone back to the initial position as it was before playing an episode
    # Note that there could be shuffling around on the positions of the pedestrians if you called the function above !
    def resetActors(self):
        # raise NotImplementedError()

        # Step 1: We keep only the initialization frame (spawn) data
        self.pedestrians_data = { FRAME_INDEX_FOR_SPAWN_DETAILS : self.pedestrians_data[FRAME_INDEX_FOR_SPAWN_DETAILS] } \
            if FRAME_INDEX_FOR_SPAWN_DETAILS in self.pedestrians_data else {}
        self.vehicles_data = { FRAME_INDEX_FOR_SPAWN_DETAILS : self.vehicles_data[FRAME_INDEX_FOR_SPAWN_DETAILS] } \
            if FRAME_INDEX_FOR_SPAWN_DETAILS in self.vehicles_data else {}

        # Step 2: We place all them in the world according to the spawning data (transforms and velocities, target points, etc).
        self.updateEntitiesFromReplayData(FRAME_INDEX_FOR_SPAWN_DETAILS, setupFutureToo=False)

    def getRandomObserverVehicleDestination(self):
        assert self.observerVehicle_model, "The vehicle model is not instantiated !"
        spawn_points = self.vehicles_spawn_points
        random.shuffle(spawn_points)
        destination = None
        if spawn_points[0].location != self.observerVehicle_model.vehicle.get_location():
            destination = spawn_points[0].location
        else:
            destination = spawn_points[1].location
        return destination

    def updateSpectator(self):
        # Don't update spectator if not needed
        if not self.dataGatherParams.use_hero_actor_forDataGathering or \
        (int(self.dataGatherParams.args.onlineEnvSettings.no_server_rendering) == 1 and int(self.dataGatherParams.args.onlineEnvSettings.no_client_rendering)==1):
            return

        # Get the main RGB camera actor and put the spectator there
        rgbCameraActor = self.s_heroCaptureCarPerspective_sensors['rgb']
        spectator_transform = rgbCameraActor.get_transform()
        self.spectator.set_transform(spectator_transform)

    # Some operation that are done after updating the environment and data gathering stuff
    def doPostUpdate(self):

        # If we are not simulating a replay...
        if not self.simulationOptions.simulateReplay:
            # Choose the next waypoint and update the car location if car perspective is used
            if self.dataGatherParams.use_hero_actor_forDataGathering:
                if not DataGatherParams.STATIC_CAR:
                    # Update info
                    self.observerVehicle_model._update_information()

                    # Check if we are at the destionation or very close. If yes, reroute
                    if self.dataGatherParams.rerouteAllowed and \
                            len(self.observerVehicle_model.get_local_planner().waypoints_queue) < DataGatherParams.MIN_WAYPOINTS_TO_DESTINATION:
                        self.observerVehicle_model.reroute(self.vehicles_spawn_points) ## ????????

                    # Set target speed to limit, as a human being, or in my case it should be ~ (120% * limit)
                    speed_limit = self.observerVehicle.get_speed_limit()
                    self.observerVehicle_model.get_local_planner().set_speed(speed_limit)

                    # Run the model and get the control to use
                    control = self.observerVehicle_model.run_step()

                    # Apply control
                    self.observerVehicle.apply_control(control)

                    # This old code is used just to modify at each frame manually from next to next waypoints without any kind of physical simulation
                    """
                    self.currWaypoint = random.choice(self.currWaypoint.next(1.5))
                    waypointTransform = self.currWaypoint.transform
                    waypointLocation = waypointTransform.location
                    waypointRotation = waypointTransform.rotation
                    modifiedWaypointTransform = carla.Location(x=waypointLocation.x, y=waypointLocation.y, z=waypointLocation.z + 5.0)
                    newTransform = carla.Transform(location=modifiedWaypointTransform, rotation=waypointRotation)
                    self.observerVehicle.set_transform(newTransform)
                    pass
                    """
                else:
                    # Apply brake
                    self.observerVehicle.apply_control(carla.VehicleControl(hand_brake=True))

        self.updateSpectator()

    # Given a list of actors, write a dictionary for each frame and actor id, the BBoxMinMax and velocity
    def addFrameData_internal(self, listOfActors, outputDS):
        for actor in listOfActors:
            assert isinstance(actor, carla.Walker) or isinstance(actor, carla.Vehicle)

            # DEBUG_REMOVE
            """
            if isinstance(actor, carla.Vehicle):
                print(actor.get_acceleration())
                print(actor.get_velocity())
                print(actor.get_location())
            """



            actorId = actor.id
            actorWorldTransform : carla.Transform = actor.get_transform()
            actorWorldLocation = actorWorldTransform.location
            actorWorldRotation = actorWorldTransform.rotation
            actorTransformMatrix = actorWorldTransform.get_matrix() # this is in world space
            actorTransformToRefWorld = np.dot(self.world_2_worldRef_matrix, actorTransformMatrix) # this is in world space but relative to reference
            actorLocation = actorTransformToRefWorld[0:3, 3]
            actorRotation = actorTransformToRefWorld[0:3,0:3]

            actorVelocity = actor.get_velocity()
            actorVelocity = np.array([actorVelocity.x, actorVelocity.y, actorVelocity.z])
            actorVelocity = np.dot(self.world_2_worldRef_matrix[0:3,0:3], actorVelocity.T)

            ai, aj, ak = transforms3d.euler.mat2euler(actorRotation)


            # Returns as [4 x 8], x,y,z1 for each of the 8 points. So all X are on row 0, Y on row 1, Z on row 2
            actorWorldBBox = getActorWorldBBox(actor, actorCustomTransformMatrix=actorTransformToRefWorld)

            xMin = np.min(actorWorldBBox[0, :])
            xMax = np.max(actorWorldBBox[0, :])
            yMin = np.min(actorWorldBBox[1, :])
            yMax = np.max(actorWorldBBox[1, :])
            zMin = np.min(actorWorldBBox[2, :])
            zMax = np.max(actorWorldBBox[2, :])
            bboxMinMax = np.array([[xMin, xMax], [yMin, yMax], [zMin, zMax]])

            assert actorId not in outputDS
            # Fill the data for this actor
            actorData = {'BBMinMax' : bboxMinMax, 'VelocityRef':actorVelocity, 'ActorOrientation' : np.array([ai, aj, ak]),
                         'WorldRefLocation' : actorLocation,#np.array([actorWorldLocation.x, actorWorldLocation.y, actorWorldLocation.z]),
                         'WorldRefRotation' : actorRotation}
            outputDS[actorId] = actorData

    # Will get maps from spawned entities ids to details when recording
    def recordSpawnData(self, carsSpawnedData : Dict[any, EntitySpawnDetails],
                        pedestriansSpawnedData : Dict[any, EntitySpawnDetails]):
        self.vehicles_data[FRAME_INDEX_FOR_SPAWN_DETAILS]       = carsSpawnedData
        self.pedestrians_data[FRAME_INDEX_FOR_SPAWN_DETAILS]    = pedestriansSpawnedData

        # If not simulation replay enabled, then just fill in some dummy data such that some underlying infrastracture works
        if self.simulationOptions.simulateReplay != True:
            assert FRAME_INDEX_FOR_SPAWN_DETAILS not in self.simulationOptions.carsData and \
                FRAME_INDEX_FOR_SPAWN_DETAILS not in self.simulationOptions.pedestriansData
            self.simulationOptions.carsData[FRAME_INDEX_FOR_SPAWN_DETAILS] = {}
            self.simulationOptions.pedestriansData[FRAME_INDEX_FOR_SPAWN_DETAILS] = {}
            for carKey, carSpawnData in carsSpawnedData.items():
                self.simulationOptions.carSpawnedIdToReplayId[carKey] = carKey
                self.simulationOptions.carsData[FRAME_INDEX_FOR_SPAWN_DETAILS][carKey]  = carSpawnData.replayDataPerFrame

            for pedKey, pedSpawnData in pedestriansSpawnedData.items():
                self.simulationOptions.pedestrianSpawnedIdToReplayId[pedKey] = pedKey
                self.simulationOptions.pedestriansData[FRAME_INDEX_FOR_SPAWN_DETAILS][pedKey]   = pedSpawnData.replayDataPerFrame

    # Given world data and where to write output for a single frame, do a snapshot of the world there
    def addFrameData(self, frameId, worldFrame, out_vehicles_data, out_pedestrians_data, isInitializationFrame=False):
        assert isInitializationFrame or frameId not in out_pedestrians_data
        assert isInitializationFrame or frameId not in out_vehicles_data

        out_vehicles_data[frameId] = {}
        out_pedestrians_data[frameId] = {}

        # Iterate over walkers and get their
        # DO NOT CACHE THESE BECAUSE THEY CAN MODIFY AT RUNTIME
        allWalkerActorsIds = [self.s_all_pedestrian_ids[walkerId] for walkerId in range(1, len(self.s_all_pedestrian_ids), 2)]
        allVehicleActors = [vehicle for vehicle in self.s_vehicles_list]
        if isinstance(self.observerVehicle, carla.Vehicle):
            allVehicleActors.append(self.observerVehicle)
        allWalkerActors = self.world.get_actors(allWalkerActorsIds)

        self.addFrameData_internal(allWalkerActors, out_pedestrians_data[frameId])
        self.addFrameData_internal(allVehicleActors, out_vehicles_data[frameId])

    # Returns the vehicles and pedestrians data for the given frame
    def getFrameData(self, frameId):
        return self.vehicles_data[frameId], self.pedestrians_data[frameId]

    def updateEntitiesFromReplayData(self, frameId, setupFutureToo=True):

        # Iterate over walkers and get their
        # DO NOT CACHE THESE BECAUSE THEY CAN MODIFY AT RUNTIME
        allWalkerActorsIds = [self.s_all_pedestrian_ids[walkerId] for walkerId in range(1, len(self.s_all_pedestrian_ids), 2)]
        allVehicleActors = [vehicle for vehicle in self.s_vehicles_list]
        if isinstance(self.observerVehicle, carla.Vehicle):
            allVehicleActors.append(self.observerVehicle)
        allWalkerActors = self.world.get_actors(allWalkerActorsIds)

        # Create a batch of apply transform commands for cars
        batch_cars = []
        for vehicleActor in allVehicleActors:
            entityReplayData : EntityReplayDataPerFrame = self.simulationOptions.getEntityDataPerFrame(frameId=frameId,
                                                                                                   spawnId=vehicleActor.id,
                                                                                                   isCar=True,
                                                                                                    customReplayData=self.vehicles_data)

            actorTargetTransform = entityReplayData.transform
            if isinstance(actorTargetTransform, np.ndarray):
                actorTargetTransform = NumpyToCarlaTransform(actorTargetTransform)
            vehicleActor.set_transform(actorTargetTransform)
            #batch_cars.append(carla.command.ApplyTransform(actor_id=vehicleActor.id, transform=actorTargetTransform))

        # Create a batch of walker controls for walkers
        batch_pedestrians = []
        USE_ROOT_CTRL = False
        for walkerActor in allWalkerActors:
            if USE_ROOT_CTRL:
                entityReplayData_currFrame : EntityReplayDataPerFrame = self.simulationOptions.getEntityDataPerFrame(frameId=frameId,
                                                                                                                     spawnId=walkerActor.id,
                                                                                                                     isCar=False,
                                                                                                                     customReplayData=self.pedestrians_data)
                #batch_pedestrians.append(carla.command.ApplyTransform(actor_id=walkerActor.id, transform=entityReplayData_currFrame.transform))

                control = carla.WalkerBoneControl()
                targetLoc = entityReplayData_currFrame.transform.location
                newRootBoneTransform = carla.Transform(location=carla.Location(targetLoc.x * 100.0, targetLoc.y * 100.0, targetLoc.z * 100.0))
                root_new_transform = ('crl_root', newRootBoneTransform)
                                                                   #rotation=entityReplayData_currFrame.transform.rotation))

                control.bone_transforms = [root_new_transform]
                walkerActor.apply_control(control)

            else:
                # Step 1: Set tranform for current frame
                if True and frameId % 1 == 0:
                    entityReplayData_currFrame : EntityReplayDataPerFrame = self.simulationOptions.getEntityDataPerFrame(frameId=frameId,
                                                                                                                         spawnId=walkerActor.id,
                                                                                                                         isCar=False,
                                                                                                                         customReplayData=self.pedestrians_data)
                    transform = entityReplayData_currFrame
                    if isinstance(entityReplayData_currFrame, EntityReplayDataPerFrame):
                        transform = entityReplayData_currFrame.transform
                    if isinstance(transform, np.ndarray):
                        transform = NumpyToCarlaTransform(transform)
                    assert isinstance(transform, carla.Transform)

                    walkerActor.set_transform(transform)
                    resTransform = walkerActor.get_transform()
                    #batch_pedestrians.append(carla.command.ApplyTransform(actor_id=walkerActor.id, transform=transform))

            if setupFutureToo:
                # Step 2: Set direction of movement for next frame
                entityReplayData_nextFrame : EntityReplayDataPerFrame = self.simulationOptions.getEntityDataPerFrame(frameId=frameId+1,
                                                                                                           spawnId=walkerActor.id,
                                                                                                           isCar=False)
                # Setup the controller
                walkerController = self.walkerIdToWalkerController[walkerActor.id]

                target_transform = entityReplayData_nextFrame.transform
                target_loc = target_transform.location
                target_rot = target_transform.rotation
                current_transform = walkerActor.get_transform()
                current_loc = current_transform.location
                current_rot = current_transform.rotation

                walkerMoveDir = [target_loc.x - current_loc.x, target_loc.y - current_loc.y]
                walkerMoveAngle = math.atan2(walkerMoveDir[1], walkerMoveDir[0]) * 180.0 / math.pi

                # Go to the desired position. TODO: should we move upfront instead ?
                walkerController.speed = np.linalg.norm(walkerMoveDir) * 0.95 # Percent of max speed as input
                walkerController.direction = carla.Vector3D(walkerMoveDir[0], walkerMoveDir[1], 0.0)
                walkerController.jump = False
                #walkerController.direction = carla.Vector3D(0.0, -1.0, 0.0)
                # walkerController.speed = 3.0
                batch_pedestrians.append(carla.command.ApplyWalkerControl(actor_id=walkerActor.id, control=walkerController))

        """
        # Apply the batch and check the results
        full_batch = batch_cars + batch_pedestrians
        results = self.carlaConnectionManager.client.apply_batch_sync(full_batch, True)
        for res_index, res_data in enumerate(results):
            if res_data.error:
                print(f"Can't solve request for actor index {res_index} while #vehicles={len(batch_cars)}, #walkers={len(batch_pedestrians)}")
        """

    # Simulate an environment frame and returns a tuple of A. actors frame data B. syncData dictionary containing registered sensors values
    def simulateFrame(self, simFrame, isInitializationFrame=False):
        # Output statistics to see where we are
        tenthNumFrames = 0
        if self.args.onlineEnvSettings.isDataGatherAndStoreEnabled:
            tenthNumFrames = (self.dataGatherParams.numFrames / 10) if self.dataGatherParams.numFrames > 0 else None
        else:
            tenthNumFrames = (self.args.seq_len_train / 10) if self.args.seq_len_train > 0 else None


        if tenthNumFrames and simFrame % tenthNumFrames == 0:
            print(f"Frame sim {(simFrame * 10.0) / tenthNumFrames}%...")

        # TODO Replay : manual set
        if self.simulationOptions.simulateReplay:
            self.updateEntitiesFromReplayData(simFrame)

        # Tick the  world
        #print(f"Ticking the clock for frame {simFrame}")
        worldFrame = self.world.tick()

        #print(f"Finished ticking {simFrame}")

        # Take this frame data when not simulating
        thisFrameData = None
        if not self.simulationOptions.simulateReplay:
            # Now take the actors and update the data and add the date for this frame
            self.addFrameData(simFrame, worldFrame, self.vehicles_data, self.pedestrians_data, isInitializationFrame)
            thisFrameData = self.getFrameData(simFrame)

        # Advance the simulation and wait for the data.
        # logging.log(logging.INFO, f"Getting data for frame {worldFrame}")
        #print(f"Ticking the data manager frame {worldFrame}")
        syncData = self.dataManager.tick(targetFrame=worldFrame, timeout=None)  # self.EnvSettings.TIMEOUT_VALUE * 100.0) # Because sometimes you forget to put the focus on server and BOOM
        # logging.log(logging.INFO, f"Data retrieved for frame {worldFrame}")

        #print(f"Finished a tick !")
        return thisFrameData, syncData

    def saveSimulatedData(self):
        self.dataGatherParams.saveSimulatedData(self.vehicles_data, self.pedestrians_data, self.world_2_worldRef_matrix)

    """
    def onConnectionSolved(self):
        self.world = self.carlaConnectionManager.world
    """
