# Utils for spawning a world sequence with different parameters

from commonUtils.RealTimeEnv.CarlaRealTimeUtils import *
from commonUtils.RealTimeEnv.CarlaServerConnection import *
from commonUtils.RealTimeEnv.CarlaEnvironmentActors import  *
from commonUtils.RealTimeEnv.CarlaEnvironmentRendering import *
import carla
import json
import shutil
from enum import Enum


# Different classes of algorithms for the 2D debugger window
class AlgorithmObserverWindow(Enum):
    FIRST_CAR = 0 # First care ever registed
    ALL_TRAINABLE_ENTITIES = 1 # Catch all trainable entities in the window
    FIRST_TRAINABLE_CAR_TRAJECTORY = 2 # Focused on the first trainable car trajectory

ALGORITHM_FOR_OBSERVERWINDOW = AlgorithmObserverWindow.FIRST_CAR

def parseScenesConfigFile(scenesConfigFile):
    scenesToCapture = []
    assert os.path.exists(scenesConfigFile), "The file with observer transform positions doesn't exists !"
    with open(scenesConfigFile, 'r') as scenesConfigStream:
        data = json.load(scenesConfigStream)
        for sceneName, sceneData in data['scenes'].items():
            mapName = sceneData['map']
            location = carla.Location(sceneData['X'], sceneData['Y'], sceneData['Z'])
            rotation = carla.Rotation(sceneData['pitch'], sceneData['yaw'], sceneData['roll'])
            observerSpawnTransform = carla.Transform(location, rotation)

            # Note: the target position could be missing and this is a valid case, it means that we want our agent to roam around the city
            hasDestinationKey = True if ('TX' in sceneData) and ('TY' in sceneData) and ('TZ' in sceneData) else False
            destination =  carla.Location(sceneData['TX'], sceneData['TY'], sceneData['TZ']) if hasDestinationKey else None

            voxelRes = sceneData['voxelRes']
            voxelsX = sceneData['voxelsX']
            voxelsY = sceneData['voxelsY']
            voxelsZ = sceneData['voxelsZ']
            heroView = sceneData['heroView']
            numEpisodesPerMap = int(sceneData['numEpisodesPerScene'])
            framesPerEpisode = int(sceneData['framesPerEpisodes'])
            numCarlaVehicles = int(sceneData['numCarlaVehicles'])
            numCarlaPedestrians = int(sceneData['numCarlaPedestrians'])
            sensorsDisplacementDist = int(sceneData['sensorsDisplacementDist'])

            randomObserverSpawnLocation = int(sceneData["randomObserverSpawnLocation"])
            isStaticCar = int(sceneData["staticCar"])

            numTrainableVehicles = 0 if "numTrainableVehicles" not in sceneData else int(sceneData["numTrainableVehicles"])
            numTrainablePedestrians = 0 if "numTrainablePedestrians" not in sceneData else int(sceneData["numTrainablePedestrians"])

            frame_step = int(sceneData['frame_step'])
            frame_step_replay = int(sceneData['frame_step_replay'])

            fixedSeed = None
            if 'simFixedSeed' in sceneData:
                fixedSeed = int(sceneData['simFixedSeed'])

            parsedSceneData = {'sceneName' : sceneName,
                               'map' : mapName, 'observerSpawnTransform' : observerSpawnTransform,
                               'numEpisodesPerScene' : numEpisodesPerMap,
                               'framesPerEpisode' : framesPerEpisode,
                               'destination' : destination,
                               'heroView' : heroView,
                               'voxelRes' : voxelRes, 'voxelsX' : voxelsX, 'voxelsY' : voxelsY, 'voxelsZ' : voxelsZ,
                               'lidarData' : sceneData['lidarData'] if 'lidarData' in sceneData else None,
                               'numCarlaVehicles' : numCarlaVehicles,
                               'numCarlaPedestrians' : numCarlaPedestrians,
                               "numTrainableVehicles": numTrainableVehicles,
                               "numTrainablePedestrians": numTrainablePedestrians,
                               'simFixedSeed' : fixedSeed,
                               'frame_step' : frame_step,
                               'frame_step_replay' : frame_step_replay,
                               'randomObserverSpawnLocation' : randomObserverSpawnLocation,
                               'isStaticCar' : isStaticCar,
                               'sensorsDisplacementDist' : sensorsDisplacementDist
                               }

            scenesToCapture.append(parsedSceneData)
    return scenesToCapture

class EnvironmentManagement:
    # Spawns a new world with settings, either for the purpose of online/offline data gathering or real time simulation
    def __init__(self, carlaConnection, renderContext, simulationOptions):
        self.dataGatherParams = None
        self.renderOptions = None
        self.envOptions = None

        # Context members for actors spawned
        self.actorsContext = None

        # Carla server connection context
        self.carlaConnection = carlaConnection

        # Render context
        self.renderContext = renderContext

        self.simFrame = None

        self.simulationOptions : SimOptions = simulationOptions

    # Basically this function will be parametrized and the entry point for real time simulation and data gathering
    def SpawnWorld(self, dataGatherSetup : DataGatherParams, envSetup : EnvSetupParams, args):
        # Step 0: set options and sanity checks
        self.dataGatherParams = dataGatherSetup
        self.envSetup = envSetup
        self.args = args

        # Step 1: create contexts, solve scene data gathering
        # Context members for actors spawned
        try:
            self.actorsContext = EnvironmentActors(self, self.carlaConnection, self.args)
            self.renderContext.actorsContext = self.actorsContext
            self.carlaConnection.actorsManager = self.actorsContext

            # Step 2: create connection to server
            self.carlaConnection.connectToServer(self.dataGatherParams.host, self.dataGatherParams.port)

            # Step 3: create the actors
            self.spawnActors()

            # Step 4: fix the 2D rendering if used
            loc = self.envSetup.observerSpawnTransform.location
            defaultWindowBbox = BBox2D.getBboxWithCenterAndExtent(loc.x, loc.y,
                                                                  width=DEFAULT_OBSERVER_WINDOW_WIDTH,
                                                                  depth=DEFAULT_OBSERVER_WINDOW_DEPTH)
            self.renderContext.setObserverWindow(defaultWindowBbox)

        except:
            self.Destroy(withError=True)
            raise RuntimeError("Destroying env")

        self.simFrame = None # Will be set on onEpisodeStartup

    def spawnActors(self):
        self.actorsContext.spawnActors(self.envSetup, self.dataGatherParams, simulationOptions=self.simulationOptions)
        # TODO: maybe keep here the initial transforms of the actors ?

    # Reset the actors to a know position - probably the one at the episode's starts
    def resetActors(self):
        # First, Reset some local data aggregation
        self.actorsContext.resetActors()


    def SimulateFrame(self, isInitializationFrame=False):
        lastFrameVehiclesData = None
        lastFramePedestriansData = None

        try:
            if not isInitializationFrame:
                self.simFrame += 1
            else:
                self.simFrame = 0

            #logging.log(logging.INFO, f"Simulating environment frame {self.simFrame}")
            frameActorsData, syncData = self.actorsContext.simulateFrame(self.simFrame, isInitializationFrame)
            lastFrameVehiclesData = None
            lastFramePedestriansData = None

            if frameActorsData != None:
                lastFrameVehiclesData = frameActorsData[0]
                lastFramePedestriansData = frameActorsData[1]

            # Initialize some of the systems if needed
            if isInitializationFrame:
                # Set the observer position for rendering context depending if we are training a real time environment or not
                bboxWindow = self.getTargetObserverWindow(lastFrameVehiclesData, lastFramePedestriansData,
                                                                    self.actorsContext.trainableWalkerIds,
                                                                    self.actorsContext.trainableVehiclesIds)
                self.renderContext.setObserverWindow(bboxWindow)

            # Take the date from world and send them to render side
            #print("Ticking render")
            self.renderContext.tick(syncData, isInitializationFrame=isInitializationFrame)

            # Save the needed stuff from sensors
            if True or not self.simulationOptions.simulateReplay:
                self.dataGatherParams.saveHeroCarPerspectiveData(syncData, self.simFrame,
                                                heroVehicleSensors=self.actorsContext.s_heroCaptureCarPerspective_sensors,
                                                heroVehicleIntrisics=self.actorsContext.s_heroCaptureCarPerspective_intrisics,
                                                world_2_worldRef=self.actorsContext.world_2_worldRef_matrix)


            #print("Post update")
            self.actorsContext.doPostUpdate()
        except:
            self.Destroy(withError=True)

        return (self.simFrame, lastFrameVehiclesData, lastFramePedestriansData)

    def getTargetObserverWindow(self, lastFrameVehiclesData, lastFramePedestriansData,
                                  trainableWalkerIds, trainableVehiclesIds):

        if lastFrameVehiclesData is None or lastFramePedestriansData is None:
            return BBox2D.getBboxWithCenterAndExtent(0, 0, DEFAULT_OBSERVER_WINDOW_WIDTH, DEFAULT_OBSERVER_WINDOW_DEPTH)

        observerWindow_width = DEFAULT_OBSERVER_WINDOW_WIDTH
        observerWindow_depth = DEFAULT_OBSERVER_WINDOW_DEPTH
        observerPos = [0,0]

        # Are we in a real time environment ?
        if self.simulationOptions.simOnlineEnv is not None:
            if ALGORITHM_FOR_OBSERVERWINDOW == AlgorithmObserverWindow.FIRST_CAR:
                # Put all trainable cars in a list or if not
                firstVehicleKey = list(lastFrameVehiclesData.keys())[0]
                firstVehicleData = lastFrameVehiclesData[firstVehicleKey]
                firstVehicleBBox = firstVehicleData['WorldRefLocation']
                #observerPos = getCenterOfBBox(firstVehicleBBox)
                entity_loc = np.dot(self.actorsContext.worldRef_2_world_matrix, np.append(firstVehicleBBox, 1.0).T)
                observerPos = list(entity_loc)

            elif ALGORITHM_FOR_OBSERVERWINDOW == AlgorithmObserverWindow.ALL_TRAINABLE_ENTITIES:
                resWindow = BBox2D()
                for trainableVehId in self.actorsContext.trainableVehiclesIds:
                    if trainableVehId in lastFrameVehiclesData:
                        entity_bbox_to_ref = lastFrameVehiclesData[trainableVehId]['WorldRefLocation']
                        entity_loc_inworld = np.dot(self.actorsContext.worldRef_2_world_matrix,
                                            np.append(entity_bbox_to_ref, 1.0).T)

                        """
                        entity_bbox_world = BBox2D.getBboxWithCenterAndExtent(xmid=entity_bbox_to_ref[0],
                                                                              ymid=entity_bbox_to_ref[1],
                                                                              width=2, depth=2)
                                                                              """
                        resWindow.extend(entity_loc_inworld)

                for trainablePedId in self.actorsContext.trainableWalkerIds:
                    if trainablePedId in lastFramePedestriansData:
                        entity_bbox_to_ref = lastFramePedestriansData[trainablePedId]['WorldRefLocation']
                        entity_loc_inworld = np.dot(self.actorsContext.worldRef_2_world_matrix,
                                                    np.append(entity_bbox_to_ref, 1.0).T)

                        """
                        entity_bbox_world = BBox2D.getBboxWithCenterAndExtent(xmid=entity_bbox_to_ref[0],
                                                                              ymid=entity_bbox_to_ref[1],
                                                                              width=2, depth=2)
                                                                              """
                        resWindow.extend(entity_loc_inworld)

                SAFETY_MARGIN = 5.0 # 5 more meters than the existing to be sure we include everything correctly
                resWindow.extend_from_margins(on_width=SAFETY_MARGIN, on_height=SAFETY_MARGIN)
                observerPos = list(resWindow.getCenter())

                resWindow_width, resWindow_depth = resWindow.getDims()
                observerWindow_width = max(resWindow_width, MIN_OBSERVER_WINDOW_WIDTH)
                observerWindow_depth = max(resWindow_depth, MIN_OBSERVER_WINDOW_DEPTH)

                observerWindow_width = observerWindow_depth = max(observerWindow_depth, observerWindow_width)

            elif ALGORITHM_FOR_OBSERVERWINDOW == AlgorithmObserverWindow.FIRST_TRAINABLE_CAR_TRAJECTORY:
                observerWindow_width = DEFAULT_OBSERVER_WINDOW_WIDTH
                observerWindow_depth = DEFAULT_OBSERVER_WINDOW_DEPTH
                observerWindow_width = observerWindow_depth = max(observerWindow_depth, observerWindow_width)

                resWindow = BBox2D()
                for trainableVehId in self.actorsContext.trainableVehiclesIds:
                    if trainableVehId in lastFrameVehiclesData:
                        entity_bbox_to_ref = lastFrameVehiclesData[trainableVehId]['WorldRefLocation']
                        entity_loc_inworld = np.dot(self.actorsContext.worldRef_2_world_matrix,
                                                    np.append(entity_bbox_to_ref, 1.0).T)

                        resWindow.extend(entity_loc_inworld)
                        break
                observerPos = list(resWindow.getCenter())

        else:
            loc = self.envSetup.observerSpawnTransform.location
            observerPos = [loc.x, loc.y, loc.z]


        print(f"Debug OBSERVER WINDOW: pos {observerPos}, width {observerWindow_width}, depth {observerWindow_depth}")
        bboxRes = BBox2D.getBboxWithCenterAndExtent(observerPos[0], observerPos[1],
                                            width=observerWindow_width,
                                            depth=observerWindow_depth)
        return bboxRes

    # Save the actors positions etc during simulation of the environment within the episode output folder
    def OnSimulationDone(self):
        if not self.simulationOptions.simulateReplay:
            self.vehicles_data = {}
            self.pedestrians_data = {}
            self.actorsContext.saveSimulatedData()
            self.dataGatherParams.prepareSceneOutput()

    def onEpisodeStartup(self, isFirstEpisode):
        self.simFrame = 0

        if True or not isFirstEpisode:
            self.resetActors() # Put actors back in place


    def onEpisodeEnd(self):
        self.simFrame = None

        # Despawn the actors
        #self.actorsContext.despawnActors(client=self.carlaConnection.client)


    def DespawnWorld(self):
        self.carlaConnection.destroyEnvironment()
        
    def Destroy(self, withError=False):
        if withError:
            print("!!!!!!!!!!!  CLOSING CONNECTION WITH ERROR !!!!!!!!!!!")
            self.carlaConnection.disconnect(withError=withError)
        self.renderContext.quit()

        if withError:
            os._exit(0)
