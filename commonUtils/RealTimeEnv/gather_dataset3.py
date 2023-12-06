from __future__ import print_function

import random
import numpy as np

import sys


from CarlaRealTimeUtils import *
from CarlaWorldManagement import *
import json
from datetime import datetime
import traceback

dirname = os.path.dirname(__file__)

sys.path.append(os.path.join(dirname, '..'))
sys.path.append(os.path.join(dirname, '../../'))
sys.path.append(os.path.join(dirname, '../../RL'))

print(sys.path)
from settings import run_settings, RLCarlaOnlineEnv

class DataCollector(object):
    def __init__(self,):
        pass

    def collectData(self, host, port,
                    outEpisodesBasePath, scenesConfigFile,
                    simulationOptions :SimOptions =None,
                    dataGatheringArgs=None):

        scenesToCapture = parseScenesConfigFile(scenesConfigFile)

        # Filter to replay only scene if enabled
        if simulationOptions.simulateReplay == True:
            scenesToCapture = [scene for scene in scenesToCapture if simulationOptions.sceneNameToReplay == scene['sceneName']]


        # Set the setttings correctly
        #--------
        allSettings = run_settings(forcedPathToCurrentDataset=outEpisodesBasePath)
        # Marked as true since we are running this from the data collection script
        allSettings.onlineEnvSettings.isDataGatherAndStoreEnabled = True
        allSettings.onlineEnvSettings.width, allSettings.onlineEnvSettings.height = [int(x) for x in dataGatheringArgs.topviewRes.split('x')]
        allSettings.onlineEnvSettings.no_server_rendering = dataGatheringArgs.no_server_rendering
        allSettings.onlineEnvSettings.no_client_rendering = True if dataGatheringArgs.no_client_rendering == 1 else False
        allSettings.onlineEnvSettings.client_simplifiedRendering = True if dataGatheringArgs.client_simplifiedRendering == 1 else False
        allSettings.onlineEnvSettings.forceSceneReconstruction = int(dataGatheringArgs.forceSceneReconstruction)
        allSettings.onlineEnvSettings.forceExistingRaycastActor = int(dataGatheringArgs.forceExistingRaycastActor)
        allSettings.onlineEnvSettings.timeStampEpisodeIndex = None

        if (dataGatheringArgs.timeStampEpisodeIndex == None or dataGatheringArgs.timeStampEpisodeIndex == ""):
            allSettings.onlineEnvSettings.timeStampEpisodeIndex = [-1]
        else:
            if isinstance(dataGatheringArgs.timeStampEpisodeIndex, list):
                allSettings.onlineEnvSettings.timeStampEpisodeIndex = dataGatheringArgs.timeStampEpisodeIndex
            else:
                assert(isinstance(dataGatheringArgs.timeStampEpisodeIndex, str))
                allSettings.onlineEnvSettings.timeStampEpisodeIndex = [int(i) for i in dataGatheringArgs.timeStampEpisodeIndex.split(",")]

        # --------

        # Create the renderer
        renderType : RenderType = RenderType.RENDER_NONE
        if allSettings.onlineEnvSettings.no_client_rendering != True:
            renderType = RenderType.RENDER_SIMPLIFIED if allSettings.onlineEnvSettings.client_simplifiedRendering == True else RenderType.RENDER_COLORED
        renderParams = RenderOptions(renderType, topViewResX=allSettings.onlineEnvSettings.width, topViewResY=allSettings.onlineEnvSettings.height)
        renderContext = EnvironmentRendering(renderParams, allSettings)

        # Create the connection
        carlaConnection = CarlaServerConnection(self, allSettings, simulationOptions=simulationOptions)

        try:
            # For each scene position, do a data capture
            for sceneData in scenesToCapture:
                mapToTestName=sceneData['map']
                sceneName = sceneData['sceneName']
                numEpisodesPerMap = sceneData['numEpisodesPerScene'] if simulationOptions.simulateReplay == False else 1 # Single episode simulation if replay...
                framesPerEpisode = sceneData['framesPerEpisode'] # Same number of frames on both data gathering and reply

                # When we are training/replaying with an agent car/ped, grab another one
                needAdditionalCar = simulationOptions.useRecordedTrainedCar == True
                needAdditionalPed = simulationOptions.useRecordedTrainedPed == True
                numCarlaVehicles = sceneData['numCarlaVehicles'] + (1 if needAdditionalCar else 0)
                numCarlaPedestrians = sceneData['numCarlaPedestrians'] + (1 if needAdditionalPed else 0)

                randomObserverSpawnLocation = sceneData["randomObserverSpawnLocation"]
                isStaticCar = sceneData["isStaticCar"]

                DataGatherParams.STATIC_CAR = isStaticCar

                """ EXPERIMENTAL"""
                player_spawn_pointsAndIndices = None
                if randomObserverSpawnLocation:
                    envParams = EnvSetupParams(controlledCarsParams=[],
                                               controlledPedestriansParams=[],
                                               NumberOfCarlaVehicles=0, #numCarlaVehicles,
                                               NumberOfCarlaPedestrians=0, #numCarlaPedestrians,
                                               observerSpawnTransform=None, #sceneData['observerSpawnTransform'],
                                               observerVoxelSize=None, #sceneData['voxelRes'],
                                               observerNumVoxelsX=None, #ceneData['voxelsX'],
                                               observerNumVoxelsY=None, #sceneData['voxelsY'],
                                               observerNumVoxelsZ=None, #sceneData['voxelsZ'],
                                               forceExistingRaycastActor=None, #args.forceExistingRaycastActor,
                                               mapToUse=mapToTestName,
                                               args=allSettings)

                    carlaConnection.connectToServer(host, port)
                    carlaConnection.reloadWorld(envParams)
                    vehicles_spawn_points = carlaConnection.map.get_spawn_points()
                    carlaConnection.releaseServerConnection()

                    player_spawn_pointsAndIndices = [(i, transform) for i, transform in
                                                     enumerate(vehicles_spawn_points)]

                    random.shuffle(player_spawn_pointsAndIndices)

                """ END EXPERIMENTAL """

                # Do the episodes. We cycle through the spawn points if not enough
                for episodeIndex in range(numEpisodesPerMap):
                    # If random observer spawn, chose a new observer point at each data gathering and fix that point in
                    forcedObserverSpawnPointIndex = None
                    forcedObserverSpawnPointTransform = None
                    if randomObserverSpawnLocation:
                        # HARDCODED DEBUG
                        if False:
                            selected_index = None
                            for debugIter in range(numEpisodesPerMap):
                                if player_spawn_pointsAndIndices[debugIter][0] == 55:
                                    selected_index = debugIter
                                    break

                            if selected_index is None:
                                selected_index = 0

                            forcedObserverSpawnPointIndex = player_spawn_pointsAndIndices[selected_index][0]
                            forcedObserverSpawnPointTransform = player_spawn_pointsAndIndices[selected_index][1]
                        else:
                            forcedObserverSpawnPointIndex = player_spawn_pointsAndIndices[episodeIndex][0]
                            forcedObserverSpawnPointTransform = player_spawn_pointsAndIndices[episodeIndex][1]

                        sceneData['observerSpawnTransform'] = forcedObserverSpawnPointTransform

                    fixedSeed = sceneData['simFixedSeed']
                    if fixedSeed is None:
                        set_fixed_seed(int(time.time()))
                    else:
                        set_fixed_seed(fixedSeed)

                    if simulationOptions.simulateReplay:
                        print("@@@@@@@ =========== Replaying Scene: ", sceneData, " from folder data ", simulationOptions.pathToReplayData, "\n\n")
                    else:
                        print(f"@@@@@@@ =========== Capturing Scene: {sceneData} forcedIndex {forcedObserverSpawnPointIndex} "
                              f"x:{forcedObserverSpawnPointTransform.location.x},"
                              f"y:{forcedObserverSpawnPointTransform.location.y},"
                              f"z:{forcedObserverSpawnPointTransform.location.z},"
                              f"yaw:{forcedObserverSpawnPointTransform.rotation.yaw}", "\n\n")

                    envParams = EnvSetupParams(controlledCarsParams=[],
                                               controlledPedestriansParams=[],
                                               NumberOfCarlaVehicles=numCarlaVehicles,
                                               NumberOfCarlaPedestrians=numCarlaPedestrians,
                                               observerSpawnTransform=sceneData['observerSpawnTransform'],
                                                observerVoxelSize=sceneData['voxelRes'],
                                               observerNumVoxelsX=sceneData['voxelsX'],
                                               observerNumVoxelsY=sceneData['voxelsY'],
                                               observerNumVoxelsZ=sceneData['voxelsZ'],
                                               forceExistingRaycastActor=allSettings.onlineEnvSettings.forceExistingRaycastActor,
                                               mapToUse=[mapToTestName],
                                               sensorsDisplacementDist=sceneData['sensorsDisplacementDist'],
                                               args=allSettings)

                    # Set the forced indices and transform for observer spawn point if any
                    envParams.forcedObserverSpawnPointIndex = forcedObserverSpawnPointIndex
                    envParams.forcedObserverSpawnPointTransform = forcedObserverSpawnPointTransform

                    dataGatheringParams = DataGatherParams(outputEpisodeDataPath=outEpisodesBasePath,
                                                           sceneName=sceneName,
                                                           episodeIndex=-1,
                                                           numFrames=framesPerEpisode,
                                                           maxNumberOfEpisodes=numEpisodesPerMap,
                                                           mapsToTest=[mapToTestName],
                                                           lidarData=sceneData['lidarData'],
                                                           copyScenePaths=True, # DO NOT USE TRUE for real time env !!
                                                           simulationOptions=simulationOptions,
                                                           host=host,
                                                           port=port,
                                                           sensorsDisplacementDist=sceneData['sensorsDisplacementDist'],
                                                           args=allSettings)

                    envManagement = EnvironmentManagement(carlaConnection=carlaConnection, renderContext=renderContext,
                                                          simulationOptions=simulationOptions)

                    envParams.mapToUse = mapToTestName

                    # Setup the hero car view parameters inside the capturing objects
                    use_hero_actor = sceneData['heroView']
                    allSettings.onlineEnvSettings.use_hero_actor_forDataGathering = renderContext.use_hero_actor = use_hero_actor

                    # TODO Replay: currently when simulating we not using hero actor at all. We must do some work around this to get from camera images too....
                    dataGatheringParams.use_hero_actor_forDataGathering = use_hero_actor #if simulationOptions.simulateReplay is False else False

                    dataGatheringParams.destination : carla.Location = sceneData['destination'] if use_hero_actor else None
                    dataGatheringParams.rerouteAllowed = True if ('rerouteAllowed' in sceneData and sceneData['rerouteAllowed'] == 1) else False

                    if ('frame_step' in sceneData) or ('frame_step_replay' in sceneData):
                        dataGatheringParams.frame_step = int(sceneData['frame_step']) if simulationOptions.simulateReplay is False else int(sceneData['frame_step_replay'])


                    envParams.episodeIndex = episodeIndex
                    logging.log(logging.INFO, f"Preparing episode {episodeIndex} on map {mapToTestName}\n=================")
                    logging.log(logging.INFO, "Spawning world")

                    # Collect scene raycast/pointcloud data only on the first episode
                    dataGatheringParams.collectSceneData = 1 if (episodeIndex == 0 and simulationOptions.simulateReplay is False) else 0

                    envManagement.SpawnWorld(dataGatherSetup=dataGatheringParams, envSetup=envParams, args=allSettings)
                    envManagement.renderContext.setupConnection(alreadySpawnedWorld=carlaConnection.world,
                                                       alreadySpawnedMapName=carlaConnection.map,
                                                       use_hero_actor=use_hero_actor)
                    try:
                        # Simulate frame by frame
                        for frameId in range(framesPerEpisode):
                            envManagement.SimulateFrame(isInitializationFrame=frameId == 0)

                        envManagement.OnSimulationDone()

                        print("Saved data, despawning the world and prepare for next episode !")
                        envManagement.DespawnWorld()
                        print("World was despawned..continue to next episode")
                        #time.sleep(10)
                        #print("I'm back")
                    except:
                        print("!!!!!!! DESPAWNING WORLD WITH EXCEPTION !!!!!!!")
                        traceback.print_exc()
                        envManagement.Destroy()

                carlaConnection.disconnect(withError=False)
                renderContext.quit()
        except:
            carlaConnection.disconnect(withError=True)
            renderContext.quit()
            os._exit(0)

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')

    argparser.add_argument(
        '--topviewRes',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        type=str,
        help='window resolution (default: 1280x720)')

    argparser.add_argument(
        '-outputDataBasePath', '--outputDataBasePath',
        metavar='P',
        type=str,
        required=True,
        help='Number of frames per episode')

    """
    argparser.add_argument(
        '-listOfMapsToTest', '--listOfMapsToTest',
        metavar='P',
        type=str,
        required=True,
        help='Number of frames per episode')
    """

    argparser.add_argument(
        '--heroCarFixedTargetPos',
        metavar='WIDTHxHEIGHT',
        default=None,
        type=str,
        help='If specified, the hero agent car will move there from starting position at stay there. If not specified, it will just do roaming in the city')

    """
    argparser.add_argument(
        '-capturePointCloud', '--capturePointCloud',
        metavar='P',
        type=int,
        help='Should we capture the point cloud ?'
    )
    """

    argparser.add_argument(
        '-scenesConfigFile', '--scenesConfigFile',
        metavar='spawnObserverTransformsFile',
        type=str,
        default=str(None),
        help='Json config file for scenes'
    )

    argparser.add_argument(
        '-no_server_rendering', '--no_server_rendering',
        metavar='no_server_rendering',
        type=int,
        default=0,
        help="1 if you don't want rendering on the server side "
    )

    argparser.add_argument(
        '-no_client_rendering', '--no_client_rendering',
        metavar='no_client_rendering',
        type=int,
        default=0,
        help="1 if you don't want rendering on the server side "
    )

    argparser.add_argument(
        '-client_simplifiedRendering', '--client_simplifiedRendering',
        metavar='client_simplifiedRendering',
        type=int,
        default=1,
        help="if using rendering on client side, simplified means a top down view high-level description, otherwise it will be a full RGB/sem img"
    )

    argparser.add_argument(
        '-forceSceneReconstruction', '--forceSceneReconstruction',
        metavar='forceSceneReconstruction',
        type=int,
        default=0,
        help="1 if you don't want rendering on the server side "
    )

    argparser.add_argument(
        '-forceExistingRaycastActor', '--forceExistingRaycastActor',
        metavar='forceExistingRaycastActor',
        type=int,
        default=0,
        help="1 if you don't want rendering on the server side "
    )

    argparser.add_argument(
        '-simulationReplayMode', '--simulationReplayMode',
        metavar='simulationReplayMode',
        type=int,
        default=0,
        help="1 if you want to simulate replay from recorded data "
    )

    argparser.add_argument(
        '-simulationReplayUseTrainedStats_car', '--simulationReplayUseTrainedStats_car',
        metavar='simulationReplayUseTrainedStats_car',
        type=int,
        default=0,
        help="1 if you want to simulate replay using statistics files from car agents"
    )

    argparser.add_argument(
        '-simulationReplayUseTrainedStats_ped', '--simulationReplayUseTrainedStats_ped',
        metavar='simulationReplayUseTrainedStats_ped',
        type=int,
        default=0,
        help="1 if you want to simulate replay using statistics files from pedestrian agents"
    )

    argparser.add_argument(
        '-simulationReplayPath', '--simulationReplayPath',
        metavar='simulationReplayPath',
        type=str,
        default=None,
        required=False,
        help="path to what you what to replay"
    )

    argparser.add_argument(
        '-sceneNameToReplay', '--sceneNameToReplay',
        metavar='sceneNameToReplay',
        type=str,
        default=None,
        required=False,
        help="path to what you what to replay"
    )

    argparser.add_argument(
        '-simulationReplayTimestamp', '--simulationReplayTimestamp',
        metavar='simulationReplayTimestamp',
        type=str,
        default=None,
        required=False,
        help="timestamp for data to read stats from"
    )

    argparser.add_argument(
        '-timeStampEpisodeIndex', '--timeStampEpisodeIndex',
        metavar='timeStampEpisodeIndex',
        type=str,
        default="",
        help="the episode indices you want to run on from stats and timestamp, as a string list"
    )

    dataGatheringArgs = argparser.parse_args()
    dataGatheringArgs.width, dataGatheringArgs.height = [int(x) for x in dataGatheringArgs.topviewRes.split('x')]
    dataGatheringArgs.no_server_rendering = 1 if dataGatheringArgs.no_server_rendering == 1 else False
    dataGatheringArgs.no_client_rendering = True if dataGatheringArgs.no_client_rendering == 1 else False
    dataGatheringArgs.client_simplifiedRendering = True if dataGatheringArgs.client_simplifiedRendering == 1 else False
    dataGatheringArgs.forceSceneReconstruction = int(dataGatheringArgs.forceSceneReconstruction)
    dataGatheringArgs.timeStampEpisodeIndex = [-1] if (dataGatheringArgs.timeStampEpisodeIndex == None or dataGatheringArgs.timeStampEpisodeIndex == "") else [int(i) for i in dataGatheringArgs.timeStampEpisodeIndex.split(",")]

    log_level = logging.DEBUG if dataGatheringArgs.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', dataGatheringArgs.host, dataGatheringArgs.port)

    simulateReplay = True if (dataGatheringArgs.simulationReplayMode == 1 and len(dataGatheringArgs.simulationReplayPath) > 0) else False

    # When only collecting data, or replaying a thing without statistics of trained agents, a dummy single invalid replay option is used
    if not simulateReplay:
        dataGatheringArgs.timeStampEpisodeIndex = [-1]

    for simulateReplay_epIndex in dataGatheringArgs.timeStampEpisodeIndex:
        print(f"Sim replay id {simulateReplay_epIndex}")

        simOptions = SimOptions(simulateReplay=simulateReplay,
                                pathToReplayData=dataGatheringArgs.simulationReplayPath,
                                simulationReplayUseTrainedStats_car=dataGatheringArgs.simulationReplayUseTrainedStats_car,
                                simulationReplayUseTrainedStats_ped=dataGatheringArgs.simulationReplayUseTrainedStats_ped,
                                simulationReplayTimestamp=dataGatheringArgs.simulationReplayTimestamp,
                                simEpisodeIndex=simulateReplay_epIndex)

        simOptions.sceneNameToReplay = dataGatheringArgs.sceneNameToReplay if simOptions.simulateReplay else None

        dc = DataCollector()
        dataGatheringArgs.useFixedWorldOffsets = False
        dc.collectData(host=dataGatheringArgs.host, port=dataGatheringArgs.port,
                       outEpisodesBasePath=dataGatheringArgs.outputDataBasePath,
                       scenesConfigFile=dataGatheringArgs.scenesConfigFile,
                       simulationOptions=simOptions,
                       dataGatheringArgs=dataGatheringArgs)

        print('\nDone!')


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nClient stoped by user.')
