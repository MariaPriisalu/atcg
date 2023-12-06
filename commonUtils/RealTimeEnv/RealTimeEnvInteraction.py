# This is for real time interaction with Carla/Unreal side.
# Work in progress but contains all hooks for interacting with the engine in a OpenAI gym compatible interface

# TODO : We need to include in this script the functionality from gather_dataset3.py
# Basically the former had the functionality for createing  an interesting scenario / world around a spawn point with pedestrians and cars and capture datasets
# The latter is used to spawn an environment for a controllable car either by Carla or Learning by cheating
# Both can be combined here in this file

"""
TODOs
-------------

4. Car collisioon
A.
- better to disable on Carla server side and let the client manage them.
																   - force the agent pos in the Carla side:
e.g. call from         self.update_agent_pos_in_episode(episode)
if realtime => force update

....
def is_next_pos_valid(self, episode, next_pos):

	valid = self.measures[self.frame, CAR_MEASURES_INDX.hit_obstacles]==0
	if self.settings.allow_car_to_live_through_collisions:


	# Check if agent is alive
def is_agent_alive(self, episode):

	-- WHy is     def is_next_pos_valid(self, episode, next_pos): in agent_car correct ??


-- Same idea probably here:
def init_car(self):
	# Find positions where cars could be initialized. This should be prefferable done
	in CARLA. \
 \
B.
Give velocity / action to Carla let it play then set back the position with the
Carla's value.
Probably in     def perform_action(self, vel, episode, prob=0): ?

But collision still simple to manage in RLAgent... check perform_action

"""



from dotmap import DotMap
from . RealTimeNullEnvInteraction import NullRealTimeEnv
from . RealTimeCarEnv import *
from . RealTimePedEnv import *
from . CarlaWorldManagement import *
from typing import Dict, Union, Set, List
from settings import run_settings, RANDOM_SEED_NP
import carla

import sys, traceback

# class NullRealTimeEnv:
# 	def __init__(self):
# 		pass
#
# 	def reset(self, initDict):
# 		obs = None
# 		return obs
#
# 	def action(self, actionDict):
# 		obs = None
# 		reward = None
# 		done = None
# 		info = None
# 		return obs, reward, done, info

class CarlaRealTimeEnv(NullRealTimeEnv):
	def __init__(self, args):
		super(NullRealTimeEnv, self).__init__()

		# TODO: Ciprian - take data from here to solve the other todos in this file
		self.pedestrianAgents = [RealTimePedEnv()]
		self.carsAgents = [RealTimeCarEnv()]

		self.envManagement: Union[None, EnvironmentManagement] = None
		self.carlaConnection : Union[None, CarlaServerConnection] = None
		self.renderContext : Union[None, EnvironmentRendering] = None
		self.args = args

	# TODO Ciprian: fix this by integrating the code from the other side
	def getObservation(self):
		ObsDict = DotMap()
		ObsDict.pedestrians = {}
		ObsDict.cars = {}
		ObsDict.engineFrame = -1

	# TODO Ciprian/Maria: Should we get a feedback from real environment ? E.g. collisions etc, this could be better than the one we compute locally !
	def getReward(self):
		return 0

	# TODO Ciprian
	def isEnvFinished(self):
		return False

	# TODO Ciprian
	def getDetails(self):
		return None

	def reset(self, initDict):
		obs = self.getObservation()
		return obs

	def action(self, inputActionDict):
		# TODO Ciprian: parse the input action requests and synchornize with Carla env to do the requested actions

		# Return output
		obs = self.getObservation()
		info = self.getDetails()
		done = self.isEnvFinished()
		reward = self.getReward()

		return obs, reward, done, info
		pass

	# Spawns a new environment given config parameters and store it persistently.
	# Don't forget to call spawnCurrentEnvironment !
	def spawnNewEnvironment(self, args, dataPath):
		scenesConfigFileName = args.onlineEnvSettings.scenesConfigFile
		#simulationSceneName = args.simulationSceneName

		try:
			# Step 1: read and fix parameters
			simOptions = SimOptions(simulateReplay=False,
									pathToReplayData=dataPath,
									simulationReplayUseTrainedStats_car=False,
									simulationReplayUseTrainedStats_ped=False,
									simulationReplayTimestamp=None,
									simEpisodeIndex=None,
									simOnlineEnv=True)

			simulationSceneName = None
			observerSpawnPointUsedForTheScene_Index = None
			observerSpawnPointUsedForTheScene_Transform = None
			with open(os.path.join(dataPath, "metadata.json"), 'r') as metadataStream:
				sceneMetaData = json.load(metadataStream)
				simulationSceneName = sceneMetaData["sceneName"]
				observerSpawnPointUsedForTheScene_Index = sceneMetaData["observerSpawnIndex"]
				observerSpawnPointUsedForTheScene_Pos = carla.Location(x=sceneMetaData["observerSpawn_X"],
																	   y=sceneMetaData["observerSpawn_Y"],
																	   z=sceneMetaData["observerSpawn_Z"])

				observerSpawnPointUsedForTheScene_Rotation = carla.Rotation(pitch=sceneMetaData["observerSpawn_Pitch"],
																			yaw=sceneMetaData["observerSpawn_Yaw"],
																			roll=sceneMetaData["observerSpawn_Roll"])
				observerSpawnPointUsedForTheScene_Transform = carla.Transform(location=observerSpawnPointUsedForTheScene_Pos,
																			  rotation=observerSpawnPointUsedForTheScene_Rotation)


			assert(simulationSceneName)
			simOptions.sceneNameToReplay = simulationSceneName


			# scenesConfigFilePath = os.path.join(datasetMainPath, scenesConfigFileName)
			# print("Scenes config file "+str(scenesConfigFilePath))

			scenesConfigFilePath = os.path.join(dataPath, scenesConfigFileName)


			scenesAvailable = parseScenesConfigFile(scenesConfigFilePath)
			sceneDataToUse = [scene for scene in scenesAvailable if simOptions.sceneNameToReplay == scene['sceneName']]
			if len(sceneDataToUse) != 1:
				assert False, "Empty scene lists !!!"

			sceneData = sceneDataToUse[0]
			framesPerEpisode = sceneData['framesPerEpisode']
			numCarlaVehicles = sceneData['numCarlaVehicles']
			numCarlaPedestrians = sceneData['numCarlaPedestrians']

			sceneData['observerSpawnTransform'] = observerSpawnPointUsedForTheScene_Transform

			"""
			fixedSeed = sceneData['simFixedSeed']
			if fixedSeed is None:
				set_fixed_seed(int(time.time()))
			else:
				set_fixed_seed(fixedSeed)
			"""
			if self.args.deterministic: # What does this do?
				set_fixed_seed(RANDOM_SEED_NP)

			mapToTestName = "Town03" # TODO Ciprian : fix this

			# Create the connection
			self.carlaConnection = CarlaServerConnection(self, args, simulationOptions=simOptions)

			# Create the renderer
			renderType: RenderType = RenderType.RENDER_NONE
			if args.onlineEnvSettings.no_client_rendering != True:
				renderType = RenderType.RENDER_SIMPLIFIED if args.onlineEnvSettings.client_simplifiedRendering == True else RenderType.RENDER_COLORED
			renderParams = RenderOptions(renderType, topViewResX=1280, topViewResY=720)
			self.renderContext = EnvironmentRendering(renderParams, args)

			# Setup the parameters/objects needed
			envParams = EnvSetupParams(controlledCarsParams=[],
									   controlledPedestriansParams=[],
									   NumberOfCarlaVehicles=numCarlaVehicles,
									   NumberOfCarlaPedestrians=numCarlaPedestrians,
									   observerSpawnTransform=sceneData['observerSpawnTransform'],
									   observerVoxelSize=sceneData['voxelRes'],
									   observerNumVoxelsX=sceneData['voxelsX'],
									   observerNumVoxelsY=sceneData['voxelsY'],
									   observerNumVoxelsZ=sceneData['voxelsZ'],
									   forceExistingRaycastActor=False,
									   mapToUse=["Town03"],
									   numberOfTrainableVehicles=sceneData['numTrainableVehicles'],
									   numberOfTrainablePedestrians=sceneData['numTrainablePedestrians'],
									   sensorsDisplacementDist=sceneData["sensorsDisplacementDist"],
									   args=self.args)

			envParams.forcedObserverSpawnPointIndex = observerSpawnPointUsedForTheScene_Index
			envParams.forcedObserverSpawnPointTransform = observerSpawnPointUsedForTheScene_Transform

			dataGatheringParams = DataGatherParams(outputEpisodeDataPath=dataPath,
												   sceneName=None, #simulationSceneName,
												   episodeIndex=-1,
												   numFrames=framesPerEpisode,
												   maxNumberOfEpisodes=1, #Doesn't matter
												   mapsToTest=[mapToTestName],
												   lidarData=sceneData['lidarData'],
												   copyScenePaths=True,  # DO NOT USE TRUE for real time env !!
												   simulationOptions=simOptions,
												   host=args.onlineEnvSettings.host,
												   port=args.onlineEnvSettings.port,
												   sensorsDisplacementDist=sceneData['sensorsDisplacementDist'],
												   args=args)

			self.envManagement = EnvironmentManagement(carlaConnection=self.carlaConnection,
													   	renderContext=self.renderContext,
												  		simulationOptions=simOptions)

			envParams.mapToUse = mapToTestName
			# Setup the hero car view parameters inside the capturing objects
			use_hero_actor_forDataGathering = sceneData['heroView'] and args.onlineEnvSettings.isDataGatherAndStoreEnabled == True
			dataGatheringParams.use_hero_actor_forDataGathering = use_hero_actor_forDataGathering  # if simulationOptions.simulateReplay is False else False
			args.onlineEnvSettings.use_hero_actor_forDataGathering = use_hero_actor_forDataGathering

			# Collect scene raycast/pointcloud data only on the first episode
			dataGatheringParams.collectSceneData = 0 #1 if (episodeIndex == 0 and simulationOptions.simulateReplay is False) else 0
			self.envManagement.SpawnWorld(dataGatherSetup=dataGatheringParams, envSetup=envParams, args=args)
			self.renderContext.setupConnection(alreadySpawnedWorld=self.carlaConnection.world, alreadySpawnedMapName=self.carlaConnection.map, use_hero_actor=use_hero_actor_forDataGathering)

		except:
			logging.exception('Fatal error')
			print("Unexpected error:", sys.exc_info()[0])
			traceback.print_exception(*sys.exc_info())
			if self.args.stop_for_errors:
				raise  # Always raise exception otherwise it will be masked and apperantly it would continue...
			self.destroyCurrentEnv(withError=True)

	def destroyCurrentEnv(self, withError=False, betweenIterations=False):
		if self.envManagement:
			self.envManagement.DespawnWorld()

		if self.carlaConnection:
			self.carlaConnection.disconnect(withError=False)

		if betweenIterations == False and self.renderContext:
			self.renderContext.quit()

	# Returns a tuple of (internal frame index, the vehicles frame data and pedestrians frame data)
	def SimulateFrame(self, isInitializationFrame=False):
		return self.envManagement.SimulateFrame(isInitializationFrame)

	def onEpisodeStartup(self, isFirstEpisode):
		self.carlaConnection.onEpisodeStartup()
		self.envManagement.onEpisodeStartup(isFirstEpisode)

	def onEpisodeEnd(self):
		pass