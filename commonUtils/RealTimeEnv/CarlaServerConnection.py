from . CarlaRealTimeUtils import *
import carla
from RL.settings import run_settings, RANDOM_SEED_NP,RANDOM_SEED

import sys
#from RL.settings import run_settings as RLAgentSettings

# This is a manager for keeping conectivity tracking with Carla Server
class CarlaServerConnection:
    def __init__(self, parent, args, simulationOptions):
        self.parent = parent
        self.actorsManager = None
        self.map = None
        self.world = None
        self.client = None
        self.traffic_manager = None
        self.TIMEOUT_VALUE = 10000.0
        self.args = args
        self.isConnected = False

        self.s_weather_presets = CarlaServerConnection.find_weather_presets()
        self.simulationOptions = simulationOptions

    # Connect to the carla server
    def connectToServer(self, host, port):
        if self.isConnected:
            return

        # Connect with the server
        self.client = carla.Client(host, port)
        serverVersion = self.client.get_server_version()
        self.client.set_timeout(self.TIMEOUT_VALUE)
        self.isConnected = True
        try:
            #self.availableMaps = ['/Game/Carla/Maps/Town01', '/Game/Carla/Maps/Town02', '/Game/Carla/Maps/Town03', '/Game/Carla/Maps/Town04', '/Game/Carla/Maps/Town05', '/Game/Carla/Maps/Town06', '/Game/Carla/Maps/Town07']
            self.availableMaps = self.client.get_available_maps()
            pass
        except:
            print("Can't get the maps !")
            raise RuntimeError('basic stuff doesnt work')

        logging.log(logging.INFO, ("Available maps are: {0}").format(self.availableMaps))
        self.orig_settings = self.client.get_world().get_settings()

    def setEnvironmentData(self):
        logging.log(logging.INFO, 'Setting some random weather and traffic management...')
        # Now set the weather
        weather_id = np.random.choice(len(self.s_weather_presets))
        preset = self.s_weather_presets[weather_id]
        self.world.set_weather(preset[0])

        if self.simulationOptions.simulateReplay == False:
            self.setupTrafficManager()


    def reloadWorld(self, envParams : EnvSetupParams):
        self.world =  self.client.get_world()
        if self.world is None or self.world.get_map() is None or self.world.get_map().name != envParams.mapToUse:
            self.client.load_world(envParams.mapToUse)
        else:
            self.client.reload_world()

        self.envParams = envParams

        self.setEnvironmentData()

        # Set settings for this episode and reload the world
        settings = carla.WorldSettings(
            no_rendering_mode=self.args.onlineEnvSettings.no_server_rendering,
            synchronous_mode=envParams.synchronousComm,
            fixed_delta_seconds=1.0 / envParams.fixedFPS,
            deterministic_ragdolls=True) #RLAgentSettings.getCarlaFramerate()) #envParams.fixedFPS)
        # settings.randomize_seeds()
        self.world.apply_settings(settings)
        self.map = self.world.get_map()

    def releaseServerConnection(self):
        # Deactivate sync mode
        if self.world == None:
            return

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

    def destroyEnvironment(self):
        if self.isConnected:
            logging.log(logging.INFO, 'Destroying the environment')
            self.actorsManager.despawnActors(self.client)

    def disconnect(self, withError : bool):
        if withError:
            print("Unexpected error:", sys.exc_info()[0])
            tb = traceback.format_exc()
            print(tb)

        if not self.isConnected:
            return

        self.releaseServerConnection()
        self.isConnected = False
        if withError is True:
            sys.exit()

    def setupTrafficManager(self):
        # Set the traffic management stuff
        # NOTE: the issue in the past with traffic manager was that cars were not moving after the second episode
        # To that end why i did was to:
        # - increase the timeout value to 10s and check the outputs from TM
        # - destroy the client each time between episodes (i.e. having a script that handles data gathering and
        # connects each time with a new client.)
        self.traffic_manager = self.client.get_trafficmanager(EnvSetupParams.tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(EnvSetupParams.distanceBetweenVehiclesCenters)
        self.traffic_manager.set_synchronous_mode(self.envParams.synchronousComm)
        self.traffic_manager.global_percentage_speed_difference(EnvSetupParams.speedLimitExceedPercent)

        self.traffic_manager.set_hybrid_physics_radius(0)
        self.traffic_manager.set_hybrid_physics_mode(True)

        if self.args.deterministic:
            self.traffic_manager.set_random_device_seed(RANDOM_SEED_NP)

    def disableTrafficLightsForHeroAgents(self, listOfHeroes):
        for heroActor in listOfHeroes:
            assert isinstance(heroActor, carla.Actor)
            self.traffic_manager.ignore_lights_percentage(heroActor, 100)

    # Get the weather presets lists
    @staticmethod
    def find_weather_presets():
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


    def onEpisodeStartup(self):
        print("Ressing the traffic manager...")
        self.traffic_manager.reset()
