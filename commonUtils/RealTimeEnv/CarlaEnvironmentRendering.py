from . CarlaRealTimeUtils import *
from . CarlaServerConnection import *
from . CarlaEnvironmentActors import  *
import carla


import commonUtils.RealTimeEnv.no_rendering_mode as SIMPLIFIED_RENDERING

class EnvironmentRendering():
    def __init__(self, renderingOptions : RenderOptions, args):
        self.renderingOptions = renderingOptions
        self.RenderImgType = None
        self.font = None
        self.clock = None
        self.display = None
        self.actorsContext = None # Getting ugly access for actors...

        # HACKS FOR RENDERING
        args.onlineEnvSettings.map = None
        args.onlineEnvSettings.show_triggers = False
        args.onlineEnvSettings.show_connections = False
        args.onlineEnvSettings.show_spawn_points = False
        args.onlineEnvSettings.filter = "vehicle.*"
        args.onlineEnvSettings.width = renderingOptions.topViewResX
        args.onlineEnvSettings.height = renderingOptions.topViewResY
        DataGatherParams.image_size[0] = args.onlineEnvSettings.width
        DataGatherParams.image_size[1] = args.onlineEnvSettings.height

        if renderingOptions.sceneRenderType != RenderType.RENDER_NONE:
            pygame.init()
            #UsePyGameRendering()
            if renderingOptions.sceneRenderType == RenderType.RENDER_COLORED:
                self.RenderImgType = RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG
                self.display = pygame.display.set_mode((DataGatherParams.image_size[0], DataGatherParams.image_size[1]),
                                                       pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self.RenderImgType = RenderUtils.EventType.EV_SIMPLIFIED_TOPVIEW
                self.display = pygame.display.set_mode((DataGatherParams.image_size[0], DataGatherParams.image_size[1]),
                                                       pygame.HWSURFACE | pygame.DOUBLEBUF)


            self.font = RenderUtils.get_font()
            self.clock = pygame.time.Clock()

            self.input_control = SIMPLIFIED_RENDERING.InputControl("test", args)
            self.hud = SIMPLIFIED_RENDERING.HUD("hud", args.onlineEnvSettings.width, args.onlineEnvSettings.height)
            self.world = SIMPLIFIED_RENDERING.World("world", args, timeout=2.0, hud=self.hud)
            # For each module, assign other modules that are going to be used inside that module
            self.input_control.start(self.hud, self.world)
            self.hud.start()


    def setupConnection(self, alreadySpawnedWorld, alreadySpawnedMapName, use_hero_actor):
        self.use_hero_actor=  use_hero_actor
        self.world.start(self.input_control,
                         alreadySpawnedWorld=alreadySpawnedWorld,
                         alreadySpawnedMapName=alreadySpawnedMapName,
                         actorsContext=self.actorsContext)

    def setObserverWindow(self, bboxWindow):
        if self.renderingOptions.sceneRenderType != RenderType.RENDER_NONE:
            self.world.setSceneWindow(bboxWindow)

    def tick(self, syncData, isInitializationFrame=False):
        if syncData is None:
            return

        if self.renderingOptions.sceneRenderType != RenderType.RENDER_NONE:
            self.clock.tick()

            # TODO: fix this call to get in input control
            self.processInputEvents()

            worldSnapshot = syncData['worldSnapshot']

            # Tick all modules
            self.world.tick(self.clock, worldSnapshot)
            self.hud.tick(self.clock)
            self.input_control.tick(self.clock)



            if self.use_hero_actor and self.renderingOptions.sceneRenderType == RenderType.RENDER_COLORED:
                # Take the date from world considering hero car view
                image_seg = syncData["seg"]
                image_rgb = syncData["rgb"]
                image_depth = syncData["depth"]

                # Draw the display and stats
                if self.RenderImgType ==  RenderUtils.EventType.EV_SWITCH_TO_DEPTH:
                    RenderUtils.draw_image(self.display, image_depth, blend=True)
                elif self.RenderImgType == RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG:
                    RenderUtils.draw_image(self.display, image_rgb)
                    RenderUtils.draw_image(self.display, image_seg, blend=True)

                assert self.actorsContext
                # self.actorsContext.updateSpectator()
            else:
                # TODO
                pass

            # Hud rendering
            self.display.blit(self.font.render('%Press D - Depth or S - Segmentation + RGB or T for topview', True, (255, 255, 255)), (8, 5))
            self.display.blit(self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)), (8, 20))

            fps = round(1.0 / worldSnapshot.timestamp.delta_seconds)
            self.display.blit(self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),(8, 38))


            # Render modules
            self.display.fill(SIMPLIFIED_RENDERING.COLOR_ALUMINIUM_4)
            self.world.render(self.display, worldSnapshot=worldSnapshot, isInitializationFrame=isInitializationFrame)
            self.hud.render(self.display)
            #input_control.render(display)

            pygame.display.flip()

    def quit(self):
        if self.renderingOptions.sceneRenderType != RenderType.RENDER_NONE:
            pygame.quit()

    def processInputEvents(self):
        inputEv = RenderUtils.get_input_event()
        if inputEv == RenderUtils.EventType.EV_QUIT:
            return False
        elif inputEv == RenderUtils.EventType.EV_SWITCH_TO_DEPTH:
            self.RenderImgType = RenderUtils.EventType.EV_SWITCH_TO_DEPTH
        elif inputEv == RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG:
            self.RenderImgType = RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG
        elif inputEv == RenderUtils.EventType.EV_SIMPLIFIED_TOPVIEW:
            self.RenderImgType = RenderUtils.EventType.EV_SIMPLIFIED_TOPVIEW
