
import numpy as np
from RL.settings import RANDOM_SEED,PEDESTRIAN_MEASURES_INDX,realTimeEnvOnline
np.random.seed(RANDOM_SEED)
import copy

# Basic agent behaviour. Implement next_action to decide the velocity of the next action.
# train and evaluate are for neural-net evaluations.
class SimplifiedAgent(object):
    def __init__(self, settings,net=None, grad_buffer=None, init_net=None, init_grad_buffer=None,goal_net=None, grad_buffer_goal=None):
        self.position=np.zeros(3, dtype=np.float)
        self.frame=0 # current frame
        self.current_init_frame=0 # initialize from this frame
        self.pos_exact = np.zeros(3, dtype=float) # exact position of agent
        self.velocity=0
        self.acceleration=settings.acceleration # Is agent taking acceleration steps?
        self.settings=settings
        self.goal=[]
        self.PFNN = None
        self.PFNN = None
        self.distracted=False
        self.distracted_len=-1
        self.isPedestrian=True
        self.id=None
        self.on_car=False

        # A collection of functors for confirmation from the real time environment
        # Will be initialized after real time environment is created, if any is used only
        self.onlinerealtime_spawnFunctor = None

        # The agent id in the online realtime environment if any is used.
        # The ID will correspond to the trainable agent in the online environment
        self.onlinerealtime_agentId = None

    # Initializes agent on position pos, with a goal

    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[], episode=None, on_car=False):
        if len(pos)>0:
            self.position=np.array(pos).astype(int)
            self.pos_exact = np.array(pos).astype(np.float)
        else:
            self.position = []
            self.pos_exact = []
        self.frame = 0
        self.current_init_frame = current_frame

        self.velocity=vel
        self.goal=goal
        self.distracted = False
        self.distracted_len = -1
        self.on_car=False

        # print(("Agent initial position "+str(self.position)+" goal "+str(self.goal)))

    def getOnlineRealtimeAgentId(self):
        return self.onlinerealtime_agentId

    def getIsPedestrian(self):
        return self.isPedestrian

    def is_distracted(self):
        add_noise=False
        # print (" Is agent distracted? " + str(self.distracted) + " distracted len " + str(self.distracted_len))
        if not self.distracted and self.settings.distracted_pedestrian:
            add_noise = np.random.binomial(1, self.settings.pedestrian_noise_probability)
            # print (" Add noise to car input? Draw random variable with prob " + str(
            #     self.settings.car_noise_probability) + " value " + str(add_noise))

            if add_noise:
                self.distracted=True
                self.distracted_len=np.random.poisson(self.settings.avg_distraction_length)
                # print (" Set agent distracted for "+str(self.distracted_len))
        elif self.distracted_len>0 and self.distracted:
            # print (" Agent is still distracted ")
            return True
        else:
            # print (" Agent is not distracted ")
            self.distracted=False
            return False

        return add_noise

    # Implement this!
    def next_action(self,  state_in, training):

        raise NotImplementedError("Please Implement this method")

    # Implement this! - for neural net, this is where gradient update is made
    def train(self, ep_itr, statistics, episode, filename, filename_weights, poses, last_frame):
        raise NotImplementedError("Please Implement this method")

    # Implement this! - for neural net, this is where loss is evaluated
    def evaluate(self,ep_itr, statistics, episode, poses, priors, last_frame):
        raise NotImplementedError("Please Implement this method")

    # Take a step forward of the agent either in a simulated or real time env
    # Make agent stand still if next step results in collision otherwise take step.
    def perform_action(self, vel, episode, prob=0):
        # We'll write environment decisions in this data structure
        if realTimeEnvOnline:
            agentLastFrameData = episode.environmentInteraction.getAgentLastFrameData(self)
        exact = True

        if self.settings.realTimeEnvOnline is False: # Simulate the planned positions and check manually for collisions
            # print ("vel before get planned " + str(vel))
            next_pos, step = self.get_planned_position(exact, vel)
            not_hitting_object, alive = self.is_next_pos_valid(episode, next_pos)
            not_hitting_object_and_alive=not_hitting_object and alive
            # print("Not hitting obj and alive not online "+str(not_hitting_object_and_alive))
        else: # Using an online real time env, so the values will come out from there as a ground truth for both real step taken, position and if it hits or not something
            step = agentLastFrameData.velUsed
            next_pos = agentLastFrameData.nextPos
            not_hitting_object = agentLastFrameData.notHittingObject
            not_hitting_object_and_alive = not_hitting_object and agentLastFrameData.notHittingObjectAndAlive
        # print("Not hitting obj and alive in agent " + str(not_hitting_object_and_alive))
        if not_hitting_object_and_alive:
            # print("Take step ")
            next_pos = self.take_step(next_pos, step, episode, force_nextPos=self.settings.realTimeEnvOnline)
        else:
            # print("Stand still ")
            next_pos, step = self.stand_still(next_pos, step, episode, force_nextPos=self.settings.realTimeEnvOnline)

        self.carry_out_step(next_pos, step, episode)

        if not realTimeEnvOnline: # i.e. running entierly without simulation
            if not_hitting_object_and_alive:
                self.mark_valid_move_in_episode(episode,self.frame)
            else:
                if not not_hitting_object:
                    self.mark_agent_hit_obstacle_in_episode(episode,self.frame)
                if not alive:
                    # print("Mark agent dead in episode")
                    self.mark_agent_dead_in_episode(episode,self.frame)

        if self.distracted:

            self.distracted_len=self.distracted_len-1
            # print(" update distracted len " + str(self.distracted_len))
        # self.update_agent_pos_in_episode(episode)
    def update_agent_pos_in_episode(self, episode, updated_frame):
        assert self.frame == updated_frame - 1, "We are not targeting the correct frame ! Agent curr frame is {self.frame} " \
                                                f"and we update the next frame as being {updated_frame} instead of {self.frame}"
        episode.pedestrian_data[self.id].agent[updated_frame] = self.pos_exact
        episode.pedestrian_data[self.id].measures[self.frame - 1, PEDESTRIAN_MEASURES_INDX.distracted] = self.distracted
        # # print(" Episode save distracted "+str(episode.measures[self.frame-1, PEDESTRIAN_MEASURES_INDX.distracted])+" frame "+str(self.frame-1))


    def update_metrics(self, episode):
        # We take the target update frame index and some data from the recorded agent data that happened on the last frame
        # this includes for example if he hits an object
        if realTimeEnvOnline:
            pedestrianLastFrameData = episode.environmentInteraction.getAgentLastFrameData(self)

            updatedFrame = pedestrianLastFrameData.frame - 1
            assert updatedFrame == self.frame - 1, "Expecting to compute metrics for previous frame!"
        else:
            updatedFrame = self.frame - 1
            #print("Frame "+str(updatedFrame))


        episode.pedestrian_data[self.id].measures[updatedFrame, PEDESTRIAN_MEASURES_INDX.distracted] = self.distracted
        if realTimeEnvOnline:
            episode.pedestrian_data[self.id].measures[updatedFrame, PEDESTRIAN_MEASURES_INDX.change_in_pose] = pedestrianLastFrameData.change_in_pose # To do: what is done instead?

        if realTimeEnvOnline:
            if pedestrianLastFrameData.not_hitting_object_and_alive:
                self.mark_valid_move_in_episode(episode, updatedFrame)
            else:
                if not pedestrianLastFrameData.not_hitting_object:
                    self.mark_agent_hit_obstacle_in_episode(episode, updatedFrame)
                else:
                    self.mark_agent_dead_in_episode(episode, updatedFrame) # agent does not die when hit by object.

    def mark_agent_hit_obstacle_in_episode(self, episode, updatedFrame):
        episode.pedestrian_data[self.id].measures[updatedFrame, PEDESTRIAN_MEASURES_INDX.hit_obstacles] = 1

    def mark_agent_dead_in_episode(self, episode, updatedFrame):
        episode.pedestrian_data[self.id].measures[updatedFrame, PEDESTRIAN_MEASURES_INDX.agent_dead] = 1

    def mark_valid_move_in_episode(self, episode, updatedFrame):
        episode.pedestrian_data[self.id].measures[updatedFrame, PEDESTRIAN_MEASURES_INDX.hit_obstacles] = 0

    # What is the next planned position given the velocity from next_action
    # Do not add side effects of the class in this function please !!!
    def get_planned_position(self, exact, vel):

        next_pos = self.pos_exact + np.array(vel)

        step = np.array(vel)
        #print ("Planned pos"+str(next_pos)+" "+str(self.frame)+" previous "+str(self.pos_exact)+" diff "+str(next_pos-self.pos_exact)+" vel "+str(vel[1:]))
        if not exact:
            step=np.array(round(vel).astype(int))
            next_pos = self.pos_exact + np.array(round(vel).astype(int))

        return next_pos, step

    # Is the planned position a valid position, or will aent collide?
    def is_next_pos_valid(self, episode, next_pos):

        valid = episode.valid_position(next_pos)

        alive=self.is_agent_alive(episode)
        #print (" Is next pos valid agent " + str(valid)+" is alive "+str(alive)+" next pos "+str(next_pos))
        return valid, alive

    # Check if agent is alive
    def is_agent_alive(self, episode):

        # print("Check if agent is dead ? ")
        # print("Hit by car "+str(PEDESTRIAN_MEASURES_INDX.hit_by_car)+" frame "+str(max(0, self.frame - 1)))
        # print (" Hit by car "+str(episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car])+" reached goal: "+str(episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.goal_reached])+" agent dead "+str(episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.agent_dead] ))
        # print("Is agent dead in previous frame "+str(max(0, self.frame - 1))+" : "+str(episode.pedestrian_data[self.id].measures[max(0, self.frame - 1),PEDESTRIAN_MEASURES_INDX.agent_dead]))
        if episode.pedestrian_data[self.id].measures[max(0, self.frame - 1),PEDESTRIAN_MEASURES_INDX.agent_dead]:
            return False
        if self.settings.end_on_bit_by_pedestrians:
            if self.settings.stop_on_goal:
                return episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] <= 0 and episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.goal_reached] == 0 and \
                       episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_pedestrians] <= 0
            return episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] <= 0  and episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_pedestrians] <= 0
        else:
            if self.settings.stop_on_goal:
                return episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] <= 0 and episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.goal_reached] == 0
            return episode.pedestrian_data[self.id].measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] <= 0

    # How should agent stand still when walking into onjects.
    def stand_still(self,next_pos, vel, episode, force_nextPos=False):
        # print ("Stand still vel "+str(vel)+" next pos "+str(next_pos))
        vel=np.zeros_like(vel)

        if force_nextPos:
            self.pos_exact = next_pos
            self.position = np.round(self.pos_exact).astype(int)

        return next_pos, vel

    # result of taking a step
    def take_step(self, next_pos, vel, episode, force_nextPos=False):
        # print ("Take step "+str(next_pos))
        self.pos_exact = next_pos
        self.position = np.round(self.pos_exact).astype(int)

        # Redundancy of code but just to be sure that the person who changes this sees clearly the logic
        # When we force the position (e.g. a real time environment where the action already took place !! , we want to force the positions there).
        if force_nextPos:
            self.pos_exact = next_pos
            self.position = np.round(self.pos_exact).astype(int)
        return next_pos

    # Pfnn agent does sometjing here.
    def carry_out_step(self,next_pos, step, episode):
        #print "Pass"
        pass

    def on_post_tick(self, episode, manual_goal=[]):
        if not self.settings.realTimeEnvOnline:#realTimeEnvOnline:
            # print("End of episode measures in agent update position "+str(self.frame))
            episode.end_of_episode_measures(self.frame, self.id)
            if self.frame>0 :
                if episode.pedestrian_data[self.id].measures[self.frame, PEDESTRIAN_MEASURES_INDX.goal_reached]==1 and not self.settings.stop_on_goal: #and self.settings.learn_goal:
                    self.set_new_goal(episode, manual_goal)

            if self.frame+1< len(episode.pedestrian_data[self.id].goal) and len(self.goal)>0:
                episode.pedestrian_data[self.id].goal[self.frame+1,:]=self.goal.copy()
                if episode.pedestrian_data[self.id].goal_time[self.frame+1]==0:
                    episode.pedestrian_data[self.id].goal_time[self.frame + 1]= episode.pedestrian_data[self.id].goal_time[self.frame]
        self.frame += 1

    def set_new_goal(self, episode, manual_goal):
        if len(manual_goal) > 0:
            self.goal = episode.get_new_goal(self.id, self.frame)
            self.goal = manual_goal

        else:
            self.goal = episode.get_new_goal(self.id, self.frame)

    def getFrame(self):
        return self.frame

    # tf session
    def set_session(self, session):
        pass


# An agent that rotates the coordinate system at each step
class ContinousAgent(SimplifiedAgent):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(ContinousAgent, self).__init__(settings, net, grad_buffer)
        self.velocity=np.zeros(3, dtype=np.float)
        self.direction=np.zeros(3, dtype=np.float)
        self.angle=0
        #self.rotation_matrix=np.identity(2)
        self.rotation_matrix_prev = np.identity(3)
        print("Continous agent--------------------------->")


    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[],episode=None):
        super(ContinousAgent, self).initial_position( pos,goal, current_frame, vel,episode)
        self.rotation_matrix_prev = np.identity(3)
        if len(init_dir)>0:
            #print "Initial direction "+str(init_dir)
            self.direction=init_dir
            self.get_rotation_matrix(init_dir)
        else:
            self.angle = 0
            #self.rotation_matrix = np.identity(2)
            self.rotation_matrix_prev = np.identity(3)

    def get_rotation_matrix(self, init_dir, angle=0):
        #print "Angular input "+str(angle)+" dir "+str(init_dir)
        if self.frame % self.settings.action_freq==0:
            if angle==0 and np.linalg.norm(init_dir[1:])>1e-5:
                # d = np.sqrt(init_dir[1] ** 2 + init_dir[2] ** 2)
                # self.rotation_matrix_prev = np.identity(3)
                # self.rotation_matrix_prev[1, 1] = init_dir[1] / d
                # self.rotation_matrix_prev[1, 2] = -init_dir[2] / d
                # self.rotation_matrix_prev[2, 2] = init_dir[1] / d
                # self.rotation_matrix_prev[2, 1] = init_dir[2] / d
                self.angle=np.arctan2(init_dir[1] ,init_dir[2] )-(np.pi/2)
                #print " Arctan dir "+str(self.angle)
                #print np.arctan2(init_dir[1] ,init_dir[2] )/np.pi
            else:
                print ("Agent angle "+str(self.angle)+" "+str(angle))
                self.angle=self.angle+angle
                #print "After addition angle " + str(self.angle) + " " + str(angle)

            self.normalize_angle()

            self.rotation_matrix_prev[1, 1] = np.cos(self.angle)
            self.rotation_matrix_prev[1, 2] = np.sin(self.angle)
            self.rotation_matrix_prev[2, 2] = np.cos(self.angle)
            self.rotation_matrix_prev[2, 1] = -np.sin(self.angle)#


        # print  self.rotation_matrix_prev
        # print "Cos "+str(np.cos(self.angle))
        # print "Sin "+str(np.sin(self.angle))

        # print "Rotation around: "+str(init_dir)
        # print "Rotation matrix "
        # print str(self.rotation_matrix_prev)
        # print "Direction: "+str(np.matmul(self.rotation_matrix_prev, np.array([0,1,0])))

    def normalize_angle(self):
        if self.angle <= -np.pi:
            # print "Before addition " + str(self.angle) + " " + str(self.angle / np.pi)
            self.angle = 2 * np.pi + self.angle
            # print "After Rotation " + str(self.angle) + " " + str(self.angle / np.pi)
        if self.angle > np.pi:
            # print "Before minus " + str(self.angle) + " " + str(self.angle / np.pi)
            self.angle = self.angle - (2 * np.pi)
            # print "After Rotation " + str(self.angle)+" "+str(self.angle/np.pi)

    def get_planned_position(self, exact, vel):

        vel_rotated=np.matmul(self.rotation_matrix_prev, np.array(vel))
        #print "Rotated direction step= "+str(vel_rotated)
        return super(ContinousAgent, self).get_planned_position( exact, vel_rotated)

    def take_step(self, next_pos, vel, episode,force_nextPos=False):
        next_pos=super(ContinousAgent, self).take_step(next_pos, vel, episode, force_nextPos)
        self.get_rotation_matrix(vel, episode.pedestrian_data[self.id].angle[self.frame])  # np.matmul(rotation_matrix,self.rotation_matrix_prev)
        if self.frame+1<len(episode.pedestrian_data[self.id].angle):
            episode.pedestrian_data[self.id].angle[self.frame+1] = copy.copy(self.angle)
        #print "Episode angle "+str(episode.angle[self.frame+1])+" "+str(self.frame+1)+" "+str(episode.angle[self.frame+1]/np.pi)
        return next_pos
