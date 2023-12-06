from agent_pfnn import AgentPFNN
from agent_net import NetAgent,AgentNetPFNN
import numpy as np
from settings import PEDESTRIAN_MEASURES_INDX

class GoalAgent(AgentNetPFNN):
    # settings, net, grad_buffer, init_net=None, init_grad_buffer=None,goal_net=None,grad_buffer_goal=None)
    def __init__(self, settings, net=None, grad_buffer=None, init_net=None, init_grad_buffer=None,goal_net=None,grad_buffer_goal=None):
        super(GoalAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer,goal_net,grad_buffer_goal)
        self.position=np.zeros(3, dtype=np.float)
        self.frame=0

    def next_action(self, episode, training=True, viz=False):
        dir=episode.get_goal_dir(episode.pedestrian_data[self.id].agent[self.frame], episode.pedestrian_data[self.id].goal[0,:])

        value=np.nonzero(dir[0,:len(dir[0])-1])
        value=value[0][0]

        episode.pedestrian_data[self.id].probabilities[self.frame, 0:9] = 1/9.0*np.ones_like(episode.pedestrian_data[self.id].probabilities[self.frame, 0:9])
        episode.pedestrian_data[self.id].velocity[self.frame] = episode.actions[value]
        episode.pedestrian_data[self.id].action[self.frame] = value
        episode.pedestrian_data[self.id].speed[self.frame] = 1
        return episode.pedestrian_data[self.id].velocity[self.frame]

    def train(self,ep_itr, statistics,episode, filename, filename_weights, poses, last_frame):
        return statistics

    def set_session(self, session):
        self.init_net.sess = session
        self.goal_net.sess = session
        #self.net.sess = session

    def evaluate(self, ep_itr, statistics, episode, poses, priors, last_frame):
        return statistics

class GoalCVAgent(AgentNetPFNN):
    #settings,  net=None, grad_buffer=None, init_net=None, init_grad_buffer=None,goal_net=None,grad_buffer_goal=None
    def __init__(self, settings,  net=None, grad_buffer=None, init_net=None, init_grad_buffer=None,goal_net=None,grad_buffer_goal=None):
        super(GoalCVAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer,goal_net,grad_buffer_goal)
        self.position=np.zeros(3, dtype=np.float)
        self.frame=0
        self.speed=0
        self.vel=[0,0,0]
        self.exact_pos=self.position.astype(np.float32).copy()
        #self, pos, goal, current_frame=0, vel=0, init_dir=[],episode=None):
    def initial_position(self, pos, goal, current_frame=0, vel=0, init_dir=[],episode=None):
        super(GoalCVAgent, self).initial_position(pos, goal, current_frame, vel, init_dir,episode=episode)
        self.position = np.array(pos)
        #print "Initial pos "+str(self.position)
        self.frame = 0
        self.current_init_frame = current_frame
        self.speed = np.random.normal(loc=1.23, scale=0.3)*5.0/17.0#5

    def next_action(self, episode, training=True, viz=False):
        #print "Random action"
        #print "Position: "+str(self.position)+" goal: "+str(episode.goal)+" "+str(episode.goal-self.position)+" "+str(self.speed*(episode.goal-self.position))
        # (self,id, pos, breadth,frame_in,vel=[], training=True, eval=True, pedestrian_view_occluded=False,temporal_scaling=0.1*0.3, field_of_view=2*np.pi)
        pos=episode.pedestrian_data[self.id].agent[self.frame]
        breadth=[self.settings.agent_s[1],self.settings.agent_s[2]]
        tensor, people, cars, people_cv, cars_cv = episode.get_agent_neighbourhood(self.id,pos,breadth,self.frame, pedestrian_view_occluded=self.settings.pedestrian_view_occluded, field_of_view=self.settings.field_of_view)

        dir=episode.get_goal_dir(episode.pedestrian_data[self.id].agent[self.frame], episode.pedestrian_data[self.id].goal[0,:])

        #print "Speed "+str(self.speed)+" "+str(dir)

        if np.linalg.norm(episode.pedestrian_data[self.id].goal[0,:]-self.position)>1e-7:
            self.vel=self.speed*(episode.pedestrian_data[self.id].goal[0,:]-self.position)/np.linalg.norm(episode.pedestrian_data[self.id].goal[0,:]-self.position)
        else:
            self.vel = self.speed * (episode.pedestrian_data[self.id].goal[0,:] - self.position)

        #print "Velocity: "+str(self.vel)
        action=episode.find_action_to_direction(np.round(self.vel),np.linalg.norm(np.round(self.vel)))
        # value=np.nonzero(dir[0,:len(dir[0])-1])
        # value=value[0][0]

        episode.pedestrian_data[self.id].probabilities[self.frame, 0:9] = 1/9.0*np.ones_like(episode.pedestrian_data[self.id].probabilities[self.frame, 0:9])
        episode.pedestrian_data[self.id].velocity[self.frame] = self.vel#episode.actions[value]*self.speed
        episode.pedestrian_data[self.id].action[self.frame] = action
        episode.pedestrian_data[self.id].speed[self.frame] = np.linalg.norm(self.vel )
        #print(" Agent pos "+str(episode.agent[self.frame])+" goal "+str(episode.goal)+" vel: "+str(episode.pedestrian_data[self.id].velocity[self.frame] ))
        return episode.pedestrian_data[self.id].velocity[self.frame]


    def train(self,ep_itr, statistics,episode, filename, filename_weights, poses, last_frame):
        return statistics

    def set_session(self, session):
        self.init_net.sess = session
        self.goal_net.sess = session
        #self.net.sess = session

    def evaluate(self, ep_itr, statistics, episode, poses, priors, last_frame):
        return statistics

class CVMAgent(AgentPFNN):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(CVMAgent, self).__init__(settings, net, grad_buffer)
        self.current_init_frame = 0
        self.vel = [0, 0, 0]  # episode.get_valid_vel(self.frame + 1, person_id=person_id)  # -self.position
        self.dir = [0, 0, 0]  # episode.get_goal_dir(episode.agent[self.frame], self.vel)
        self.exact_pos = [0, 0, 0]  # self.position.copy()

    def initial_position(self, pos, goal, current_frame=0, vel=0):
        self.position = np.array(pos)
        self.frame = 0
        self.current_init_frame = current_frame
        # self.vel=episode.get_valid_vel(self.frame + 1, person_id=person_id)#-self.position
        # self.dir = episode.get_goal_dir(episode.agent[self.frame], self.vel)
        self.exact_pos = self.position.copy()
        # print "Agent init frame"+str(self.frame)+"  current frame "+str(self.current_frame)

        # To do: follow pfnnagent!
    def perform_action(self, vel, episode, prob=0, person_id=-1, training=True):
        pedestrianLastFrameData = episode.environmentInteraction.GetAgentLastFrameData(self)
        pedestrianLastFrameData.Reset()

        self.vel = episode.get_valid_vel(0, person_id=person_id)  # -self.position

        # self.dir = episode.get_goal_dir(episode.agent[self.frame], self.vel)
        # vel = episode.get_valid_pos(self.frame + 1, person_id=person_id)-self.position
        self.exact_pos = self.exact_pos + self.vel

        valid = episode.valid_position(self.exact_pos.astype(int))  # self.position + np.array(self.vel))
        if valid and episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] < 0.15 and episode.measures[
            max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.goal_reached] == 0:

            episode.agent[self.frame + 1] = self.position + np.array(self.vel)
            pedestrianLastFrameData.not_hitting_object = True
            # print "Take action " + str(vel) + " in " + str(self.position)+" to get "+str(episode.agent[self.frame+1])
        else:
            episode.agent[self.frame + 1] = self.position
            pedestrianLastFrameData.not_hitting_object = False
            self.exact_pos = self.exact_pos - self.vel
        self.position = episode.get_valid_pos(self.frame + 1, person_id=person_id)
        episode.velocity[self.frame] = self.vel

        pedestrianLastFrameData.velUsed = self.vel
        pedestrianLastFrameData.frame = self.frame

        self.frame += 1

    def next_action(self, episode, training=True):
        # pos = episode.get_valid_pos(self.frame + 1, person_id=episode.goal_person_id) - self.position
        # self.vel = episode.get_valid_vel(0, person_id=person_id)  # -self.position

        next_pos = self.exact_pos + self.vel
        # vel=next_pos.astype(int)-self.position
        dir = episode.get_goal_dir(self.position, next_pos.astype(int))
        value = np.nonzero(dir[0, :len(dir[0]) - 1])
        value = value[0][0]

        episode.pedestrian_data[self.id].probabilities[self.frame, 0:9] = 1 / 9.0 * np.ones_like(
            episode.pedestrian_data[self.id].probabilities[self.frame, 0:9])
        episode.pedestrian_data[self.id].velocity[self.frame] = episode.actions[value]
        episode.action[self.frame] = value
        episode.pedestrian_data[self.id].speed[self.frame] = 1
        return episode.pedestrian_data[self.id].velocity[self.frame]

    def train(self, ep_itr,statistics, episode, filename, filename_weights, poses, last_frame):
        pass

    def set_session(self, session):
        self.net.sess = session

    def evaluate(self, ep_itr, statistics, episode, poses, last_frame):
        return statistics

# Agent moving in one direction.
class DirectionalAgent(AgentPFNN):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(DirectionalAgent, self).__init__(settings, net, grad_buffer)
        self.direction = np.array([0, 0, 0])

    # To do: follow pfnnagent!
    def perform_action(self, vel, episode, prob=0, training=True):

        if episode.useRealTimeEnv == False or episode.environmentInteraction.isOnline == False:
            pedestrianLastFrameData = episode.environmentInteraction.GetAgentLastFrameAction(self)
            pedestrianLastFrameData.Reset()
            if np.linalg.norm(vel) == 1:  # Move!
                if vel[1] >= 0 and vel[2] >= 0:
                    move_dir = self.direction
                else:
                    move_dir = -self.direction
                valid = episode.valid_position(self.position + np.array(move_dir))
                if valid and episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] < 0.15:
                    self.position = self.position + np.array(move_dir)
                    pedestrianLastFrameData.not_hitting_object = True
                else:
                    if np.random.rand(1) < prob:
                        self.position = self.position + np.array(move_dir)
                    if not valid:
                        pedestrianLastFrameData.not_hitting_object = False
            else:
                self.direction = self.direction + vel
                if np.linalg.norm(self.direction) > 1:
                    self.direction = np.ceil(self.direction / np.linalg.norm(self.direction))

            pedestrianLastFrameData.velUsed = vel
            pedestrianLastFrameData.frame = self.frame
        else:
            super(DirectionalAgent, self).perform_action()

        if episode.useRealTimeEnv == False:
            self.frame = self.frame + 1
            episode.agent[self.frame] = self.position
