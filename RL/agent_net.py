import numpy as np


from agent import SimplifiedAgent, ContinousAgent
from agents_dummy import DummyAgent
from net import Net
from agent_pfnn import AgentPFNN
from agent_externals_flask import Agent_SGAN, Agent_STGAT, Agent_SOCIALSTGCNN



import numpy as np


from memory_profiler import profile as profilemem
from settings import run_settings

memoryLogFP_decisions_agentpfnn = None
if run_settings.memoryProfile:
    memoryLogFP_decisions_agentpfnn = open("memoryLogFP_decisions_agentpfnn.log", "w+")

class NetAgent(SimplifiedAgent):

    def __init__(self, settings, net, grad_buffer, init_net=None, init_grad_buffer=None,goal_net=None,grad_buffer_goal=None):
        super(NetAgent, self).__init__(settings)
        self.net=net
        if init_net:
            self.init_net=init_net
        if goal_net:
            self.goal_net=goal_net

        if grad_buffer:
            net.set_grad_buffer(grad_buffer)
        if self.settings.learn_init and init_grad_buffer:
            self.init_net.set_grad_buffer(init_grad_buffer)
        if self.settings.separate_goal_net and grad_buffer_goal:
            self.goal_net.set_grad_buffer(grad_buffer_goal)
        self.current_init_frame = 0



    def init_agent(self, episode, training=True,  viz=False, current_frame=0):
        self.current_init_frame =current_frame
        self.frame=0
        init_value=self.init_net.feed_forward(self.id, episode, self.current_init_frame + self.frame, training,0 ,  manual=self.settings.manual_init)

        if self.settings.separate_goal_net:
            self.goal_net.feed_forward(self.id,episode, self.current_init_frame + self.frame, training, 0)
        return init_value

    def set_new_goal(self, episode, manual_goal):
        if not self.settings.separate_goal_net:
            return super(NetAgent, self).set_new_goal(episode, manual_goal)
        else:
            return self.goal_net.feed_forward(self.id, episode, self.current_init_frame + self.frame, training, 0)
    # @profilemem(stream=memoryLogFP_decisions_agentpfnn)
    def next_action(self, episode, training=True,  viz=False):

        self.is_distracted()
        if training:
            agent_frame=self.frame
        else:
            agent_frame = -1
        if self.frame % self.settings.action_freq==0 or  episode.pedestrian_data[self.id].goal_person_id >= 0:
            return self.net.feed_forward(self.id,episode, self.current_init_frame + self.frame, training, agent_frame, distracted=self.distracted)#,  viz=viz)
        else:
            self.update_episode(episode, episode.pedestrian_data[self.id].speed[self.frame-1], episode.pedestrian_data[self.id].action[self.frame-1], episode.pedestrian_data[self.id].velocity[self.frame-1],  episode.pedestrian_data[self.id].probabilities[self.frame-1] )
            return episode.pedestrian_data[self.id].velocity[self.frame-1]


    def update_episode(self, episode, speed, value, velocity, probabilities):
        episode.pedestrian_data[self.id].velocity[self.frame] = velocity
        episode.pedestrian_data[self.id].action[self.frame] = value
        episode.pedestrian_data[self.id].probabilities[self.frame] = probabilities
        episode.pedestrian_data[self.id].speed[self.frame] = speed

    def train(self, ep_itr, statistics,episode, filename, filename_weights, poses, last_frame):

        return self.net.train(self.id, ep_itr, statistics,episode, filename, filename_weights, poses, [], [], seq_len=last_frame)

    def init_net_train(self, ep_itr, statistics, episode, filename, filename_weights, poses,priors, initialization_car, last_frame, statistics_car=[], initialization_goal=[]):
        if self.settings.separate_goal_net and self.goal_net!=None:
            goal_filename=filename[:len(".pkl")]+"_goal.pkl"
            filename_weights_goal=filename_weights[:len(".pkl")]+"_goal.pkl"
            return self.goal_net.train(self.id, ep_itr, statistics, episode, goal_filename, filename_weights_goal, poses, initialization_goal, initialization_car,statistics_car, seq_len=last_frame)
        #
        return self.init_net.train(self.id,ep_itr, statistics, episode, filename, filename_weights, poses, priors, initialization_car,statistics_car, seq_len=last_frame)

    def set_session(self, session):
        self.net.sess = session
        self.init_net.sess = session
        self.goal_net.sess = session

    def evaluate(self, ep_itr, statistics, episode, poses, priors, last_frame):
        return self.net.evaluate(self.id, ep_itr, statistics, episode, poses, priors, [], last_frame)

    def init_net_evaluate(self,ep_itr, statistics, episode, poses, priors,initialization_car, last_frame, statistics_car=[]):
        return self.init_net.evaluate(self.id, ep_itr, statistics, episode, poses, priors, initialization_car, statistics_car, last_frame)

class AgentNetPFNN(AgentPFNN, NetAgent):
    def __init__(self, settings, net, grad_buffer, init_net, init_grad_buffer,goal_net=None,grad_buffer_goal=None):

        super(AgentNetPFNN, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer,goal_net,grad_buffer_goal)

    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[],episode=None):
        super(AgentNetPFNN, self).initial_position(pos, goal, current_frame=current_frame, vel=vel,episode=episode)

class RandomAgentPFNN(AgentNetPFNN):
    def __init__(self, settings, net, grad_buffer, init_net, init_grad_buffer,goal_net=None,grad_buffer_goal=None):

        super(RandomAgentPFNN, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer,goal_net,grad_buffer_goal)

    def next_action(self,episode,training=True, viz=False):
        #print "Random action"
        value=np.random.randint(9)
        episode.pedestrian_data[self.id].probabilities[self.frame, 0:9] = 1/9.0*np.ones_like(episode.pedestrian_data[self.id].probabilities[self.frame, 0:9])
        episode.pedestrian_data[self.id].velocity[self.frame] = episode.actions[value]#*self.settings.ped_reference_speed
        episode.pedestrian_data[self.id].action[self.frame] = value
        episode.pedestrian_data[self.id].speed[self.frame] = 1#self.settings.ped_reference_speed
        return episode.pedestrian_data[self.id].velocity[self.frame]

class ContinousNetAgent(ContinousAgent, NetAgent):
    def __init__(self, settings, net, grad_buffer, init_net, init_grad_buffer):

        super(ContinousNetAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer)

    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[],episode=None):
        super(ContinousNetAgent, self).initial_position(pos, goal, current_frame=current_frame, vel=vel, init_dir=init_dir,episode=episode)

class ContinousNetPFNNAgent(ContinousAgent, AgentNetPFNN):
    def __init__(self, settings, net, grad_buffer, init_net, init_grad_buffer):

        super(ContinousNetPFNNAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer)

    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[],episode=None):
        super(ContinousNetPFNNAgent, self).initial_position(pos, goal, current_frame=current_frame, vel=vel, init_dir=init_dir,episode=episode)

