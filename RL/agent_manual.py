from RL.agent_manual_simplified import ManualAgent
import copy

import numpy as np

from RL.agent_manual_simplified import ManualAgent
from agent_net import AgentNetPFNN, ContinousNetAgent

np.set_printoptions(precision=2)


class PFNNManualAgent(AgentNetPFNN, ManualAgent):
    def __init__(self, settings, net, grad_buffer,init_net=None, init_grad_buffer=None, goal_net=None, grad_buffer_goal=None):
        super(PFNNManualAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer, goal_net, grad_buffer_goal)

    def initial_position(self, pos, goal, current_frame=0, vel=0, init_dir=[],episode=None):

        # First we reset the position of the agent, set him a target position and desired speed
        # Remember: You can always change these values at runtime !
        dir_x = 0 if len(init_dir) < 1 else init_dir[0]
        dir_z = 0 if len(init_dir) < 2 else init_dir[1]
        init_dir=[dir_x, dir_z]
        super(PFNNManualAgent, self).initial_position(pos, goal, current_frame=current_frame, vel=vel, init_dir=init_dir,episode=episode)
        #self.PFNN.resetPosAndOrientation(0, 0, 0, dir_x, dir_z)
        self.init_pos = copy.deepcopy(self.pos_exact)
        self.distanceReachedThreshold = self.PFNN.getTargetReachedThreshold()
        print(("Reset position PFNN: " + str(self.pos_exact)))
        self.maxSpeedReached = 0
        self.poseData = []
        self.Time_at60fps = 0
        self.Time_inMySimulation = 0  # Let's say you have 30 fps in your simulation
        # print "Agent position in agent class: "+str(self.position)+" position "+str(pos)+" "+str(goal)

    def next_action(self, episode, training=True,viz=False):
        # print "Random action"
        # [4,1,0,3,6,7,8,5,2]
        print ("Manual agent get next action "+str(self.net))
        if self.net:
            if training:
                agent_frame = self.frame
            else:
                agent_frame = -1

            if self.frame % self.settings.action_freq == 0 or  episode.goal_person_id >= 0:
                print(("Agent input agent frame "+str(self.frame)+" "+str(agent_frame)))

                self.net.feed_forward(self.id, episode, self.current_init_frame + self.frame, training, agent_frame)
                print ("Model output "+str(episode.pedestrian_data[self.id].velocity[self.frame]))
                self.get_next_step(episode)

                return episode.pedestrian_data[self.id].velocity[self.frame]
            else:
                self.net.feed_forward(self.id,episode, self.current_init_frame + self.frame, training, agent_frame)
                self.update_episode(episode,episode.pedestrian_data[self.id].speed[self.frame-1],episode.pedestrian_data[self.id].action[self.frame-1], episode.pedestrian_data[self.id].velocity[self.frame-1], episode.pedestrian_data[self.id].probabilities[self.frame, :] )
                return episode.pedestrian_data[self.id].velocity[self.frame-1]

    def get_next_step(self, episode):

        # value = self.get_action(episode)
        # speed = float(eval(input("Next speed:")))
        try:
            if self.is_agent_alive(episode):
                # print "Random action"
                # [4,1,0,3,6,7,8,5,2]
                value_y = eval(input("Next action: y  test ------------------------------------"))
                #print("Input "+str(value_y)+" len "+str(len(value_y)))
                if len(value_y)==0:
                    return np.array(episode.pedestrian_data[self.id].velocity[self.frame])
                value_z = eval(input("Next action: z test "))
            else:
                print(" Agent dead")
                # input("Prese enter to continue: y ")

                value_y = 0
                value_z = 0
            episode.pedestrian_data[self.id].velocity[self.frame] = np.array([0, float(value_y), float(value_z)])
            episode.pedestrian_data[self.id].speed[self.frame] = np.sqrt(float(value_y) ** 2 + float(value_z) ** 2)
            print(("Action " + str(episode.pedestrian_data[self.id].velocity[self.frame])))
            episode.pedestrian_data[self.id].action[self.frame] = episode.find_action_to_direction(episode.pedestrian_data[self.id].velocity[self.frame],
                                                                                episode.pedestrian_data[self.id].speed[self.frame])
            episode.pedestrian_data[self.id].probabilities[self.frame, 0] = np.copy(episode.pedestrian_data[self.id].speed[self.frame])
            # episode.pedestrian_data[self.id].probabilities[self.frame, 1] = np.copy(angle)
            episode.pedestrian_data[self.id].probabilities[self.frame, 3] = np.copy(self.settings.sigma_vel)
            # episode.pedestrian_data[self.id].action[self.frame] = np.copy(angle)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            return np.array(episode.pedestrian_data[self.id].velocity[self.frame])
        return np.array(episode.pedestrian_data[self.id].velocity[self.frame])
        # value = self.get_action(episode)
        # speed = float(eval(input("Next speed:")))
        # self.update_episode(episode, speed, value)

    def train(self, ep_itr, statistics, episode, filename, filename_weights, poses, last_frame):
        # if self.frame % self.settings.action_freq == 0 or episode.goal_person_id >= 0:
        return self.net.train(self.id, ep_itr, statistics, episode, filename, filename_weights, poses, [], [], seq_len=last_frame)

# Rotating Agent

class ManualAgentContinousActions(ManualAgent):
    def __init__(self, settings, net, grad_buffer,init_net=None, init_grad_buffer=None, goal_net=None, grad_buffer_goal=None):
        print("ManualActionsContinousAgent")
        super(ManualAgentContinousActions, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer,goal_net, grad_buffer_goal)



    def next_action(self, episode, training=True, viz=False):
        # print "Random action"G137
        # [4,1,0,3,6,7,8,5,2]
        if self.net:
            if training:
                agent_frame = self.frame
            else:
                agent_frame = -1
            if self.frame % self.settings.action_freq == 0:
                # print(("Pedestrian id "+str(episode.goal_person_id)))
                print(("Agent input agent frame "+str(self.frame)+" "+str(agent_frame)))
                self.net.feed_forward(self.id,episode, self.current_init_frame + self.frame, training, agent_frame)
                print(("Agent takes step: " + str(episode.pedestrian_data[self.id].velocity[self.frame])))
                # if episode.goal_person_id<0:
                self.get_next_step(episode)
                return episode.pedestrian_data[self.id].velocity[self.frame]
            else:
                self.update_episode(episode,episode.pedestrian_data[self.id].speed[self.frame-1],episode.pedestrian_data[self.id].action[self.frame-1])
                return episode.pedestrian_data[self.id].velocity[self.frame-1]

    def get_next_step(self, episode):
        try:
            if self.is_agent_alive( episode):
                #print "Random action"
                #[4,1,0,3,6,7,8,5,2]
                value_y = eval(input("Next action: y "))
                value_z = eval(input("Next action: z "))
            else:
                print(" Agent dead")
                # input("Prese enter to continue: y ")

                value_y=0
                value_z=0
            episode.pedestrian_data[self.id].velocity[self.frame] = np.array([0,float(value_y),float(value_z)])
            episode.pedestrian_data[self.id].speed[self.frame] =np.sqrt(float(value_y)**2+float(value_z)**2)
            print(("Action " + str(episode.pedestrian_data[self.id].velocity[self.frame])))
            episode.pedestrian_data[self.id].action[self.frame] = episode.find_action_to_direction(episode.pedestrian_data[self.id].velocity[self.frame], episode.pedestrian_data[self.id].speed[self.frame] )
            episode.pedestrian_data[self.id].probabilities[self.frame, 0] = np.copy(episode.pedestrian_data[self.id].speed[self.frame] )
            #episode.pedestrian_data[self.id].probabilities[self.frame, 1] = np.copy(angle)
            episode.pedestrian_data[self.id].probabilities[self.frame, 3] = np.copy(self.settings.sigma_vel)
            #episode.pedestrian_data[self.id].action[self.frame] = np.copy(angle)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            return np.array(episode.pedestrian_data[self.id].velocity[self.frame])
        return np.array(episode.pedestrian_data[self.id].velocity[self.frame])

    def train(self,ep_itr,  statistics,episode, filename, filename_weights, poses, last_frame):
        return self.net.train(self.id, ep_itr,  statistics,episode, filename, filename_weights, poses,[], [], seq_len=last_frame)

class ManualActionsContinousAgent(ContinousNetAgent, ManualAgent):
    def __init__(self, settings, net, grad_buffer, init_net=None, init_grad_buffer=None, goal_net=None, grad_buffer_goal=None):
        print("ManualActionsContinousAgent")
        super(ManualActionsContinousAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer, goal_net, grad_buffer_goal)

    def next_action(self, episode, training=True, viz=False):
        # print "Random action"
        # [4,1,0,3,6,7,8,5,2]
        if self.net:
            if training:
                agent_frame = self.frame
            else:
                agent_frame = -1
            if self.frame % self.settings.action_freq == 0:
                print(("Agent input agent frame "+str(self.frame)+" "+str(agent_frame)))
                self.net.feed_forward(self.id,episode, self.current_init_frame + self.frame, training, agent_frame)

                self.get_next_step(episode)
                return episode.pedestrian_data[self.id].velocity[self.frame]
            else:
                self.update_episode(episode,episode.pedestrian_data[self.id].speed[self.frame-1],episode.pedestrian_data[self.id].action[self.frame-1])
                return episode.pedestrian_data[self.id].velocity[self.frame-1]

    def get_next_step(self, episode):
        if False:
            #print "Random action"
            #[4,1,0,3,6,7,8,5,2]
            value_y = eval(input("Next action: y "))
            value_z = eval(input("Next action: z "))
            episode.pedestrian_data[self.id].velocity[self.frame] = np.array([0,float(value_y),float(value_z)])
            episode.pedestrian_data[self.id].action[self.frame] = 0
            episode.pedestrian_data[self.id].speed[self.frame] =np.sqrt(float(value_y)**2+float(value_z)**2)
        else: # angular agent


            speed_value = float(eval(input("Next action speed: ")))
            angle =  float(eval(input("Next action angle: ")))*np.pi

            episode.pedestrian_data[self.id].speed[self.frame] = speed_value
            episode.pedestrian_data[self.id].velocity[self.frame] = np.zeros(3)

            episode.pedestrian_data[self.id].velocity[self.frame][1] = copy.copy(speed_value * np.cos(angle))
            episode.pedestrian_data[self.id].velocity[self.frame][2] = copy.copy(-speed_value * np.sin(angle))
            print(("Action " + str(episode.pedestrian_data[self.id].velocity[self.frame])))
            episode.pedestrian_data[self.id].action[self.frame] = 0
            episode.pedestrian_data[self.id].probabilities[self.frame, 0] = np.copy(speed_value)
            episode.pedestrian_data[self.id].probabilities[self.frame, 1] = np.copy(angle)
            episode.pedestrian_data[self.id].probabilities[self.frame, 3] = np.copy(self.settings.sigma_vel)
            episode.pedestrian_data[self.id].action[self.frame] = np.copy(angle)
        return np.array(episode.pedestrian_data[self.id].velocity[self.frame])



    def update_episode(self, episode, speed, value,velocity, probabilities):
        AgentNetPFNN.update_episode(self, episode, speed, value,velocity, probabilities)

    def train(self,ep_itr,  statistics,episode, filename, filename_weights, poses, last_frame):
        return self.net.train(self.id, ep_itr,  statistics,episode, filename, filename_weights, poses, seq_len=last_frame)

class ManualContinousAgent(ManualActionsContinousAgent, ManualAgent):
    def __init__(self, settings, net, grad_buffer,init_net=None, init_grad_buffer=None, goal_net=None, grad_buffer_goal=None):
        super(ManualContinousAgent, self).__init__(settings, net, grad_buffer,init_net, init_grad_buffer, goal_net, grad_buffer_goal)

class ManualActionsContinousAgent(ContinousNetAgent, ManualAgent):
    def __init__(self, settings, net, grad_buffer,init_net=None, init_grad_buffer=None, goal_net=None, grad_buffer_goal=None):
        print("ManualActionsContinousAgent")
        super(ManualActionsContinousAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer, goal_net, grad_buffer_goal)



    def next_action(self, episode, training=True, viz=False):
        # print "Random action"
        # [4,1,0,3,6,7,8,5,2]
        if self.net:
            if training:
                agent_frame = self.frame
            else:
                agent_frame = -1
            if self.frame % self.settings.action_freq == 0:
                print(("Agent input agent frame "+str(self.frame)+" "+str(agent_frame)))
                self.net.feed_forward(self.id,episode, self.current_init_frame + self.frame, training, agent_frame)

                self.get_next_step(episode)
                return episode.pedestrian_data[self.id].velocity[self.frame]
            else:
                self.update_episode(episode,episode.pedestrian_data[self.id].speed[self.frame-1],episode.pedestrian_data[self.id].action[self.frame-1], episode.pedestrian_data[self.id].velocity[self.frame-1],episode.pedestrian_data[self.id].probabilities[self.frame-1,:])
                return episode.pedestrian_data[self.id].velocity[self.frame-1]

    def get_next_step(self, episode):
        if False:
            #print "Random action"
            #[4,1,0,3,6,7,8,5,2]
            value_y = eval(input("Next action: y "))
            value_z = eval(input("Next action: z "))
            episode.pedestrian_data[self.id].velocity[self.frame] = np.array([0,float(value_y),float(value_z)])
            episode.pedestrian_data[self.id].action[self.frame] = 0
            episode.pedestrian_data[self.id].speed[self.frame] =np.sqrt(float(value_y)**2+float(value_z)**2)
        else: # angular agent


            speed_value = float(eval(input("Next action speed: ")))
            angle =  float(eval(input("Next action angle: ")))*np.pi

            episode.pedestrian_data[self.id].speed[self.frame] = speed_value
            episode.pedestrian_data[self.id].velocity[self.frame] = np.zeros(3)

            episode.pedestrian_data[self.id].velocity[self.frame][1] = copy.copy(speed_value * np.cos(angle))
            episode.pedestrian_data[self.id].velocity[self.frame][2] = copy.copy(-speed_value * np.sin(angle))
            print(("Action " + str(episode.pedestrian_data[self.id].velocity[self.frame])))
            episode.pedestrian_data[self.id].action[self.frame] = 0
            episode.pedestrian_data[self.id].probabilities[self.frame, 0] = np.copy(speed_value)
            episode.pedestrian_data[self.id].probabilities[self.frame, 1] = np.copy(angle)
            episode.pedestrian_data[self.id].probabilities[self.frame, 3] = np.copy(self.settings.sigma_vel)
            episode.pedestrian_data[self.id].action[self.frame] = np.copy(angle)
        return np.array(episode.pedestrian_data[self.id].velocity[self.frame])



    def update_episode(self, episode, speed, value,velocity, probabilities):
        AgentNetPFNN.update_episode(self, episode, speed, value,velocity, probabilities)

    def train(self, ep_itr, statistics,episode, filename, filename_weights, poses, last_frame):
        return self.net.train(self.id,  ep_itr, statistics,episode, filename, filename_weights, poses, seq_len=last_frame)
