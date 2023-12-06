import sys

import numpy as np


from RL.agent_net import NetAgent

class ManualAgent(NetAgent):
    def __init__(self, settings, net=None, grad_buffer=None ,init_net=None, init_grad_buffer=None, goal_net=None, grad_buffer_goal=None):
        super(ManualAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer, goal_net, grad_buffer_goal)
        self.position=np.zeros(3, dtype=np.float)
        self.frame=0



    def next_action(self,episode, training=True,viz=False):
        value = self.get_action(episode)

        speed = float(eval(input("Next speed:")))

        self.update_episode(episode, speed, value, None, None)

        return episode.pedestrian_data[self.id].velocity[self.frame]

    def update_episode(self, episode, speed, value,velocity, probabilities):
        episode.pedestrian_data[self.id].probabilities[self.frame, value] = 1
        episode.pedestrian_data[self.id].velocity[self.frame] = np.array(episode.pedestrian_data[self.id].actions[value])
        episode.pedestrian_data[self.id].action[self.frame] = value
        if self.settings.velocity:
            if np.linalg.norm(episode.pedestrian_data[self.id].velocity[self.frame]) != 1 and np.linalg.norm(
                    episode.pedestrian_data[self.id].velocity[self.frame]) > 0.1:
                episode.pedestrian_data[self.id].velocity[self.frame] = episode.pedestrian_data[self.id].velocity[self.frame] / np.linalg.norm(
                    episode.pedestrian_data[self.id].velocity[self.frame])
            if not self.settings.acceleration:
                print("Velocity "+str(episode.pedestrian_data[self.id].velocity[self.frame])+" speed: "+str(speed))
                episode.pedestrian_data[self.id].velocity[self.frame] = episode.pedestrian_data[self.id].velocity[self.frame] * speed * 5 / episode.frame_rate
        elif self.settings.controller_len > 0:
            episode.pedestrian_data[self.id].velocity[self.frame] = np.array(episode.actions[value]) * (speed + 1)

    def get_action(self, episode):
        if np.linalg.norm(episode.actions[0]) == 0:
            value = eval(input(
                "Next action: [1:'stand',2:'down',3:'downL'4:'left',5:'upL',6:'up', 7:'upR', 8:'right', 9:'downR', 0: stop excecution] "))
        else:
            value = eval(input(
                "Next action: [ 1:'downL', 2:'down', 3:'downR', 4:'left', 5:'stand', 6:'right',7:'upL', 8:'up', 9:'upR', 0: stop excecution] "))
        value = int(value) - 1
        if value == -1:
            sys.exit(0)
        return value

    def train(self,ep_itr,  statistics,episode, filename, filename_weights, poses, last_frame):
        return statistics


    def set_session(self, session):
        self.net.sess = session

    def evaluate(self,ep_itr,  statistics, episode, poses, last_frame):
        return statistics