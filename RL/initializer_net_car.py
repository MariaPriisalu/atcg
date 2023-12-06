import numpy as np

from initializer_net import InitializerNet
from settings import NBR_MEASURES,RANDOM_SEED, STATISTICS_INDX_MAP,STATISTICS_INDX_CAR_INIT,STATISTICS_INDX,PEDESTRIAN_MEASURES_INDX,STATISTICS_INDX_CAR, STATISTICS_INDX_MAP_CAR

import tensorflow as tf

tf.random.set_seed(RANDOM_SEED)
import tensorflow_probability as tfp
tfd = tfp.distributions

import pickle

import copy


class InitializerNetCar(InitializerNet):
    def __init__(self, settings, weights_name="place_car") :
        super(InitializerNetCar, self).__init__(settings, weights_name=weights_name)

    def get_sample(self, id, statistics,ep_itr, agent_frame,  initialization_car):
        # Check that statistics_car is used
        #print("Get sample "+str(([int(statistics[ep_itr,id, 0,STATISTICS_INDX_CAR.agent_pos[0]+1])], [int(statistics[ep_itr,id, 0,STATISTICS_INDX_CAR.agent_pos[0]+2])])) +" size "+str(self.settings.env_shape[1:]) )
        return np.ravel_multi_index(([int(statistics[ep_itr,id, 0,STATISTICS_INDX_CAR.agent_pos[0]+1])], [int(statistics[ep_itr,id, 0,STATISTICS_INDX_CAR.agent_pos[0]+2])]), self.settings.env_shape[1:] )

    def apply_net(self,id,  feed_dict, episode, frame, training, max_val=False, viz=False, manual=False):
        episode.calculate_car_prior(id, self.settings.field_of_view_car)
        probabilities, probabilities_goal = self.run_net(episode, feed_dict, id, manual,episode.initializer_car_data[id].prior)
        episode.car_data[id].car_dir=episode.valid_directions_cars[ int(episode.car_data[id].car[0][1]) ,  int(episode.car_data[id].car[0][2]),: ]
        episode.car_data[id].car_goal=np.zeros(3)
        episode.car_data[id].car_goal[0]=episode.car_data[id].car[0][0]
        episode.car_data[id].car_goal[1:]=self.find_end_of_scene_along_direction( episode.car_data[id].car[0][1:], episode.car_data[id].car_dir[1:], episode.reconstruction.shape[1:])
        #  car_pos, goal_car, car_dir, car_key, online_car_key, on_car
        return episode.car_data[id].car[0], episode.car_data[id].car_goal,episode.car_data[id].car_dir, -1, -1,False #  pos, indx, vel_init

    def find_end_of_scene_along_direction(self, pos, vel, dims):
        pos_cur=pos.copy()
        speed=np.linalg.norm(vel)
        if speed<1e-5:
            print("Car's goal is too close !? Give a random goal?")
            return [0,0]
        else:
            vel=vel*1/speed
        counter=0
        while(pos_cur[0]>0 and pos_cur[1]>0 and  pos_cur[0] <dims[0] and pos_cur[1]<dims[1]) and counter< max(dims):
            pos_cur=pos_cur+vel
            counter=counter+1
        return pos_cur

    def place_pos_in_episode(self, episode, id, pos):
        episode.car_data[id].car[0][0] = episode.get_height_init()
        episode.car_data[id].car[0][1] = pos[0]
        episode.car_data[id].car[0][2] = pos[1]

    def place_initializer_in_episode(self, episode, id, probabilities):
        episode.initializer_car_data[id].init_distribution = np.copy(probabilities)

    def evaluate(self, id, ep_itr, statistics, episode, poses, priors,initialization_car, statistics_car, seq_len=-1):
        if seq_len == -1:
            seq_len = self.settings.seq_len_test
        # print "Evaluate"
        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_vel = self.stats(id,statistics, statistics_car, initializer=True)
        reward = agent_reward_d  # np.zeros_like(agent_reward)


        self.reset_mem()
        feed_dict = self.grad_feed_dict(id, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                     episode, 0, reward, statistics_car,poses,priors,  initialization_car,agent_speed=agent_vel)
        loss, summaries_loss, responsible_output,responsible_output_prior  = self.sess.run([self.loss, self.loss_summaries, self.responsible_output, self.responsible_output_prior], feed_dict)  # self.merged
        # episode.loss[frame] = loss
        self.save_loss(ep_itr, id, loss, statistics, statistics_car)
        if self.writer:
            self.test_writer.add_summary(summaries_loss, global_step=self.num_grad_itrs)
        if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
            self.feed_dict = []
        return statistics

    def train(self,id,  ep_itr, statistics, episode, filename, filename_weights,poses, priors,initialization_car,statistics_car, seq_len=-1):

        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_vel = self.stats(id, statistics,statistics_car, initializer=True)


        if self.settings.normalize:
            reward=self.normalize_reward(agent_reward_d, agent_measures)
        else:
            reward=agent_reward_d
        self.reset_mem()



        feed_dict = self.grad_feed_dict(id,agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                         episode, 0, reward, statistics_car,poses,priors,  initialization_car,agent_speed=agent_vel)
        self.do_gradient_debug_printout(id,ep_itr, feed_dict)

        self.calculate_gradients( id, feed_dict, statistics_car, 0, ep_itr) #                 self.calculate_gradients(episode, frame, feed_dict) %


        if ep_itr==statistics.shape[0]-1 and id==self.settings.number_of_car_agents-1:
            if not self.settings.overfit or( self.settings.overfit and self.num_grad_itrs%20==0):
                with open(filename, 'wb') as f:
                    pickle.dump(self.gradBuffer, f, pickle.HIGHEST_PROTOCOL)
                [weights]=self.sess.run([self.tvars])
                with open(filename_weights, 'wb') as f:
                    pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
            else:
                [weights] = self.sess.run([self.tvars])
            self.update_gradients()

            if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):

                self.feed_dict=[]
        return statistics

    def grad_feed_dict(self,id,  agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode, frame,
                       reward, statistics,poses,priors,initialization_car, agent_speed=None, training=True, agent_frame=-1):
        if agent_frame < 0 or not training:
            agent_frame = frame
        r=reward[ep_itr, 0]
        #print  ("Reward "+str(r)+" rewards "+str(reward[ep_itr, :]))
        if agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.agent_dead]:
            return {}
        # print ("Frame "+str(frame)+" Episode "+str(ep_itr)+" Reward "+str(r)+" hit by car:"+str(agent_measures[ep_itr,agent_frame, 0])+" Reached goal "+str(agent_measures[ep_itr,agent_frame, 13]))

        feed_dict = {self.state_in: self.get_input_init(id,episode),
                     self.advantages: r,
                     self.sample: self.get_sample(id,statistics, ep_itr, agent_frame,  initialization_car)}

        feed_dict[self.prior] = np.reshape(priors[ep_itr,id, :,STATISTICS_INDX_MAP_CAR.prior]*(1.0/max(priors[ep_itr,id,:,STATISTICS_INDX_MAP_CAR.prior])), (self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2],1))

        return feed_dict

    def save_loss(self, ep_itr, id, loss, statistics, statistics_car):
        statistics_car[ep_itr, id, 0, STATISTICS_INDX_CAR.loss_initializer] = loss

