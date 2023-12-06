import numpy as np

from initializer_net import InitializerNet
from RL.initializer_gaussian_net import InitializerGaussianNet
from settings import NBR_MEASURES,STATISTICS_INDX,STATISTICS_INDX_MAP,PEDESTRIAN_MEASURES_INDX
import tensorflow as tf
from settings import RANDOM_SEED
tf.random.set_seed(RANDOM_SEED)
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils.utils_functions import get_goal_frames


class GoalNet(InitializerNet):
    def __init__(self, settings, weights_name="goal") :
        super(GoalNet, self).__init__(settings, weights_name="goal")

    def define_loss(self, dim_p):
        self.goal = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="sample")
        self.responsible_output = tf.slice(self.probabilities, self.goal, [1]) + np.finfo(
            np.float32).eps
        self.prior_flat = tf.reshape(self.prior, [
            self.settings.batch_size * self.settings.env_shape[1] * self.settings.env_shape[2]])
        self.responsible_output_prior = tf.slice(self.prior_flat, self.goal, [1]) + np.finfo(
            np.float32).eps
        self.distribution = self.prior_flat * self.probabilities

        self.loss = -tf.reduce_mean(
            (tf.math.log(self.responsible_output) + tf.math.log(self.responsible_output_prior)) * self.advantages)

        if self.settings.entr_par_goal:
            y_zeros = tf.zeros_like(self.distribution)
            y_mask = tf.math.greater(self.distribution, y_zeros)
            res = tf.boolean_mask(self.distribution, y_mask)
            logres = tf.math.log(res)

            self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=res, logits=logres)
            self.loss = -tf.reduce_mean(self.settings.entr_par_init * self.entropy)
        if self.settings.learn_time :
            self.time_requirement = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 1], dtype=self.DTYPE,
                                                        name="time_requirement")
            self.beta_distr=tfd.Beta(self.alpha, self.beta)
            self.time_prob=self.beta_distr.prob(self.time_requirement)
            self.loss = self.loss - tf.reduce_mean(self.advantages * tf.log(self.time_prob))
        return self.goal, self.loss


    #return [statistics[ep_itr, agent_frame, 6]]
    def get_sample(self,id, statistics,ep_itr, agent_frame,  initialization_car):
        if self.settings.printdebug_network:
            print ("Get sample "+str([np.ravel_multi_index( statistics[ep_itr,id, agent_frame, STATISTICS_INDX.goal[0]:STATISTICS_INDX.goal[1]].astype(int),
                                    self.settings.env_shape[1:])]))
        return [np.ravel_multi_index( statistics[ep_itr,id, agent_frame, STATISTICS_INDX.goal[0]:STATISTICS_INDX.goal[1]].astype(int),
                                    self.settings.env_shape[1:])]


    def apply_net(self,id, feed_dict, episode, frame, training, max_val=False, viz=False, manual=False):

        flat_prior=episode.calculate_goal_prior(id, frame).flatten()
        if self.settings.printdebug_network:
            print ("Max value of goal prior "+str( max(flat_prior)))
        feed_dict[self.prior] = np.expand_dims(np.expand_dims(episode.initializer_data[id].goal_priors[frame] * (1 / max(flat_prior)), axis=0), axis=-1)
        if self.settings.printdebug_network:
            print ("goal prior shape " + str(feed_dict[self.prior].shape))
            print ("Goal prior min "+str(np.min(flat_prior))+" max "+str(np.max(flat_prior)))
            print ("After normalizing goal prior min " + str(np.min(feed_dict[self.prior][0, :, :, 0])) + " max " + str(np.max(feed_dict[self.prior][0, :, :, 0])))
        if self.settings.learn_time:
            probabilities, flattend_layer, conv_1,alpha, beta, summary_train = self.sess.run(
                [self.probabilities, self.flattened_layer, self.conv1,self.alpha, self.beta, self.train_summaries], feed_dict)
            episode.pedestrian_data[id].goal_time[frame] = np.random.beta(alpha, beta)
            if self.settings.printdebug_network:
                print ("Model outputs alpha " + str(alpha) + " beta " + str(beta) + " factor " + str(episode.goal_time))

        else:
            probabilities,flattend_layer,conv_1, summary_train = self.sess.run([self.probabilities,self.flattened_layer,self.conv1, self.train_summaries], feed_dict)

        episode.initializer_data[id].goal_distributions[frame] = np.copy(probabilities)
        if self.settings.printdebug_network:
            pos_max=np.argmax(episode.initializer_data[id].goal_distributions[frame])
            pos_min = np.argmin(episode.initializer_data[id].goal_distributions[frame])

            print ("Init distr "+str(np.sum(np.abs(episode.init_distribution)))+"  max: "+str(pos_max)+" pos in 2D: "+str(np.unravel_index(pos_max, self.settings.env_shape[1:])))
            print ("Init distr  min pos "+ str(pos_min) +" pos in 2D: "+str(np.unravel_index(pos_min, self.settings.env_shape[1:])))
            print (" max value "+str(episode.initializer_data[id].goal_distribution[pos_max])+"  min value "+str(episode.initializer_data[id].goal_distribution[pos_min]) )

            reshaped_init=np.reshape(episode.initializer_data[id].goal_distributions[frame], episode.prior.shape)

            print ("10th row:"+str(reshaped_init[10,:10]))
            print("Max of flattened layer "+str(max(flattend_layer))+" min of flattened layer "+str(min(flattend_layer)))
            print ("10th row of conv "+str(conv_1[0,10,:10,0]))

        if self.settings.printdebug_network:
            pos_max = np.argmax(flat_prior)
            pos_min = np.argmin(flat_prior)
            print ("Prior distr " + str(np.sum(np.abs(flat_prior))) + "  max: " + str(
                pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:])))
            print ("Prior distr  min pos " + str(pos_min) + " pos in 2D: " + str(
                np.unravel_index(pos_min, self.settings.env_shape[1:])))
            print (" max value " + str(flat_prior[pos_max]) + "  min value " + str(flat_prior[pos_min]))

            print ("10th row:" + str(episode.prior[10, :10]))

        probabilities = probabilities*flat_prior#episode.prior.flatten()
        if self.settings.printdebug_network:
            probabilities_reshaped = np.reshape(probabilities, episode.prior.shape)
            pos_max = np.argmax(probabilities)
            pos_min = np.argmin(probabilities)
            for car in episode.cars[0]:
                print ("Probabilities reshaped " + str(car))
                print (np.sum(np.abs(probabilities_reshaped[car[2]:car[3], car[4]:car[5]])))

            pos_max = np.argmax(probabilities)
            print ("After prior Init distr " + str(np.sum(np.abs(probabilities))) + "  max: " + str(
                pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:])))



        probabilities=probabilities*(1/np.sum(probabilities))
        if self.settings.printdebug_network:
            print ("Final distr " + str(np.sum(np.abs(probabilities))) + "  max: " + str(
                pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:])))
            print ("Final distr  min pos " + str(pos_min) + " pos in 2D: " + str(
                np.unravel_index(pos_min, self.settings.env_shape[1:])))
            print (" max value " + str(probabilities[pos_max]) + "  min value " + str(probabilities[pos_min]))

        pos_max = np.argmax(probabilities)

        if max_val and not training:
            #print ("Maximal evaluation init net")
            pos = np.unravel_index(pos_max, self.settings.env_shape[1:])
            episode.goal[0] = episode.get_height_init()
            episode.goal[1] = pos[0]
            episode.goal[2] = pos[1]
        else:
            if self.settings.printdebug_network:
                print ("After normalization " + str(np.sum(np.abs(probabilities))) + "  max: " + str(
                    pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:]))+" no prior max "+str(episode.init_distribution[pos_max]))

            indx = np.random.choice(range(len(probabilities)), p=np.copy(probabilities))
            pos = np.unravel_index(indx, self.settings.env_shape[1:])
            episode.pedestrian_data[id].goal[frame,0] = episode.get_height_init()
            episode.pedestrian_data[id].goal[frame,1] = pos[0]
            episode.pedestrian_data[id].goal[frame,2] = pos[1]
        if self.settings.speed_input:
            if self.settings.learn_time:
                episode.pedestrian_data[id].goal_time[frame] = episode.pedestrian_data[id].goal_time[frame] * 3 * 5 * episode.frame_time
                if self.settings.printdebug_network:
                    print ("Episode speed " + str(episode.pedestrian_data[id].goal_time[frame]) + " factor " + str(
                        15 * episode.frame_time) + " frametime " + str(episode.frame_time))
                    episode.pedestrian_data[id].goal_time[frame] = np.linalg.norm(episode.pedestrian_data[id].goal[frame,1:] - episode.pedestrian_data[id].agent[frame][1:]) / episode.pedestrian_data[id].goal_time[frame]
                    print ("Episode goal time " + str(episode.pedestrian_data[id].goal_time[frame]))
            else:
                episode.pedestrian_data[id].goal_time[frame] = min(np.linalg.norm(episode.pedestrian_data[id].goal[frame,1:] - episode.pedestrian_data[id].agent[frame][1:]) / episode.pedestrian_data[id].speed_init,
                                        episode.seq_len - 2)
        if self.settings.printdebug_network:
            print ("Goal "+str(episode.pedestrian_data[id].goal))
        return episode.pedestrian_data[id].agent[frame], 11,episode.pedestrian_data[id].vel_init #  pos, indx, vel_init


    def grad_feed_dict(self,id,  agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode, frame,
                       reward, statistics,poses,priors,initialization_car, agent_speed=None, training=True, agent_frame=-1, goal_frames=[]):
        if agent_frame < 0 or not training:
            agent_frame = frame
        r=reward[ep_itr, agent_frame]
        # print  ("Reward "+str(r)+" rewards "+str(reward[ep_itr, :]))
        if sum(agent_measures[ep_itr, :agent_frame, PEDESTRIAN_MEASURES_INDX.agent_dead])>0:
            r=0
        # print ("Frame "+str(frame)+" Episode "+str(ep_itr)+" Reward "+str(r)+" hit by car:"+str(agent_measures[ep_itr,agent_frame, 0])+" Reached goal "+str(agent_measures[ep_itr,agent_frame, 13]))

        feed_dict = {self.state_in: self.get_input_init(id,episode, frame),
                     self.advantages: r,
                     self.goal: self.get_sample(id, statistics, ep_itr, agent_frame,  initialization_car)}
        local_frame=np.argwhere(goal_frames==frame)
        prior=priors[num_episode][id][local_frame][:,STATISTICS_INDX_MAP_GOAL.goal_prior]
        feed_dict[self.prior]= np.reshape(prior*(1.0/max(prior)), (self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2],1))
        if self.settings.learn_time and self.goal_net:
            goal = statistics[ep_itr, frame, STATISTICS_INDX.goal[0]:STATISTICS_INDX.goal[1]]
            agent_pos = statistics[ep_itr, frame, STATISTICS_INDX.agent_pos[0]:STATISTICS_INDX.agent_pos[1]]
            goal_dist = np.linalg.norm(goal - agent_pos)
            goal_time = statistics[ep_itr, frame, STATISTICS_INDX.goal_time]
            if self.settings.printdebug_network:
                print("Agent position " + str(agent_pos) + " goal " + str(goal))
                print ("Episode goal time " + str(goal_time) + " goal time " + str(goal_dist) + " fraction " + str(
                    goal_dist / goal_time) + " ratio " + str(17 / 15))
            if goal_time == 0:
                feed_dict[self.time_requirement] = np.array([[0]])
            else:
                feed_dict[self.time_requirement] = np.array([[goal_dist / goal_time * (17 / 15)]])
            if self.settings.printdebug_network:
                print ("Feed dict input " + str(feed_dict[self.time_requirement]))
        return feed_dict


    def train(self,id,  ep_itr, statistics, episode, filename, filename_weights,poses, priors,initialization_car,statistics_car, seq_len=-1):

        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_vel = self.stats(id, statistics,statistics_car, initializer=True)


        if self.settings.normalize:
            reward=self.normalize_reward(agent_reward_d, agent_measures)
        else:
            reward=agent_reward_d
        self.reset_mem()
        goal_frames = get_goal_frames(statistics,ep, id)
        for frame in goal_frames:
            feed_dict = self.grad_feed_dict(id,agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                             episode, frame, reward, statistics,poses,priors,  initialization_car,agent_speed=agent_vel, goal_frames=goal_frames)
            self.do_gradient_debug_printout(id,ep_itr, feed_dict)

            self.calculate_gradients( id, feed_dict, statistics, frame, ep_itr) #                 self.calculate_gradients(episode, frame, feed_dict) %


        if ep_itr==statistics.shape[0]-1 and id==self.settings.number_of_agents-1:
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



    def evaluate(self, id, ep_itr, statistics, episode, poses, priors,initialization_car, statistics_car, seq_len=-1):
        if seq_len == -1:
            seq_len = self.settings.seq_len_test
        # print "Evaluate"
        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_vel = self.stats(id,statistics, statistics_car, initializer=True)
        reward = agent_reward_d  # np.zeros_like(agent_reward)


        self.reset_mem()
        goal_frames = get_goal_frames(statistics, ep, id)
        for frame in goal_frames:
            feed_dict = self.grad_feed_dict(id, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                         episode, frame, reward, statistics,poses,priors,  initialization_car,agent_speed=agent_vel, goal_frames=goal_frames)
            loss, summaries_loss, responsible_output,responsible_output_prior  = self.sess.run([self.loss, self.loss_summaries, self.responsible_output, self.responsible_output_prior], feed_dict)  # self.merged
            # episode.loss[frame] = loss
            self.save_loss(ep_itr, id, loss, statistics, statistics_car, frame)
            if self.writer:
                self.test_writer.add_summary(summaries_loss, global_step=self.num_grad_itrs)
            if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
                self.feed_dict = []
        return statistics