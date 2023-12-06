import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow.compat.v1.nn.rnn_cell as rnn

from settings import RANDOM_SEED,RANDOM_SEED_NP
tf.random.set_seed(RANDOM_SEED)
#from tensorflow.contrib import rnn

from net import Net, variable_summaries
from settings import NBR_MEASURES,POSE_DIM, PEDESTRIAN_REWARD_INDX,STATISTICS_INDX,STATISTICS_INDX_POSE, PEDESTRIAN_MEASURES_INDX, run_settings


class SoftMaxNet(Net):
    def __init__(self,settings):
        self.action_size_mem=9+1
        super(SoftMaxNet, self).__init__( settings)
        #self.merge_summaries()
        np.random.seed(RANDOM_SEED_NP)
        tf.random.set_seed(RANDOM_SEED)
        self.actions = []

        v = [-1, 0, 1]

        j=0
        if self.settings.actions_3d:
            for z in range(3):
                for y in range(3):
                    for x in range(3):

                        self.actions.append([v[x], v[y], v[z]])
                        j+=1
        else:
            for y in range(3):
                for x in range(3):

                    self.actions.append([0, v[y],v[x]])
                    j += 1
        j=0
        if settings.reorder_actions:

            self.actions = [self.actions[k] for k in [4, 1, 0, 3, 6, 7, 8, 5, 2]]

    def define_loss(self, dim_p):
        y_zeros = tf.zeros_like(self.probabilities)
        y_mask = tf.math.greater(self.probabilities, y_zeros)
        #self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(res), logits=logres)
        res = tf.boolean_mask(self.probabilities, y_mask)
        logres = tf.math.log(res)

        self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=res, logits=logres)
        #self.entropy = - tf.reduce_sum(self.mu * tf.math.log(self.mu))
        self.sample = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="sample")
        self.responsible_output = tf.slice(self.probabilities, self.sample, [1])+ np.finfo(
                np.float32).eps
        self.loss = -tf.reduce_mean(input_tensor=tf.math.log(self.responsible_output) * self.advantages)
        if self.settings.entr_par>0:
            self.loss = -tf.reduce_mean(input_tensor=tf.math.log(self.responsible_output) * self.advantages+self.settings.entr_par * self.entropy)
        if self.settings.controller_len > 0:
            self.sample_c = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="sample_c")
            self.responsible_output_c = tf.slice(self.probabilities_c, self.sample_c, [1]) * np.finfo(
                np.float32).eps
            self.loss = tf.reduce_mean(input_tensor=(-tf.math.log(self.responsible_output)-tf.math.log(self.responsible_output_c))*self.advantages+self.settings.entr_par*self.entropy)
        # loglik = tf.log(self.sample * (self.sample - probability) + (1 - self.sample) * (self.sample + probability))
        # loss = -tf.reduce_mean(loglik * self.advantages)
        if self.settings.velocity:
            self.velocity = tf.compat.v1.placeholder(shape=[1], dtype=self.DTYPE,name="velocity")
            if self.settings.velocity_sigmoid:

                self.normal = tfd.Normal(loc=self.probabilities_v, scale=self.settings.sigma_vel)
                import math as m
                pi = tf.constant(m.pi)
                self.probability_of_velocity =self.normal.prob(self.velocity)
                self.l2_loss=tf.nn.l2_loss(self.velocity-self.probabilities_v)
                self.normal_loss=-tf.math.log(self.settings.sigma_vel)-(0.5*tf.math.log(2*pi))-(self.l2_loss/(self.settings.sigma_vel*self.settings.sigma_vel))
                self.policy_loss =tf.math.log(self.responsible_output)
                self.loss = -tf.reduce_mean(input_tensor=self.advantages*(self.policy_loss+self.normal_loss))

            else:
                #self.cat = tfd.Categorical(probs=self.mu_vel),
                self.mus = [0.2, 1.4, 2.5, 3.7]
                self.var = [0.1, 0.2, 0.1, 0.4]
                self.normal = tfd.Normal(loc=[0.2, 1.4, 2.5, 3.7], scale=[0.1, 0.2, 0.1, 0.4])

                self.probability_of_velocity = tf.reduce_sum(input_tensor=tf.multiply(self.normal.prob(self.velocity), self.probabilities_v))
                self.loss = -tf.reduce_mean(input_tensor=tf.math.log(self.responsible_output) * self.advantages + tf.math.log(
                    self.probability_of_velocity) * self.advantages)
        if self.settings.detect_turning_points:
            self.loss=self.loss+ (0.1*self.probabilities_point)
        return self.sample, self.loss



    def importance_sample_weight(self,id, responsible, statistics, ep_itr, frame,responsible_v=0):
        probabilities=statistics[ep_itr,id, frame,STATISTICS_INDX.probabilities[0]:STATISTICS_INDX.probabilities[1]]
        action=statistics[ep_itr,id, frame,STATISTICS_INDX.action]
        #print action

        if self.settings.velocity:
            mu=probabilities[9]
            sigma=probabilities[10]
            vel=statistics[ep_itr,id, frame,STATISTICS_INDX.speed]

            return np.exp(np.log(responsible)- np.log(probabilities[int(action)])+np.log( responsible_v) - np.log(self.normalpdf(mu,sigma, vel)))
        return np.exp(np.log(responsible)- np.log(probabilities[int(action)]))

    # def apply_ablation_on_input(self,  feed_dict, episode, frame, ablation):
    #     if ablation%2==0:
    #         feed_dict[self.state_in]=feed_dict[np.swapaxes(self.state_in, 1,2)]
    #         for
    #     if ablation % 3 == 0:
    #
    #     if ablation % 5 == 0:
    def normal_dist(self, x, mean, sd):
        prob_density = np.exp(-0.5 * ((x - mean) / sd) ** 2)/(sd*np.sqrt(2*np.pi))
        return prob_density

    def log_normal_dist(self, x, mean, sd):
        l2=(x-mean)**2
        return -log(sd )-(0.5 * tf.math.log(2 * pi)) - (l2/ (sd * sd))


    def apply_net(self,id,  feed_dict, episode, frame, training, max_val=False, viz=False,manual=False):
        attention=1
        if self.settings.velocity:
            if self.settings.velocity_sigmoid:
                # probabilities, mean_vel,sigma_vel, summary_train = self.sess.run(
                #     [self.probabilities, self.probabilities_v,self.sigm_vel, self.train_summaries], feed_dict)
                if self.settings.detect_turning_points:
                    probabilities, mean_vel, turning_point, summary_train = self.sess.run(
                        [self.probabilities, self.probabilities_v,self.probabilities_point, self.train_summaries], feed_dict)
                elif self.settings.attention:
                    probabilities, mean_vel, attention, summary_train = self.sess.run(
                        [self.probabilities, self.probabilities_v, self.probabilities_att, self.train_summaries],
                        feed_dict)
                else:
                    # probabilities, mean_vel, summary_train = self.sess.run(
                    #     [self.probabilities, self.probabilities_v, self.train_summaries], feed_dict)
                    probabilities,mean_vel ,state_in, conv_1_out, conv_1_after_relu, conv_2, conv_2_relu, conv_out, summary_train = self.sess.run(
                        [self.probabilities, self.probabilities_v,self.state_in, self.bias1, self.conv1, self.conv_out2, self.conv2,
                         self.conv_output,
                         self.train_summaries], feed_dict)

                    #print("mean_vel " + str(mean_vel) + " probabilities " + str(probabilities))
                    # print("State in output size " + str(state_in.shape))
                    # for i in range(state_in.shape[-1]):
                    #     print("Input " + str(i) + " input ------------ ")
                    #     print(state_in[:,:,:, i])
                    # if len(self.save_probabilities)<=frame:
                    #     self.save_probabilities.append({})
                    # else:
                    #     self.save_probabilities[frame]={}
                    # self.save_probabilities[frame]['probabilities']=probabilities
                    # self.save_probabilities[frame]['mean_vel'] = mean_vel


                        # print ("mean_vel "+str(mean_vel)+" probabilities "+str(probabilities))
                if max_val and (not training or self.settings.learn_init):
                    speed_value = mean_vel
                    if self.settings.detect_turning_points:
                        episode.pedestrian_data[id].turning_point[frame] =turning_point-0.5
                    #print ("Max speed " + str(speed_value))
                else:
                    speed_value =np.random.normal(mean_vel, self.settings.sigma_vel)[0]
                    # print ("Random speed value "+str(speed_value))
                    speed_value =max(min(speed_value, 3), 0.1)
                    # print("Random speed after minmax " + str(speed_value) + " probability " + str(
                    #     self.normal_dist(speed_value, mean_vel, self.settings.sigma_vel)) + " mean " + str(
                    #     mean_vel) + " std " + str(self.settings.sigma_vel)+ " probability log " + str(
                    #     np.log(self.normal_dist(speed_value, mean_vel, self.settings.sigma_vel))))
                    # self.save_probabilities[frame]['speed_sample'] = speed_value
                    # self.save_probabilities[frame]['speed_probability'] = self.normal_dist(speed_value, mean_vel, self.settings.sigma_vel)
                    if self.settings.detect_turning_points:
                        episode.pedestrian_data[id].turning_point[frame] = np.random.uniform(0, 1)-turning_point
                #print "Model output"+str(mean_vel)+" speed sample: "+str( speed_value)


                episode.pedestrian_data[id].probabilities[frame, len(probabilities)] = np.copy(mean_vel)
                episode.pedestrian_data[id].probabilities[frame, len(probabilities) + 1] = np.copy(self.settings.sigma_vel)

                episode.pedestrian_data[id].speed[frame] = speed_value

            else:


                probabilities, probability_of_velocity, summary_train = self.sess.run(
                    [self.probabilities, self.probabilities_v, self.train_summaries], feed_dict)

                if abs(sum(probability_of_velocity) - 1.0) > 1e-7 and sum(probability_of_velocity) > np.finfo(
                        np.float32).eps:
                    probability_of_velocity = probability_of_velocity / sum(probability_of_velocity)

                distr = np.random.choice(list(range(len(probability_of_velocity))), p=np.copy(probability_of_velocity))

                episode.pedestrian_data[id].probabilities[frame,
                len(probabilities):len(probabilities) + len(probability_of_velocity)] = np.copy(probability_of_velocity)

                speed_value = np.random.normal(self.mus[distr], self.var[distr], 1)

                episode.pedestrian_data[id].speed[frame] = speed_value
                # print "Speed drawn:"+str(speed_value)+" actual speed: "+str(episode.speed[frame])+" probabilities: "+str(probability_of_velocity)+"  distribution: "+str(distr)+" "+str(self.mus[distr])+" "+str( self.var[distr])

        elif self.settings.controller_len <= 0:
            if self.settings.attention:
                probabilities,  attention, summary_train = self.sess.run(
                    [self.probabilities, self.probabilities_att, self.train_summaries], feed_dict)
                # print(("Attention: "+str(attention)))
            else:
                probabilities, summary_train = self.sess.run([self.probabilities, self.train_summaries], feed_dict)
                # print("Network output " + str(probabilities))
        else:


            probabilities, probabilities_c, summary_train = self.sess.run(
                [self.probabilities, self.probabilities_c, self.train_summaries], feed_dict)
            if abs(sum(probabilities_c) - 1.0) > 1e-7 and sum(probabilities_c) > np.finfo(np.float32).eps:
                probabilities_c = probabilities_c / sum(probabilities_c)

            if max_val and (not training or self.settings.learn_init):
                speed_value = np.argmax(probabilities_c)
            else:
                speed_value = np.random.choice(list(range(len(probabilities_c))), p=np.copy(probabilities_c))
            # speed_value=np.random.choice(range(len(probabilities_c)), p=np.copy(probabilities_c))
            episode.pedestrian_data[id].speed[frame] = speed_value
            # episode.probabilities_c[frame, 0:len(probabilities)] = np.copy(probabilities)
            # print "Pos "+str(episode.agent[frame])
        if self.writer:
            if training:
                self.writer.add_summary(summary_train, self.num_iterations)
            else:
                self.test_writer.add_summary(summary_train, self.num_iterations)

        if abs(sum(probabilities) - 1.0) > 1e-7 and sum(probabilities) > np.finfo(np.float32).eps:
            probabilities = probabilities / sum(probabilities)

        if max_val and (not training or self.settings.learn_init):
            value = np.argmax(probabilities)
            #print ("Max value action " + str(value)+" proabilities "+str(probabilities))
        else:
            value = np.random.choice(list(range(len(probabilities))), p=np.copy(probabilities))
            # print ("Random action " + str(value)+" probability  "+str(probabilities[value]))
        if episode.pedestrian_data[id].goal_person_id >= 0 and frame+1<len(episode.valid_people_tracks[episode.pedestrian_data[id].goal_person_id_val]) and not viz:  # np.linalg.norm(self.actions[value]) >0:


            track_length = len(episode.valid_people_tracks[episode.pedestrian_data[id].goal_person_id_val])
            next_frame = min(frame + 1, track_length - 1)
            current_frame = min(frame, track_length - 1)
            # print ("Next position of pedestrian "+str(np.mean(episode.valid_people_tracks[episode.goal_person_id_val][next_frame],axis=1))+"  current pos "+str(episode.agent[frame])+" dif "+str(np.mean(episode.valid_people_tracks[episode.goal_person_id_val][next_frame],axis=1)-episode.agent[frame]))
            episode.pedestrian_data[id].velocity[frame] = np.mean(episode.valid_people_tracks[episode.pedestrian_data[id].goal_person_id_val][next_frame],axis=1)-episode.pedestrian_data[id].agent[frame]
            episode.pedestrian_data[id].speed[frame] = np.linalg.norm(episode.pedestrian_data[id].velocity[frame])
            episode.pedestrian_data[id].action[frame] = episode.find_action_to_direction(episode.pedestrian_data[id].velocity[frame], episode.pedestrian_data[id].speed[frame])
            episode.pedestrian_data[id].probabilities[frame, 0:len(probabilities)] = np.copy(probabilities)
            #print ("Follow pedestrian action: " + str(episode.action[frame]) + " " + str(episode.velocity[frame])+"")
            #
            # print ("mode " + str(np.argmax(probabilities))+ " "+ "value "+str(value))
        else:

            episode.pedestrian_data[id].velocity[frame] = np.array(self.actions[value])
            episode.pedestrian_data[id].action[frame] = value
            # if self.settings.printdebug_network_input:
            #     self.save_probabilities[frame]['sample'] = value
            #     self.save_probabilities[frame]['sample_probability'] = probabilities[value]
            #print ("Do not follow pedestrian action: " + str(episode.action[frame]) + " " + str(
            #    episode.velocity[frame]) + "")
            #print ("Goal id "+str(episode.goal_person_id_val)+" Len of tracks "+str(len(episode.valid_people_tracks[episode.goal_person_id_val]) )+" frame+1: "+str(frame+1)+" viz "+str(viz))
            if self.settings.velocity:

                if np.linalg.norm(episode.pedestrian_data[id].velocity[frame]) != 1 and np.linalg.norm(episode.pedestrian_data[id].velocity[frame]) > 0.1:
                    episode.pedestrian_data[id].velocity[frame] = episode.pedestrian_data[id].velocity[frame] / np.linalg.norm(episode.pedestrian_data[id].velocity[frame])

                # print ("Agent velocity planned without time and scaling "+str( speed_value*episode.velocity[frame] ))
                if not self.settings.acceleration:
                    episode.pedestrian_data[id].velocity[frame] = episode.pedestrian_data[id].velocity[frame] * speed_value * 5 / episode.frame_rate
                    # print ("Agent velocity after tim and scaling " + str(speed_value * 5 / episode.frame_rate))
                    # print episode.velocity[frame]
            elif self.settings.controller_len > 0:
                episode.pedestrian_data[id].velocity[frame] = np.array(self.actions[value]) * (speed_value + 1)
            episode.pedestrian_data[id].probabilities[frame, 0:len(probabilities)] = np.copy(probabilities)
            if len(episode.pedestrian_data[id].velocity[frame]) == 0:
                print("No vel!")
        # print ("Agent velocity final " + str(episode.velocity[frame]))
        return episode.pedestrian_data[id].velocity[frame]

    def fully_connected_size(self, dim_p):
        if  self.settings.actions_3d:
            return 27
        else:
            # if self.settings.velocity_sigmoid:
            #     return 4
            return 9

    def calc_probabilities(self, fc_size):
        #self.prev_probabilities = tf.compat.v1.placeholder(shape=[fc_size], dtype=tf.int32, name="prev_probabilities")
        if self.settings.attention:
            self.probabilities_att = tf.reshape(tf.sigmoid(self.mu_att + np.finfo(np.float32).eps), [1,1])
            self.attention =tf.tile(self.probabilities_att, tf.constant([1,9], tf.int32))# tf.greater(self.probabilities_att, tf.constant([0.5]))
            self.attention_neg = tf.tile(1-self.probabilities_att, tf.constant([1, 9], tf.int32))
            self.val=self.mu*self.attention + tf.slice(self.action_mem, [0, 0], [1, 9])*self.attention_neg
            self.probabilities = tf.reshape(tf.nn.softmax(self.val), [fc_size])  # +tf.slice(self.action_mem, [0,0],[1,9])

        elif self.settings.resnet:
            self.probabilities_old = tf.compat.v1.placeholder(shape=[1, 9], dtype=self.DTYPE, name="old_prob")
            self.probabilities = tf.reshape( tf.nn.softmax(self.mu+ self.probabilities_old),[fc_size])  # +tf.slice(self.action_mem, [0,0],[1,9])
        # This is the only branch used here.
        elif  self.settings.lstm:
            if self.settings.goal_dir and self.settings.extend_lstm_net and not self.settings.pose :
                self.probabilities = tf.reshape(
                    tf.nn.softmax(self.mu_dir + np.finfo(np.float32).eps + tf.slice(self.action_mem, [0, 0], [1, 9])),
                    [fc_size])
            elif self.settings.pose and (self.settings.extend_lstm_net or self.settings.extend_lstm_net_further):
                self.probabilities = tf.reshape(
                    tf.nn.softmax(self.mu_pose + np.finfo(np.float32).eps + tf.slice(self.action_mem, [0, 0], [1, 9])),
                    [fc_size])  # +tf.slice(self.action_mem, [0,0],[1,9])
            else:
                self.probabilities = tf.reshape(tf.nn.softmax(self.mu + np.finfo(np.float32).eps+tf.slice(self.action_mem, [0,0],[1,9])), [fc_size])#+tf.slice(self.action_mem, [0,0],[1,9])
        else:
            self.probabilities = tf.reshape(tf.nn.softmax(self.mu+np.finfo(np.float32).eps), [fc_size])
            if self.settings.controller_len > 0:
                self.probabilities_c = tf.reshape(tf.nn.softmax(self.mu_c+np.finfo(np.float32).eps), [self.settings.controller_len])

            if  self.settings.goal_dir:
                self.probabilities = tf.reshape(tf.nn.softmax(self.mu_dir + np.finfo(np.float32).eps), [fc_size])
            if self.settings.pose:
                self.probabilities = tf.reshape(tf.nn.softmax(self.mu_pose + np.finfo(np.float32).eps), [fc_size])
            if self.settings.old_lstm:
                self.probabilities = tf.reshape(tf.nn.softmax(self.mu_rnn + np.finfo(np.float32).eps), [fc_size])
        # And this branch
        if  self.settings.velocity:
            if self.settings.velocity_sigmoid:
                if self.settings.acceleration:
                    self.probabilities_v =tf.reshape(tf.sigmoid(self.mu_vel + np.finfo(np.float32).eps), [1])*8-4
                else:
                    #print "Defined probabilities_v"
                    self.probabilities_v = tf.reshape(tf.sigmoid(self.mu_vel + np.finfo(np.float32).eps), [1]) * 3
            else:
                self.probabilities_v = tf.reshape(tf.nn.softmax(self.mu_vel + np.finfo(np.float32).eps), [4])
        if self.settings.detect_turning_points:
            self.probabilities_point = tf.reshape(tf.sigmoid(self.mu_point + np.finfo(np.float32).eps), [1])



        variable_summaries(self.probabilities, name='softmax', summaries=self.train_summaries)

    def get_velocity(self, velocity, action, frame):
        return [action[frame]]

    def fully_connected(self, dim_p, prev_layer):
        with tf.compat.v1.variable_scope('fully_connected') as scope:
            # Concatenate pose and previous layer
            if not self.settings.old_fc:
                # print(("Convolutional output: : " + str(prev_layer.get_shape)))
                try:
                    dim = np.prod(prev_layer.get_shape().as_list()[1:])
                    prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
                except AttributeError:
                    dim = 0
                    prev_layer_flat = []
                # print(("Convolutional output flattened : " + str(prev_layer_flat.get_shape)))


                # with tf.compat.v1.variable_scope('car_var') as scope:
                if self.settings.car_var:

                    self.cars = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 8], dtype=self.DTYPE, name="cars")
                    if dim > 0:
                        prev_layer_flat = tf.concat([prev_layer_flat, self.cars], axis=1)
                    else:
                        prev_layer_flat = self.cars
                    dim = dim + 8
                # print(("Convolutional output+ cars : " + str(prev_layer_flat.get_shape)))

                if self.settings.angular:
                    self.angle = tf.compat.v1.placeholder(shape=[1, 1], dtype=self.DTYPE,
                                                name="angle")
                    if dim > 0:
                        prev_layer_flat = tf.concat([prev_layer_flat, self.angle], axis=1)
                    else:
                        prev_layer_flat = self.angle
                    dim = dim + 1

                # print(("Convolutional output+ angle : " + str(prev_layer_flat.get_shape)))



                #if self.settings.lstm or self.settings.old_lstm:
                # print("lstm")
                input_no_temp = prev_layer_flat
                input_no_temp_dim = dim
                # print(("Input no temp " + str(input_no_temp.get_shape)+" "+str(input_no_temp_dim)))



                # with tf.compat.v1.variable_scope('action_mem') as scope:
                if self.settings.action_mem:
                    self.action_mem = tf.compat.v1.placeholder(dtype=self.DTYPE, shape=(
                        1, self.action_size_mem * self.settings.action_mem) , name="action_mem_1")
                    if dim > 0:
                        prev_layer_flat = tf.concat([prev_layer_flat, self.action_mem], axis=1)
                    else:
                        prev_layer_flat = self.action_mem
                    dim = dim + (self.action_size_mem * self.settings.action_mem)
                    # print(("Convolutional output+ mem : " + str(prev_layer_flat.get_shape)))

                fc_size = self.fully_connected_size(9)
                if not self.settings.lstm:
                    weights = tf.compat.v1.get_variable('weights', [dim, fc_size], self.DTYPE,
                                              initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                    biases = self.bias_variable('biases', [fc_size])

                    self.mu = tf.matmul(prev_layer_flat, weights)
                    self.mu =tf.add(self.mu, biases, name=scope.name)
                    self.fc_no_addons = self.mu
                    # print(("Mu "+str(self.mu.get_shape)))

                # Additional memory
                # with tf.compat.v1.variable_scope('additional_mem') as scope:
                if self.settings.action_mem and self.settings.nbr_timesteps > 0:
                    self.action_mem_further = tf.compat.v1.placeholder(dtype=self.DTYPE, shape=(
                        1, self.action_size_mem * self.settings.nbr_timesteps), name="action_mem_further")
                    if not self.settings.lstm:
                        dim_w = (self.action_size_mem * self.settings.nbr_timesteps)
                        weights_mem = tf.compat.v1.get_variable('weights_mem', [dim_w, fc_size], self.DTYPE,
                                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.mu =self.mu + tf.matmul(self.action_mem_further, weights_mem)
                        last_layer = self.mu
                        self.fc_memory = self.mu
                        # print(("Mu + mem" + str(self.mu.get_shape)))
                        # print(("Last layer + mem" + str(self.mu.get_shape)))
                    else:
                        last_layer=prev_layer_flat

                if self.settings.goal_dir and not self.settings.extend_lstm_net:
                    # print("Define goal")
                    input_no_temp, input_no_temp_dim, last_layer = self.define_goal_fc(fc_size, input_no_temp,
                                                                                       input_no_temp_dim, last_layer,
                                                                                       scope, not self.settings.lstm,self.settings.lstm )

                if self.settings.pose and not (self.settings.extend_lstm_net_further or self.settings.extend_lstm_net):
                    input_no_temp, input_no_temp_dim, last_layer = self.define_pose_fc(fc_size, input_no_temp,
                                                                                       input_no_temp_dim, last_layer,
                                                                                       scope, not self.settings.lstm,self.settings.lstm)


                if self.settings.lstm or self.settings.old_lstm:

                    temp_in = tf.concat([self.action_mem, self.action_mem_further], axis=1)
                    temp_in = tf.split(temp_in, self.settings.nbr_timesteps + self.settings.action_mem, 1)

                    # self.rnn_in_state=tf.compat.v1.placeholder(dtype=self.DTYPE, shape=(
                    #     1, self.action_size_mem * self.settings.action_mem) , name="action_mem_1")
                    # print(("RNN input: "+str(temp_in)))
                    lstm_cell = rnn.BasicLSTMCell(32)

                    outputs, self.rnn_states = tf.compat.v1.nn.static_rnn(lstm_cell, temp_in, dtype=tf.float32)#, initial_state=self.rnn_in_state)
                    self.lstm_out = outputs[-1]
                    # print(("LSTM input no temp: "+str(input_no_temp.shape)))
                    input_no_temp = tf.concat([input_no_temp, outputs[-1]], axis=1)
                    input_no_temp_dim = input_no_temp_dim + 32
                    # print(("LSTM inout no temp+rnn: " + str(input_no_temp.shape)+" "+str(input_no_temp_dim)))



                    if self.settings.lstm:
                        # print(("LSTM input: " + str(input_no_temp.shape)+" "+str(input_no_temp_dim)))
                        # print
                        self.weights_rnn = tf.compat.v1.get_variable('weights_rnn', [input_no_temp_dim, fc_size], self.DTYPE,
                                                           initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.biases_rnn = self.bias_variable('biases_rnn', [fc_size])

                        self.mu = tf.matmul(input_no_temp, self.weights_rnn)
                        self.mu =tf.add(self.mu, self.biases_rnn, name=scope.name)
                        # print(("New lstm mu: " + str(self.mu.shape)))
                        last_layer=self.mu
                        if self.settings.goal_dir and self.settings.extend_lstm_net:
                            input_no_temp, input_no_temp_dim, last_layer = self.define_goal_fc(fc_size, input_no_temp,
                                                                                               input_no_temp_dim,
                                                                                               self.mu,
                                                                                               scope,
                                                                                               True,
                                                                                               True)
                            last_layer = self.mu_dir

                        if self.settings.pose and (self.settings.extend_lstm_net_further or self.settings.extend_lstm_net):
                            input_no_temp, input_no_temp_dim, last_layer = self.define_pose_fc(fc_size, input_no_temp,
                                                                                               input_no_temp_dim,
                                                                                               last_layer,
                                                                                               scope,
                                                                                               True,
                                                                                               True)
                            # self.pose = tf.compat.v1.placeholder(shape=[self.settings.batch_size, POSE_DIM], dtype=self.DTYPE,
                            #                            name="pose_fc_layer")
                            # self.weights_pose = tf.get_variable('weights_pose', [fc_size+POSE_DIM, fc_size], self.DTYPE,
                            #                                    initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                            # self.biases_pose = self.bias_variable('biases_pose', [fc_size])
                            # self.mu_pose = tf.matmul(tf.concat([self.mu, self.pose], axis=1), self.weights_pose)
                            # self.mu_pose = tf.add(self.mu_pose, self.biases_pose, name=scope.name)
                            # input_no_temp=tf.concat([input_no_temp,self.pose], axis=1)
                            # input_no_temp_dim=input_no_temp_dim+POSE_DIM


                    if self.settings.old_lstm:
                        input_rnn = tf.concat([last_layer, outputs[-1]], axis=1)
                        # print(("LSTM mu input: " + str(input_rnn.shape)))
                        weights_rnn = tf.compat.v1.get_variable('weights_rnn', [fc_size + 32, fc_size], self.DTYPE,
                                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        biases_rnn = self.bias_variable('biases_rnn', [fc_size])

                        self.mu_rnn = tf.matmul(input_rnn, weights_rnn)
                        self.mu_rnn =tf.add(self.mu_rnn, biases_rnn, name=scope.name)


                    if self.settings.velocity:
                        # print("VElocity! "+str(self.settings.velocity))
                        if self.settings.lstm:
                            input_layer = input_no_temp
                            input_layer_dim = input_no_temp_dim
                        else:
                            input_layer = input_for_vel
                            input_layer_dim = input_dim
                        if not self.settings.velocity_sigmoid:

                            weights_vel = tf.compat.v1.get_variable('weights_vel', [input_layer_dim, 4], self.DTYPE,
                                                          initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                            biases_vel = self.bias_variable('biases_vel', [4])

                            self.mu_vel = tf.matmul(input_layer, weights_vel)
                            self.mu_vel = tf.add(self.mu_vel, biases_vel, name=scope.name)
                        else:
                            #if not self.settings.extend_lstm_net_further:
                            self.weights_vel = tf.compat.v1.get_variable('weights_vel', [input_layer_dim, 1], self.DTYPE,
                                                               initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                            self.biases_vel = self.bias_variable('biases_vel', [1])

                            self.mu_vel = tf.matmul(input_layer, self.weights_vel)
                            self.mu_vel = tf.add(self.mu_vel, self.biases_vel, name=scope.name)
                            # else:
                            #     self.weights_vel = tf.concat([tf.get_variable('weights_vel', [input_layer_dim-POSE_DIM, 1], self.DTYPE,
                            #                                        initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED)), tf.get_variable('weights_vel2', [POSE_DIM, 1], self.DTYPE,
                            #                                        initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))], axis=0)
                            #     self.biases_vel = self.bias_variable('biases_vel', [1])
                            #
                            #     self.mu_vel = tf.matmul(input_layer, self.weights_vel)
                            #     self.mu_vel = tf.add(self.mu_vel, self.biases_vel, name=scope.name)
                    # if self.settings.detect_turning_points:
                    #     weights_point = tf.get_variable('weights_point', [input_layer_dim, 1], self.DTYPE,
                    #                                   initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                    #     biases_point = self.bias_variable('biases_point', [1])
                    #
                    #     self.mu_point = tf.matmul(input_layer, weights_point)
                    #     self.mu_point = tf.add(self.mu_point, biases_point, name=scope.name)
                    # if self.settings.attention:
                    #     weights_att = tf.get_variable('weights_point', [input_layer_dim, 1], self.DTYPE,
                    #                                     initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                    #     biases_att = self.bias_variable('biases_point', [1])
                    #
                    #     self.mu_att = tf.matmul(input_layer, weights_att)
                    #     self.mu_att = tf.add(self.mu_att, biases_att, name=scope.name)
            else:

                try:
                    dim = np.prod(prev_layer.get_shape().as_list()[1:])
                    prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
                except AttributeError:
                    dim = 0
                    prev_layer_flat = []

                # with tf.compat.v1.variable_scope('car_var') as scope:
                if self.settings.car_var:

                    self.cars = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 8], dtype=self.DTYPE, name="cars")
                    if dim > 0:
                        prev_layer_flat = tf.concat([prev_layer_flat, self.cars], axis=1)
                    else:
                        prev_layer_flat = self.cars
                    dim = dim + 8

                # if self.settings.pfnn:
                #
                #     self.pose = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 93], dtype=self.DTYPE, name="cars")
                #     if dim>0:
                #         prev_layer_flat =tf.concat([prev_layer_flat, self.pose], axis=1)
                #     else:
                #         prev_layer_flat =self.pose
                #     dim=93+8

                input_no_temp = prev_layer_flat
                input__no_temp_dim = dim
                # with tf.compat.v1.variable_scope('action_mem') as scope:
                if self.settings.action_mem:
                    self.action_mem = tf.compat.v1.placeholder(dtype=self.DTYPE, shape=(
                    1, self.action_size_mem * self.settings.action_mem))
                    if dim > 0:
                        prev_layer_flat = tf.concat([prev_layer_flat, self.action_mem], axis=1)
                    else:
                        prev_layer_flat = self.action_mem
                    dim = dim + (self.action_size_mem * self.settings.action_mem)
                input = prev_layer_flat
                input_dim = dim

                # Initialize weights
                # with tf.compat.v1.variable_scope('layer_2') as scope:

                # if self.settings.fully_connected_layer>0:
                #     fc_size=self.settings.fully_connected_layer
                #     weights_fc = tf.get_variable('fc_weights', [dim, fc_size], self.DTYPE,
                #                               initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                #     biases_fc = self.bias_variable('fc_biases', [fc_size])
                #     self.fc = tf.matmul(prev_layer_flat, weights_fc)
                #     self.fc = tf.add(self.fc, biases_fc, name=scope.name)
                #     prev_layer_flat=self.fc
                #     dim=self.settings.fully_connected_layer

                fc_size = self.fully_connected_size(dim_p)
                if not self.settings.lstm:
                    weights = tf.compat.v1.get_variable('weights', [dim, fc_size], self.DTYPE,
                                              initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                    biases = self.bias_variable('biases', [fc_size])

                    self.mu = tf.matmul(prev_layer_flat, weights)
                    self.mu = tf.add(self.mu, biases, name=scope.name)

                # Additional memory
                # with tf.compat.v1.variable_scope('additional_mem') as scope:
                if self.settings.action_mem and self.settings.nbr_timesteps > 0:
                    self.action_mem_further = tf.compat.v1.placeholder(dtype=self.DTYPE, shape=(
                        1, (self.action_size_mem ) * self.settings.nbr_timesteps))
                    if not self.settings.lstm:
                        dim_w = ((self.action_size_mem ) * self.settings.nbr_timesteps)
                        weights_mem = tf.compat.v1.get_variable('weights_mem', [dim_w, fc_size], self.DTYPE,
                                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.mu = self.mu + tf.matmul(self.action_mem_further, weights_mem)
                        input = tf.concat([input, self.action_mem_further], axis=1)
                        input_dim = input_dim + dim_w
                    last_layer = self.mu
                # with tf.compat.v1.variable_scope('additional_mem') as scope:
                # if self.settings.estimate_advantages:
                #     fc_size = 1
                #     weights_val = tf.get_variable('weights_advantage', [dim, fc_size], self.DTYPE,
                #                               initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                #     biases_val = self.bias_variable('biases_advantage', [fc_size])
                #
                #     self.val = tf.matmul(prev_layer_flat, weights_val)
                #     self.val = tf.add(self.val, biases_val, name=scope.name)
                #
                #     last_layer=self.mu
                #
                #
                #
                #     # Add to summaries
                #     if self.writer:
                #         variable_summaries(last_layer, name='fc1_out', summaries=self.train_summaries)
                #
                #     if self.writer:
                #         variable_summaries(biases, name='fc1b', summaries=self.train_summaries)
                #         variable_summaries(weights, name='fc1k', summaries=self.train_summaries)
                #         variable_summaries(last_layer, name='softmax', summaries=self.train_summaries)
                # with tf.compat.v1.variable_scope('controller') as scope:
                # if self.settings.controller_len>0:
                #     print "Define mu_c"
                #     weights_c_1 = tf.get_variable('weights_c1', [dim, self.settings.controller_len], self.DTYPE,
                #                               initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                #     weights_c_2 = tf.get_variable('weights_c2', [fc_size, self.settings.controller_len], self.DTYPE,
                #                                 initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                #     biases_c = self.bias_variable('biases_c', [ self.settings.controller_len])
                #
                #
                #     self.mu_c = tf.add(tf.matmul(prev_layer_flat, weights_c_1),tf.matmul(self.mu, weights_c_2))
                #     self.mu_c = tf.add(self.mu_c, biases_c, name=scope.name)
                #     prev_layer_flat_c = self.mu_c
                # if not self.settings.lstm:
                #     prev_layer_flat=self.mu
                #     dim=fc_size

                # with tf.compat.v1.variable_scope('semantic_class') as scope:
                # if len(self.settings.sem_class) > 0:
                #     self.sem_class = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 9], dtype=self.DTYPE, name="sem_class")
                #     prev_layer_flat = tf.concat([prev_layer_flat, self.sem_class], axis=1)
                #     dim = dim + 9
                #
                #     weights_sem = tf.get_variable('weights_sem', [dim, fc_size], self.DTYPE,
                #                                   initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                #     biases_sem = self.bias_variable('biases_sem', [fc_size])
                #
                #     self.mu_sem = tf.matmul(prev_layer_flat, weights_sem)
                #     self.mu_sem = tf.add(self.mu_sem, biases_sem, name=scope.name)

                # with tf.compat.v1.variable_scope('goal_direction') as scope:

                if self.settings.goal_dir:
                    self.goal_dir = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 10], dtype=self.DTYPE,
                                                   name="goal_direction")
                    prev_layer_flat = tf.concat([prev_layer_flat, self.goal_dir], axis=1)
                    dim = dim + 10

                    input = tf.concat([input, self.goal_dir], axis=1)
                    input_dim = input_dim + 10

                    input_no_temp = tf.concat([input_no_temp, self.goal_dir], axis=1)
                    input__no_temp_dim = input__no_temp_dim + 10
                    if not self.settings.lstm:
                        weights_dir = tf.compat.v1.get_variable('weights_dir', [fc_size + 10, fc_size], self.DTYPE,
                                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        biases_dir = self.bias_variable('biases_dir', [fc_size])
                        self.mu_dir = tf.matmul(tf.concat([self.mu, self.goal_dir], axis=1), weights_dir)
                        self.mu_dir = tf.add(self.mu_dir, biases_dir, name=scope.name)
                        last_layer = self.mu_dir

                if self.settings.pose:
                    self.pose = tf.compat.v1.placeholder(shape=[self.settings.batch_size, POSE_DIM], dtype=self.DTYPE, name="pose")

                    last_layer = tf.concat([last_layer, self.pose], axis=1)
                    weights_pose = tf.compat.v1.get_variable('weights_pose', [fc_size + POSE_DIM, fc_size], self.DTYPE,
                                                   initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                    self.biases_pose = self.bias_variable('biases_pose', [fc_size])


                    self.mu_pose = tf.matmul(last_layer, weights_pose)
                    self.mu_pose = tf.add(self.mu_pose, self.biases_pose, name=scope.name)

                if self.settings.lstm or self.settings.old_lstm:
                    # print self.action_mem

                    # print self.action_mem_further

                    temp_in = tf.concat([self.action_mem, self.action_mem_further], axis=1)
                    # temp_in=tf.reshape(temp_in, [1,self.settings.nbr_timesteps+self.settings.action_mem,self.fully_connected_size(dim_p) + 1])
                    temp_in = tf.split(temp_in, self.settings.nbr_timesteps + self.settings.action_mem, 1)
                    # print temp_in

                    lstm_cell = rnn.BasicLSTMCell(32)

                    outputs, states = rnn.static_rnn(lstm_cell, temp_in, dtype=tf.float32)
                    # print outputs[-1]
                    # print input_no_temp

                    input_no_temp = tf.concat([input_no_temp, outputs[-1]], axis=1)
                    input = input_no_temp
                    # print input_no_temp
                    input__no_temp_dim = input__no_temp_dim + 32
                    input_dim = input__no_temp_dim

                    if self.settings.lstm:

                        # print
                        weights_rnn = tf.compat.v1.get_variable('weights_rnn', [input__no_temp_dim, fc_size], self.DTYPE,
                                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        biases_rnn = self.bias_variable('biases_rnn', [fc_size])

                        self.mu = tf.matmul(input_no_temp, weights_rnn)
                        self.mu = tf.add(self.mu, biases_rnn, name=scope.name)
                    else:
                        # with tf.compat.v1.variable_scope('lstm') as scope:
                        input_rnn = tf.concat([last_layer, outputs[-1]], axis=1)

                        weights_rnn = tf.compat.v1.get_variable('weights_rnn', [fc_size + 32, fc_size], self.DTYPE,
                                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        biases_rnn = self.bias_variable('biases_rnn', [fc_size])

                        self.mu_rnn = tf.matmul(input_rnn, weights_rnn)
                        self.mu_rnn = tf.add(self.mu_rnn, biases_rnn, name=scope.name)

                # with tf.compat.v1.variable_scope('velocity') as scope:
                if self.settings.lstm:
                    input_layer = input_no_temp
                    input_layer_dim = input__no_temp_dim
                else:
                    input_layer = input
                    input_layer_dim = input_dim
                if self.settings.velocity or self.settings.attention:
                    if self.settings.goal_dir and self.settings.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] > 0:
                        self.time_alloted = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 1], dtype=self.DTYPE,
                                                           name="time_alotted")
                        input_layer = tf.concat([input_layer, self.time_alloted], axis=1)
                        input_layer_dim = input_layer_dim + 1
                    if not self.settings.velocity_sigmoid:
                        weights_vel = tf.compat.v1.get_variable('weights_vel', [input_layer_dim, 4], self.DTYPE,
                                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        biases_vel = self.bias_variable('biases_vel', [4])

                        self.mu_vel = tf.matmul(input_layer, weights_vel)
                        self.mu_vel = tf.add(self.mu_vel, biases_vel, name=scope.name)
                    else:
                        weights_vel = tf.compat.v1.get_variable('weights_vel', [input_layer_dim, 1], self.DTYPE,
                                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        biases_vel = self.bias_variable('biases_vel', [1])

                        self.mu_vel = tf.matmul(input_layer, weights_vel)
                        self.mu_vel = tf.add(self.mu_vel, biases_vel, name=scope.name)

                        if False:
                            weights_vel_sig = tf.compat.v1.get_variable('weights_vel_s', [input_layer_dim, 1], self.DTYPE,
                                                              initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                            biases_vel_sig = self.bias_variable('biases_vel_s', [1])

                            self.sigm_vel = tf.matmul(input_layer, weights_vel_sig)
                            self.sigm_vel = tf.nn.relu(tf.add(self.sigm_vel, biases_vel_sig, name=scope.name))

    def define_pose_fc(self, fc_size, input_no_temp, input_no_temp_dim, last_layer, scope,define_mu_pose, add_to_no_temp):
        self.pose = tf.compat.v1.placeholder(shape=[self.settings.batch_size, POSE_DIM], dtype=self.DTYPE, name="pose")
        if define_mu_pose:#not self.settings.lstm:
            last_layer = tf.concat([last_layer, self.pose], axis=1)
            weights_pose = tf.compat.v1.get_variable('weights_pose', [fc_size + POSE_DIM, fc_size], self.DTYPE,
                                           initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            biases_pose = self.bias_variable('biases_pose', [fc_size])
            self.mu_pose = tf.matmul(last_layer, weights_pose)
            self.mu_pose = tf.add(self.mu_pose, biases_pose, name=scope.name)
            last_layer = self.mu_pose
            # print(("Mu_pose" + str(self.mu_pose.get_shape)))
        if add_to_no_temp:
            # print(("Last layer + pose" + str(last_layer.get_shape)))
            input_no_temp = tf.concat([input_no_temp, self.pose], axis=1)
            # print(("input_no_temp + pose" + str(input_no_temp.get_shape) + " " + str(input_no_temp_dim)))
            input_no_temp_dim = input_no_temp_dim + POSE_DIM
        return input_no_temp, input_no_temp_dim, last_layer

    def define_goal_fc(self, fc_size, input_no_temp, input_no_temp_dim, last_layer, scope, define_mu_dir, add_to_no_temp):
        self.goal_dir = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 10], dtype=self.DTYPE,
                                       name="goal_direction")
        goal_var = self.goal_dir
        goal_dim = 10
        if self.settings.speed_input:
            self.time_alloted = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 1], dtype=self.DTYPE,
                                               name="time_alotted")
            goal_var = tf.concat([goal_var, self.time_alloted], axis=1)
            goal_dim = goal_dim + 1
        if define_mu_dir:
            # print("Mu dir")
            weights_dir = tf.compat.v1.get_variable('weights_dir', [fc_size + goal_dim, fc_size], self.DTYPE,
                                          initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            biases_dir = self.bias_variable('biases_dir', [fc_size])
            self.mu_dir = tf.matmul(tf.concat([self.mu, goal_var], axis=1), weights_dir)
            self.mu_dir = tf.add(self.mu_dir, biases_dir, name=scope.name)
            last_layer = self.mu_dir
        if add_to_no_temp:
            input_no_temp = tf.concat([input_no_temp, goal_var], axis=1)
            input_no_temp_dim = input_no_temp_dim + goal_dim
            # print(("Input no temp + goal" + str(input_no_temp.get_shape) + " " + str(input_no_temp_dim)))
        return input_no_temp, input_no_temp_dim, last_layer

    def get_feature_vectors_gradient(self, id,agent_action, agent_frame, agent_measures, agent_pos, agent_speed,agent_velocity, ep_itr,
                                     episode, feed_dict, frame, poses, statistics, training, statistics_car=[]):
        if self.settings.resnet:
            if agent_frame==0:
                feed_dict[self.probabilities_old] = np.zeros((1,9))
            else:
                feed_dict[self.probabilities_old]=np.expand_dims(statistics[ep_itr,id,  agent_frame-1, STATISTICS_INDX.probabilities[0]:STATISTICS_INDX.probabilities[1]], axis=0)

        if self.settings.car_var:
            if agent_frame == 0:
                vel=statistics[ep_itr, id, -1, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]]
            else:
                vel=agent_velocity[ep_itr, agent_frame-1, :]
            distracted=agent_measures[ep_itr, agent_frame,PEDESTRIAN_MEASURES_INDX.distracted]
            #print (" Gradient distracted "+str(distracted)+" frame "+str(frame)+" ")
            feed_dict[self.cars] = episode.get_input_cars_smooth(agent_pos[ep_itr, agent_frame, :], frame, vel, self.settings.field_of_view, distracted=distracted )
            #print (" Gradient car var "+str(feed_dict[self.cars]))
        # if self.settings.pose:
        #     feed_dict[self.pose] = np.expand_dims(poses[ep_itr, agent_frame, :93] * (1 / 100.0), axis=0)
        if self.settings.pose:
            itr = int(poses[ep_itr,id, agent_frame, STATISTICS_INDX_POSE.agent_pose_frames])
            # print("Get grad feature Frame " + str(agent_frame) + " pfnn itr " + str(itr))
            pose_input=poses[ep_itr,id, itr, STATISTICS_INDX_POSE.agent_pose_hidden[0]:STATISTICS_INDX_POSE.agent_pose_hidden[1]]

            #print( "Get grad feature Frame " + str(agent_frame) + " " + str(itr)+" indx "+str(STATISTICS_INDX_POSE.agent_pose_hidden[0]) +" - "+str(STATISTICS_INDX_POSE.agent_pose_hidden[1]))
            #feed_dict[self.pose] = np.expand_dims(poses[ep_itr, itr, STATISTICS_INDX_POSE.agent_pose_hidden[0]:STATISTICS_INDX_POSE.agent_pose_hidden[1]]* (1 / 100.0), axis=0)
            # print( pose_input[np.abs(pose_input) > 1e3])
            pose_input[np.abs(pose_input) > 1e3] = 0
            pose_input[np.abs(pose_input) < 1e-4] = 0
            if np.sum(np.abs(pose_input))< 1e-4:
                pose_input=np.zeros_like(pose_input)
            feed_dict[self.pose] = np.expand_dims(pose_input * (1 / 100.0), axis=0)

            # print "Hidden layer gradient: " + str(feed_dict[self.pose][0,:5])
        if self.settings.controller_len>0:
            if np.isnan(agent_speed[ep_itr, agent_frame]):
                feed_dict[self.sample_c] = [0]
            else:
                feed_dict[self.sample_c]=[int(agent_speed[ep_itr, agent_frame])]

        # if self.settings.pose:
        #     itr= episode.agent_pose_frames[self.frame]
        #     feed_dict[self.pose] = np.expand_dims( [episode.agent_pose[itr,:]])

        if self.settings.goal_dir:
            feed_dict[self.goal_dir] = np.zeros((1, 10))
            goal = [0, 0, 0]
            goal[1] = statistics[ep_itr,id, agent_frame, STATISTICS_INDX.goal[0]]
            goal[2] = statistics[ep_itr,id, agent_frame, STATISTICS_INDX.goal[0]+1]
            feed_dict[self.goal_dir] = episode.get_goal_dir_smooth(agent_pos[ep_itr, agent_frame, :], goal)
            if self.settings.speed_input:
                feed_dict[self.time_alloted] =np.expand_dims(
                    [episode.get_time(id, agent_pos[ep_itr, agent_frame, :], goal, agent_frame, )], axis=0)
        if self.settings.velocity:

            goal = [0, 0, 0]
            goal[1] = statistics[ep_itr,id, agent_frame, STATISTICS_INDX.goal[0]]
            goal[2] = statistics[ep_itr,id, agent_frame, STATISTICS_INDX.goal[0]+1]

            goal_time = statistics[ep_itr,id, agent_frame, STATISTICS_INDX.goal_time]
            if self.settings.goal_dir and self.settings.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] > 0:
                feed_dict[self.time_alloted] = np.expand_dims(
                    [episode.get_time(id,agent_pos[ep_itr, agent_frame, :], goal, agent_frame)], axis=0)
            # print("Speed "+str(agent_speed[ep_itr, agent_frame])+" times "+str(episode.frame_rate/5.0)+" = "+str( [agent_speed[ep_itr, agent_frame]*episode.frame_rate/5.0]))
            feed_dict[self.velocity] = [agent_speed[ep_itr, agent_frame]]
        if self.settings.action_mem:
            dim_in = 9  # self.fully_connected_size(0)
            values = np.zeros((dim_in + 1) * self.settings.action_mem)
            if self.old or self.old_mem:
                for past_frame in range(1, self.settings.action_mem + 1):
                    if agent_frame - past_frame >= 0:
                        values[(past_frame - 1) * (dim_in + 1) + int(
                            agent_action[ep_itr, max(agent_frame - past_frame, 0)])] = 1
                        values[past_frame * past_frame - 1] = agent_measures[
                            ep_itr, max(agent_frame - past_frame, 0), PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
                if self.settings.nbr_timesteps > 0:
                    values = np.zeros((dim_in + 1) * self.settings.nbr_timesteps)
                    for past_frame in range(self.settings.action_mem, self.settings.nbr_timesteps + 1):
                        if agent_frame - past_frame >= 0:
                            values[(past_frame - 1) * (dim_in + 1) + int(
                                agent_action[ep_itr, max(agent_frame - past_frame, 0)])] = 1
                            values[past_frame * past_frame - 1] = agent_measures[
                                ep_itr, max(agent_frame - past_frame, 0), PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                    feed_dict[self.action_mem_further] = np.expand_dims(values, axis=0)
            else:
                #if self.settings.old_fc:
                    for past_frame in range(1, self.settings.action_mem + 1):
                        if frame - past_frame >= 0:
                            pos = (past_frame - 1) * (dim_in + 1) + int(
                                agent_action[ep_itr, max(agent_frame - past_frame, 0)])
                            values[pos] = 1
                            values[past_frame * (dim_in + 1) - 1] = agent_measures[
                                ep_itr, max(frame - past_frame, 0), PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                        feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
                    feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
                    if self.settings.nbr_timesteps > 0:
                        values = np.zeros((dim_in + 1) * self.settings.nbr_timesteps)
                        vel_init=statistics[ep_itr,id, -1, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]]
                        speed_init = np.linalg.norm(vel_init)
                        dir_init = episode.find_action_to_direction(vel_init, speed_init)
                        for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 1):
                            if frame - past_frame >= 0:
                                values[(past_frame - 2) * (dim_in + 1) + int(
                                    agent_action[ep_itr, max(frame - past_frame, 0)])] = 1
                                values[(past_frame - 1) * (dim_in + 1) - 1] = agent_measures[ep_itr, max(frame - past_frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                            elif self.settings.past_filler:
                                values[(past_frame - 2) * (dim_in + 1) + int(dir_init)] = 1
                                values[(past_frame - 1) * (dim_in + 1) - 1] = speed_init
                        feed_dict[self.action_mem_further] = np.expand_dims(values, axis=0)
                # else:
                #
                #     # if self.settings.nbr_timesteps > 0:
                #     values = np.zeros(self.settings.nbr_timesteps * (self.fully_connected_size(9)+1))
                #     for past_frame in range(1, self.settings.nbr_timesteps + 1):
                #         if frame - past_frame >= 0:
                #             pos = (past_frame - 1) * (dim_in + 1) + int(
                #                 agent_action[ep_itr, max(agent_frame - past_frame, 0)])
                #             values[pos] = 1
                #             values[past_frame * (dim_in + 1) - 1] = agent_measures[
                #                 ep_itr, max(frame - past_frame, 0), 3]
                #     feed_dict[self.action_mem] = np.expand_dims(values, axis=0)

    def get_feature_vectors(self, id,agent_frame, episode, feed_dict, frame, distracted=False):
        #print ("Get feature vectors")
        # print len(self.settings.sem_class)
        if self.settings.resnet:
            if agent_frame==0:
                feed_dict[self.probabilities_old] = np.zeros((1,9))
            else:
                feed_dict[self.probabilities_old]=np.expand_dims(episode.pedestrian_data[id].probabilities[agent_frame-1,:9 ], axis=0)
        if self.settings.car_var:
            if agent_frame==0:
                velocity=episode.pedestrian_data[id].vel_init
            else:
                velocity = episode.pedestrian_data[id].velocity[agent_frame-1]
            if self.settings.old_fc:
                feed_dict[self.cars] = episode.get_input_cars(episode.pedestrian_data[id].agent[agent_frame], frame, velocity, self.settings.field_of_view,distracted=distracted)
            else:
                feed_dict[self.cars] = episode.get_input_cars_smooth(episode.pedestrian_data[id].agent[agent_frame], frame,velocity, self.settings.field_of_view, distracted=distracted)
            #print (" Cars var " + str(feed_dict[self.cars]))
        if self.settings.goal_dir:
            if self.settings.old_fc:
                feed_dict[self.goal_dir] = episode.get_goal_dir(episode.pedestrian_data[id].agent[agent_frame],
                                                                          episode.pedestrian_data[id].goal[agent_frame,:])
            else:
                feed_dict[self.goal_dir] = episode.get_goal_dir_smooth(episode.pedestrian_data[id].agent[agent_frame], episode.pedestrian_data[id].goal[agent_frame,:])
            if self.settings.speed_input:
                feed_dict[self.time_alloted] = np.expand_dims(
                    [episode.get_time(id,episode.pedestrian_data[id].agent[agent_frame], episode.pedestrian_data[id].goal[agent_frame,:], agent_frame)], axis=0)
        if self.settings.velocity and self.settings.goal_dir and self.settings.reward_weights_pedestrian[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] > 0:
            feed_dict[self.time_alloted] = np.expand_dims(
                [episode.get_time(id,episode.pedestrian_data[id].agent[agent_frame], episode.pedestrian_data[id].goal[agent_frame,:], agent_frame)], axis=0)
        # if self.settings.pose:
        #     itr = episode.agent_pose_frames[agent_frame]
        #     feed_dict[self.pose] = np.expand_dims(np.array(episode.agent_pose[itr, :]) * (1 / 100.0), axis=0)
        if self.settings.pose:
            itr = int( episode.pedestrian_data[id].agent_pose_frames[agent_frame])
            pose_input=episode.pedestrian_data[id].agent_pose_hidden[itr, :]
            # print("Get pose feature Frame " + str(agent_frame) + " pfnn itr " + str(itr))
            # print("Get grad feature Frame " + str(pose_input[0:5]) + " pfnn itr " + str(np.sum(np.abs(pose_input))))
            # print("Get pose of frame "+str(agent_frame)+" is index "+str(episode.agent_pose_frames[agent_frame]))
            # #feed_dict[self.pose] = np.expand_dims(np.array(episode.agent_pose_hidden[itr, :])* (1 / 100.0), axis=0)
            # print("Before Thresholding values " + str(pose_input[:5]) + " " + str(
            #     np.sum(np.abs(pose_input))))
            # print(np.histogram(pose_input, bins=50))
            pose_input[abs(pose_input)>1e3]=0
            pose_input[np.abs(pose_input) < 1e-4] = 0
            if  np.sum(np.abs(pose_input))< 1e-4:
                pose_input=np.zeros_like(pose_input)
            # print("Get feature Frame " + str(agent_frame) + " " + str(itr) + " sum " + str(
            #     np.sum(np.abs(pose_input))))
            # print("After thresholding")
            # print(np.histogram(pose_input, bins=50))
            #print ("Hidden layer: "+str(feed_dict[self.pose][0,:5]))
            feed_dict[self.pose] = np.expand_dims(np.array(pose_input) * (1 / 100.0), axis=0)
        if self.settings.action_mem:
            dim_in=9#self.fully_connected_size(0)
            values = np.zeros((dim_in + 1) * self.settings.action_mem)
            if self.old or self.old_mem:
                for past_frame in range(1, self.settings.action_mem + 1):
                    if agent_frame - past_frame >= 0:
                        # print str(len(episode.action))+" "+str(max(agent_frame - past_frame, 0))
                        # print (past_frame-1)*(self.fully_connected_size(0)+1)+int(episode.action[max(agent_frame - past_frame, 0)])
                        # print values.shape
                        values[(past_frame - 1) * (dim_in + 1) + int(
                            episode.pedestrian_data[id].action[max(agent_frame - past_frame, 0)])] = 1
                        values[past_frame * past_frame - 1] = episode.pedestrian_data[id].measures[max(agent_frame - past_frame, 0), PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
                if self.settings.nbr_timesteps > 0:
                    values = np.zeros((dim_in + 1) * self.settings.nbr_timesteps)
                    for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 1):
                        if agent_frame - past_frame >= 0:
                            # print str(len(episode.action))+" "+str(max(agent_frame - past_frame, 0))
                            # print (past_frame-1)*(dim_in+1)+int(episode.action[max(agent_frame - past_frame, 0)])
                            # print values.shape
                            values[(past_frame - 1) * (dim_in + 1) + int(
                                episode.pedestrian_data[id].action[max(agent_frame - past_frame, 0)])] = 1
                            values[past_frame * past_frame - 1] = episode.pedestrian_data[id].measures[
                                max(agent_frame - past_frame, 0), PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                    feed_dict[self.action_mem_further] = np.expand_dims(values, axis=0)
            else:
                #print ("History")
                #if self.settings.old_fc:
                for past_frame in range(1, self.settings.action_mem + 1):
                    if frame - past_frame >= 0:
                        #print ("past frame "+str(past_frame)+" ROW "+str((past_frame - 1) * (dim_in + 1))+" action "+str(episode.action[max(frame - past_frame, 0)])+" relative pos "+str(max(frame - past_frame, 0)))

                        pos = (past_frame - 1) * (dim_in + 1) + int(episode.pedestrian_data[id].action[max(frame - past_frame, 0)])
                        values[pos] = 1
                        values[past_frame * (dim_in + 1) - 1] = episode.pedestrian_data[id].measures[
                            max(frame - past_frame, 0), PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                        #print (" pos "+str(pos)+" speed pos "+str(past_frame * (dim_in + 1) - 1)+" ")


                feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
                if self.settings.nbr_timesteps > 0:
                    values = np.zeros((dim_in + 1) * self.settings.nbr_timesteps)
                    speed_init=np.linalg.norm(episode.pedestrian_data[id].vel_init)
                    # if frame==0:
                    #     print ("Vel init "+str(episode.vel_init))
                    dir_init=episode.find_action_to_direction(episode.pedestrian_data[id].vel_init, speed_init)
                    for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 1):
                        if frame - past_frame >= 0:
                            # print ("frame " + str(past_frame) + " relative " + str(max(agent_frame - past_frame, 0)))
                            # print ("pos " + str((past_frame - 2) * (dim_in + 1) + int(
                            #     episode.action[max(agent_frame - past_frame, 0)])) + " row: " + str(
                            #     (past_frame - 2) * (dim_in + 1)) + " action " + str(
                            #     episode.action[max(agent_frame - past_frame, 0)]))
                            # print ("speed: pos " + str(past_frame * past_frame - 2) + " val " + str(
                            #     episode.measures[max(agent_frame - past_frame, 0), 3]))
                            values[(past_frame - 2) * (dim_in + 1) + int(
                                episode.pedestrian_data[id].action[max(frame - past_frame, 0)])] = 1
                            values[(past_frame - 2) * (dim_in + 1) - 1] = episode.pedestrian_data[id].measures[
                                max(frame - past_frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                        elif self.settings.past_filler:
                            values[(past_frame - 2) * (dim_in + 1) + int(dir_init)] = 1
                            values[(past_frame - 2) * (dim_in + 1) - 1] = speed_init
                    feed_dict[self.action_mem_further] = np.expand_dims(values, axis=0)
                # else:
                #     # if self.settings.nbr_timesteps > 0:
                #     values = np.zeros(self.settings.nbr_timesteps * (self.fully_connected_size(9) + 1))
                #     for past_frame in range(1, self.settings.nbr_timesteps + 1):
                #         if frame - past_frame >= 0:
                #             # print str(len(episode.action))+" "+str(max(agent_frame - past_frame, 0))
                #             # print (past_frame-1)*(dim_in+1)+int(episode.action[max(agent_frame - past_frame, 0)])
                #             # print values.shape
                #             values[(past_frame - 1) * (dim_in + 1) + int(
                #                 episode.action[max(agent_frame - past_frame, 0)])] = 1
                #             values[past_frame * past_frame - 1] = episode.measures[
                #                 max(agent_frame - past_frame, 0), 3]
                #     feed_dict[self.action_mem] = np.expand_dims(values, axis=0)


    def get_sample(self,id, statistics, ep_itr, agent_frame):
        return [statistics[ep_itr, id,agent_frame, STATISTICS_INDX.action]]
class PoPSoftMaxNet(SoftMaxNet):

    def define_loss(self, dim_p):

        self.policy_old_log = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="policy_old_log")
        self.prob_log=tf.math.log(self.probabilities)
        self.sample = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="sample")
        self.responsible_output_log = tf.slice(self.prob_log, self.sample, [1])+ np.finfo(
                np.float32).eps
        self.ratio=tf.exp(self.responsible_output_log-self.policy_old_log)
        self.loss1 =-self.ratio * self.advantages
        self.loss2 = -tf.clip_by_value(self.ratio, 1-self.epsilon, 1+self.epsilon) * self.advantages
        self.loss=tf.reduce_mean(input_tensor=tf.maximum(self.loss1, self.loss2))
        return self.sample, self.loss

    def construct_feed_dict(self,id, episode, frame,agent_frame, training=True):
        feed_dict = super(PoPSoftMaxNet, self).construct_feed_dict(id,episode, frame, agent_frame)

        feed_dict[self.policy_old_log] = np.log(episode.probabilities[max(frame-2), episode.action[max(frame-1,0)]])
        return feed_dict

    def grad_feed_dict(self, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode, agent_frame,
                       reward, statistics, agent_speed=None, training=True, frame=-1):
        probabilities = statistics[ep_itr, frame-2,STATISTICS_INDX.probabilities[0]:STATISTICS_INDX.probabilities[1]]
        grad_feed_dict=super(agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode, agent_frame,
                       reward, statistics, agent_speed=agent_speed, training=training, frame=frame)
        feed_dict[self.policy_old_log] = np.log(probabilities[grad_feed_dict[self.sample]])

