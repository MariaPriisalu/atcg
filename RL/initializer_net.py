import numpy as np

from net import Net
from settings import NBR_MEASURES,RANDOM_SEED, STATISTICS_INDX_MAP,STATISTICS_INDX_CAR_INIT,STATISTICS_INDX,PEDESTRIAN_MEASURES_INDX

import tensorflow as tf

tf.random.set_seed(RANDOM_SEED)
import tensorflow_probability as tfp
tfd = tfp.distributions
from dotmap import DotMap
import pickle

import copy
from commonUtils.ReconstructionUtils import NUM_SEM_CLASSES, CHANNELS,cityscapes_labels_dict




class InitializerNet(Net):
    # Softmax Simplified
    def __init__(self, settings, weights_name="init") :
        self.labels_indx = {11: 0, 13: 1, 14: 2, 4: 2, 5: 2, 15: 2, 16: 2, 17: 3, 18: 3, 7: 4, 9: 4, 6: 4, 10: 4,
                            8: 5, 21: 6, 22: 6, 12: 7, 20: 8, 19: 8}#,34 :9, 35:10}


        self.valid_pos=[]
        self.probabilities_saved=[]
        self.carla = settings.carla
        self.set_nbr_channels()
        self.channels = DotMap()
        self.channels.rgb = [0, 3]
        self.channels.pedestrian_trajectory = 3
        self.channels.cars_trajectory = 4
        self.channels.pedestrians = 5
        self.channels.cars = 6
        self.channels.semantic = 7

        #self.temporal_scaling = 0.1 * 0.3

        super(InitializerNet, self).__init__(settings, weights_name=weights_name)
        if settings.printdebug_network_input and self.is_net_type(settings.printdebug_network):
            self.feed_dict=[]

    def get_goal(self, statistics, ep_itr, agent_frame, initialization_car):
        #print ("Get goal "+str(statistics[ep_itr, 3:5, 38 + NBR_MEASURES].astype(int))+" size "+str(self.settings.env_shape[1:]) )
        if self.settings.goal_gaussian:
            goal=statistics[ep_itr, agent_frame, STATISTICS_INDX.goal[0]:STATISTICS_INDX.goal[1]]
            agent_pos=statistics[ep_itr, 0,STATISTICS_INDX.agent_pos[0]+1:STATISTICS_INDX.agent_pos[1]]
            manual_goal=initialization_car[ep_itr,STATISTICS_INDX_CAR_INIT.manual_goal[0]:STATISTICS_INDX_CAR_INIT.manual_goal[1]]
            goal_dir=goal-manual_goal#agent_pos
            goal_dir[0]=goal_dir[0]#/self.settings.env_shape[1]
            goal_dir[1] = goal_dir[1]# / self.settings.env_shape[2]
            # print ("Get goal " + str(goal_dir))
            return goal_dir
        else:
            # print ("Get goal " + str(
            #     np.ravel_multi_index( statistics[ep_itr, 3:5, 38 + NBR_MEASURES].astype(int),
            #                          self.settings.env_shape[1:])))
            return [np.ravel_multi_index( statistics[ep_itr, agent_frame, STATISTICS_INDX.goal[0]:STATISTICS_INDX.goal[1]].astype(int),
                                    self.settings.env_shape[1:])]

    def set_nbr_channels(self):
        self.nbr_channels = 9 + 7

    def define_conv_layers(self):
        #tf.reset_default_graph()
        output_channels=1

        self.state_in = tf.compat.v1.placeholder(shape=[ self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], self.nbr_channels],
                                       dtype=self.DTYPE,
                                       name="reconstruction")
        self.prior = tf.compat.v1.placeholder(
            shape=[self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2],1],
            dtype=self.DTYPE,
            name="prior")

        #mean = tf.constant(self.mean, dtype=self.DTYPE)  # (3)
        padding = 'SAME'
        if self.settings.inilitializer_interpolate:
            padding = 'VALID'
        prev_layer = tf.concat([self.state_in, self.prior], axis=3)  # - mean
        if self.settings.num_layers_init==1:
            with tf.compat.v1.variable_scope('conv1') as scope:
                out_filters = self.settings.outfilters[0]
                # print("Define 2D weights") # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
                #kernel1 = tf.get_variable('weights', self.DTYPE,[ 3, 3, self.nbr_channels, out_filters],initializer=#tf.constant_initializer(1))#tf.keras.initializers.GlorotNormal(RANDOM_SEED))

                # kernel1 = tf.compat.v1.get_variable('weights', [self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, output_channels], self.DTYPE,
                #                           initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                kernel1 = tf.compat.v1.get_variable('weights', [self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, output_channels], self.DTYPE,
                                          initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                self.conv1 = tf.nn.conv2d(prev_layer, kernel1,strides=[1,1,1,1], padding=padding)  # [1, 2, 2, 2, 1]

                #biases1 = self.bias_variable('biases', [out_filters])

                #self.bias1 =self.conv_out1#tf.nn.bias_add(self.conv_out1, biases1)
                #self.conv1 #=tf.nn.relu(self.bias1, name=scope.name)

                if self.settings.learn_goal and not self.settings.goal_gaussian:
                    kernel1_goal = tf.compat.v1.get_variable('weights_goal',
                                              [self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels + 1,output_channels], self.DTYPE,initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
                                              #initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

                    self.conv1_goal = tf.nn.conv2d(prev_layer, kernel1_goal,strides=[1,1,1,1], padding=padding)

                    prev_layer = tf.concat([self.conv1, self.conv1_goal],axis=3)
                else:
                    prev_layer = self.conv1

                if self.settings.inilitializer_interpolate:
                    prev_layer = tf.image.resize(prev_layer,[self.settings.env_shape[1],
                                                                  self.settings.env_shape[2]])
        else:

            # print ("Two layers!")
            with tf.compat.v1.variable_scope('conv1') as scope:
                out_filters = self.settings.outfilters[0]
                # print("Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
                kernel1 = tf.compat.v1.get_variable('weights',
                                          [3, 3, self.nbr_channels + 1,
                                           output_channels], self.DTYPE,
                                          initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED)) # initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                self.conv1 = tf.nn.conv2d(prev_layer, kernel1,strides=[1,1,1,1], padding=padding)  # [1, 2, 2, 2, 1]
                prev_layer = self.conv1
                if 1 in self.settings.pooling:
                    self.pooled = tf.nn.max_pool(prev_layer, [1, 2, 2, 1], [1, 1, 1, 1], padding) # tf.nn.max_pool2d(input=prev_layer, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding=padding)  # , 'SAME')

                    prev_layer = self.pooled
                    # print("Pooling 1")
                if self.settings.inilitializer_interpolate and self.settings.inilitializer_interpolated_add:
                    layer_1_interpolated = tf.image.resize(prev_layer,[self.settings.env_shape[1],
                                                                  self.settings.env_shape[2]])
                    # print(" After reshaping size " + str(layer_1_interpolated.shape))
                in_filters = out_filters

                with tf.compat.v1.variable_scope('conv2') as scope:
                    out_filters = self.settings.outfilters[1]
                    kernel2 = tf.compat.v1.get_variable('weights_conv2', [2, 2, in_filters, out_filters], self.DTYPE,
                                              initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))

                    #self.conv_out2 = tf.nn.conv2d(input=prev_layer, filters=kernel2, strides=[1, 1, 1, 1], padding=padding)  # [1, 2, 2, 2, 1]
                    self.conv_out2 = tf.nn.conv2d(prev_layer, kernel2, strides=[1,1,1,1], padding=padding)  # [1, 2, 2, 2, 1]
                    if self.settings.conv_bias:
                        biases2 = self.bias_variable('biases', [out_filters])

                        self.bias2 = tf.nn.bias_add(self.conv_out2, biases2)
                    else:
                        self.bias2 = self.conv_out2

                    self.conv2 = tf.nn.relu(self.bias2, name=scope.name)
                    # variable_summaries(biases2, name='conv2b', summaries=self.conv_summaries)

                prev_layer = self.conv2
                if 2 in self.settings.pooling:
                    self.pooled2 = tf.nn.max_pool2d(input=prev_layer, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding=padding)  # , 'SAME')
                    self.pooled2 = tf.nn.max_pool(prev_layer, [1, 2, 2, 1], [1, 1, 1, 1], padding)  # , 'SAME')

                    prev_layer = self.pooled2
                    # print("Pooling 2")

                if self.settings.inilitializer_interpolate:
                    layer_2_interpolated = tf.image.resize(prev_layer, [self.settings.env_shape[1],
                                                                              self.settings.env_shape[2]])
                    # print(" After reshaping size "+str(layer_2_interpolated.shape))
                    if self.settings.inilitializer_interpolated_add:
                        self.conv_output = layer_2_interpolated+ layer_1_interpolated
                    else:
                        self.conv_output = layer_2_interpolated
                    prev_layer=self.conv_output
                else:
                    self.conv_output = prev_layer

        return prev_layer


    def define_loss(self, dim_p):

        self.sample = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="sample")
        self.responsible_output = tf.slice(self.probabilities, self.sample, [1]) + np.finfo(
            np.float32).eps
        if self.settings.learn_goal and not self.settings.goal_gaussian:
            self.prior_goal = tf.compat.v1.placeholder(
                shape=[self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], 1],
                dtype=self.DTYPE,
                name="prior_goal")
            self.prior_flat_goal = tf.reshape(self.prior_goal, [
                self.settings.batch_size * self.settings.env_shape[1] * self.settings.env_shape[2]])
            self.goal = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="goal")
            self.responsible_output_goal = tf.slice(self.probabilities_goal, self.goal, [1]) + np.finfo(
                np.float32).eps
            self.responsible_output_prior_goal = tf.slice(self.prior_flat_goal, self.goal, [1]) + np.finfo(
                np.float32).eps
        self.prior_flat= tf.reshape(self.prior, [self.settings.batch_size*self.settings.env_shape[1]*self.settings.env_shape[2]])
        self.responsible_output_prior=tf.slice(self.prior_flat, self.sample, [1]) + np.finfo(
            np.float32).eps
        self.distribution=self.prior_flat*self.probabilities


        self.loss = -tf.reduce_mean(input_tensor=(tf.math.log(self.responsible_output) )* self.advantages)
        if self.settings.learn_goal and not self.settings.goal_gaussian:
            self.loss -= tf.reduce_mean(
                input_tensor=(tf.math.log(self.responsible_output_goal) ) * self.advantages)

        if self.settings.entr_par_init: # To do: add entropy for goal!
            y_zeros = tf.zeros_like(self.distribution)
            y_mask = tf.math.greater(self.distribution, y_zeros)
            res = tf.boolean_mask(tensor=self.distribution, mask=y_mask)
            logres = tf.math.log(res)

            # self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(res), logits=logres)
            # self.loss =self.loss -tf.reduce_mean(input_tensor=self.settings.entr_par_init*self.entropy)
            self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=res, logits=logres)
            self.loss =self.loss -tf.reduce_mean(self.settings.entr_par_init*self.entropy)

        if self.settings.learn_goal and self.settings.goal_gaussian:
            self.goal = tf.compat.v1.placeholder(shape=[2], dtype=self.DTYPE, name="goal")
            self.normal_dist = tfd.Normal(self.probabilities_goal, self.settings.goal_std)
            self.responsible_output = self.normal_dist.prob(self.goal)

            self.l2_loss = tf.nn.l2_loss(self.probabilities_goal - self.goal)  # tf.nn.l2_loss
            # self.loss = tf.reduce_mean(self.advantages * (2 * tf.log(self.settings.sigma_vel) + tf.log(2 * tf.pi) + (
            # self.l2_loss / (self.settings.sigma_vel * self.settings.sigma_vel))))
            self.loss =self.loss + tf.reduce_mean(input_tensor=self.advantages * self.l2_loss)

        if self.settings.learn_time and self.goal_net:
            self.time_requirement = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 1], dtype=self.DTYPE,
                                                        name="time_requirement")
            self.beta_distr=tfd.Beta(self.alpha, self.beta)
            self.time_prob=self.beta_distr.prob(self.time_requirement)
            self.loss = self.loss - tf.reduce_mean(input_tensor=self.advantages * tf.math.log(self.time_prob))

        return self.sample, self.loss

    def get_feature_vectors_gradient(self,id, agent_action, agent_frame, agent_measures, agent_pos, agent_speed,agent_velocity,ep_itr,
                                     episode, feed_dict, frame, poses, statistics, training,statistics_car=[]):
        pass

    def fully_connected(self, dim_p, prev_layer):
        # print ("Fully connected flattened layer: "+str(prev_layer))

        if self.settings.learn_goal and not self.settings.goal_gaussian:
            dim = np.prod(prev_layer.get_shape().as_list()[1:-1])
            # print ("Flattened size "+str(dim) +" shape "+str(prev_layer.get_shape()))
            self.flattened_layer = tf.reshape(prev_layer[:,:,:,0], [dim])
            self.flattened_layer_goal = tf.reshape(prev_layer[:, :, :, 1], [ dim])
        else:
            dim = np.prod(prev_layer.get_shape().as_list()[1:])
            self.flattened_layer=tf.reshape(prev_layer, [-1])
            if self.settings.learn_goal:

                weights = tf.compat.v1.get_variable('weights_goal', [dim, 2], self.DTYPE,
                                          initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                biases = self.bias_variable('biases_goal', [2])
                self.flattened_layer_goal = tf.matmul(tf.reshape(self.flattened_layer, [1,dim]), weights)
                self.flattened_layer_goal = tf.add(self.flattened_layer_goal, biases)

        if self.settings.learn_time and self.goal_net:
            weights_time = tf.compat.v1.get_variable('weights_time', [dim, 2], self.DTYPE,
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            biases_time = self.bias_variable('biases_time', [2])
            self.flattened_layer_time = tf.matmul(tf.reshape(self.flattened_layer, [1, dim]), weights_time)
            self.flattened_layer_time = tf.add(self.flattened_layer_time, biases_time)


    #return [statistics[ep_itr, agent_frame, 6]]
    def get_sample(self, id, statistics,ep_itr, agent_frame,  initialization_car):
        #print ("Get sample "+str(np.ravel_multi_index(([int(statistics[ep_itr,id, 0, STATISTICS_INDX.agent_pos[0]+1])], [int(statistics[ep_itr,id, 0, STATISTICS_INDX.agent_pos[0]+2])]), self.settings.env_shape[1:] ))+" "+str([int(statistics[ep_itr,id, 0, STATISTICS_INDX.agent_pos[0]+1])])+" "+str([int(statistics[ep_itr,id, 0, STATISTICS_INDX.agent_pos[0]+2])]))
        return np.ravel_multi_index(([int(statistics[ep_itr,id, 0,STATISTICS_INDX.agent_pos[0]+1])], [int(statistics[ep_itr,id, 0,STATISTICS_INDX.agent_pos[0]+2])]), self.settings.env_shape[1:] )


    def get_input(self, id, episode, agent_pos_cur, frame_in=-1, training=True):

        return episode.reconstruction

    def calc_probabilities(self, fc_size):
        #print ("Define probabilities: " + str(self.flattened_layer))
        if self.settings.learn_goal :
            if not self.settings.goal_gaussian:
                self.probabilities_goal=tf.nn.softmax(self.flattened_layer_goal)
            else:
                self.probabilities_goal = tf.nn.sigmoid(self.flattened_layer_goal)

            if self.settings.learn_time and self.goal_net:
                self.probabilities_time = self.flattened_layer_time
                #print (self.probabilities_time.shape)
                self.alpha=tf.math.abs(self.probabilities_time[0,0])+1e-5
                self.beta = tf.nn.relu(self.probabilities_time[0,1])+1e-5

        self.probabilities = tf.nn.softmax(self.flattened_layer)

    def importance_sample_weight(self,id,responsible, statistics, ep_itr, frame, responsible_v=0):
        pass

    def get_feature_vectors(self,id,  agent_frame, episode, feed_dict, frame):
        pass

    def adapt_goal(self,time_to_goal, seq_len, goal, vel_init):
        new_goal=goal+ (abs(seq_len-time_to_goal)*vel_init)
        return new_goal
        # points=[[0, shape[1]], [0,shape[2]]]
        # for dim, point_list in enumerate(points):
        #     other_dim=1-dim
        #     if abs(vel_init[dim])>1e-5:
        #         for point in point_list:
        #             t=(point-goal[dim+1])/vel_init[dim+1]
        #             other_coordinate=goal[other_dim+1] +(t*vel_init[other_dim+1])
        #             if t>0 and points[other_dim][0]<=other_coordinate and other_coordinate<points[other_dim][1]:
        #                 return goal +(t*vel_init)

    def apply_net(self,id,  feed_dict, episode, frame, training, max_val=False, viz=False, manual=False):
        # Choose car
        # print(" Apply initializer")
        episode.pedestrian_data[id].init_method = 7
        self.get_car(episode, id)

        episode.calculate_prior(id, self.settings.field_of_view_car)

        probabilities, probabilities_goal = self.run_net(episode, feed_dict, id, manual, episode.initializer_data[id].prior)
        # Vector from pedestrian to car in voxels
        vector_car_to_pedestrian = episode.pedestrian_data[id].agent[0][1:] - episode.initializer_data[id].init_car_pos

        # print ("Car velocity "+str(episode.init_car_vel)+ " vector car to pedestrian "+str(vector_car_to_pedestrian))
        # print ("Vector car to pedestrian "+str(vector_car_to_pedestrian))
        if np.linalg.norm(episode.initializer_data[id].init_car_vel) < 0.01:
            speed=3 * 5  # Speed voxels/second
            # print ("Desired speed pedestrian " + str(speed * .2) + " car vel " + str(episode.init_car_vel * .2))

            # Unit orthogonal direction
            unit = -vector_car_to_pedestrian * (
                1 / np.linalg.norm(vector_car_to_pedestrian))

            # Set ortogonal direction to car
            episode.pedestrian_data[id].vel_init = np.array([0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly
            episode.pedestrian_data[id].speed_init = np.linalg.norm(episode.pedestrian_data[id].vel_init)
            if episode.follow_goal:
                episode.pedestrian_data[id].goal[0,:]=episode.pedestrian_data[id].agent[0].copy()
                episode.pedestrian_data[id].goal[0,1:] = episode.initializer_data[id].init_car_pos
                if not self.settings.stop_on_goal and self.settings.longer_goal:
                    time=np.linalg.norm(episode.pedestrian_data[id].goal[0,1:]-episode.pedestrian_data[id].agent[0][1:])/np.linalg.norm(episode.pedestrian_data[id].vel_init[1:])
                    # time_to_goal, seq_len, goal, vel_init):

                    episode.pedestrian_data[id].goal[0,1:]=self.adapt_goal(time, episode.seq_len, episode.pedestrian_data[id].goal[0,1:],  episode.pedestrian_data[id].vel_init[1:])#, episode.reconstruction.shape)

                if self.settings.printdebug_network_input:
                    print (" car still Pedestrian goal init " + str(episode.pedestrian_data[id].goal))
                episode.pedestrian_data[id].manual_goal = episode.pedestrian_data[id].goal[0,1:].copy()
                if self.settings.learn_goal:
                    if self.settings.goal_gaussian:


                        #print ("Before adding random goal : " + str(episode.goal)+" model output "+str(probabilities_goal))
                        random_numbers = [np.random.normal(probabilities_goal[0][0], self.settings.goal_std, 1),np.random.normal(probabilities_goal[0][1], self.settings.goal_std, 1)]
                        # pr1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,int ("Randon numbers " + str(random_numbers)+ " mean "+str(probabilities_goal[0][0]))
                        random_numbers[0] = random_numbers[0]#* self.settings.env_shape[1]
                        random_numbers[1] = random_numbers[1] #* self.settings.env_shape[2]
                        # print ("Before adding random goal : " + str(episode.goal) + " model output " + str(probabilities_goal))
                        # print ("Randon numbers after scaling " + str(random_numbers)+ "  scaled mean "+str(probabilities_goal[0][0]* self.settings.env_shape[1])+ "  scaled mean "+str(probabilities_goal[0][1]* self.settings.env_shape[2]))
                        # print ("Final mean " + str(
                        #     probabilities_goal[0][0] +episode.goal[1]) + "  scaled mean " + str(
                        #     probabilities_goal[0][1] +episode.goal[2]))

                        episode.goal[0,1]+=random_numbers[0]
                        episode.goal[0,2]+= random_numbers[1]
                        # print ("After addition: "+str(episode.goal))
                    else:
                        indx = np.random.choice(range(len(probabilities)), p=np.copy(probabilities_goal))
                        pos = np.unravel_index(indx, self.settings.env_shape[1:])
                        episode.goal[0,0] = episode.get_height_init()
                        episode.goal[0,1] = pos[0]
                        episode.goal[0,2] = pos[1]
                #episode.goal[2] += goal[1][0] * 256
                # print (" car still Pedestrian goal " + str(episode.goal))
            if self.settings.speed_input:
                if self.settings.learn_time:
                    episode.pedestrian_data[id].goal_time[0] = episode.pedestrian_data[id].goal_time[0]*3*5*episode.frame_time

                    episode.pedestrian_data[id].goal_time[0] = np.linalg.norm(episode.pedestrian_data[id].goal[0,1:] - episode.pedestrian_data[id].agent[0][1:])/episode.pedestrian_data[id].goal_time[0]
                    if self.settings.printdebug_network_input:
                        print("Episode speed " + str(episode.pedestrian_data[id].goal_time[0]) + " factor " + str(
                            15 * episode.frame_time) + " frametime " + str(episode.frame_time))
                        print ("Episode goal time " + str(episode.pedestrian_data[id].goal_time[0] ))
                else:

                    episode.pedestrian_data[id].goal_time[0] = min(np.linalg.norm(vector_car_to_pedestrian)/episode.pedestrian_data[id].speed_init, episode.seq_len-2)
                    if self.settings.printdebug_network_input:
                        print("Episode goal time " + str(episode.pedestrian_data[id].goal_time[0]))
                        print ("Car standing, Pedestrian initial speed: voxels / frame " + str(episode.pedestrian_data[id].speed_init) + " dist " + str(
                            np.linalg.norm(vector_car_to_pedestrian)) + " timeframe " + str(
                            episode.pedestrian_data[id].goal_time[0]))
                        # if self.settings.learn_goal:
                        #     episode.goal_time = episode.seq_len - 2



            #print (" prior "+str(np.sum(feed_dict[self.prior])))

            return episode.pedestrian_data[id].agent[0], 11, episode.pedestrian_data[id].vel_init  # pos, indx, vel_init
        # set initial pedestrian velocity orthogonal to car!- voxels
        # time to collision in seconds

        time_to_collision=np.dot(vector_car_to_pedestrian, episode.initializer_data[id].init_car_vel) / (np.linalg.norm(episode.initializer_data[id].init_car_vel)** 2 )
        vector_travelled_by_car_to_collision = time_to_collision* episode.initializer_data[id].init_car_vel
        #time to collision in frames:
        time_to_collision = time_to_collision*episode.frame_rate

        # print ("scalar product between car direction "+str(episode.init_car_vel)+" and vector car to pedestrian "+str(vector_car_to_pedestrian)+ " : "+ str(np.dot(vector_car_to_pedestrian, episode.init_car_vel)))
        # # #
        # print ("Car speed "+str(np.linalg.norm(episode.init_car_vel))+" time to collision car "+str(time_to_collision)+" in s "+str(time_to_collision/episode.frame_rate))
        vector_travelled_by_pedestrian_to_collision =  vector_travelled_by_car_to_collision-vector_car_to_pedestrian

        # print ("Vector to collision by car "+str(vector_travelled_by_car_to_collision)+" pos at collision "+str(vector_travelled_by_car_to_collision+episode.init_car_pos))
        # print ("Vector to collision by ped " + str(vector_travelled_by_pedestrian_to_collision) + " pos at collision " + str(
        #     vector_travelled_by_pedestrian_to_collision + episode.agent[0][1:]))

        # pedestrian speed in voxels per frame
        speed =np.linalg.norm(vector_travelled_by_pedestrian_to_collision)/time_to_collision
        if speed>0.01:

            # print ("Desired speed pedestrian " + str(speed ) + " dist to travel "+str(np.linalg.norm(vector_travelled_by_pedestrian_to_collision)) )

            speed=min(speed, 3*5*episode.frame_time)  # Speed voxels/second, max speed= 3m/s, 5 voxels per meter

            time_to_collision=np.linalg.norm(vector_travelled_by_pedestrian_to_collision)/speed

            # Unit orthogonal direction
            unit = vector_travelled_by_pedestrian_to_collision * (1 / np.linalg.norm(vector_travelled_by_pedestrian_to_collision))

            # Set ortogonal direction to car
            episode.pedestrian_data[id].vel_init = np.array([0, unit[0], unit[1]]) * speed  # set this correctly

            # print ("Pedestrian intercepting car? "+str(episode.intercept_car(0, all_frames=False)))
            episode.pedestrian_data[id].speed_init = np.linalg.norm(episode.pedestrian_data[id].vel_init)
        else:
            speed = 3 * 5  # Speed voxels/second

            # Set agent movement to car position
            unit = -vector_car_to_pedestrian * (
                1 / np.linalg.norm(vector_car_to_pedestrian))

            # Set ortogonal direction to car
            episode.pedestrian_data[id].vel_init = np.array(
                [0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly
            episode.pedestrian_data[id].speed_init = np.linalg.norm(episode.pedestrian_data[id].vel_init)

        if episode.follow_goal:
            episode.pedestrian_data[id].goal[0,:]=episode.pedestrian_data[id].agent[0]+(2*episode.pedestrian_data[id].vel_init*time_to_collision)
            if not self.settings.stop_on_goal and self.settings.longer_goal:
                # time_to_goal, seq_len, goal, vel_init):
                episode.pedestrian_data[id].goal[0, 1:] = self.adapt_goal(2*time_to_collision,episode.seq_len, episode.pedestrian_data[id].goal[0, 1:],
                                                                         episode.pedestrian_data[id].vel_init[1:])
                                                                         #episode.reconstruction.shape)
            episode.pedestrian_data[id].manual_goal = episode.pedestrian_data[id].goal[0,1:].copy()
            if self.settings.printdebug_network_input:
                print ("Pedestrian goal init " + str(episode.pedestrian_data[id].goal[0,:]))
            if self.settings.learn_goal:

                if self.settings.goal_gaussian:

                    # print ("Before adding random goal : " + str(episode.goal) + " model output " + str(probabilities_goal))
                    random_numbers = [np.random.normal(probabilities_goal[0][0], self.settings.goal_std, 1),
                                      np.random.normal(probabilities_goal[0][1], self.settings.goal_std, 1)]
                    # print ("Randon numbers " + str(random_numbers) + " mean " + str(probabilities_goal[0]))
                    random_numbers[0] = random_numbers[0]#"* self.settings.env_shape[1]
                    random_numbers[1] = random_numbers[1]# * self.settings.env_shape[2]
                    # print ("Init goal : " + str(episode.goal))
                    # print ("Randon numbers after scaling " + str(random_numbers) + "  scaled mean " + str(
                    #     probabilities_goal[0][0]) + "  scaled mean " + str(
                    #     probabilities_goal[0][1] ))
                    # print ("Final mean " + str(
                    #     probabilities_goal[0][0]  + episode.goal[
                    #         1]) + "  scaled mean " + str(
                    #     probabilities_goal[0][1]  + episode.goal[2]))

                    episode.pedestrian_data[id].goal[0,1] += random_numbers[0]
                    episode.pedestrian_data[id].goal[0,2] += random_numbers[1]
                else:
                    indx = np.random.choice(range(len(probabilities_goal)), p=np.copy(probabilities_goal))
                    pos = np.unravel_index(indx, self.settings.env_shape[1:])
                    episode.pedestrian_data[id].goal[0,0] = episode.get_height_init()
                    episode.pedestrian_data[id].goal[0,1] = pos[0]
                    episode.pedestrian_data[id].goal[0,2] = pos[1]
                    # print ("Goal pos "+str(indx)+" in pos "+str(pos))

            # print ("Pedestrian goal "+str(episode.goal))
        if self.settings.speed_input:
            if self.settings.learn_time:
                episode.pedestrian_data[id].goal_time[0] = episode.pedestrian_data[id].goal_time[0] * 3 * 5 * episode.frame_time

                if episode.pedestrian_data[id].goal_time[0]!=0:
                    episode.pedestrian_data[id].goal_time[0] =  np.linalg.norm(episode.pedestrian_data[id].goal[0,1:] - episode.pedestrian_data[id].agent[0][1:])/episode.pedestrian_data[id].goal_time[0]
                if self.settings.printdebug_network_input:
                    print("Episode speed " + str(episode.pedestrian_data[id].goal_time[0]) + " factor " + str(
                        15 * episode.frame_time) + " frametime " + str(episode.frame_time))
                    print ("Episode goal time " + str(episode.pedestrian_data[id].goal_time[0])+" dist "+str(np.linalg.norm(episode.pedestrian_data[id].goal[0,1:] - episode.pedestrian_data[id].agent[0][1:])))
            else:
                episode.pedestrian_data[id].goal_time[0] = min(time_to_collision*2,  episode.seq_len-2)
                if self.settings.learn_goal:
                    episode.goal_time[0] =  episode.seq_len - 2
                    if self.settings.printdebug_network_input:
                        print("Episode goal time " + str(episode.pedestrian_data[id].goal_time[0]))
        if self.settings.printdebug_network_input:
            print ("Pedestrian initial speed: voxels / frame "+str(episode.pedestrian_data[id].speed_init)+" dist "+str(np.linalg.norm(vector_travelled_by_pedestrian_to_collision) )+" timeframe "+str(episode.pedestrian_data[id].goal_time[0]))
            print ("Pedestrian vel " + str(episode.pedestrian_data[id].vel_init[1:])+" time to goal "+str(episode.pedestrian_data[id].goal_time[0])+" dist travelled "+str(episode.pedestrian_data[id].goal_time[0]*episode.pedestrian_data[id].vel_init)+" final pos "+str(episode.pedestrian_data[id].agent[0]+episode.pedestrian_data[id].goal_time[0]*episode.pedestrian_data[id].vel_init) )

        # print (" prior " + str(np.sum(feed_dict[self.prior])))
        return episode.pedestrian_data[id].agent[0], 11,episode.pedestrian_data[id].vel_init #  pos, indx, vel_init

    def run_net(self, episode, feed_dict, id, manual, prior):
        probabilities_goal=None
        flat_prior =prior.flatten()
        feed_dict[self.prior] = np.expand_dims(np.expand_dims(prior * (1 / max(flat_prior)), axis=0), axis=-1)
        self.do_debug_printout(id, feed_dict)
        if self.settings.random_init:
            flat_prior[flat_prior>0]=1
        if (self.settings.evaluate_prior and self.init_net) or (self.settings.evaluate_prior_car and self.init_net_car):
            probabilities = np.ones_like(flat_prior)
        else:
            if self.settings.learn_goal and not self.init_net_car:
                if self.settings.learn_time:
                    probabilities, probabilities_goal, flattend_layer, conv_1, alpha, beta, summary_train = self.sess.run(
                        [self.probabilities, self.probabilities_goal, self.flattened_layer, self.conv1, self.alpha,
                         self.beta,
                         self.train_summaries, ], feed_dict)
                    episode.pedestrian_data[id].goal_time[0] = np.random.beta(alpha, beta)
                    # print ("Model outputs alpha "+str(alpha)+" beta "+str(beta)+" factor "+str(episode.goal_time))

                else:
                    probabilities, probabilities_goal, flattend_layer, conv_1, summary_train = self.sess.run(
                        [self.probabilities, self.probabilities_goal, self.flattened_layer, self.conv1,
                         self.train_summaries], feed_dict)
                if self.settings.goal_gaussian:
                    # print ( "Gaussian model output: "+str(probabilities_goal))
                    episode.initializer_data[id].goal_distribution = np.copy(probabilities_goal)
                    # probabilities_goal[0][0]=probabilities_goal[0][0]
                    # probabilities_goal[0][1] = probabilities_goal[0][1] * self.settings.env_shape[2]
                    # print ("After scaling: " + str(probabilities_goal))
                else:
                    episode.initializer_data[id].goal_distribution = np.copy(probabilities_goal)
                    flat_prior_goal = episode.initializer_data[id].goal_prior.flatten()
                    probabilities_goal = probabilities_goal * flat_prior_goal
                    probabilities_goal = probabilities_goal * (1 / np.sum(probabilities_goal))

            else:
                probabilities, flattend_layer, conv_1, conv_2, pooled, pooled_2, summary_train = self.sess.run(
                    [self.probabilities, self.flattened_layer, self.conv1, self.conv2, self.pooled, self.pooled2,
                     self.train_summaries], feed_dict)

        if self.settings.printdebug_network_input:
            print(" Probabilities max pos " + str(
                np.unravel_index(np.where(probabilities == np.max(probabilities[:])), self.settings.env_shape[1:])))
            print(" Prior max pos " + str(
                np.unravel_index(np.where(flat_prior == np.max(flat_prior[:])), self.settings.env_shape[1:])))
        self.place_initializer_in_episode(episode, id, probabilities)
        # print (" Prior less than 0 " + str(np.unravel_index(np.where(flat_prior<0), self.settings.env_shape[1:])))
        # print("Prior sum "+str(np.sum(probabilities)))
        if self.settings.printdebug_network_input:
            print(" Prior less than 0 " + str(np.unravel_index(np.where(flat_prior < 0), self.settings.env_shape[1:])))
            print("Prior sum " + str(np.sum(probabilities)))
        sampling_probabilities = probabilities * flat_prior  # episode.prior.flatten()
        sampling_probabilities = sampling_probabilities * (1 / np.sum(sampling_probabilities))
        if self.settings.printdebug_network_input:
            print("After multiplication with prior probabilities max pos " + str(
                np.unravel_index(np.where(sampling_probabilities == np.max(sampling_probabilities[:])),
                                 self.settings.env_shape[1:])))
        indx = np.random.choice(range(len(sampling_probabilities)), p=np.copy(sampling_probabilities))
        pos = np.unravel_index(indx, self.settings.env_shape[1:])
        self.place_pos_in_episode(episode, id, pos)
        if manual:
            print("Initialize pedestrian pos in 2D: " + str(pos) + " pos " + str(
                indx))

            value_y = input("Agent initial pos: y ")
            value_z=input("Agent initial pos: z ")
            pos=[int(eval(value_y)), int(eval(value_z))]
            if not len(value_y) == 0:
                self.place_pos_in_episode(episode, id, pos)
                indx = np.ravel_multi_index(([pos[0]],
                                             [pos[1]]),
                                            self.settings.env_shape[1:])
        self.do_debug_printout_sample(id, indx, probabilities)
        return probabilities, probabilities_goal

    def place_pos_in_episode(self, episode, id, pos):
        episode.pedestrian_data[id].agent[0][0] = episode.get_height_init()
        episode.pedestrian_data[id].agent[0][1] = pos[0]
        episode.pedestrian_data[id].agent[0][2] = pos[1]

    def place_initializer_in_episode(self, episode, id, probabilities):
        episode.initializer_data[id].init_distribution = np.copy(probabilities)

    def do_debug_printout_sample(self, id, indx, probabilities):
        if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
            import math
            self.feed_dict[-1][id]['responsible'] = probabilities[indx]
            self.feed_dict[-1][id]['ln(responsible)'] = math.log(probabilities[indx])
            self.feed_dict[-1][id][self.sample]=[indx]
            print("Responsible indx "+str(indx)+" probability " +str(probabilities[indx])+" log of probability "+str(math.log(probabilities[indx])) )

    def do_debug_printout(self, id, feed_dict):
        if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
            print("Feed dict init  ")

            for key, value in feed_dict.items():
                if key == self.state_in:
                    print("input size " + str(value.shape))

                    print("Input cars " + str(np.sum(value[0, :, :, 6])) + " people " + str(
                        np.sum(value[0, :, :, 5])) + " Input cars traj" + str(
                        np.sum(value[0, :, :, 4])) + " people traj " + str(
                        np.sum(value[0, :, :, 3])))
                    print("Input building " + str(np.sum(value[0, :, :, 7])) + " fence " + str(
                        np.sum(value[0, :, :, 7 + 1])) + " static " + str(
                        np.sum(value[0, :, :, 7 + 2])) + " pole " + str(
                        np.sum(value[0, :, :, 7 + 3])))
                    print("Input sidewalk " + str(np.sum(value[0, :, :, 7 + 5])) + " road " + str(
                        np.sum(value[0, :, :, 7 + 4])) + " veg. " + str(
                        np.sum(value[0, :, :, 7 + 6])) + " wall " + str(np.sum(value[0, :, :, 7 + 7])) + " sign " + str(
                        np.sum(value[0, :, :, 7 + 8])))
                    print("R " + str(np.sum(value[0, :, :, 0])) + "G " + str(np.sum(value[0, :, :, 1])) + "B " + str(
                        np.sum(value[0, :, :, 1])))
                    # print ("Cars: ")
                    # print (value[0, :, :, 6])
                    # print ("Cars traj : ")
                    # print (value[0, :, :, 4])
                    # print ("People:")
                    # print (value[0, :, :, 5])
                    # print ("People traj : ")
                    # print (value[0, :, :, 3])
                elif key == self.prior:
                    print("prior " + str(np.sum(value[0, :, :])))

                else:
                    print(key)
                    print(value)



            if id == 0:
                feed_dicts = []
                for id_local in range(self.settings.number_of_agents):
                    feed_dicts.append({})
                self.feed_dict.append(feed_dicts)
            feed_dict_copy = {}
            for key, value in feed_dict.items():
                feed_dict_copy[key] = copy.deepcopy(value)
            self.feed_dict[-1][id] = feed_dict_copy

    def get_car(self, episode, id):
        if episode.useRealTimeEnv and episode.number_of_car_agents > 0:
            # print(" Use ReaL TIME ")
            episode.initializer_data[id].init_car_id =0
            if self.settings.attack_random_car:
                episode.initializer_data[id].init_car_id = np.random.randint(0,episode.number_of_car_agents)  # which car to attack?
            car_id = episode.initializer_data[id].init_car_id
            episode.initializer_data[id].init_car_pos = episode.car_data[car_id].car[0][1:]

            episode.initializer_data[id].init_car_vel = episode.car_data[car_id].car_dir[1:] * episode.frame_rate  # voxels / second


            if self.settings.printdebug_network_input:
                if id==0:
                    print("Car pos " + str(episode.initializer_data[id].init_car_pos)+" Car vel " + str(episode.initializer_data[id].init_car_vel) + " m/s " + str(
                        episode.initializer_data[id].init_car_vel * .2))
            car_dim = episode.car_dim[1:]
            if abs(episode.initializer_data[id].init_car_vel[0]) > abs(episode.initializer_data[id].init_car_vel[1]):
                car_dim = [episode.car_dim[1], episode.car_dim[0]]
            episode.initializer_data[id].car_max_dim = max(car_dim)
            episode.initializer_data[id].car_min_dim = min(car_dim)
            episode.initializer_data[id].init_car_bbox = episode.car_data[car_id].car_bbox[0]
            episode.initializer_data[id].car_goal = episode.car_data[car_id].car_goal

        else:
            episode.initializer_data[id].init_car_id = np.random.choice(episode.init_cars)
            episode.initializer_data[id].init_car_pos = np.array(
                [np.mean(episode.cars_dict[episode.initializer_data[id].init_car_id][0][2:4]),
                 np.mean(episode.cars_dict[episode.initializer_data[id].init_car_id][0][4:])])
            car_pos_next = np.array([np.mean(episode.cars_dict[episode.initializer_data[id].init_car_id][1][2:4]),
                                     np.mean(episode.cars_dict[episode.initializer_data[id].init_car_id][1][4:])])
            episode.initializer_data[id].init_car_vel = (car_pos_next - episode.initializer_data[
                id].init_car_pos) / episode.frame_time
            # print ("Car pos "+str(episode.init_car_pos))
            # print ("Car vel " + str(episode.init_car_vel))
            car_dim = [episode.cars_dict[episode.initializer_data[id].init_car_id][0][3] -
                       episode.cars_dict[episode.initializer_data[id].init_car_id][0][2],
                       episode.cars_dict[episode.initializer_data[id].init_car_id][0][5] -
                       episode.cars_dict[episode.initializer_data[id].init_car_id][0][4]]
            episode.initializer_data[id].car_max_dim = max(car_dim)
            episode.initializer_data[id].car_min_dim = min(car_dim)
            episode.initializer_data[id].init_car_bbox = episode.cars_dict[episode.initializer_data[id].init_car_id][0]
            goal = episode.cars_dict[episode.initializer_data[id].init_car_id][-1]
            episode.initializer_data[id].car_goal = np.array(np.mean(goal[:2]), np.mean(goal[2:4]), np.mean(goal[4:]))

    def get_vel(self,id,  episode, frame):
        pass

    def fully_connected_size(self, dim_p):
        return 0

    def construct_feed_dict(self,id,  episode, frame, agent_frame,training=True,distracted=False):
        feed_dict ={}

        # print(" Ep use real time construct feed dict? " + str(episode.useRealTimeEnv))
        feed_dict[self.state_in] = self.get_input_init(id, episode, frame)
        return feed_dict

    def get_input_init(self, id, episode, frame=0):
        # print(" Ep use real time get_input_init? " + str(episode.useRealTimeEnv))
        sem = np.zeros(( self.settings.env_shape[1], self.settings.env_shape[2], self.nbr_channels))
        segmentation = (episode.reconstruction_2D[ :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(np.int)
        sem[:,:,self.channels.rgb[0]:self.channels.rgb[-1]]=episode.reconstruction_2D[ :, :, CHANNELS.rgb[0]:CHANNELS.rgb[-1]].copy()


        if not episode.useRealTimeEnv:
            # print(" Not using real time env")
            temp_people = episode.people_predicted[frame+1].copy()
            temp_cars = episode.cars_predicted[frame+1].copy()

        elif self.goal_net and frame>0:

            temp_people = episode.people_predicted[frame].copy()
            temp_cars = episode.cars_predicted[frame].copy()
            # To Do: add linear forward predictions here for people and cars in the frame!
        else:
            # print (" People predicted init ")
            temp_people = episode.people_predicted_init.copy()
            temp_cars = episode.cars_predicted_init.copy()

        if self.init_net or self.goal_net:
            if frame > 0 and self.goal_net:
                temp_people = episode.predict_controllable_pedestrian(id, temp_people, frame, add_all=True)
            else:
                temp_people=episode.predict_controllable_pedestrian(id, temp_people, frame)
            temp_cars = episode.predict_controllable_car(self.settings.number_of_car_agents, temp_cars, frame)
        else:
            temp_cars = episode.predict_controllable_car(id, temp_cars)

        if self.goal_net and frame > 0:
            temp_people = episode.predict_pedestrians( temp_people, frame)
            temp_cars = episode.predict_cars(temp_cars, frame)

        # print ("People predicted shape "+str(temp_people.shape)+" frame "+str(0)+ " sum "+str(np.sum(np.abs(temp_people))))
        # print ("Cars predicted shape " + str(temp_cars.shape) + " frame " + str(0) + " sum " + str(np.sum(np.abs(temp_cars))))
        temp_people[temp_people!=0]=1.0/temp_people[temp_people!=0]
        sem[:, :, self.channels.pedestrian_trajectory ]=temp_people#.copy()#*self.temporal_scaling

        temp_cars[temp_cars != 0] = 1.0 / temp_cars[temp_cars != 0]
        sem[:, :, self.channels.cars_trajectory] = temp_cars#.copy()

        self.get_people_and_cars_in_frame(episode, frame, id, sem)

        # Do this faster somehow.
        for x in range(sem.shape[0]):
            for y in range(sem.shape[1]):
                if segmentation[ x, y]>0 and segmentation[ x, y]!=cityscapes_labels_dict['sky']:
                    if segmentation[ x, y] in self.labels_indx :
                        sem[x, y,self.labels_indx[segmentation[ x, y]] + self.channels.semantic] = 1
        if self.settings.normalize_channels_init:
            for channel in range(self.nbr_channels):
                sum_value=np.sum(np.abs(sem[:,:, channel]))
                if sum_value>1e-2:
                    sem[:, :, channel]=sem[:,:, channel]*(1/sum_value)
                if self.settings.printdebug_network_input:
                    print("Normalize channel "+str(channel)+" original sum "+str(sum_value)+" after norm "+str(np.sum(sem[:,:, channel])))
        return np.expand_dims( sem, axis=0)

    def get_people_and_cars_in_frame(self, episode, frame, id, sem):
        # Not necessary any longer to have a separate channel!
        for person in episode.people[frame]:
            person_index = np.round(person).astype(int)
            # print ("Pedestrian " + str([person[1][0], person[1][1], person[2][0], person[2][1]]))
            sem[person_index[1][0]:person_index[1][1] + 1, person_index[2][0]:person_index[2][1] + 1,
            self.channels.pedestrians] = np.ones_like(
                sem[person_index[1][0]:person_index[1][1] + 1, person_index[2][0]:person_index[2][1] + 1,
                self.channels.pedestrians])
        if self.init_net or self.goal_net:
            for person_id in range(id):
                person = episode.pedestrian_data[person_id].agent[frame]
                person_int = [int(round(person[0])), int(round(person[1])), int(round(person[2]))]
                person_bbox = [person_int[0] - self.settings.agent_shape[0],
                               person_int[0] + self.settings.agent_shape[0] + 1,
                               person_int[1] - self.settings.agent_shape[1],
                               person_int[1] + self.settings.agent_shape[1] + 1,
                               person_int[2] - self.settings.agent_shape[2],
                               person_int[2] + self.settings.agent_shape[2] + 1]
                sem[person_bbox[2]:person_bbox[3], person_bbox[4]:person_bbox[5],
                self.channels.pedestrians] = np.ones_like(
                    sem[person_bbox[2]:person_bbox[3], person_bbox[4]:person_bbox[5], self.channels.pedestrians])
        # print(" Episode cars "+str(episode.cars[0]))
        for car in episode.cars[frame]:
            car_index = np.round(car).astype(int)
            # print("Car " + str([car[2], car[3], car[4], car[5]]))
            sem[car_index[2]:car_index[3], car_index[4]:car_index[5], self.channels.cars] = np.ones_like(
                sem[car_index[2]:car_index[3], car_index[4]:car_index[5], self.channels.cars])
        if self.init_net or self.goal_net:
            max_car_id = self.settings.number_of_car_agents
        else:
            max_car_id = id
        for car_id in range(max_car_id):
            car_index = np.round(episode.car_data[car_id].car_bbox[frame]).astype(int)

            sem[car_index[2]:car_index[3], car_index[4]:car_index[5], self.channels.cars] = np.ones_like(
                sem[car_index[2]:car_index[3], car_index[4]:car_index[5], self.channels.cars])

    #construct_feed_dict(self, episode, frame, agent_frame, training=True):
    def grad_feed_dict(self,id,  agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode, frame,
                       reward, statistics,poses,priors,initialization_car, agent_speed=None, training=True, agent_frame=-1):
        if agent_frame < 0 or not training:
            agent_frame = frame
        r=reward[ep_itr, agent_frame]
        #print  ("Reward "+str(r)+" rewards "+str(reward[ep_itr, :]))
        if agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.agent_dead]:
            return {}
        # print ("Frame "+str(frame)+" Episode "+str(ep_itr)+" Reward "+str(r)+" hit by car:"+str(agent_measures[ep_itr,agent_frame, 0])+" Reached goal "+str(agent_measures[ep_itr,agent_frame, 13]))

        feed_dict = {self.state_in: self.get_input_init(id,episode, agent_frame),
                     self.advantages: r,
                     self.sample: self.get_sample(id,statistics, ep_itr, agent_frame,  initialization_car)}

        if self.settings.learn_goal and self.settings.goal_gaussian:
            feed_dict[self.goal]=self.get_goal(statistics, ep_itr, agent_frame, initialization_car)

        feed_dict[self.prior] = np.reshape(priors[ep_itr,id, :,STATISTICS_INDX_MAP.prior]*(1.0/max(priors[ep_itr,id,:,STATISTICS_INDX_MAP.prior])), (self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2],1))
        if self.settings.learn_goal and not self.settings.goal_gaussian:
            feed_dict[self.prior_goal]=np.reshape(priors[ep_itr,id,:,STATISTICS_INDX_MAP.goal_prior]*(1.0/max(priors[ep_itr,id,:,STATISTICS_INDX_MAP.goal_prior])), (self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2],1))

        if self.settings.learn_time and self.goal_net:
            goal = statistics[ep_itr,id, agent_frame, STATISTICS_INDX.goal[0]:STATISTICS_INDX.goal[1]]
            agent_pos = statistics[ep_itr,id, agent_frame,STATISTICS_INDX.agent_pos[0]+1:STATISTICS_INDX.agent_pos[1]]
            goal_dist =np.linalg.norm( goal - agent_pos)
            goal_time=statistics[ep_itr,id,agent_frame, STATISTICS_INDX.goal_time]
            # episode.goal_time = episode.goal_time * 3 * 5 * episode.frame_time * np.linalg.norm(
            #     episode.goal[1:] - episode.agent[1:])
            # print("Agent position "+str(agent_pos)+" goal "+str(goal))
            # print ("Episode goal time " + str(goal_time)+" goal time "+str(goal_dist)+" fraction "+str(goal_dist/goal_time)+" ratio "+str(17/15))
            if goal_time==0:
                feed_dict[self.time_requirement] = np.array([[0]])
            else:
                feed_dict[self.time_requirement]=np.array([[goal_dist/goal_time*(17/15)]])
            # print ("Feed dict input " + str(feed_dict[self.time_requirement]))

        return feed_dict

    def evaluate(self, id, ep_itr, statistics, episode, poses, priors,initialization_car, statistics_car, seq_len=-1):
        if seq_len == -1:
            seq_len = self.settings.seq_len_test
        # print "Evaluate"
        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_vel = self.stats(id,statistics, statistics_car, initializer=True)
        reward = agent_reward_d  # np.zeros_like(agent_reward)


        self.reset_mem()
        feed_dict = self.grad_feed_dict(id, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                     episode, 0, reward, statistics,poses,priors,  initialization_car,agent_speed=agent_vel)
        loss, summaries_loss, responsible_output,responsible_output_prior  = self.sess.run([self.loss, self.loss_summaries, self.responsible_output, self.responsible_output_prior], feed_dict)  # self.merged
        # episode.loss[frame] = loss
        self.save_loss(ep_itr, id, loss, statistics, statistics_car)
        if self.writer:
            self.test_writer.add_summary(summaries_loss, global_step=self.num_grad_itrs)
        if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
            self.feed_dict = []
        return statistics

    def save_loss(self, ep_itr, id, loss, statistics, statistics_car, frame=0):
        statistics[ep_itr, id, frame, STATISTICS_INDX.loss_initializer] = loss

    def train(self,id,  ep_itr, statistics, episode, filename, filename_weights,poses, priors,initialization_car,statistics_car, seq_len=-1):

        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_vel = self.stats(id, statistics,statistics_car, initializer=True)


        if self.settings.normalize:
            reward=self.normalize_reward(agent_reward_d, agent_measures)
        else:
            reward=agent_reward_d
        self.reset_mem()



        feed_dict = self.grad_feed_dict(id,agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                         episode, 0, reward, statistics,poses,priors,  initialization_car,agent_speed=agent_vel)
        self.do_gradient_debug_printout(id,ep_itr, feed_dict)

        self.calculate_gradients( id, feed_dict, statistics, 0, ep_itr) #                 self.calculate_gradients(episode, frame, feed_dict) %


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

    def do_gradient_debug_printout(self, id,ep_itr, feed_dict):
        if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
            print("Grad Feed dict---------------------------------------------------------")
            print(" Feed dict len " + str(len(self.feed_dict)))
            local_feed_dict=self.feed_dict[ep_itr][id]
            for key, value in feed_dict.items():
                if key == self.state_in:
                    print(value.shape)

                    print("Input cars " + str(np.sum(value[0, :, :, 6])) + " people " + str(
                        np.sum(value[0, :, :, 5])) + " Input cars traj" + str(
                        np.sum(value[0, :, :, 4])) + " people traj " + str(
                        np.sum(value[0, :, :, 3])))
                    print("Input building " + str(np.sum(value[0, :, :, 7])) + " fence " + str(
                        np.sum(value[0, :, :, 7 + 1])) + " static " + str(
                        np.sum(value[0, :, :, 7 + 2])) + " pole " + str(
                        np.sum(value[0, :, :, 7 + 3])))
                    print("Input sidewalk " + str(np.sum(value[0, :, :, 7 + 5])) + " road " + str(
                        np.sum(value[0, :, :, 7 + 4])) + " veg. " + str(
                        np.sum(value[0, :, :, 7 + 6])) + " wall " + str(np.sum(value[0, :, :, 7 + 7])) + " sign " + str(
                        np.sum(value[0, :, :, 7 + 8])))

                    print("R " + str(np.sum(value[0, :, :, 0])) + "G " + str(np.sum(value[0, :, :, 1])) + "B " + str(
                        np.sum(value[0, :, :, 1])))
                    if self.settings.printdebug_network_input:
                        equal = np.array_equal(value, local_feed_dict[self.state_in])
                        print("Equal to feed dict state " + str(equal))
                        if not equal:
                            diff = local_feed_dict[self.state_in] - value
                            not_equal_pos = np.where(diff != 0)

                            print("Not equal: " + str(np.sum(local_feed_dict[self.state_in] - value)))
                            print(" Size " + str(len(not_equal_pos)) + " diff size " + str(diff.shape))
                            print(" Not equal x " + str(np.unique(not_equal_pos[1])))
                            print(" Not equal y " + str(np.unique(not_equal_pos[2])))
                            print("Not equal at " + str(np.unique(not_equal_pos[3])))
                            print(
                                " Value in orig feed dict " + str(local_feed_dict[self.state_in][not_equal_pos]))
                            print(" Value in grad feed dict " + str(value[not_equal_pos]))

                    # print ("Cars: ")
                    # print (value[0, :, :, 6])
                    # print ("Cars traj : ")
                    # print (value[0, :, :, 4])
                    # print ("People:")
                    # print (value[0, :, :, 5])
                    # print ("People traj : ")
                    # print (value[0, :, :, 3])
                elif key == self.prior:
                    print("prior " + str(np.sum(value[0, :, :])) + " shape " + str(value.shape))
                    if self.settings.printdebug_network_input:
                        print("Equal to feed dict prior " + str( np.array_equal(value, local_feed_dict[self.prior])))

                    # print ("Equal to feed dict ")
                    # print (np.array_equal(value, self.feed_dict[self.prior]))
                    # indxs=self.feed_dict[self.prior]-value
                    # print ("nonzero  grad dict "+str(np.sum(value)))
                    # print ("nonzero  feed dict "+str(np.sum(self.feed_dict[self.prior])))
                else:
                    print(str(key) + " " + str(value))
                    if self.settings.printdebug_network_input:
                        if key in local_feed_dict:
                            equal=np.array_equal(value, local_feed_dict[key])
                            print("Equal to feed dict " + str(str(key)) + " " + str(
                                    np.array_equal(value, local_feed_dict[key])))
                            if not equal:
                                print("Saved value feed dict " + str(str(key)) + " " + str( local_feed_dict[key]))



