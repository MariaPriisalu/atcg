import numpy as np
from settings import NBR_MEASURES,RANDOM_SEED

from initializer_net import InitializerNet


import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions

tf.random.set_seed(RANDOM_SEED)



import copy


#
# np.set_printoptions(precision=5)




class InitializerArchNet(InitializerNet):
    # Softmax Simplified
    def __init__(self, settings, weights_name="policy"):
        self.labels_indx = {11: 0, 13: 1, 14: 2, 4: 2, 5: 2, 15: 2, 16: 2, 17: 3, 18: 3, 7: 4, 9: 4, 6: 4, 10: 4,
                            8: 5, 21: 6, 22: 6, 12: 7, 20: 8, 19: 8}
        if self.debugging and self.is_net_type(settings.printdebug_network):
            self.feed_dict = {}
        self.valid_pos = []
        self.probabilities_saved = []
        self.carla = settings.carla
        self.set_nbr_channels()
        # self.temporal_scaling = 0.1 * 0.3

        super(InitializerArchNet, self).__init__(settings, weights_name="init")


    def define_conv_layers(self):
        # tf.reset_default_graph()
        output_channels = 1

        self.state_in = tf.compat.v1.placeholder(
            shape=[self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], self.nbr_channels],
            dtype=self.DTYPE,
            name="reconstruction")
        self.prior = tf.compat.v1.placeholder(
            shape=[self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], 1],
            dtype=self.DTYPE,
            name="prior")

        # mean = tf.constant(self.mean, dtype=self.DTYPE)  # (3)

        prev_layer = tf.concat([self.state_in, self.prior], axis=3)  # - mean
        out_channels = (self.nbr_channels + 1)
        with tf.compat.v1.variable_scope('conv1_1') as scope:
            out_channels = (self.nbr_channels + 1) * 2

            # print("Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel1 = tf.compat.v1.get_variable('weights', [3, 3, self.nbr_channels + 1, out_channels], self.DTYPE,
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=prev_layer, filters=kernel1, strides=[1, 2, 2, 1], padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv2_1') as scope:
            # print(
            # "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel1_2 = tf.compat.v1.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=prev_layer, filters=kernel1_2,strides=[1,1,1,1], padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv3_1') as scope:
            # print(
            # "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel1_3 = tf.compat.v1.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=prev_layer, filters=kernel1_3,strides=[1,1,1,1], padding='SAME')  # [1, 2, 2, 2, 1]
        self.block1=copy.copy(prev_layer)
        self.out_channels_1 = copy.copy(out_channels)

        # Second Block
        with tf.compat.v1.variable_scope('conv1_2') as scope:
            out_channels = out_channels * 2

            # print(
            #     "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel2 = tf.compat.v1.get_variable('weights', [3, 3, self.nbr_channels + 1, out_channels], self.DTYPE,
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=prev_layer, filters=kernel2, strides=[1, 2, 2, 1], padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv2_2') as scope:
            # print(
            #     "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel2_2 = tf.compat.v1.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=prev_layer, filters=kernel2_2,strides=[1,1,1,1], padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv3_2') as scope:
            # print(
            #     "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel2_3 = tf.compat.v1.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=prev_layer, filters=kernel2_3,strides=[1,1,1,1], padding='SAME')  # [1, 2, 2, 2, 1]
        self.block2 = copy.copy(prev_layer)
        self.out_channels_2=copy.copy(out_channels)

        # Third Block
        with tf.compat.v1.variable_scope('conv1_3') as scope:
            out_channels = out_channels * 2

            # print(
            #     "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel3 = tf.compat.v1.get_variable('weights', [3, 3, self.nbr_channels + 1, out_channels], self.DTYPE,
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=prev_layer, filters=kernel3, strides=[1, 2, 2, 1], padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv2_3') as scope:
            # print(
            #     "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel3_2 = tf.compat.v1.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=prev_layer, filters=kernel3_2, padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv3_3') as scope:
            print(
                "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel3_3 = tf.compat.v1.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=prev_layer, filters=kernel3_3, padding='SAME')  # [1, 2, 2, 2, 1]
        self.block3 = copy.copy(prev_layer)
        self.out_channels_3 = copy.copy(out_channels)


        self.block1_resized=tf.image.resize(self.block1, [self.settings.env_shape[1], self.settings.env_shape[2]])

        # print ("Block 1 before resizing "+str(self.block1.shape)+" block 1 after resizing "+str(self.block1_resized.shape))

        self.block2_resized = tf.image.resize(self.block2,
                                                     [self.settings.env_shape[1], self.settings.env_shape[2]])

        # print ("Block 2 before resizing " + str(self.block2.shape) + " block 2 after resizing " + str(self.block2_resized.shape))

        self.block3_resized = tf.image.resize(self.block3,
                                                     [self.settings.env_shape[1], self.settings.env_shape[2]])

        # print ("Block 3 before resizing " + str(self.block3.shape) + " block 3 after resizing " + str(self.block3_resized.shape))

        self.out_channels=tf.concat([self.block1_resized, self.block2_resized, self.block3_resized], axis=3)

        # print ("Size after concatenation " + str(self.out_channels.shape))
        out_channels=self.out_channels_1+self.out_channels_2+self.out_channels_3
        with tf.compat.v1.variable_scope('out_conv') as scope:
            # print("Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel_out = tf.compat.v1.get_variable('weights', [1, 1, out_channels, 1], self.DTYPE,
                                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            prev_layer = tf.nn.conv2d(input=self.out_channels, filters=kernel_out,strides=[1,1,1,1], padding='SAME')  # [1, 2, 2, 2, 1]

        # print ("Size after 1-conv " + str(prev_layer.shape))
        return prev_layer




class InitializerArchGaussianNet(InitializerNet):
    def __init__(self, settings, weights_name="policy"):
        super(InitializerArchGaussianNet, self).__init__(settings, weights_name="init")

    def define_loss(self, dim_p):
        self.sample = tf.compat.v1.placeholder(shape=[2], dtype=self.DTYPE, name="sample")
        self.normal_dist = tfd.Normal(self.probabilities, self.settings.init_std)
        self.responsible_output = self.normal_dist.prob(self.sample)

        self.l2_loss = tf.nn.l2_loss(self.probabilities - self.sample)  # tf.nn.l2_loss
        # self.loss = tf.reduce_mean(self.advantages * (2 * tf.log(self.settings.sigma_vel) + tf.log(2 * tf.pi) + (
        # self.l2_loss / (self.settings.sigma_vel * self.settings.sigma_vel))))
        self.loss = tf.reduce_mean(input_tensor=self.advantages * self.l2_loss)

        return self.sample, self.loss

    def get_feature_vectors_gradient(self,id, agent_action, agent_frame, agent_measures, agent_pos, agent_speed,agent_velocity, ep_itr,
                                     episode, feed_dict, frame, poses, statistics, training,statistics_car=[]):
        pass

    def fully_connected(self, dim_p, prev_layer):
        #print ("Fully connected flattened layer: " + str(prev_layer))
        self.flattened_layer = tf.reshape(prev_layer, [-1])
        # dim = np.prod(prev_layer.get_shape().as_list()[1:])
        # prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        # weights = tf.get_variable('weights', [dim, 2], self.DTYPE,
        #                           initializer=tf.keras.initializers.GlorotNormal(RANDOM_SEED))
        # biases = self.bias_variable('biases', [2])

        # self.mu = tf.matmul(tf.expand_dims(prev_layer_flat,0), weights)
        # self.mu = tf.add(self.mu, biases)

    # return [statistics[ep_itr, agent_frame, 6]]
    def get_sample(self,id, statistics, ep_itr, agent_frame, init_car):
        # print ([statistics[ep_itr, 0, 1]], [statistics[ep_itr, 0, 2]])
        # print ("Get Sample "+str(statistics[ep_itr, 0, 1]*self.settings.env_shape[1]+statistics[ep_itr, 0, 2]))
        # np.ravel_multi_index(([statistics[ep_itr, 0, 1]], [statistics[ep_itr, 0, 2]]), self.settings.env_shape[1:] )
        # print ("Get Sample "+str([statistics[ep_itr, 0, 1:2]])+" flattened: "+str(np.ravel_multi_index(([int(statistics[ep_itr, 0, 1])], [int(statistics[ep_itr, 0, 2])]), self.settings.env_shape[1:] )))
        sample = statistics[ep_itr,id, 0, 1:3] - init_car[ep_itr, 1:3]
        # sample[0]=sample[0]/self.settings.env_shape[1]
        # sample[1] = sample[1] / self.settings.env_shape[2]
        return sample
        # [statistics[ep_itr, 0, 1]*self.settings.env_shape[1]+statistics[ep_itr, 0, 2]]

    # def get_input(self, episode, agent_pos_cur, frame_in=-1, training=True):
    #     return episode.reconstruction

    def calc_probabilities(self, fc_size):
        # print ("Define probabilities: " + str(self.flattened_layer))
        self.probabilities = self.mu  # tf.sigmoid(self.mu)

    def importance_sample_weight(self, id,responsible, statistics, ep_itr, frame, responsible_v=0):
        pass

    def get_feature_vectors(self,id, agent_frame, episode, feed_dict, frame):
        pass


    def apply_net(self,id, feed_dict, episode, frame, training, max_val=False, viz=False,manual=False):
        # Choose car
        episode.initializer_data[id].init_car_id = np.random.choice(episode.init_cars)
        episode.initializer_data[id].init_car_pos = np.array([np.mean(episode.cars_dict[episode.initializer_data[id].init_car_id][0][2:4]),
                                         np.mean(episode.cars_dict[episode.initializer_data[id].init_car_id][0][4:])])
        car_pos_next = np.array([np.mean(episode.cars_dict[episode.initializer_data[id].init_car_id][1][2:4]),
                                 np.mean(episode.cars_dict[episode.initializer_data[id].init_car_id][1][4:])])
        episode.initializer_data[id].init_car_vel = (car_pos_next - episode.initializer_data[id].init_car_pos) / episode.frame_time

        episode.pedestrian_data[id].init_method = 7

        car_dim = [episode.cars_dict[episode.initializer_data[id].init_car_id][0][3] - episode.cars_dict[episode.initializer_data[id].init_car_id][0][2],
                    episode.cars_dict[episode.initializer_data[id].init_car_id][0][5] - episode.cars_dict[episode.initializer_data[id].init_car_id][0][4]]
        car_max_dims = max(car_dim)
        car_min_dims = min(car_dim)

        episode.calculate_prior(id, self.settings.field_of_view_car)

        flat_prior = episode.initializer_data[id].prior.flatten()
        feed_dict[self.prior] = np.expand_dims(np.expand_dims(episode.initializer_data[id].prior * (1 / max(flat_prior)), axis=0), axis=-1)

        mean_vel, summary_train = self.sess.run([self.mu, self.train_summaries], feed_dict)
        print ("Mean vel " + str(mean_vel))
        episode.init_distribution = copy.copy(mean_vel[0])

        episode.pedestrian_data[id].agent[0] = np.zeros(3)
        episode.pedestrian_data[id].agent[0][0] = episode.get_height_init()
        episode.pedestrian_data[id].agent[0][1] = episode.initializer_data[id].init_car_pos[0] + np.random.normal(mean_vel[0][0], self.settings.init_std, 1)
        episode.pedestrian_data[id].agent[0][2] = episode.initializer_data[id].init_car_pos[1] + np.random.normal(mean_vel[0][1], self.settings.init_std, 1)
        episode.pedestrian_data[id].agent[0][1] = max(min(episode.pedestrian_data[id].agent[0][1], self.settings.env_shape[1] - 1), 0)
        episode.pedestrian_data[id].agent[0][2] = max(min(episode.pedestrian_data[id].agent[0][2], self.settings.env_shape[2] - 1), 0)

        while (episode.initializer_data[id].prior[int(episode.pedestrian_data[id].agent[0][1]), int(episode.pedestrian_data[id].agent[0][1])] == 0):
            episode.pedestrian_data[id].agent[0][1] = episode.initializer_data[id].init_car_pos[0] + np.random.normal(mean_vel[0][0], self.settings.init_std, 1)
            episode.pedestrian_data[id].agent[0][2] = episode.initializer_data[id].init_car_pos[1] + np.random.normal(mean_vel[0][1], self.settings.init_std, 1)
            episode.pedestrian_data[id].agent[0][1] = max(min(episode.pedestrian_data[id].agent[0][1], self.settings.env_shape[1] - 1), 0)
            episode.pedestrian_data[id].agent[0][2] = max(min(episode.pedestrian_data[id].agent[0][2], self.settings.env_shape[2] - 1), 0)
        # make sure agent is in environment borders
        episode.pedestrian_data[id].agent[0][1] = max(min(episode.pedestrian_data[id].agent[0][1], self.settings.env_shape[1] - 1), 0)
        episode.pedestrian_data[id].agent[0][2] = max(min(episode.pedestrian_data[id].agent[0][2], self.settings.env_shape[2] - 1), 0)

        # print ("Initialize pedestrian "  + str( indx) + " pos in 2D: " + str(episode.pedestrian_data[id].agent[0][1:])+" no prior probability: "+str( episode.init_distribution[indx])+" prior probability"+str(probabilities[[indx]]))

        # Vector from pedestrian to car in voxels
        vector_car_to_pedestrian = episode.pedestrian_data[id].agent[0][1:] - episode.initializer_data[id].init_car_pos
        if np.linalg.norm(episode.initializer_data[id].init_car_vel) < 0.01:
            speed = 3 * 5  # Speed voxels/second
            # print ("Desired speed pedestrian " + str(speed * .2) + " car vel " + str(episode.initializer_data[id].init_car_vel * .2))

            # Unit orthogonal direction
            unit = -vector_car_to_pedestrian * (
                1 / np.linalg.norm(vector_car_to_pedestrian))

            # Set ortogonal direction to car
            episode.pedestrian_data[id].vel_init = np.array([0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly
            return episode.pedestrian_data[id].agent[0], 11, episode.pedestrian_data[id].vel_init  # pos, indx, vel_init
        # set initial pedestrian velocity orthogonal to car!- voxels
        vector_travelled_by_car_to_collision = np.dot(vector_car_to_pedestrian, episode.initializer_data[id].init_car_vel) / np.linalg.norm(
            episode.initializer_data[id].init_car_vel) * episode.initializer_data[id].init_car_vel

        vector_travelled_by_pedestrian_to_collision = vector_car_to_pedestrian - vector_travelled_by_car_to_collision

        ratio_pedestrian_to_car_dist_to_collision = np.linalg.norm(
            vector_travelled_by_pedestrian_to_collision) / np.linalg.norm(vector_travelled_by_car_to_collision)

        speed = min(ratio_pedestrian_to_car_dist_to_collision * np.linalg.norm(episode.initializer_data[id].init_car_vel),
                    3 * 5)  # Speed voxels/second
        # print ("Desired speed pedestrian "+str(speed*.2)+" ratio_pedestrian_to_car_dist_to_collision "+str(ratio_pedestrian_to_car_dist_to_collision)+" "+str(episode.initializer_data[id].init_car_vel*.2))

        # Unit orthogonal direction
        unit = -vector_travelled_by_pedestrian_to_collision * (
            1 / np.linalg.norm(vector_travelled_by_pedestrian_to_collision))

        # Set ortogonal direction to car
        episode.pedestrian_data[id].vel_init = np.array([0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly

        # print ("Pedestrian intercepting car? "+str(episode.intercept_car(0, all_frames=False)))

        return episode.pedestrian_data[id].agent[0], 11, episode.pedestrian_data[id].vel_init  # pos, indx, vel_init



