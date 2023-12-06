
import tensorflow as tf
from settings import RANDOM_SEED
tf.random.set_seed(RANDOM_SEED)
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from net import Net
from RL.settings import STATISTICS_INDX_CAR,STATISTICS_INDX, CAR_MEASURES_INDX
import copy
import math as m
from utils.utils_functions import overlap



class SimpleCarNet(Net):
    def __init__(self, settings):
        super(SimpleCarNet, self).__init__(settings, "car")



    def define_conv_layers(self):

        if self.settings.car_input == "difference" :
            self.state_in = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 6], dtype=self.DTYPE,
                                                     name="time-to-collision")
        elif self.settings.car_input == "distance_to_car_and_pavement_intersection" or self.settings.car_input=="distance_to_car_and_pavement_pedestrian_speed_intersection":

            self.state_in = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 3], dtype=self.DTYPE,
                                                     name="time-to-collision")
        elif self.settings.car_input == "distance" or self.settings.car_input == "time_to_collision" or self.settings.car_input == "scalar_product":
            self.state_in = tf.compat.v1.placeholder(shape=[ self.settings.batch_size, 1], dtype=self.DTYPE,
                                       name="time-to-collision")

        if self.settings.car_input == "distance_to_car_and_pavement_pedestrian_speed_intersection":
            self.state_in_additional = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 3], dtype=self.DTYPE,
                                                     name="additional")
        if self.settings.angular_car:
            self.state_in_angle = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 3], dtype=self.DTYPE,
                                                     name="angles")
        return self.state_in

    def bias_variable(self, name, shape):
        print(" Initialize car bias as -.1")
        return tf.compat.v1.get_variable(name, shape, self.DTYPE, tf.compat.v1.constant_initializer(-0.1, dtype=self.DTYPE))

    def fully_connected(self, dim_p, prev_layer):

        init_1 = tf.compat.v1.constant_initializer(1.0)
        if self.settings.car_input == "difference":
            weights_dim = [6, 2]
            biases_dim = [2]
        elif self.settings.car_input == "distance_to_car_and_pavement_intersection" or self.settings.car_input=="distance_to_car_and_pavement_pedestrian_speed_intersection":
            weights_dim = [3, 1]
            biases_dim = [1]
        elif self.settings.car_input == "distance" or self.settings.car_input == "time_to_collision" or self.settings.car_input == "scalar_product":
            weights_dim = [1, 1]
            biases_dim = [1]
        self.weight = tf.compat.v1.get_variable('weights', weights_dim, self.DTYPE,
                                          initializer=init_1)
        self.bias = self.bias_variable('biases', biases_dim)
        #prev_layer=tf.add(self.mu, self.bias)
        self.mu = tf.matmul(prev_layer, self.weight)
        self.mu=tf.add(self.mu, self.bias)
        if self.settings.car_input=="distance_to_car_and_pavement_pedestrian_speed_intersection":
            weights_dim_additional = [3, 1]
            self.weight_additional_input = tf.get_variable('weight_angular_input', weights_dim_additional, self.DTYPE,
                                                        initializer=init_1)
            self.sum_addition = tf.matmul(self.state_in_additional, self.weight_additional_input)
            self.mu = tf.add(self.mu, self.sum_addition)

        if self.settings.angular_car:
            weights_for_angular_input = [3, 1]
            weights_for_angular_output = [weights_dim[0]+3, 1]
            biases_angular = [1]
            self.weight_angular_input = tf.compat.v1.get_variable('weight_angular_input', weights_for_angular_input, self.DTYPE,
                                          initializer=init_1)
            self.angular_addition = tf.matmul(self.state_in_angle, self.weight_angular_input)
            self.mu = tf.add(self.mu, self.angular_addition)

            self.weight_angular = tf.compat.v1.get_variable('weight_angular', weights_for_angular_output, self.DTYPE,
                                          initializer=init_1)
            self.bias_angular = self.bias_variable('biases_angular', biases_angular)
            self.full_input=tf.concat([prev_layer,self.state_in_angle], axis=1)
            self.mu_angular = tf.matmul(self.full_input, self.weight_angular)
            self.mu_angular = tf.add(self.mu_angular, self.bias_angular)

        return self.mu

    def calc_probabilities(self, fc_size):
        if self.settings.car_input == "difference":
            self.past_move=tf.compat.v1.placeholder(shape=[self.settings.batch_size, 2], dtype=self.DTYPE,
                                                     name="past_move")
            self.past_move_fraction=self.past_move/(1-self.past_move)
            self.past_move_logit=tf.math.log(self.past_move_fraction)
            self.sigmoid_output=tf.sigmoid(self.past_move_logit+self.mu)
            self.probability = self.sigmoid_output * 2 - 1
            self.probabilities = self.probability
        elif self.settings.linear_car:
            self.probability = tf.sigmoid(self.mu)#tf.sigmoid(self.mu)

            self.probabilities= self.probability
        else:
            self.probability = tf.sigmoid(self.mu)
            self.probabilities = tf.reshape(tf.concat([1 - self.probability, self.probability], axis=1), [-1])

        if self.settings.angular_car:
            self.probability_angular = tf.sigmoid(self.mu_angular )*2-1  # tf.sigmoid(self.mu)
            self.probabilities_angular = self.probability_angular
        return self.probability

    def define_loss(self, dim_p):

        if self.settings.car_input == "difference" :
            self.sample = tf.compat.v1.placeholder(shape=[2], dtype=self.DTYPE, name="sample")
            pi = tf.constant(m.pi)
            self.normal_dist = tfd.Normal(self.probability, self.settings.sigma_car)
            self.responsible_output = self.normal_dist.prob(self.sample)
            self.diff =  self.sample- self.probability
            self.l2_loss = tf.nn.l2_loss(self.diff)  # tf.nn.l2_loss
            self.inner_loss = (self.l2_loss / (self.settings.sigma_car * self.settings.sigma_car))
            self.loss = tf.reduce_mean(self.advantages * (self.inner_loss))  # 2 * tf.log(self.settings.sigma_vel) + tf.log(2 * pi) +
            # self.loss = tf.reduce_mean(self.advantages*self.l2_loss)
        elif self.settings.linear_car:
            self.sample = tf.compat.v1.placeholder(shape=[1], dtype=self.DTYPE, name="sample")
            pi = tf.constant(m.pi)
            self.normal_dist = tfd.Normal(self.probability, self.settings.sigma_car)
            self.responsible_output = self.normal_dist.prob(self.sample)
            self.diff= self.probability-self.sample
            self.l2_loss = tf.nn.l2_loss(self.diff)  # tf.nn.l2_loss
            self.inner_loss=(self.l2_loss / (self.settings.sigma_car * self.settings.sigma_car))
            self.loss = tf.reduce_mean(self.advantages * ( self.inner_loss)) #2 * tf.log(self.settings.sigma_vel) + tf.log(2 * pi) +
            # self.loss = tf.reduce_mean(self.advantages*self.l2_loss)
        else:
            self.sample = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="sample")

            self.responsible_output = tf.slice(self.probabilities, self.sample, [1]) + np.finfo(np.float32).eps
            self.loss = -tf.reduce_mean(tf.log(self.responsible_output) * self.advantages)
        if self.settings.angular_car:
            self.sample_angular = tf.compat.v1.placeholder(shape=[1], dtype=self.DTYPE, name="sample_angular")
            self.normal_dist_angular = tfd.Normal(self.probability_angular, self.settings.sigma_car_angular)
            self.responsible_output_angular = self.normal_dist.prob(self.sample_angular)
            self.diff_angular = self.probability_angular - self.sample_angular
            self.l2_loss_angular = tf.nn.l2_loss(self.diff_angular)  # tf.nn.l2_loss
            self.inner_loss_angular = self.l2_loss_angular / (self.settings.sigma_car_angular * self.settings.sigma_car_angular)
            self.angular_loss=tf.reduce_mean(self.advantages * self.inner_loss_angular)  # 2 * tf.log(self.s
            self.loss =self.loss+ self.angular_loss # 2 * tf.log(self.s
        return self.sample, self.loss

    def construct_feed_dict(self, id,episode, frame, agent_frame, training=True,distracted=False):

        # print (" distracted constrcut feed dict "+str(distracted))
        feed_dict = {}

        id_closest_agent=episode.car_data[id].id_closest_agent
        if id_closest_agent>=0:
            pedestrian=episode.pedestrian_data[id_closest_agent]
            pedestrian_pos=pedestrian.agent[frame]
            if frame == 0:
                pedestrian_velocity = pedestrian.init_dir
            else:
                pedestrian_velocity = pedestrian.velocity[frame-1]
        else:
            pedestrian_velocity=[]
            pedestrian_pos=[]

        if frame==0:
            car_velocity = episode.car_data[id].car_dir
        else:
            car_velocity = episode.car_data[id].velocity_car
        if self.settings.car_input == "difference" :
            state_in =self.get_difference(pedestrian_pos, episode.car_data[id].car[frame], pedestrian_velocity, episode.car_data[id].car_goal,distracted)
            if frame>=1:
                # print ("Frame "+str(frame)+" "+str(car_velocity))
                car_velocity_cur = car_velocity[frame-1]
            else:
                # print("Frame " + str(frame) + " " + str(car_velocity))
                car_velocity_cur = car_velocity
            # print ("Normalized vel "+str(car_velocity_cur[1:])+" max speed "+str(self.settings.car_max_speed_voxelperframe)+" frame "+str(frame))
            normalized_vel=np.array(car_velocity_cur[1:])/self.settings.car_max_speed_voxelperframe
            feed_dict[self.past_move] =((normalized_vel+1)*.5).reshape((1,2))
            #print(" Feed dict  " + str(feed_dict[self.past_move]) )
        elif self.settings.car_input == "scalar_product":
            # print ("Frame " + str(frame) + " car " + str(episode.car))
            state_in = self.get_scalar_product( pedestrian_pos, episode.car_data[id].car[frame],  episode.car_data[id].car_dir, distracted)
            #print (" input: " + str(state_in))
        elif self.settings.car_input == "distance" :
            state_in = self.get_distance_to_agent( pedestrian_pos, episode.car_data[id].car[frame], distracted)

        elif self.settings.car_input == "distance_to_car_and_pavement_intersection" or self.settings.car_input == "distance_to_car_and_pavement_pedestrian_speed_intersection":
            state_in = self.get_distance_to_agent_cars_and_pavement(pedestrian_pos, episode.car_data[id].car[frame], distracted, episode.car_data[id].dist_to_closest_car, episode.car_data[id].dist_to_closest_ped, episode.car_data[id].intersection_with_pavement)
        elif self.settings.car_input == "time_to_collision":
            state_in = self.get_time_to_collision( pedestrian_pos, episode.car_data[id].car[frame], pedestrian_velocity, car_velocity, distracted, max(self.settings.agent_shape[1:])+max(self.settings.car_dim[1:]))

        # print(" State in  " + str(state_in))
        feed_dict[self.state_in] = state_in

        if self.settings.angular_car:#  agent_pos, car_pos,  velocity_car, frame,closest_car, goal, distracted)
            closest_car=episode.car_data[id].closest_car[frame]
            # print (" Feed dict input "+str(frame))
            # print (" Closest cars "+str(episode.closest_car[frame]))
            feed_dict[self.state_in_angle]=self.get_angles_in( pedestrian_pos, episode.car_data[id].car[frame],  car_velocity, closest_car, episode.car_data[id].car_goal, distracted)
        if self.settings.car_input == "distance_to_car_and_pavement_pedestrian_speed_intersection" :


            feed_dict[self.state_in_additional]= self.get_distance_to_agent_cars_and_pavement_speed(pedestrian_pos, episode.car_data[id].car[frame],
                                                                    distracted, episode.car_data[id].dist_to_closest_car,
                                                                    episode.car_data[id].dist_to_closest_ped,
                                                                    episode.car_data[id].intersection_with_pavement, pedestrian_velocity,car_velocity)
        # if self.settings.linear_car:
        #     feed_dict[self.state_in]=feed_dict[self.state_in] +0.5
        return feed_dict


    def apply_net(self, id, feed_dict, episode, frame, training, max_val=False, viz=False,manual=False):
        if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
            self.feed_dict.append({})
            for key, value in feed_dict.items():
                self.feed_dict[-1][key] = copy.deepcopy(value)
        episode.car_data[id].action_car_angle = 0

        # print(" Apply car")
        if self.settings.car_input == "difference":

            sigmoid_output, mean_vel, weight, mu, summary_train = self.sess.run([self.sigmoid_output, self.probability,self.weight, self.mu, self.train_summaries], feed_dict)
            mean_vel=mean_vel[0]
            # print (" Mean vel "+str(mean_vel))
            if training:
                if len(episode.car_data[id].supervised_car_vel) > 0:
                    # print( " supervised car vel "+str(episode.car_data[id].supervised_car_vel))
                    speed_value_y = [episode.car_data[id].supervised_car_vel[1]/self.settings.car_max_speed_voxelperframe]
                    speed_value_z = [ episode.car_data[id].supervised_car_vel[2]/self.settings.car_max_speed_voxelperframe]
                    # print (" Choose car input according to supervised car "+str(speed_value_y)+" "+str(speed_value_z))
                else:
                    speed_value_y = np.random.normal(mean_vel[0], self.settings.sigma_car, 1)
                    # print ("Random y " + str(speed_value_y))
                    speed_value_z = np.random.normal(mean_vel[1], self.settings.sigma_car, 1)
                    # print(" Car speed random " + str(speed_value_y) + " " + str(speed_value_z))
            else:
                speed_value_y = [mean_vel[0]]
                #print ("Random y " + str(speed_value_y))
                speed_value_z = [mean_vel[1]]
                # print (" Car speed " + str(speed_value_y)+" "+str(speed_value_z))
            # print ("Random z " + str(speed_value_y)+" "+str(speed_value_z)+" mean: "+str(mean_vel[0])+" "+str(mean_vel[1])+" std "+str(self.settings.sigma_car))
            speed_value = np.sqrt(speed_value_y[0] ** 2 + speed_value_z[0] ** 2)
            episode.car_data[id].velocity_car = np.zeros(3)
            if speed_value> 1.0 or self.settings.car_constant_speed and speed_value>1e-2 and len(episode.car_data[id].supervised_car_vel) <0:
                default_speed = 1.0
                speed_value_y[0] = speed_value_y[0] * default_speed / speed_value
                speed_value_z[0] = speed_value_z[0] * default_speed / speed_value
                speed_value = default_speed
                # print (" Adjust speed "+str( speed_value_y[0])+" "+str(speed_value_z[0])+" norm "+str(np.sqrt(speed_value_y[0] ** 2 + speed_value_z[0] ** 2)))
            if self.settings.car_constant_speed and len(episode.car_data[id].supervised_car_vel) <0:
                speed=self.settings.car_reference_speed
            else:
                speed = self.settings.car_max_speed_voxelperframe
            episode.car_data[id].velocity_car[1] = copy.copy(speed_value_y[0] *speed)
            episode.car_data[id].velocity_car[2] = copy.copy(speed_value_z[0] * speed)
            episode.car_data[id].speed_car = speed_value *speed
            episode.car_data[id].probabilities_car = np.copy(mean_vel)
            if self.settings.car_constant_speed and speed_value>1e-2:
                episode.car_data[id].action_car = np.copy(speed_value)*speed/self.settings.car_max_speed_voxelperframe

            else:
                episode.car_data[id].action_car=np.copy(speed_value)
            #print (" Car takes step "+str(speed_value_y[0] )+" "+str(speed_value_z[0])+" speed "+str(episode.speed_car)+" probabilities "+str(episode.probabilities_car))
        elif  self.settings.linear_car:

            probability, weight,mu, summary_train = self.sess.run([self.probability,self.weight,self.mu, self.train_summaries], feed_dict)
            if training:
                if len(episode.car_data[id].supervised_car_vel) > 0:
                    episode.car_data[id].action_car =np.linalg.norm(episode.car_data[id].supervised_car_vel[1:])/self.settings.car_max_speed_voxelperframe
                    # print(" Choose car speed according to supervised car " + str(episode.action_car))
                else:
                    episode.car_data[id].action_car = np.random.normal(probability[0],self.settings.sigma_car)[0]
                #print(" Car speed random " + str(episode.action_car))
            else:
                episode.car_data[id].action_car =probability[0]
                #print(" Car speed " + str(episode.action_car))

            speed=episode.car_data[id].action_car*self.settings.car_max_speed_voxelperframe
            # print ("Sample car mean" + str(probability[0]) + " std " + str(self.settings.sigma_car) + " action " + str(
            #     episode.action_car)+" speed "+str(speed)+" weight "+str(weight)+" input "+str(feed_dict[self.state_in])+" mu "+str(mu))
            episode.car_data[id].probabilities_car = np.copy(probability[0])
            episode.car_data[id].speed_car = speed
            # print (" Car speed "+str(episode.speed_car))
            if self.settings.angular_car:

                probability_angular, weight_angular, mu_angular, full_input, summary_train = self.sess.run(
                    [self.probability_angular, self.weight_angular, self.mu_angular, self.full_input, self.train_summaries], feed_dict)
                # print (" Apply net: angular probability  "+str(probability_angular)+" angular mu "+str(mu_angular )+" weight mu "+str(weight_angular)+"full input "+str(full_input) )
                if training:
                    if len(episode.car_data[id].supervised_car_vel) > 0:

                        if self.settings.angular_car_add_previous_angle:
                            if frame == 0:
                                car_velocity = episode.car_data[id].car_dir
                            else:
                                car_velocity = episode.car_data[id].velocity_car[frame - 1]
                            # print (" Input previous dir "+str(car_velocity)+" input "+str(episode.car_data[id].supervised_car_vel[1:]))
                            episode.car_data[id].action_car_angle = self.signed_angle_between(car_velocity[1:], episode.car_data[id].supervised_car_vel[1:])/np.pi
                        else:
                            # print(" Arctan of "+str(episode.car_data[id].supervised_car_vel[1:])+" "+str(np.arctan2(episode.car_data[id].supervised_car_vel[1], episode.car_data[id].supervised_car_vel[2])))
                            episode.car_data[id].action_car_angle =np.arctan2(episode.car_data[id].supervised_car_vel[1], episode.car_data[id].supervised_car_vel[2])/np.pi

                        # print(" Choose car angle according to supervised car " + str(episode.action_car_angle ))
                    else:
                        episode.car_data[id].action_car_angle = np.random.normal(probability_angular[0], self.settings.sigma_car_angular)[0]
                else:
                    episode.car_data[id].action_car_angle = probability_angular[0]
                # print(" Save angle in fake episode "+str( episode.action_car_angle))
                if self.settings.angular_car_add_previous_angle:
                    if frame == 0:
                        car_velocity = episode.car_data[id].car_dir
                    else:
                        car_velocity = episode.car_data[id].velocity_car[frame-1]

                    angle=np.arctan2(car_velocity[1], car_velocity[2])/np.pi
                    # print (" Previous angle "+str(angle)+" dir "+str(car_velocity))
                    angle = angle+episode.car_data[id].action_car_angle
                    # print(" After adding  angle " + str(episode.action_car_angle) + " dir " + str(angle))
                    angle=self.normalize_angle(angle*np.pi)/np.pi
                    # print(" After normalizing " + str(angle))
                else:
                    angle=episode.car_data[id].action_car_angle
                episode.car_data[id].velocity_car =np.zeros(3)
                episode.car_data[id].velocity_car[1]= np.sin( angle*np.pi)# y
                episode.car_data[id].velocity_car[2]=np.cos( angle*np.pi)# x
                #print(" Draw random angle  " +str(episode.action_car_angle)+" times pi "+str(episode.action_car_angle*np.pi)+" vel" +str(episode.velocity_car) )
                episode.car_data[id].velocity_car = episode.car_data[id].velocity_car  * speed # / np.linalg.norm(episode.velocity_car)
                # print(" Final velocity  " + " vel" + str(episode.velocity_car))
            else:
                if len(episode.car_data[id].external_car_vel)>0:
                    car_dir= episode.car_data[id].external_car_vel
                else:
                    if frame == 0:
                        car_velocity = episode.car_data[id].car_dir
                    else:
                        car_velocity = episode.car_data[id].velocity_car[frame-1]
                    pos=episode.car_data[id].car[frame]
                    if int(pos[1])>=0 and int(pos[2])>=0 and int(pos[1])<episode.valid_directions.shape[0] and  int(pos[2])<episode.valid_directions.shape[1] and np.linalg.norm(episode.valid_directions[int(pos[1]),int(pos[2]),1:])>1e-5:
                        car_dir =episode.valid_directions[int(pos[1]),int(pos[2]),:]*(1/np.linalg.norm(episode.valid_directions[int(pos[1]),int(pos[2]),1:]))
                    else:

                        car_dir =episode.car_data[id].car_dir
                    if self.settings.car_motion_smoothing:
                        car_dir=car_dir+car_velocity
                if np.linalg.norm(car_dir)> 1e-5:
                    episode.car_data[id].velocity_car =car_dir*speed/np.linalg.norm(car_dir)
                else:
                    episode.car_data[id].velocity_car = car_dir* speed

        else:
            probability, summary_train = self.sess.run([self.probabilities, self.train_summaries], feed_dict)
            if training:
                move = np.random.binomial(1, probability[1])
                #print(" Car action random " + str(episode.action_car))
            else:
                move = probability[1]>=0.5
                #print(" Car action " + str(move))
            # print ("Random action " + str(move)+" probabilities "+str(probability)+" episode car_dir "+str(episode.car_dir))
            if move == 0:
                episode.car_data[id].speed_car = 0
                episode.car_data[id].probabilities_car = np.copy(probability[1])
                episode.car_data[id].velocity_car = np.zeros(3)
                episode.car_data[id].action_car = 0
            else:
                episode.car_data[id].probabilities_car = np.copy(probability[1])
                episode.car_data[id].speed_car = np.linalg.norm(episode.car_data[id].car_dir)
                episode.car_data[id].velocity_car = episode.car_data[id].car_dir
                episode.car_data[id].action_car = 1


        #print("Car Take vel "+str(np.linalg.norm(episode.velocity_car)/self.settings.car_max_speed)+" action "+str(episode.action_car)+" move "+str(episode.velocity_car))#+" probability "+str(probability))
        return episode.car_data[id].velocity_car

    def grad_feed_dict(self,id, agent_action_all, car_measures, agent_pos_all, agent_velocity_all, ep_itr, episode, frame,
                       reward, statistics, poses, priors, agent_speed=None, statistics_car=None, training=True, agent_frame=-1):
        if agent_frame < 0 or not training:
            agent_frame = frame
        # print (" Get gradient feed dict")

        r = reward[ep_itr, agent_frame]
        #print ( "Car net Reward "+str(r))
        if car_measures[ep_itr, agent_frame,CAR_MEASURES_INDX.agent_dead]:
            r = 0
            return {}
        #print ("Car net Frame "+str(frame)+" Episode "+str(ep_itr)+" Reward "+str(r)+" hit by car:"+str(agent_measures[ep_itr,agent_frame, 0])+" Reached goal "+str(agent_measures[ep_itr,agent_frame, 13]))
        velocity=statistics_car[ep_itr,id,:,STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]]
        pedestrian_id = car_measures[ep_itr, agent_frame, CAR_MEASURES_INDX.id_closest_agent].astype(int)
        if pedestrian_id>=0:
            agent_pos = agent_pos_all[ep_itr, pedestrian_id, agent_frame, :]
            agent_vel_input=agent_velocity_all[ep_itr,pedestrian_id,agent_frame-1,:]
            if frame ==0:
                agent_vel_input=agent_velocity_all[ep_itr,pedestrian_id, -1, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]]
        else:
            agent_pos =[]
            agent_vel_input =[]
        if frame == 0:
            velocity = statistics_car[ep_itr, id, 3:6, STATISTICS_INDX_CAR.goal]
        car_pos= statistics_car[ep_itr,id,frame,STATISTICS_INDX_CAR.agent_pos[0]:STATISTICS_INDX_CAR.agent_pos[1]]

        car_dir=statistics_car[ep_itr,id, 3:6, STATISTICS_INDX_CAR.goal]
        car_goal=statistics_car[ep_itr,id, 0:3, STATISTICS_INDX_CAR.goal]





        # print (statistics_car[ep_itr, 3:6, STATISTICS_INDX_CAR.goal])
        # print ("Car init dir "+str(statistics_car[ep_itr, 3:6, STATISTICS_INDX_CAR.goal]))
        distracted=car_measures[ep_itr, frame, CAR_MEASURES_INDX.distracted]
        dist_to_closest_car=car_measures[ep_itr, frame, CAR_MEASURES_INDX.dist_to_closest_car]
        dist_to_closest_pedestrian=car_measures[ep_itr, frame, CAR_MEASURES_INDX.dist_to_closest_pedestrian]
        intersection_with_pavement=car_measures[ep_itr, frame, CAR_MEASURES_INDX.iou_pavement]
        car_action_angle = statistics_car[ep_itr,id, :, STATISTICS_INDX_CAR.angle]

        closest_car =episode.car_data[id].closest_car[frame]
        # print (" Distracted gradient ? "+str(distracted)+" frame "+str(frame))
        if self.settings.car_input == "difference":
            state_in = self.get_difference(agent_pos , car_pos, agent_vel_input,car_goal, distracted )

        elif self.settings.car_input == "scalar_product":
            # print ("Frame "+str(frame))
            state_in = self.get_scalar_product(agent_pos, car_pos,  car_dir, distracted)
            #print (" Gradient input: "+str(state_in))
        elif self.settings.car_input == "distance":
            state_in =self.get_distance_to_agent( agent_pos, car_pos, distracted)
            #state_in=np.array([[np.linalg.norm(car_pos[ frame, 1:]-agent[frame,1:])/np.linalg.norm(self.settings.env_shape[1:])]])
        elif self.settings.car_input == "distance_to_car_and_pavement_intersection" or self.settings.car_input == "distance_to_car_and_pavement_pedestrian_speed_intersection":
            state_in = self.get_distance_to_agent_cars_and_pavement(agent_pos, car_pos, distracted, dist_to_closest_car, dist_to_closest_pedestrian,
                                                                    intersection_with_pavement)

        else:

            state_in= self.get_time_to_collision(agent_pos , car_pos, agent_velocity,velocity, distracted , max(self.settings.agent_shape[1:])+max(self.settings.car_dim[1:]), dist_to_closest_car)
        # print("Grad state in gradient "+str(state_in)+" reward "+str(r))
        feed_dict = {self.state_in: state_in,
                     self.advantages: r,
                     self.sample: self.get_sample(id,statistics_car, ep_itr, agent_frame)}  # , priors)}
        if self.settings.car_input == "difference":

            if frame >= 1:
                car_velocity_cur = velocity[frame - 1]
            else:
                car_velocity_cur = velocity
            feed_dict[self.past_move] = (
                        (np.array(car_velocity_cur[1:]) / self.settings.car_max_speed_voxelperframe + 1) / 2.0).reshape((1, 2))
            #print("Grad feed dict  " + str(feed_dict[self.past_move]))
        if self.settings.angular_car:  # agent_pos, car_pos,  velocity_car, frame,closest_car, goal, distracted)
            # print(" Grad feed dict input " + str(frame)+" sample " +str(car_action_angle[frame]))
            feed_dict[self.state_in_angle] = self.get_angles_in(agent_pos, car_pos,
                                                                velocity, closest_car,
                                                                car_goal, distracted)
            feed_dict[self.sample_angular] =np.array([car_action_angle[frame]])

        if self.settings.car_input == "distance_to_car_and_pavement_pedestrian_speed_intersection":

            feed_dict[self.state_in_additional]  = self.get_distance_to_agent_cars_and_pavement_speed(agent_pos, car_pos,  distracted,
                                                                    dist_to_closest_car, dist_to_closest_pedestrian,
                                                                    intersection_with_pavement, agent_velocity,velocity)

        return feed_dict


    def get_sample(self, id, statistics_car,ep_itr, agent_frame):
        if self.settings.car_input == "difference":
            velocity = statistics_car[ep_itr,id, agent_frame, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]]
            # print (" Velocity sample "+str(velocity))
            return np.array(velocity[1:]*(1/self.settings.car_max_speed_voxelperframe))
        return np.array([statistics_car[ep_itr,id, agent_frame,STATISTICS_INDX_CAR.action]])


    def fully_connected_size(self, dim_p):
        return 1

    def get_distance_to_agent(self, agent_pos, car_pos, distracted):
        if distracted or  len(agent_pos)==0:
            return np.array([[1]])
        disp_car_to_ped = car_pos[1:] - agent_pos[1:]
        return np.array([[np.linalg.norm(disp_car_to_ped)/np.linalg.norm(self.settings.env_shape[1:])]])#-0.5

    def get_distance_to_agent_cars_and_pavement(self, agent_pos, car_pos, distracted, dist_to_closest_car,dist_to_closest_ped, intersection_with_pavement):
        difference = np.zeros((1, 3))
        if distracted or len(agent_pos)==0:
            difference[0, 0]=1
        else:
            disp_car_to_ped = car_pos[1:] - agent_pos[1:]
            difference[0, 0] =np.linalg.norm(disp_car_to_ped)/np.linalg.norm(self.settings.env_shape[1:])

            if dist_to_closest_ped/np.linalg.norm(self.settings.env_shape[1:])<difference[0, 0]:

                difference[0, 0] =dist_to_closest_ped/np.linalg.norm(self.settings.env_shape[1:])

        difference[0, 1] = dist_to_closest_car/np.linalg.norm(self.settings.env_shape[1:])
        difference[0, 2] = intersection_with_pavement

        return difference#-0.5
    def get_distance_to_agent_cars_and_pavement_speed(self,agent_pos, car_pos, distracted, dist_to_closest_car,dist_to_closest_ped, intersection_with_pavement,pedestrian_velocity, velocity_car):
        difference = np.zeros((1, 3))
        if len(agent_pos)>0:
            difference[0, 0] =np.linalg.norm(pedestrian_velocity[1:])/(15/17)
            vector_car_to_agent = agent_pos - car_pos
            difference[0, 1] = self.signed_angle_between(velocity_car[1:], vector_car_to_agent[1:])
            difference[0, 2] = self.signed_angle_between(velocity_car[1:], pedestrian_velocity[1:])
        return difference

    def get_scalar_product(self, agent_pos, car_pos,  velocity_car, distracted):
        MAX_TIME=1
        if distracted or len(agent_pos)==0:
            return np.array([[MAX_TIME]])

        disp_car_to_ped = agent_pos-car_pos

        car_vel = velocity_car
        return np.array([[np.dot(car_vel[1:], disp_car_to_ped[1:])/np.linalg.norm(self.settings.env_shape[1:])]])

    def get_difference(self, agent_pos, car_pos, agent_vel, car_goal, distracted):
        if distracted or len(agent_pos) == 0:
            disp_to_pedestrian_normalized = np.zeros_like(disp_to_pedestrian_normalized)
            agent_vel = np.zeros_like(agent_vel)
        else:
            disp_car_to_ped =  agent_pos-car_pos
            disp_to_pedestrian_normalized=disp_car_to_ped/np.linalg.norm(self.settings.env_shape[1:])
            disp_to_goal=car_goal-car_pos
            disp_to_goal_normalized=disp_to_goal/np.linalg.norm(self.settings.env_shape[1:])

        difference=np.zeros((1,6))
        difference[0,0:2]=disp_to_goal_normalized[1:]
        difference[0, 2:4] = disp_to_pedestrian_normalized[1:]
        difference[0, 4:] = agent_vel[1:]

        return difference

    def get_time_to_collision(self, agent_pos, car_pos, velocity, velocity_car, distracted,collision_radius):

        MAX_TIME=1000.0
        if distracted or len(agent_pos)==0:
            #print ("Distracted True ")
            return np.array([[MAX_TIME/MAX_TIME]])

        disp_car_to_ped =  car_pos- agent_pos

        difference_in_vel = velocity_car-velocity
        collision_time_limits=[[],[]]
        for i in range(1,3):
            if abs(difference_in_vel[i])<1e-5:
                if abs(difference_in_vel[i])<1e-5:
                    collision_time_limits[i - 1] = [0, MAX_TIME]
                else:
                    collision_time_limits[i - 1] = [MAX_TIME,0]
            elif difference_in_vel[i]<0:
                collision_time_limits[i-1] = [(collision_radius - disp_car_to_ped[i]) / difference_in_vel[i],
                     (-collision_radius - disp_car_to_ped[i]) / difference_in_vel[i]]
            else:
                collision_time_limits[i - 1] = [(-collision_radius - disp_car_to_ped[i]) / difference_in_vel[i],
                                                (collision_radius - disp_car_to_ped[i]) / difference_in_vel[i]]
        # print ( " collision time limits "+str(collision_time_limits))
        t=[max(collision_time_limits[0][0], collision_time_limits[1][0]),min(collision_time_limits[0][1], collision_time_limits[1][1])]
        # print (" t " + str(t)+" mean "+str(np.mean(t))+" scaled "+str(max(0,np.mean(t))/MAX_TIME)+" centered "+str(max(0,np.mean(t))/MAX_TIME))

        if t[0]<=t[1]:

            return np.array([[max(0,np.mean(t))/MAX_TIME]])

        return np.array([[MAX_TIME/MAX_TIME]])

    def get_angles_in(self, agent_pos, car_pos,  velocity_car, closest_car, goal, distracted):

        if len(closest_car)>0:
            vector_car_to_closest_car = closest_car - car_pos
        else:
            vector_car_to_closest_car=car_pos-car_pos
        vector_car_to_goal = goal - car_pos
        # Now find angular difference between vector to agent and velocity_car

        # print(" Get angles: pedestrian agent " + str(agent_pos) + " closest car " + str(
        #     closest_car) + " goal location " + str(goal) + " car position " + str(car_pos))
        # print(" Vectors to agent: " + str(vector_car_to_agent) + " closest car " + str(
        #     vector_car_to_closest_car) + " vector to goal " + str(vector_car_to_goal))
        angular_input=np.zeros((1,3))
        if distracted or len(agent_pos)==0:
            angular_input[0, 0]=np.pi
        else:
            vector_car_to_agent = agent_pos - car_pos
            angular_input[0, 0] = self.signed_angle_between(velocity_car[1:], vector_car_to_agent[1:])
        angular_input[0, 1] = self.signed_angle_between(velocity_car[1:], vector_car_to_closest_car[1:])
        angular_input[0, 2] = self.signed_angle_between(velocity_car[1:], vector_car_to_goal[1:])


        # print(" Angle differnces : " + str(angular_input[0, 0]) + " closest car " + str(angular_input[0, 1]) + " vector to goal " + str(angular_input[0, 2]))

        return angular_input

    def signed_angle_between(self, direction_a, direction_b):
        angle_a=np.arctan2(direction_a[0], direction_a[1])
        angle_b = np.arctan2(direction_b[0], direction_b[1])
        # print (" Car direction "+str(direction_a)+" angle "+str(angle_a/np.pi)+" input direction "+str(direction_b)+" angle "+str(angle_b/np.pi)+" difference "+str((angle_b-angle_a)/np.pi))
        difference=self.normalize_angle(angle_b-angle_a)/np.pi
        # print(" Normalized difference " + str(difference) )

        return difference

    def normalize_angle(self,angle):
        # print (" Angle "+str(angle )+" smaller than "+str(-np.pi)+"? "+str(angle <= -np.pi)+" larger than "+str(np.pi)+"? "+str(angle > np.pi))
        if angle <= -np.pi:
            # print ("Before addition " + str(self.angle) + " " + str(self.angle / np.pi))
            angle = 2 * np.pi + angle
            # print ("After Rotation " + str(self.angle) + " " + str(self.angle / np.pi))
        if angle > np.pi:
            # print ("Before minus " + str(self.angle) + " " + str(self.angle / np.pi))
            angle = angle - (2 * np.pi)
            # print ("After Rotation " + str(self.angle)+" "+str(self.angle/np.pi))
        return angle
