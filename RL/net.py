import numpy as np
from settings import RANDOM_SEED, RANDOM_SEED_NP, NBR_MEASURES,STATISTICS_INDX, PEDESTRIAN_MEASURES_INDX, STATISTICS_INDX_CAR, CAR_MEASURES_INDX
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)


from datetime import datetime
import pickle
import logging
import copy
#
# np.set_printoptions(precision=5)


def variable_summaries(var, name='', summaries=[]):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        summaries.append(tf.compat.v1.summary.scalar(name + 'mean', mean))
        with tf.compat.v1.name_scope(name + 'stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        summaries.append(
            tf.compat.v1.summary.scalar(name + 'stddev', stddev))
        summaries.append(
            tf.compat.v1.summary.scalar(name + 'max', tf.reduce_max(input_tensor=var)))
        summaries.append(
            tf.compat.v1.summary.scalar(name + 'min', tf.reduce_min(input_tensor=var)))
        summaries.append(
            tf.compat.v1.summary.histogram(name+'histogram', var))


class Net(object):
    # Softmax Simplified
    def weight_variable(self, name, shape):
        return tf.compat.v1.get_variable(name, shape, self.DTYPE, tf.compat.v1.truncated_normal_initializer(stddev=0.1))

    def bias_variable(self, name, shape):
        return tf.compat.v1.get_variable(name, shape, self.DTYPE, tf.compat.v1.constant_initializer(0.01, dtype=self.DTYPE))

    def __init__(self, settings, weights_name=""):

        print ("weights_name "+str(weights_name))
        # Design choices
        self.set_nbr_channels()
        self.settings=settings
        np.random.seed(RANDOM_SEED_NP)
        tf.random.set_seed(RANDOM_SEED)

        self.old=self.settings.old
        self.old_mem=self.settings.old_mem

        # Counter in training
        self.num_iterations=0
        self.num_grad_itrs=0
        self.grad_counter=0


        self.position = np.zeros(3)

        self.DTYPE = tf.float32
        if weights_name=="":
            self.init_net_car = False
            self.init_net=False
            self.goal_net = False
            self.car_net = False
        elif weights_name=="init":
            self.init_net_car = False
            self.init_net = True
            self.goal_net = False
            self.car_net = False
        elif weights_name=="car":
            self.init_net_car = False
            self.init_net = False
            self.goal_net = False
            self.car_net=True
        elif weights_name == "place_car":
            self.init_net_car = True
            self.init_net = False
            self.goal_net = False
            self.car_net = False
        else:
            self.init_net_car = False
            self.init_net = False
            self.goal_net = True
            self.car_net = False
        self.mean= [103.939/255.0, 116.779/255.0, 123.68/255.0,16.5/33.0, 0.5, 0.5]
        self.sess=None
        self.log=None
        self.writer=None
        self.test_writer=None

        # Summaries
        self.train_summaries=[]
        self.loss_summaries=[]
        self.conv_summaries=[]
        self.grad_summaries=[]

        self.gradBuffer=[]
        self.traj_forget_rate=1#self.settings.people_traj_gamma
        with tf.compat.v1.variable_scope(weights_name) as scope_outer:
            print ("scope_name " + str(weights_name))
            prev_layer = self.define_conv_layers()

            dim_p = self.get_dim_p()
            self.fully_connected(dim_p, prev_layer)
            self.calc_probabilities(self.fully_connected_size(dim_p))
            self.advantages = tf.compat.v1.placeholder(shape=[], dtype=self.DTYPE, name="advantage")
            if self.writer:
                self.loss_summaries.append(tf.compat.v1.summary.scalar("reward", self.advantages))
            self.loss_setup(dim_p)


        vars = tf.compat.v1.trainable_variables()
        self.gradient_holders = []
        self.tvars = []
        idx = 0

        if self.settings.printdebug_network_input and self.is_net_type(settings.printdebug_network):
            self.feed_dict = []# For debugging can be removed!
        self.save_probabilities=[]


        for var in vars:
            is_goal_net_var=self.goal_net  and not  self.init_net_car and not self.init_net and not self.car_net and "goal" in var.name
            is_init_net_var=self.init_net  and not  self.init_net_car and not self.goal_net and not self.car_net and "init" in var.name
            is_car_net_var = self.car_net and not  self.init_net_car  and not self.goal_net and not self.init_net and "car" in var.name and "place_car" not in var.name
            is_agent_net_var=not self.init_net and not  self.init_net_car  and not self.car_net and  not self.goal_net and "init" not in var.name and "goal" not in var.name and "place_car" not in var.name
            is_init_car_net_var = self.init_net_car and not self.init_net  and not self.goal_net and not self.car_net and "place_car" in var.name
            if is_goal_net_var or is_init_net_var  or is_agent_net_var or is_car_net_var or is_init_car_net_var:
                if is_car_net_var:
                    print ("Added to buffer " +str(var.name)+" var.shape "+str(var.shape))
                placeholder = tf.compat.v1.placeholder(self.DTYPE, name=str(idx) + '_holder_gradient')
                self.gradient_holders.append(placeholder)
                self.tvars.append(var)
                idx = idx + 1
        #print ("Trainable variables "+str(self.tvars))



        self.gradients = tf.gradients(self.loss, self.tvars)

        # if  self.car_net:
        #     optimizer = tf.train.AdamOptimizer(learning_rate=self.settings.lr_car)
        #     print(" use car learning rate "+str(self.settings.lr_car))
        # elif settings.sem_arch_init:
        #
        #     self.gradients = tf.compat.v1.gradients(ys=self.loss, xs=self.tvars)

        if  self.car_net:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.settings.lr_car)
            print(" use car learning rate "+str(self.settings.lr_car))
        elif settings.sem_arch_init:

            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.settings.lr, momentum=0.9)  # self.settings.lr)
        elif self.settings.continous and self.settings.polynomial_lr:
            decay_steps = 10000
            self.learningrate=tf.compat.v1.train.polynomial_decay(self.settings.lr,self.num_grad_itrs,decay_steps)
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learningrate)#self.settings.lr)
        else:
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.settings.lr)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.settings.lr)

        self.define_update_gradients( optimizer)


        # Summaries
        if self.writer:
            self.create_summary_holders()

    def set_nbr_channels(self):
        self.nbr_channels = 6

    def is_net_type(self, weights_name):
        if weights_name=="":
            return self.init_net_car== False and self.init_net==False  and self.goal_net == False and self.car_net == False
        elif weights_name=="init":
            return self.init_net_car == False and self.init_net ==True and self.goal_net== False and self.car_net == False
        elif weights_name=="car":
            return self.init_net_car == False and self.init_net== False and  self.goal_net == False and self.car_net==True
        elif weights_name == "place_car":
            return self.init_net_car == True and self.init_net == False and self.goal_net == False and self.car_net == False
        else:
            return self.init_net_car == False and self.init_net == False and self.goal_net == True and self.car_net == False

    def create_summary_holders(self):
        self.num_cars = tf.compat.v1.placeholder(shape=[], dtype=self.DTYPE, name="num_cars")
        self.loss_summaries.append(tf.compat.v1.summary.scalar("num_cars", self.num_cars))
        self.num_people = tf.compat.v1.placeholder(shape=[], dtype=self.DTYPE, name="num_people")
        self.loss_summaries.append(
            tf.compat.v1.summary.scalar("num_people", self.num_people))
        self.pavement = tf.compat.v1.placeholder(shape=[], dtype=self.DTYPE, name="pavement")
        self.loss_summaries.append(
            tf.compat.v1.summary.scalar("pavement", self.pavement))
        self.num_of_obj = tf.compat.v1.placeholderr(shape=[], dtype=self.DTYPE, name="num_of_obj")
        self.loss_summaries.append(
            tf.compat.v1.summary.scalar("num_of_obj", self.num_of_obj))
        self.dist_travelled = tf.compat.v1.placeholder(shape=[], dtype=self.DTYPE, name="dist_travelled")
        self.loss_summaries.append(
            tf.compat.v1.summary.scalar("dist_travelled", self.dist_travelled))
        self.out_of_axis = tf.compat.v1.placeholder(shape=[], dtype=self.DTYPE, name="out_of_axis")
        self.loss_summaries.append(
            tf.compat.v1.summary.scalar("out_of_axis", self.out_of_axis))
        self.tot_reward = tf.compat.v1.placeholder(shape=[], dtype=self.DTYPE, name="tot_reward")
        self.loss_summaries.append(tf.compat.v1.summary.scalar("tot_reward", self.tot_reward))

    def define_conv_layers(self):

        raise NotImplementedError("Please Implement this method")

    def feed_forward(self,id, episode, frame, training, agent_frame=-1,  viz=False, distracted=False, manual=False):
        if agent_frame<0 or not training:
            agent_frame=frame

        # print(" Ep use real time feed forward? " + str(episode.useRealTimeEnv))
        feed_dict=self.construct_feed_dict(id, episode , frame, agent_frame,training=training, distracted=distracted)
        if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network) and not self.init_net and not self.init_net_car:
            if id==0:
                feed_dicts=[]
                for id_local in range(self.settings.number_of_agents):
                    feed_dicts.append({})
                self.feed_dict.append(feed_dicts)
            feed_dict_copy={}
            for key, value in feed_dict.items():
                feed_dict_copy[key] = copy.deepcopy(value)
            self.feed_dict[-1][id]=feed_dict_copy
        value=self.apply_net(id, feed_dict, episode, agent_frame, training, max_val=self.settings.deterministic_test, viz=viz, manual=manual)#, viz=viz)
        if self.settings.printdebug_network_input:
            print ("model out: "+str(value))
        self.num_iterations+=1
        return value

    def set_grad_buffer(self, grad_buffer):
        self.gradBuffer=grad_buffer

    def construct_feed_dict( self,id, episode, frame, agent_frame,training=True,distracted=False):

        feed_dict ={}
        #print ("Construct feed dict frame " + str(frame) + " agent frame " + str(agent_frame)+" pos "+str(episode.agent[agent_frame] ))

        feed_dict[self.state_in] = self.get_input(id=id, episode=episode, agent_pos_cur=episode.pedestrian_data[id].agent[agent_frame], frame_in=frame, training= training)
        if self.settings.car_var and not self.init_net and not self.goal_net and not self.init_net_car:
            if agent_frame==0:
                vel=episode.pedestrian_data[id].vel_init
            else:
                vel=episode.pedestrian_data[id].velocity[frame-1]
            feed_dict[self.cars] = episode.get_input_cars(episode.pedestrian_data[id].agent[agent_frame],frame,vel, self.settings.field_of_view,  distracted)

        self.get_feature_vectors(id, agent_frame, episode, feed_dict, frame, distracted)
        return feed_dict #self.sample: episode.velocity[self.frame][0]

    def get_feature_vectors(self, id, agent_frame, episode, feed_dict, frame, distracted=False):
        raise NotImplementedError("Please Implement this method")


    def apply_net(self,id,  feed_dict, episode, frame, training, max_val=False, viz=False, manual=False):
        raise NotImplementedError("Please Implement this method")

    def get_vel(self,id,  episode, frame):
        raise NotImplementedError("Please Implement this method")

    def set_session(self, session):
        self.sess = session

    def set_logs(self, writer,test_writer, log):
        self.writer=writer
        self.test_writer=test_writer
        self.log=log

    def calculate_gradients(self, id, feed_dict, stat, frame, ep_itr):
        responsible_v=0
        if self.settings.velocity and not self.init_net and not self.goal_net and not self.car_net and not self.init_net_car:
            loss, grads, weights, responsible , responsible_v , probabilities, mu, normal_loss, sample, sample_v,reward = self.sess.run(
                [self.loss, self.gradients, self.tvars, self.responsible_output,self.probability_of_velocity,self.probabilities, self.probabilities_v, self.normal_loss,self.sample, self.velocity,self.advantages], feed_dict)  # self.merged
            # if self.settings.printdebug_network_input:
            #     print(" Probabilities " + str(probabilities) + " mu " + str(mu))
            #     print(" sample of action  " + str(sample) + " sample of velocity " + str(sample_v))
            #     print(" probability of action  " + str(responsible)+ " probability of velocity " + str(responsible_v) )
            #     print(" normal loss " +str(normal_loss)+" normal loss exponential "+str(np.exp(normal_loss)))
            #
            #
            #     if np.sum(np.array(self.save_probabilities[frame]['probabilities']) -np.array(probabilities))<1e-10:
            #         print("Probabilities are the same! " + str(probabilities))
            #     else:
            #         print("Not equal. Probabilities are not the same! gradient: " + str(probabilities) + " before " + str(
            #             self.save_probabilities[frame]['probabilities'])+" error:"+str(np.sum(np.array(self.save_probabilities[frame]['probabilities']) -np.array(probabilities))))
            #
            #     if self.save_probabilities[frame]['mean_vel'] == mu:
            #         print("Mean vel are the same! " + str(mu))
            #     else:
            #         print(
            #             "Not equal. Mean vel  are not the same! gradient: " + str(mu) + " before " + str(
            #                 self.save_probabilities[frame]['mean_vel']))
            #
            #
            #     if self.save_probabilities[frame]['sample'] ==sample:
            #         print("Action samples are the same! "+str(sample))
            #     else:
            #         print("Not equal. Action samples are not the same! gradient: " + str(sample)+ " before "+str(self.save_probabilities[frame]['sample']))
            #
            #     if abs(responsible-self.save_probabilities[frame]['sample_probability'] )< 1e-5:
            #         print("Action probability are the same! " + str(responsible))
            #     else:
            #         print("Not equal. Action probability are not the same! gradient: " + str(responsible) + " before " + str(
            #             self.save_probabilities[frame]['sample_probability'])+" error: "+str(abs(responsible-self.save_probabilities[frame]['sample_probability'] )))
            #
            #
            #
            #     if abs(sample_v-self.save_probabilities[frame]['speed_sample'] )< 1e-5:
            #         print("Speed sample are the same! " + str(sample_v))
            #     else:
            #         print("Not equal. Speed sample are not the same! gradient: " + str(sample_v) + " before " + str(
            #             self.save_probabilities[frame]['speed_sample'])+" error: "+str(abs(sample_v-self.save_probabilities[frame]['speed_sample'] )))
            #
            #     if abs(responsible_v-self.save_probabilities[frame]['speed_probability'])< 1e-5:
            #         print("Speed probability are the same! " + str(responsible_v))
            #     else:
            #         print("Not equal. Speed probability are not the same! gradient: " + str(responsible_v) + " before " + str(
            #             self.save_probabilities[frame]['speed_probability'])+" error: "+str(abs(responsible_v-self.save_probabilities[frame]['speed_probability'])))



        elif self.init_net or self.init_net_car:
            loss,grads,weights, responsible_output,responsible_prior  = self.sess.run([self.loss,self.gradients, self.tvars, self.responsible_output, self.responsible_output_prior], feed_dict)  # self.merged
            if  self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
                print("loss "+str(loss))
                print(" responsible output "+str(responsible_output))
                print(" responsible prior " + str(responsible_prior))
                print("Saved responsible output "+str(self.feed_dict[ep_itr][id]['responsible'] ) +" in gradient used: "+str(responsible_output*responsible_prior))
                print(self.feed_dict[ep_itr][id]['ln(responsible)'])
                print(feed_dict[self.advantages])
                print("Loss should be"+str( -self.feed_dict[ep_itr][id]['ln(responsible)']*feed_dict[self.advantages]) )
        else:

            loss, grads, weights, = self.sess.run(
                [self.loss, self.gradients, self.tvars],
                feed_dict)  # self.merged

        if self.car_net:
            stat[ep_itr,id, frame, STATISTICS_INDX_CAR.loss]=loss
        elif self.init_net:
            stat[ep_itr, id, frame, STATISTICS_INDX.loss_initializer] = loss
        elif self.init_net_car:
            stat[ep_itr, id, frame, STATISTICS_INDX_CAR.loss_initializer] = loss
        else:
            stat[ep_itr,id, frame, STATISTICS_INDX.loss]=loss
        # if self.car_net:
        #
        #     print( "Loss "+str(loss)+" weights "+str(weights)+" gradient "+str(grads)+" input "+str(feed_dict[self.state_in])+" action input "+str(feed_dict[self.sample])+" responsible output "+str(responsible))
        #     print( " Probabilities "+str(probabilities)+" mu "+str(mu)+" probability grad "+str(grads_prob)+" sample "+str(sample))

        # if self.writer:
        #     self.writer.add_summary(summaries_loss, global_step=self.num_grad_itrs)
        #     self.writer.add_summary(summaries_grad, global_step=self.num_grad_itrs)

        self.num_grad_itrs += 1

        indx=0
        for idx, gradient in enumerate(grads):

            if np.isnan(gradient).any():
                logging.warning('Gradient is '+str(gradient)+" given input "+str(feed_dict))
                if np.isnan(gradient).all():
                    import sys
                    sys.exit()
            else:
                multiplier=1
                if self.settings.replay_buffer>0:
                    multiplier=self.importance_sample_weight(id, responsible, stat, ep_itr, frame, responsible_v)
                    if np.isnan(multiplier):
                        multiplier=0

                if idx< len(self.gradBuffer):
                    if self.settings.printdebug_network_input:
                        print(" Buffer before addition "+str(np.sum(np.abs(self.gradBuffer[idx]))))#+" reward "+str(reward))
                    self.gradBuffer[idx] += multiplier * gradient
                    if self.settings.printdebug_network_input:
                        print(" Add gradient "+str(np.sum(np.abs(gradient)))+" multiplier "+str(multiplier)+" buffer "+str(np.sum(np.abs(self.gradBuffer[idx])))+" grad counter "+str(self.grad_counter))
        self.grad_counter=self.grad_counter+1

    def importance_sample_weight(self,id,responsible, statistics, ep_itr, frame, responsible_v=0):
        raise NotImplementedError("Please Implement this method")

    def normalpdf(self, mu,sigma, x):
        return 1 / (sigma * np.sqrt(2 * np.pi)) *np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        # May need to add logartihms


    def train(self,id, ep_itr, statistics, episode, filename, filename_weights,poses,priors,statistics_car, seq_len=-1):
        if seq_len==-1:
            seq_len=self.settings.seq_len_train-1
        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_speed = self.stats(id,statistics, statistics_car)

        if self.car_net or self.init_net_car:
            stat=statistics_car
        else:
            stat = statistics

        if self.settings.normalize:
            reward=self.normalize_reward(agent_reward_d, agent_measures)
        else:
            reward=agent_reward_d

        if self.settings.replay_buffer>0:
            for _ in range(self.settings.replay_buffer):
                self.reset_mem()
                for frame in range(seq_len):
                    feed_dict = self.grad_feed_dict(id, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                                    episode, frame, reward, statistics, poses,priors,
                                                    agent_speed=agent_speed,statistics_car=statistics_car)
                    if len(feed_dict)>0:

                        self.calculate_gradients(id, feed_dict, stat, frame,
                                                 ep_itr)

            if ep_itr==len(statistics.shape[0])-1:
                if not self.settings.overfit or (self.settings.overfit and self.num_grad_itrs % 20 == 0):
                    with open(filename, 'wb') as f:
                        pickle.dump(self.gradBuffer, f, pickle.HIGHEST_PROTOCOL)

                    [weights] = self.sess.run([self.tvars])


                    with open(filename_weights, 'wb') as f:
                        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)


                self.update_gradients()
        else:

            #print "Gradient: "+str(seq_len-1)

            self.reset_mem()

            for frame in range(seq_len):
                #if frame % self.settings.action_freq == 0 and not (statistics[ep_itr, frame, 0]==1 and statistics[ep_itr, max(frame-1, 0), 0]==1 ) and not (statistics[ep_itr, frame, 13]==1 and statistics[ep_itr, max(frame-1, 0), 13]==1 ):
                #print "COnstruct Gradient feed dict "+str(ep_itr)+" "+str(frame)

                feed_dict = self.grad_feed_dict(id, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                                episode, frame, reward, statistics,poses,priors, agent_speed=agent_speed,statistics_car=statistics_car)
                if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
                    if len(feed_dict) == 0:
                        print("Empty feed dict "+str(len(feed_dict)))



                if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network): #not self.goal_net and not self.init_net and not
                    print("Grad Feed dict frame " + str(frame))
                    for key, value in feed_dict.items():
                        self.print_feed_dict(key, value)
                        frame_indx=ep_itr*(seq_len)+ int(frame)

                        if key in self.feed_dict[frame_indx][id]:
                            if np.sum(np.abs(self.feed_dict[frame_indx][id][key]-feed_dict[key]))<1e-5:
                                print ("Equal agent feed net "+str(key))
                            else:
                                print ("Not Equal agent feed net " + str(key))
                                print("In training used see below")
                                self.print_feed_dict(key, self.feed_dict[frame_indx][id][key])

                if len(feed_dict) > 0:
                    self.calculate_gradients( id, feed_dict, stat, frame, ep_itr) #                 self.calculate_gradients(episode, frame, feed_dict) %


            if ep_itr == statistics.shape[0] - 1 and ((id==self.settings.number_of_agents-1 and not self.car_net) or (id==self.settings.number_of_car_agents-1 and self.car_net) ):
                if self.settings.printdebug_network_input and self.is_net_type(self.settings.printdebug_network):
                    self.feed_dict = []
                if not self.settings.overfit or( self.settings.overfit and self.num_grad_itrs%20==0):
                    with open(filename, 'wb') as f:
                        pickle.dump(self.gradBuffer, f, pickle.HIGHEST_PROTOCOL)
                        sum=0
                        for idx, gradient in enumerate(self.gradBuffer):
                            sum=+np.sum(np.abs(gradient))
                        print ("Save gradient " + str(sum))


                    [weights]=self.sess.run([self.tvars])
                    with open(filename_weights, 'wb') as f:
                        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
                else:
                    [weights] = self.sess.run([self.tvars])
                    # print "Weights "
                    # print weights
                #if self.num_grad_itrs % (self.update_frequency) == 0 and self.num_grad_itrs != 0:
                self.update_gradients()


        return statistics

    def print_feed_dict(self, key, value):
        if key == self.state_in and len(value.shape)==4:

            print("Input cars " + str(np.sum(value[0, :, :, 6])) + " people " + str(
                np.sum(value[0, :, :, 5])) + " Input cars traj" + str(
                np.sum(value[0, :, :, 4])) + " people traj " + str(
                np.sum(value[0, :, :, 3])))
            print("Input building " + str(np.sum(value[0, :, :, 7])) + " fence " + str(
                np.sum(value[0, :, :, 7 + 1])) + " static " + str(np.sum(value[0, :, :, 7 + 2])) + " pole " + str(
                np.sum(value[0, :, :, 7 + 3])))
            print("Input sidewalk " + str(np.sum(value[0, :, :, 7 + 5])) + " road " + str(
                np.sum(value[0, :, :, 7 + 4])) + " veg. " + str(
                np.sum(value[0, :, :, 7 + 6])) + " wall " + str(np.sum(value[0, :, :, 7 + 7])) + " sign " + str(
                np.sum(value[0, :, :, 7 + 8])))
            # print "Cars: "
            # print value[0,:, :, 6]
            # print "Cars traj : "
            # print value[0,:, :, 4]
            # print "People:"
            # print value[0,:, :, 5]
            # print "People traj : "
            # print value[0,:, :, 3]

        elif self.car_net==False and self.settings.pfnn and key == self.pose:
            print("Pose " + str(value[0, :5]) + " value " + str(np.sum(np.abs(value))))
        else:
            print(key)
            print(value)

    def evaluate(self,id,  ep_itr, statistics, episode, poses,priors,statistics_car, seq_len=-1):
        if seq_len==-1:
            seq_len=self.settings.seq_len_test-1
        #print "Evaluate"
        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_speed = self.stats(id, statistics,statistics_car)

        reward = agent_reward_d#np.zeros_like(agent_reward)
        #for ep_itr in range(statistics.shape[0]):
        for frame in range(seq_len ):
            feed_dict = self.grad_feed_dict(id, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                            episode, frame, reward, statistics,poses,priors, training=True, agent_speed=agent_speed,statistics_car=statistics_car)
            if len(feed_dict) > 0:
                loss, summaries_loss = self.sess.run([self.loss, self.loss_summaries],feed_dict)  # self.merged
                #episode.loss[frame] = loss
                if self.car_net:

                    statistics_car[ep_itr,id, frame, STATISTICS_INDX_CAR.loss] = loss
                    # elif self.init_net:
                    #     statistics[ep_itr, frame, STATISTICS_INDX.loss] = loss
                else:
                    statistics[ep_itr,id, frame, STATISTICS_INDX.loss]=loss
                if self.writer:
                    self.test_writer.add_summary(summaries_loss, global_step=self.num_grad_itrs)
        return statistics


    def normalize_reward(self, agent_reward_d, agent_measures, seq_len=-1):
        if seq_len==-1:
            seq_len=self.settings.seq_len_train
        reward = np.zeros_like(agent_reward_d)

        mask=np.full(agent_reward_d.shape, False, dtype=bool)

        hit_by_car_indx=PEDESTRIAN_MEASURES_INDX.hit_by_car
        hit_by_pedestrian_indx=PEDESTRIAN_MEASURES_INDX.hit_pedestrians
        reached_goal_indx=PEDESTRIAN_MEASURES_INDX.goal_reached
        if self.car_net:
            hit_by_car_indx = CAR_MEASURES_INDX.hit_by_car
            hit_by_pedestrian_indx = CAR_MEASURES_INDX.hit_pedestrians
            reached_goal_indx = CAR_MEASURES_INDX.goal_reached

        for ep in range(agent_reward_d.shape[0]):
            if sum(agent_measures[ep,:,hit_by_car_indx])>0:
                i=0
                while i < len(agent_measures[ep,:,hit_by_car_indx]) and agent_measures[ep,i,hit_by_car_indx]==0:
                    i=i+1

                for p in range(i,agent_reward_d.shape[1]):
                    mask[ep,p]=True
        if self.settings.end_on_bit_by_pedestrians or self.car_net:
            for ep in range(agent_reward_d.shape[0]):
                if sum(agent_measures[ep, :, hit_by_pedestrian_indx]) > 0:
                    i = 0
                    while i < len(agent_measures[ep, :, hit_by_pedestrian_indx]) and agent_measures[ep, i, hit_by_pedestrian_indx] == 0:
                        i = i + 1
                    for p in range(i, agent_reward_d.shape[1]):
                        mask[ep, p] = True
        if self.settings.stop_on_goal or self.car_net:
            for ep in range(agent_reward_d.shape[0]):
                if sum(agent_measures[ep, :, reached_goal_indx]) > 0:
                    i = 0
                    while i < len(agent_measures[ep, :, reached_goal_indx]) and agent_measures[ep, i, reached_goal_indx] == 0:
                        i = i + 1
                    for p in range(i,agent_reward_d.shape[1]):
                        mask[ep,p]=True

        masked_array=np.ma.array(agent_reward_d, mask=mask)
        mean_reward =masked_array.mean(axis=1)

        std_reward = np.std(agent_reward_d, axis=1)+np.finfo(np.float32).eps
        std_reward[np.abs(std_reward)<=np.finfo(np.float32).eps] = 1
        #print(" Before normalizing rewards: mean: " + str(np.mean(agent_reward_d, axis=1))+" std "+ str(np.std(agent_reward_d, axis=1)))
        for ep in range(agent_reward_d.shape[0]):
            for frame in range(0, seq_len - 1):
                if not mask[ep, frame]:
                    #reward[:, frame] = (agent_reward_d[:, frame] - mean_reward[frame])/ std_reward[frame]
                    reward[ep, frame] = (agent_reward_d[ep, frame] - mean_reward[ep]) / std_reward[ep]
        #print(" Before normalizing rewards: mean: " + str(np.mean(reward, axis=1))+" std "+ str(np.std(reward, axis=1)))
        return reward

    def stats(self,id,  statistics,statistics_car, initializer=False):
        if len(statistics_car)>0:
            agent_pos = np.copy(statistics[:, :, :, STATISTICS_INDX_CAR.agent_pos[0]:STATISTICS_INDX_CAR.agent_pos[1]])
            agent_velocity = np.copy(statistics[:, :, :, STATISTICS_INDX_CAR.velocity[0]:STATISTICS_INDX_CAR.velocity[1]])
            agent_action = np.copy(statistics[:, :, :, STATISTICS_INDX_CAR.action])
            agent_probabilities = np.copy( statistics[:, :, :, STATISTICS_INDX_CAR.probabilities[0]:STATISTICS_INDX_CAR.probabilities[1]])
            agent_loss = np.copy(statistics[:, :, :, STATISTICS_INDX.loss])
            agent_speed = np.copy(statistics[:, :, :, STATISTICS_INDX.speed])
            if initializer:
                agent_reward = np.copy(statistics_car[:, id, :, STATISTICS_INDX_CAR.reward_initializer])
                agent_reward_d = np.copy(statistics_car[:, id, :, STATISTICS_INDX_CAR.reward_initializer_d])
            else:
                agent_reward = np.copy(statistics_car[:,id,  :, STATISTICS_INDX_CAR.reward])
                agent_reward_d = np.copy(statistics_car[:,id,  :, STATISTICS_INDX_CAR.reward_d])
            agent_measures = np.copy(statistics_car[:,id,  :, STATISTICS_INDX_CAR.measures[0]:STATISTICS_INDX_CAR.measures[1]])
        else:
            agent_pos = np.copy(statistics[:, id, :, STATISTICS_INDX.agent_pos[0]:STATISTICS_INDX.agent_pos[1]])
            agent_velocity = np.copy(statistics[:, id, :, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]])
            agent_action = np.copy(statistics[:, id, :, STATISTICS_INDX.action])
            agent_probabilities = np.copy(
                statistics[:, id, :, STATISTICS_INDX.probabilities[0]:STATISTICS_INDX.probabilities[1]])
            if initializer:
                agent_reward = np.copy(statistics[:, id, :, STATISTICS_INDX.reward_initializer])
                agent_reward_d = np.copy(statistics[:, id, :, STATISTICS_INDX.reward_initializer_d])
            else:
                agent_reward = np.copy(statistics[:,id,  :, STATISTICS_INDX.reward])
                agent_reward_d = np.copy(statistics[:,id,  :, STATISTICS_INDX.reward_d])
            agent_measures = np.copy(statistics[:,id,  :, STATISTICS_INDX.measures[0]:STATISTICS_INDX.measures[1]])

            agent_loss = np.copy(statistics[:,id,  :, STATISTICS_INDX.loss])
            agent_speed=np.copy(statistics[:,id,  :, STATISTICS_INDX.speed])


        return agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_speed

    def update_gradients(self):


        # if self.settings.clip_gradients:
        #     for idx, gradient in enumerate(self.gradBuffer):
        #         self.gradBuffer[idx] = np.clip(self.gradBuffer[idx], -50, 50)

        if self.log:
            self.log.write(datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + " Updating gradients")
        for idx, gradient in enumerate(self.gradBuffer):
            self.gradBuffer[idx] = gradient * (1/self.grad_counter)
            if idx==0:
                print("Normalize gradient "+str(np.sum(np.abs(gradient))) +" by "+str(1/self.grad_counter)+" "+str(np.sum(np.abs(self.gradBuffer[idx])))+ " inverted: "+str(self.grad_counter))
        grad_dict = dict(list(zip(self.gradient_holders, self.gradBuffer)))

        if self.car_net:
            # print ("Update gradient car net")
            # for holder, gradient in enumerate(grad_dict):
            #     print ("Update gradient from dict : " + str(holder)+ "  indx "+str(np.sum(grad_dict[gradient]))+" shape "+str(grad_dict[gradient].shape))
            _ = self.sess.run([self.settings.update_batch_car], feed_dict=grad_dict)
        elif self.init_net_car:
            print("Update gradient init net")
            # for holder, gradient in enumerate(grad_dict):
            #     print ("Update gradient from dict : " + str(holder) + "  indx " + str(
            #         np.sum(grad_dict[gradient])) + " shape " + str(grad_dict[gradient].shape))
            _ = self.sess.run([self.settings.update_batch_init_car], feed_dict=grad_dict)
        elif self.init_net:
            print ("Update gradient init net")
            # for holder, gradient in enumerate(grad_dict):
            #     print ("Update gradient from dict : " + str(holder) + "  indx " + str(
            #         np.sum(grad_dict[gradient])) + " shape " + str(grad_dict[gradient].shape))
            _ = self.sess.run([self.settings.update_batch_init], feed_dict=grad_dict)
        elif self.goal_net:
            print ("Update gradient goal net")
            # for holder, gradient in enumerate(grad_dict):
            #     print ("Update gradient from dict : " + str(holder) + "  indx " + str(
            #         np.sum(grad_dict[gradient])) + " shape " + str(grad_dict[gradient].shape))
            _ = self.sess.run([self.settings.update_batch_goal], feed_dict=grad_dict)
        else:
            print("Update gradient----------------------------------------------------")
            _= self.sess.run([self.settings.update_batch], feed_dict=grad_dict)

        for idx, gradient in enumerate(self.gradBuffer):
            self.gradBuffer[idx] = gradient * 0
        if self.writer:
            self.writer.flush()
        if not self.settings.temporal:
            self.traj_forget_rate = self.traj_forget_rate * self.settings.people_traj_gamma
        if self.traj_forget_rate< self.settings.people_traj_tresh:
            self.traj_forget_rate=0
        self.grad_counter=0


    def reset_mem(self):
        pass

    def grad_feed_dict(self, id, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode, frame,
                       reward, statistics,poses,priors, agent_speed=None, training=True, agent_frame=-1, statistics_car=None):
        if agent_frame < 0 or not training:
            agent_frame = frame


        r=reward[ep_itr, agent_frame]
        # print  ("Reward "+str(r))

        if agent_measures[ep_itr,agent_frame,PEDESTRIAN_MEASURES_INDX.agent_dead]:

            return {}

        feed_dict = {self.state_in: self.get_input(id, episode, agent_pos[ep_itr, agent_frame, :], frame_in=frame,training=training),
                     self.advantages: r,
                     self.sample: self.get_sample(id, statistics, ep_itr, agent_frame)}#, priors)}

        if self.writer:
            feed_dict+={
                self.num_cars: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.hit_by_car].copy(),
                self.num_people: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory].copy(),
                self.pavement: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.iou_pavement].copy(),
                self.num_of_obj: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles].copy(),
                self.dist_travelled: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init].copy(),
                self.out_of_axis: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.out_of_axis].copy(),
                self.tot_reward: np.sum(statistics[ep_itr,id,  :agent_frame, STATISTICS_INDX.reward]) #  self.pose: poses,

            }
        self.get_feature_vectors_gradient(id, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, agent_velocity, ep_itr,
                                          episode, feed_dict, frame, poses, statistics, training , statistics_car)
        return feed_dict



    def get_input(self, id, episode, agent_pos_cur, frame_in=-1, training=True):
        raise NotImplementedError("Please Implement this method")

    def define_update_gradients(self,  optimizer):
        if self.car_net:
            self.settings.update_batch_car = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))
        elif self.init_net_car:
            # print ("Define init update batch , holders:")
            # print (str(self.gradient_holders))
            # print ("Trainable variables list")
            # print (str(self.tvars))
            self.settings.update_batch_init_car = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))
        elif self.init_net:
            # print ("Define init update batch , holders:")
            # print (str(self.gradient_holders))
            # print ("Trainable variables list")
            # print (str(self.tvars))
            self.settings.update_batch_init = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))
        elif self.goal_net:
            # print ("Define init update batch , holders:")
            # print (str(self.gradient_holders))
            # print ("Trainable variables list")
            # print (str(self.tvars))
            self.settings.update_batch_goal = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))
        else:
            if self.settings.batch_normalize:
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.settings.update_batch = optimizer.apply_gradients(list(zip( self.gradient_holders, self.tvars)))
            else:
                self.settings.update_batch = optimizer.apply_gradients(list(zip(self.gradient_holders, self.tvars)))

    def merge_summaries(self):
        self.train_summaries.extend(self.conv_summaries)
        self.train_summaries = tf.compat.v1.summary.merge(self.train_summaries)
        self.loss_summaries = tf.compat.v1.summary.merge(self.loss_summaries)
        self.grad_summaries = tf.compat.v1.summary.merge(self.grad_summaries)


    def loss_setup(self, dim_p):
        self.sample, self.loss=self.define_loss(dim_p)
        if self.writer:
            self.loss_summaries.append(tf.compat.v1.summary.scalar("calculated_loss", self.loss))
            self.loss_summaries.append(tf.compat.v1.summary.scalar("sample", self.sample))



    def define_loss(self, dim_p):
        raise NotImplementedError("Please Implement this method")

    def get_feature_vectors_gradient(self, id, agent_action, agent_frame, agent_measures, agent_pos, agent_speed,agent_velocity, ep_itr,
                                     episode, feed_dict, frame, poses, statistics, training, statistics_car=[]):
        raise NotImplementedError("Please Implement this method")

    def fully_connected(self, dim_p, prev_layer):
        raise NotImplementedError("Please Implement this method")

    #return [statistics[ep_itr, agent_frame, 6]]
    def get_sample(self,id,  statistics,ep_itr, agent_frame):
        raise NotImplementedError("Please Implement this method")


    def calc_probabilities(self, fc_size):
        raise NotImplementedError("Please Implement this method")


    def fully_connected_size(self, dim_p):
        raise NotImplementedError("Please Implement this method")

    def get_dim_p(self):
        return 3

    def get_velocity(self, velocity, action, frame):
        raise NotImplementedError("Please Implement this method")


