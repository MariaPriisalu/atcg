
#  TODO: CPaduraru: TEMPORARY GET RID OF THIS !!!
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Continue impl by following: https://www.tensorflow.org/guide/effective_tf2 and https://www.tensorflow.org/guide/migrate guides !!
# Rendering options for no-x: https://carla.readthedocs.io/en/latest/adv_rendering_options/

import os
if True:
    import sys

    sys.path.insert(0,os.path.join(os.path.dirname( os.path.abspath(__file__)), '..'))
    import matplotlib as mpl
    print(('current backend: ', mpl.get_backend()))
    backend = 'agg' #'cairo'  # 'agg'
    mpl.use(backend, force=True)
    print(('current backend: ', mpl.get_backend()))
else:
    import sys
    sys.path.insert(0,os.path.join(os.path.dirname( os.path.abspath(__file__)), '..'))

import tensorflow as tf
import math
from settings import RANDOM_SEED

tf.random.set_seed(RANDOM_SEED)
from tensorflow.python import pywrap_tensorflow
from agents_goal import GoalAgent, GoalCVAgent
from environment_test import TestEnvironment
from carla_environment import CARLAEnvironment
from environment import Environment
from agent_net import NetAgent, AgentNetPFNN, ContinousNetAgent, ContinousNetPFNNAgent,RandomAgentPFNN
from agents_dummy import RandomAgent, PedestrianAgent
from agent_car import CarAgent
from net_sem_2d import Seg_2d,  Seg_2d_min_softmax

from net_simple_car import SimpleCarNet

from goal_net import GoalNet
from initializer_net import InitializerNet
import glob
import os
import numpy as np
np. set_printoptions(suppress=True)
import scipy.io as sio
from settings import run_settings, CARLA_CACHE_PREFIX_EVALUATE, CARLA_CACHE_PREFIX_EVALUATE_SUPERVISED,  \
    CARLA_CACHE_PREFIX_TRAIN, CARLA_CACHE_PREFIX_TRAIN_SUPERVISED, \
    CARLA_CACHE_PREFIX_TEST, CARLA_CACHE_PREFIX_TEST_SUPERVISED, \
    CARLA_CACHE_PREFIX_TEST_TOY, CARLA_CACHE_PREFIX_EVALUATE_TOY, CARLA_CACHE_PREFIX_TRAIN_TOY,\
    CARLA_CACHE_PREFIX_TEST_NEW, CARLA_CACHE_PREFIX_EVALUATE_NEW, CARLA_CACHE_PREFIX_TRAIN_NEW, CAR_REWARD_INDX, \
    CARLA_CACHE_PREFIX_EVALUATE_REALTIME, CARLA_CACHE_PREFIX_TRAIN_REALTIME, CARLA_CACHE_PREFIX_TEST_REALTIME, \
    CARLA_CACHE_PREFIX_EVALUATE_ONLINE,CARLA_CACHE_PREFIX_TRAIN_ONLINE, CARLA_CACHE_PREFIX_TEST_ONLINE, RANDOM_SEED_NP, \
    RANDOM_SEED,CARLA_CACHE_PREFIX_TRAIN_REALTIME_NO_EXTR_CARS_OR_PEDS,CARLA_CACHE_PREFIX_TRAIN_NO_EXTR_CARS_OR_PEDS, \
    CARLA_CACHE_PREFIX_TEST_REALTIME_NO_EXTR_CARS_OR_PEDS,CARLA_CACHE_PREFIX_TEST_NO_EXTR_CARS_OR_PEDS,\
    CARLA_CACHE_PREFIX_EVALUATE_REALTIME_NO_EXTR_CARS_OR_PEDS,CARLA_CACHE_PREFIX_EVALUATE_NO_EXTR_CARS_OR_PEDS

def get_carla_prefix_eval(supervised, useToyEnv, new_carla, realtime_carla,useOnlineEnv, no_external):
    if no_external:
        if realtime_carla:
            return CARLA_CACHE_PREFIX_EVALUATE_NO_EXTR_CARS_OR_PEDS
        else:
            return CARLA_CACHE_PREFIX_EVALUATE_REALTIME_NO_EXTR_CARS_OR_PEDS
    if useOnlineEnv:
        return CARLA_CACHE_PREFIX_EVALUATE_ONLINE
    if realtime_carla:
        return CARLA_CACHE_PREFIX_EVALUATE_REALTIME
    if new_carla:
        return CARLA_CACHE_PREFIX_EVALUATE_NEW
    if useToyEnv:
        return CARLA_CACHE_PREFIX_EVALUATE_TOY
    return (CARLA_CACHE_PREFIX_EVALUATE if not supervised else CARLA_CACHE_PREFIX_EVALUATE_SUPERVISED)


def get_carla_prefix_train(supervised, useToyEnv, new_carla, realtime_carla, useOnlineEnv, no_external):
    if no_external:
        if realtime_carla:
            return CARLA_CACHE_PREFIX_TRAIN_NO_EXTR_CARS_OR_PEDS
        else:
            return CARLA_CACHE_PREFIX_TRAIN_REALTIME_NO_EXTR_CARS_OR_PEDS
    if useOnlineEnv:
        return CARLA_CACHE_PREFIX_TRAIN_ONLINE
    if realtime_carla:
        return CARLA_CACHE_PREFIX_TRAIN_REALTIME
    if new_carla:
        return CARLA_CACHE_PREFIX_TRAIN_NEW
    if useToyEnv:
        return CARLA_CACHE_PREFIX_TRAIN_TOY
    return (CARLA_CACHE_PREFIX_TRAIN if not supervised else CARLA_CACHE_PREFIX_TRAIN_SUPERVISED)


def get_carla_prefix_test( supervised, useToyEnv, new_carla, realtime_carla,useOnlineEnv, no_external):
    if no_external:
        if realtime_carla:
            return CARLA_CACHE_PREFIX_TEST_NO_EXTR_CARS_OR_PEDS
        else:
            return CARLA_CACHE_PREFIX_TEST_REALTIME_NO_EXTR_CARS_OR_PEDS
    if useOnlineEnv:
        return CARLA_CACHE_PREFIX_TEST_ONLINE
    if realtime_carla:
        return CARLA_CACHE_PREFIX_TEST_REALTIME
    if new_carla:
        return CARLA_CACHE_PREFIX_TEST_NEW
    if useToyEnv:
        return CARLA_CACHE_PREFIX_TEST_TOY
    return (CARLA_CACHE_PREFIX_TEST if not supervised else CARLA_CACHE_PREFIX_TEST_SUPERVISED)

from dotmap import DotMap
import time

import logging


##
# Main script- train an RL agent on real or toy case.
#

import sys
import traceback

class TracePrints(object):
  def __init__(self):
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)




class RL(object):
    def __init__(self):
        # Load H3.6M database of valid poses.
        # Log path
        self.save_model_path="localUserData/Models/rl_model/"

        self.test_counter=-1
        self.car_test_counter=0
        self.real_counter=0
        self.save_counter=0

        self.log_path =  "localUserData/Results/agent/"
        self.save_model_path="localUserData/Results/agent/"
        self.poses_db=None


    def eraseCachedEpisodes(self, folderPath):
        import os, shutil
        for filename in os.listdir(folderPath):
            file_path = os.path.join(folderPath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(('Failed to delete %s. Reason: %s' % (file_path, e)))

    # Tensorflow session.
    def run(self,  settings, time_file=None, supervised=False):
        #print("Supervised? "+supervised)
        # Check if episodes cached data should be deleted
        if settings.deleteCacheAtEachRun:
            self.eraseCachedEpisodes(settings.target_episodesCache_Path)
            print("I have deleted the cache folder !")

        counter, filecounter, images_path, init, net, saver,  tmp_net, init_net,goal_net,car_net, init_car_net = self.pre_run_setup(settings)
        tf.random.set_seed(RANDOM_SEED)
        #from tensorflow.python import debug as tf_debug
        config = tf.compat.v1.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config) as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            tf.random.set_seed(RANDOM_SEED)
            sess.run(init)



            grad_buffer, grad_buffer_init,grad_buffer_goal,grad_buffer_car, grad_buffer_init_car = self.create_gradient_buffer(sess)

            print((settings.load_weights))
            if "model.ckpt" in settings.load_weights:
                self.load_net_weights(saver, sess, settings, latest=False)#, exact=True)
            else:
                self.load_net_weights(saver, sess, settings, latest=True)
            self.create_setting_folder(settings)

            sess.graph.finalize()
            trainableCars, trainablePedestrians = self.get_cars_and_agents(net,car_net, init_net, goal_net, init_car_net, grad_buffer, grad_buffer_car, grad_buffer_init,grad_buffer_goal,grad_buffer_init_car,settings)


            if not settings.cv_agent:
                net.set_session(sess)
            if settings.learn_init:
                init_net.set_session(sess)
            if settings.separate_goal_net:
                goal_net.set_session(sess)
            if settings.useRLToyCar:
                car_net.set_session(sess)
            if settings.learn_init_car:
                init_car_net.set_session(sess)

            writer=None
            log_file=None

            self.start_run(trainablePedestrians, trainableCars, counter, filecounter, grad_buffer, images_path, log_file, saver, sess, settings,
                           supervised, tmp_net, writer)

    def  get_cars_and_agents(self, net,car_net, init_net, goal_net, init_car_net, grad_buffer, grad_buffer_car, grad_buffer_init,grad_buffer_goal,grad_buffer_init_car,settings):
        trainablePedestrians = []
        trainableCars = []
        for i in range(settings.number_of_agents):
            trainablePedestrians.append(
                self.get_agent(grad_buffer, net, grad_buffer_init, init_net, grad_buffer_goal, goal_net, settings))
            trainablePedestrians[-1].id = i
        if settings.useRLToyCar:
            for i in range(settings.number_of_car_agents):
                trainableCars.append(CarAgent(settings, car_net, grad_buffer_car, init_net=init_car_net, init_grad_buffer=grad_buffer_init_car))
                trainableCars[-1].id = i

        else:
            trainableCars = [None]
        return trainableCars, trainablePedestrians

    def start_run(self, trainablePedestrians, trainableCars, counter, filecounter, grad_buffer, images_path, log_file, saver, sess, settings,
                  supervised, tmp_net, writer):
        saved_files_counter = 0
        # print(("Before run seq len " + str(settings.seq_len_train) + " " + str(settings.seq_len_train_final)))
        if settings.carla:

            counter = self.train_on_carla(trainablePedestrians, counter, filecounter, grad_buffer, images_path,
                                          log_file, saved_files_counter, saver, sess, settings, writer, supervised, trainableCars)
        elif settings.toy_case:

            counter = self.train_on_test_set(trainablePedestrians, counter, grad_buffer, images_path, log_file, saver,
                                             sess, settings, tmp_net, writer)
        elif settings.temp_case:

            counter = self.train_on_temp_case(trainablePedestrians, counter, filecounter, grad_buffer, images_path,
                                              log_file, saved_files_counter, saver, sess, settings, writer)
        saver.save(sess,
                   self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                   global_step=counter)

    def train_on_temp_case(self, trainablePedestrians,counter, filecounter, grad_buffer, images_path,
                                              log_file, saved_files_counter, saver, sess, settings, writer):
        if settings.old:
            settings.run_2D = False
        epoch, filename_list, pos = self.get_carla_files(settings)
        train_set, val_set, test_set = self.get_train_test_fileIndices(filename_list)
        num_epochs = 20  # 10  # 23
        if settings.overfit:
            train_set = [0, 4, 8]
            val_set = [100, 104]  # [100]
            num_epochs = 200

        filepath = filename_list[0]

        # print "Hej "
        env = self.get_environments_CARLA(grad_buffer, images_path, log_file,sess, settings, writer)

        num_epochs = 500  # 10  # 23

        for epoch in range(0, num_epochs):
            # print "Epoch "+str(epoch)
            self.adapt_variables(epoch, settings)

            if epoch % 100 == 0:

                self.test_counter = self.test_counter + 1

                saver.save(sess,
                           self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                           global_step=epoch)
                print(("Save model : " + self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt-" + str(
                    epoch)))
                filecounter += 1
                print(("filecounter " + str(filecounter)))

            self.test_counter = self.test_counter + 1
            saver.save(sess, self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                       global_step=epoch)

            return counter





    def train_on_test_set(self, trainablePedestrians, counter, grad_buffer, images_path, log_file, saver, sess, settings, tmp_net, writer):
        env = TestEnvironment(images_path, sess, writer, grad_buffer, log_file, settings, net=tmp_net)

        counter = env.work("", trainablePedestrians, self.poses_db)
        saver.save(sess, self.save_model_path + settings.name_movie, global_step=counter)
        return counter

    def train_on_carla(self, trainablePedestrians,  counter, filecounter, grad_buffer, images_path, log_file,
                       saved_files_counter, saver, sess, settings, writer, supervised, trainableCars):
        print("CARLA ")
        if settings.old:
            settings.run_2D=False
        env= self.get_environments_CARLA( grad_buffer, images_path, log_file,
                                                               sess, settings, writer)
        epoch, filename_list, pos = self.get_carla_files(settings, False)
        train_set, val_set, test_set = self.get_train_test_fileIndices(filename_list, carla=True, new_carla=settings.new_carla, realtime_carla=settings.realtime_carla_only)

        num_epochs = 20 #10#10  # 23

        num_epochs, train_set, val_set = self.adjust_training_set(num_epochs, settings, train_set, val_set)
        if settings.realtime_carla:
            epoch_rt, filename_list_realtime, pos_rt = self.get_carla_files(settings, settings.realtime_carla)


        if settings.overfit:
            save_stats=False
        else:
            save_stats=True


        car_epoch=0
        init_epoch=0
        train_car=settings.useHeroCar
        train_initializer=settings.learn_init or settings.learn_goal
        success_rate_car=0
        collision_rate_initializer=0
        if len(settings.train_car_and_initialize)>0 and (settings.learn_goal or  settings.learn_init or settings.useRLToyCar) and settings.useHeroCar:
            iterative_training=DotMap()
        else:
            iterative_training=None


        print (" Initially: iterative training "+str(iterative_training)+" train_car "+str(train_car)+" train initializer "+str(train_initializer))
        for epoch in range(0, num_epochs):
            print(("$$$$$$$$$$ Epoch "+str(epoch)))

            if settings.overfit and num_epochs%settings.save_frequency==0:
                save_stats=True

                

            if  iterative_training!=None:
                print ("Switch training ")
                train_car, train_initializer, car_epoch, init_epoch=self.switch_training(epoch, car_epoch, init_epoch, settings, train_car, train_initializer, success_rate_car, collision_rate_initializer)
                iterative_training.train_car=train_car
                iterative_training.train_initializer = train_initializer
                print("Done switch training train car:"+str(iterative_training.train_car)+"  train initializer "+str(iterative_training.train_initializer))


            for pos in sorted(train_set):
                print(("-------- Epoch {0}, train environment index {1}/{2}".format(epoch, pos, len(train_set))))
                self.adapt_variables(epoch, settings, success_rate_car)
                filepath = filename_list[pos]

                try:
                    prefix=get_carla_prefix_train(supervised,  settings.useRLToyCar or settings.useHeroCar , settings.new_carla, settings.realtime_carla_only, settings.realTimeEnvOnline, settings.ignore_external_cars_and_pedestrians)
                    counter, saved_files_counter,initializer_stats = env.work(prefix, filepath, trainablePedestrians, self.poses_db, epoch, saved_files_counter, trainableCars=trainableCars,training=True, save_stats=save_stats, iterative_training=iterative_training, useRealTimeEnv=settings.useRealTimeEnv, realtime_carla=settings.realtime_carla_only)

                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    logging.exception('Fatal error in main loop ' + settings.timestamp)
                    print("Unexpected error:", sys.exc_info()[0])
                    traceback.print_exception(*sys.exc_info())
                    if settings.stop_for_errors:
                        raise



                car = trainableCars


                if settings.realtime_carla  and filecounter > settings.realtime_freq * (self.real_counter + 1):
                    for pos_rt in range(4):
                        filepath = filename_list_realtime[pos_rt]
                        print (" Filepath "+str(filepath))
                        try:
                            prefix=get_carla_prefix_train(supervised, settings.useRLToyCar or settings.useHeroCar, settings.new_carla,
                                                       settings.realtime_carla,settings.realTimeEnvOnline, settings.ignore_external_cars_and_pedestrians)
                            counter, saved_files_counter, initializer_stats = env.work(prefix
                                , filepath, trainablePedestrians, self.poses_db, epoch,
                                saved_files_counter, trainableCars=trainableCars, training=True, save_stats=save_stats,
                                iterative_training=iterative_training, useRealTimeEnv=settings.useRealTimeEnv, realtime_carla=True)

                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except:
                            logging.exception('Fatal error in main loop ' + settings.timestamp)
                            print("Unexpected error:", sys.exc_info()[0])
                            traceback.print_exception(*sys.exc_info())
                            if settings.stop_for_errors:
                                raise  # Always raise exception otherwise it will be masked and apperantly it would continue...
                    self.real_counter = self.real_counter + 1



                if filecounter > settings.test_freq * (self.test_counter + 1):

                    success_rates_car = []
                    collision_rates_initializer = []
                    for test_index, test_pos in enumerate(val_set):
                        if test_pos % settings.VAL_JUMP_STEP_CARLA == 0:

                            filepath = filename_list[test_pos]

                            try:
                                prefix=get_carla_prefix_test(supervised,  settings.useRLToyCar or settings.useHeroCar, settings.new_carla, settings.realtime_carla_only,settings.realTimeEnvOnline, settings.ignore_external_cars_and_pedestrians)
                                _, saved_files_counter,initializer_stats = env.work(prefix, filepath, trainablePedestrians, self.poses_db, epoch,
                                                                  saved_files_counter,trainableCars=trainableCars, training=False, useRealTimeEnv=settings.useRealTimeEnv, realtime_carla=settings.realtime_carla_only)

                                if initializer_stats:
                                    print (" From initializers stats get success rate "+str(initializer_stats.success_rate_car)+" collision rate "+str(initializer_stats.collision_rate_initializer))
                                    success_rates_car.append(initializer_stats.success_rate_car)
                                    collision_rates_initializer.append(initializer_stats.collision_rate_initializer)
                            except (KeyboardInterrupt, SystemExit):
                                raise
                            except:
                                logging.exception('Fatal error in main loop ' + settings.timestamp)
                                print("Exception")

                                if settings.stop_for_errors:
                                    raise
                    if settings.realtime_carla:
                        for pos_rt in range(4,6):
                            print(" Filepath " + str(filepath))
                            filepath = filename_list_realtime[pos_rt]

                            try:
                                prefix= get_carla_prefix_test(supervised, settings.useRLToyCar or settings.useHeroCar, settings.new_carla,
                                                          settings.realtime_carla,settings.realTimeEnvOnline, settings.ignore_external_cars_and_pedestrians)
                                _, saved_files_counter, initializer_stats = env.work(prefix,
                                    filepath, trainablePedestrians, self.poses_db,epoch,saved_files_counter, trainableCars=trainableCars,
                                    training=False, useRealTimeEnv=settings.useRealTimeEnv , realtime_carla=True)
                            except (KeyboardInterrupt, SystemExit):
                                raise
                            except:
                                logging.exception('Fatal error in main loop ' + settings.timestamp)
                                print("Unexpected error:", sys.exc_info()[0])
                                traceback.print_exception(*sys.exc_info())
                                if settings.stop_for_errors:
                                    raise  # Always raise exception otherwise it will be masked and apperantly it would continue...
                    success_rate_car=np.mean(success_rates_car)
                    collision_rate_initializer=np.mean(collision_rates_initializer)

                    print(" Update success rate "+str(success_rate_car)+" Update collision rate "+str(collision_rate_initializer))



                    self.test_counter = self.test_counter + 1
                    if not settings.overfit or epoch%100==0:
                        saver.save(sess,
                                   self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                                   global_step=epoch)
                    print(("Save model : " + self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt-" + str(
                        epoch)))
                    self.save_counter = self.save_counter + 1
                filecounter += 1
                print(("filecounter " + str(filecounter)))

        print ("Final testing ")
        for test_pos in val_set:
            if test_pos % settings.VAL_JUMP_STEP_CARLA == 0:
                if test_pos in filename_list:
                    filepath = filename_list[test_pos]
                else:
                    filepath = filename_list[test_index]
                    print(f"WARNING ! I was requesting dataset key {test_pos} but i will failsafe to item {test_index}")

                try:
                    prefix=get_carla_prefix_test(supervised,  settings.useRLToyCar or settings.useHeroCar, settings.new_carla,settings.realtime_carla_only,settings.realTimeEnvOnline, settings.ignore_external_cars_and_pedestrians)
                    _, saved_files_counter,initializer_stats = env.work(prefix, filepath, trainablePedestrians, self.poses_db, epoch,
                                                      saved_files_counter,trainableCars=trainableCars, training=False, useRealTimeEnv=settings.useRealTimeEnv)

                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    logging.exception('Fatal error in main loop ' + settings.timestamp)
                    print("Exception")
                    if settings.stop_for_errors:
                        raise

        saver.save(sess, self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                   global_step=epoch)

        self.test_counter = self.test_counter + 1

        if not settings.overfit:
            self.evaluate_method_carla(trainablePedestrians,  sess, settings, trainableCars=trainableCars)
        return counter

    def adjust_training_set(self, num_epochs, settings, train_set, val_set):
        if settings.fastTrainEvalDebug is True:
            num_epochs = 5 #2
            if settings.new_carla:
                return num_epochs, train_set, val_set
            train_set = [1, 0, 2]
            if settings.useRealTimeEnv:
                val_set = [3, 4]
            else:
                val_set = [100]#, 101]
        if settings.overfit:
            print("Overfit! ")
            num_epochs = 300
            if settings.new_carla:
                return num_epochs, train_set, val_set
            if settings.realtime_carla:
                train_set = [ 1, 0]
                val_set = []  # [100]
                return num_epochs, train_set, val_set
            train_set = [0, 1, 2]
            val_set = [101]  # [100]

            if settings.learn_init or settings.useHeroCar or settings.useRLToyCar:
                train_set = [0, 1, 2, 19]
                val_set = [104, 101]  # [100]
                num_epochs = 300#300
                if len(settings.train_car_and_initialize)>0:
                    num_epochs = 1000  # 300

        return num_epochs, train_set, val_set



    def switch_training(self, epoch, car_epoch, init_epoch, settings, train_car, train_initializer, success_rate_car, collision_rate_initializer):
        print (" In switch trainig ")
        if "simultaneously" in settings.train_car_and_initialize :
            print(" Simultaneous")
            train_car=True
            train_initializer=True
            car_epoch=car_epoch+1
            init_epoch=init_epoch+1
        elif "alternatively" in settings.train_car_and_initialize:
            print(" Alternative")
            if car_epoch< (epoch//(settings.num_car_epochs+settings.num_init_epochs)+1)*settings.num_car_epochs:
                print("Set train car don't train initializer")
                train_car = True
                train_initializer = False
                car_epoch = car_epoch + 1
            elif init_epoch< (epoch//(settings.num_car_epochs+settings.num_init_epochs)+1)*settings.num_init_epochs:
                print("Set train initializer don't train car")
                train_car = False
                train_initializer = True
                init_epoch = init_epoch + 1
        elif "according_to_stats":
            print(" According to statistics")
            print ("Train car "+str(train_car)+" and reached success rate? "+str(success_rate_car>settings.car_success_rate_threshold)+" rate "+str(success_rate_car)+" threshold "+str(settings.car_success_rate_threshold))
            print ("Train initializer "+str(train_initializer)+" and reached success rate? " + str(collision_rate_initializer>settings.initializer_collision_rate_threshold) + " rate " + str(
                collision_rate_initializer) + " threshold " + str(settings.initializer_collision_rate_threshold))
            if settings.initializer_collision_rate_threshold<1.0 or settings.car_success_rate_threshold<1.0:
                if train_initializer and collision_rate_initializer>=settings.initializer_collision_rate_threshold :
                    train_car = True
                    train_initializer = False
                    car_epoch = car_epoch + 1
                    print("Set train initializer don't train car")
                elif train_car and success_rate_car>=settings.car_success_rate_threshold:
                    train_car = False
                    train_initializer = True
                    print("Set train car don't train car")
                    init_epoch = init_epoch + 1
                elif success_rate_car==0 and collision_rate_initializer==0:
                    print("Set train initializer do train car- init")
                    train_car = False
                    train_initializer = True
                    init_epoch = init_epoch + 1
            else:
                print("Not switching training")
                train_car = False
                train_initializer = True
                init_epoch = init_epoch + 1

        return train_car, train_initializer, car_epoch, init_epoch

    def adapt_variables(self, epoch, settings, success_rate_car):
        if settings.goal_dir and (settings.learn_goal or  settings.learn_init or settings.useHeroCar or settings.useRLToyCar):
            if epoch/2==0:
                settings.init_std=max(settings.init_std-0.016, 0.1)
                settings.goal_std = max(settings.goal_std - 0.016, 0.1)

        if settings.pfnn:
            if epoch / 2 == 0:  # 5

                settings.sigma_vel = max(settings.sigma_vel / 2.0, 0.1)

        if settings.overfit or settings.temp_case:
            if epoch%3==0:
                settings.sigma_vel=max(settings.sigma_vel-0.016, 0.1)#-0.005

        else:
            print(("Adapt variables epochs: "+str(epoch)+" seq len "+str(settings.seq_len_train)))
            settings.sigma_vel = max(settings.sigma_vel - 0.00533, 0.1)

        print("Sigma: "+str(settings.sigma_vel))


    def get_carla_files(self, settings, realtime=False):
        if settings.new_carla:
            ending = "*/*"
        else:
            ending = "test_*"
        if realtime:
            filespath = settings.carla_path_realtime
        else:
            filespath = settings.carla_path

        epoch = 0
        filename_local_list = {}
        # Get files to run on.
        print(filespath + ending)
        for filepath in glob.glob(filespath + ending):
            parts = os.path.basename(filepath).split('_')
            pos = int(parts[-1])
            if settings.new_carla:
                if pos not in filename_local_list:
                    filename_local_list[pos]=[]
                filename_local_list[pos].append(filepath)
            else:
                filename_local_list[pos] = filepath
            print (" Pos "+str(pos)+" path "+str(filepath)+" "+str(parts))
        return epoch, filename_local_list, pos

    def get_agent(self, grad_buffer, net, grad_buffer_init, init_net, grad_buffer_goal, goal_net, settings):
        if settings.random_agent:
            agent = RandomAgentPFNN(settings, net, grad_buffer, init_net, grad_buffer_init, goal_net, grad_buffer_goal)
            return agent
        if settings.cv_agent:
            return GoalCVAgent(settings, net, grad_buffer, init_net, grad_buffer_init,goal_net,grad_buffer_goal )
        if settings.carla_agent:
            return PedestrianAgent(settings)
        if settings.angular:
            if settings.pfnn:
                return ContinousNetPFNNAgent(settings, net, grad_buffer, init_net, grad_buffer_init)
            agent = ContinousNetAgent(settings, net, grad_buffer, init_net, grad_buffer_init)
            return agent
        if settings.pfnn:
            agent = AgentNetPFNN(settings, net, grad_buffer, init_net, grad_buffer_init,goal_net,grad_buffer_goal)
            return agent
        agent = NetAgent(settings, net, grad_buffer, init_net, grad_buffer_init,goal_net,grad_buffer_goal )
        return agent


    def get_environments_CARLA(self,  grad_buffer, images_path, log_file, sess, settings, writer):
        env = CARLAEnvironment(images_path,  sess, writer, grad_buffer, log_file, settings)
        return env

    def pre_run_setup(self, settings):
        car_net, goal_net, init, init_net, net, saver, tmp_net,init_car = self.get_models_and_tf_saver(settings)
        images_path = settings.colmap_path
        counter = 0
        car_counter = 0
        people_counter = 0
        filecounter = 0
        settings.model_path = self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt"
        settings.save_settings("")
        logging.basicConfig(filename=os.path.join(settings.path_settings_file, settings.timestamp + '.log'))
        return  counter, filecounter, images_path, init, net, saver, tmp_net, init_net, goal_net, car_net, init_car

    def get_models_and_tf_saver(self, settings):
        net, tmp_net, init_net, init, goal_net, car_net, init_car= self.get_model(
            settings)  # Initialize various constants for 3D reconstruction.
        # Tensorboard saver.
        saver = tf.compat.v1.train.Saver(max_to_keep=50)
        return car_net, goal_net, init, init_net, net, saver, tmp_net,init_car

    def create_setting_folder(self, settings):
        if not os.path.exists(self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/"):
            os.makedirs(self.save_model_path + settings.name_movie + "_" + settings.timestamp)

    def load_net_weights(self, saver, sess, settings, latest=False, exact=False):
        print(latest)

        ignore_cars=False
        ignore_init=False
        ignore_init_car=False
        if len(settings.load_weights_car) > 0:
            ignore_cars = True
            print(" Ignore cars ")
        if len(settings.load_weights_init) > 0:
            ignore_init = True
            ignore_init_car=True
            print(" Ignore init ")
        print("Restore weights " + str(settings.load_weights))
        if len(settings.load_weights) > 0:
            if latest:
                latest_model = tf.train.latest_checkpoint(settings.load_weights)
            else:
                latest_model=settings.load_weights
            if exact and False:
                saver.restore(sess, latest_model)
                print("Exact")
            else:
                print(" Load weights, ignore car? "+str(ignore_cars)+" "+str(latest_model))
                restored_vars = self.get_tensors_in_checkpoint_file(file_name=latest_model, ignore_cars=ignore_cars, ignore_init=ignore_init, ignore_init_car=ignore_init_car)
                tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
                loader = tf.compat.v1.train.Saver(tensors_to_load)
                loader.restore(sess, settings.load_weights)
            print((settings.load_weights))

        print("Restore weights car "+str(settings.load_weights_car))
        if len(settings.load_weights_car) > 0:
            if 'model.ckpt' in settings.load_weights_car:
                latest_car=settings.load_weights_car
            else:
                latest_car = tf.train.latest_checkpoint(settings.load_weights_car)
            print("Load weights car "+str(settings.load_weights_car) )
            loader_car = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='car'))
            print (" Car variables")
            print(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='car'))
            if exact:
                loader_car.restore(sess, latest_car)
                ignore_cars=True
                print("Restored car weights")
            else:
                restored_vars = self.get_tensors_in_checkpoint_file(file_name=latest_car, ignore_cars=False, ignore_model=True, ignore_init=True, ignore_init_car=True)
                tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
                loader = tf.compat.v1.train.Saver(tensors_to_load)
                loader.restore(sess, settings.load_weights_car)

        print("Restored weights init" + str(settings.load_weights_init))
        if len(settings.load_weights_init) > 0:
            if 'model.ckpt' in settings.load_weights_init:
                latest_init = settings.load_weights_init
            else:
                latest_init = tf.train.latest_checkpoint(settings.load_weights_init)
            print("Load weights car " + str(settings.load_weights_init))
            loader_init = tf.compat.v1.train.Saver(
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='init'))
            print(" Car variables")
            print(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='init'))
            if exact:
                loader_init.restore(sess, latest_init)
                ignore_cars = True
                ignore_init=True
                print("Restored init weights")
            else:
                restored_vars = self.get_tensors_in_checkpoint_file(file_name=latest_init, ignore_cars=True,
                                                                    ignore_model=True, ignore_init=False, ignore_init_car=False)
                tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
                loader = tf.compat.v1.train.Saver(tensors_to_load)
                loader.restore(sess, settings.load_weights_init)
            print("Restored weights " + str(settings.load_weights_init))

    def get_tensors_in_checkpoint_file(self,file_name, all_tensors=True, tensor_name=None, ignore_cars=True, ignore_model=False, ignore_init=False, ignore_init_car=False):
        varlist = []

        var_value = []
        reader = tf.compat.v1.train.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                if not ignore_model:
                    if ignore_init_car and "place_car" in key:
                        print("Ignore Key " + str(key))
                    elif ignore_cars and "car" in key:
                        print ("Ignore Key "+str(key))
                    elif ignore_init and "init" in key:
                        print("Ignore Key " + str(key))
                    else:
                        print("Append " + str(key))
                        varlist.append(key)
                        var_value.append(reader.get_tensor(key))
                else:
                    if (ignore_cars==False and "car" in key) or (ignore_init==False and "init" in key) or  (ignore_init_car==False and "place_car" in key):
                        print("Add Key " + str(key))
                        varlist.append(key)
                        var_value.append(reader.get_tensor(key))
        else:
            varlist.append(tensor_name)
            var_value.append(reader.get_tensor(tensor_name))
        return (varlist, var_value)

    def build_tensors_in_checkpoint_file(self,loaded_tensors):
        full_var_list = list()

        # Loop all loaded tensors
        for i, tensor_name in enumerate(loaded_tensors[0]):
            # Extract tensor
            #if "fully_connected" not in tensor_name:

            if True:#not "vel" in tensor_name:
                try:
                    print(("recover " + tensor_name))
                    tensor_aux = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name + ":0")
                    full_var_list.append(tensor_aux)
                except:
                    print(('Not found: ' + tensor_name))
            else:
                print(("Don't recover " + tensor_name))


        return full_var_list

    def create_gradient_buffer(self, sess):
        if sess is None:
            return None, None, None, None
        variables_policy = []
        variables_init = []
        variables_goal = []
        variables_car = []
        variables_init_car=[]
        for v in tf.compat.v1.trainable_variables():
            if "place_car" in v.name:
                variables_init_car.append(v)
            elif "goal" in v.name:
                variables_goal.append(v)
            elif "init" in v.name:
                variables_init.append(v)
            elif "car" in v.name:
                variables_car.append(v)
            else:
                variables_policy.append(v)
        [car_vars]=sess.run([variables_car])
        for var in car_vars:
            print ("Car var "+str(var))

        # print ("Create gradient buffer for policy ")
        grad_buffer = sess.run(variables_policy)
        for ix, grad in enumerate(grad_buffer):
            grad_buffer[ix] = grad * 0
            # print ("Buffer indx "+str(ix) +" name "+str(variables_policy[ix].name))

        # print ("Create gradient buffer for init ")
        grad_buffer_init = sess.run(variables_init)
        for ix, grad in enumerate(grad_buffer_init):
            grad_buffer_init[ix] = grad * 0
            # print ("Buffer indx " + str(ix) + " name " + str(variables_init[ix].name))
            # print ("Create gradient buffer for init ")
        grad_buffer_goal = sess.run(variables_goal)
        for ix, grad in enumerate(grad_buffer_goal):
            grad_buffer_goal[ix] = grad * 0
            # print ("Buffer indx " + str(ix) + " name " + str(variables_init[ix].name))
        grad_buffer_car = sess.run(variables_car)
        for ix, grad in enumerate(grad_buffer_car):
            grad_buffer_car[ix] = grad * 0
            # print ("Buffer indx " + str(ix) + " name " + str(variables_init[ix].name))
        grad_buffer_init_car = sess.run(variables_init_car)
        for ix, grad in enumerate(grad_buffer_init_car):
            grad_buffer_init_car[ix] = grad * 0
            # print ("Buffer indx " + str(ix) + " name " + str(variables_init[ix].name))
        return grad_buffer, grad_buffer_init,grad_buffer_goal, grad_buffer_car,grad_buffer_init_car

    def get_model(self, settings):
        print("Set seed")
        tmp_net = None
        init_net=None
        goal_net = None
        car_net=None
        net=None
        init_net_car = None

        if not settings.cv_agent:
            if settings.min_seg:
                net = Seg_2d_min_softmax(settings)
            else:
                net = Seg_2d(settings)

        print("Set seed")
        if settings.learn_init_car:
            from initializer_net_car import InitializerNetCar
            init_net_car = InitializerNetCar(settings)

        if settings.learn_init:
            init_net=InitializerNet(settings)

        if settings.separate_goal_net:
            goal_net=GoalNet(settings)

        if settings.useRLToyCar:
            car_net=SimpleCarNet(settings)

        init = tf.compat.v1.global_variables_initializer()

        return net, tmp_net,init_net, init, goal_net, car_net, init_net_car

    def get_evaluation_agent(self, settings, net, init_net, goal_net):
        if settings.goal_agent:
            agent = GoalAgent(settings)
        elif settings.pedestrian_agent:
            agent = PedestrianAgent(settings)
        elif settings.random_agent:
            agent = RandomAgent(settings)
        else:
            agent = self.get_agent(None, net, None, init_net, None, goal_net, settings)
        return agent

    def evaluate(self, settings, time_file=None, viz=False):

        #if not settings.random_agent:
        net, tmp_net, init_net, init,goal_net, car_net,init_car_net = self.get_model(settings)# Initialize various constants for 3D reconstruction.
        saver = tf.compat.v1.train.Saver()

        settings.model_path = self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt"
        settings.save_settings("")

        latest=True
        if "model.ckpt" in settings.load_weights:
            latest=False

        sess=None
        if  not settings.goal_agent  and not settings.social_lstm:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess=tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
            sess.run(init)
            self.load_net_weights(saver, sess, settings, latest=latest, exact=True)

            sess.graph.finalize()
        sess.graph.finalize()
        trainableCars, trainablePedestrians = self.get_cars_and_agents(net, car_net, init_net, goal_net, init_car_net,
                                                                       None, None, None,
                                                                       None, None, settings)


        if not settings.cv_agent and not settings.random_agent:
            net.set_session(sess)
        if settings.learn_init:
            init_net.set_session(sess)
        if settings.separate_goal_net:
            goal_net.set_session(sess)
        if settings.useRLToyCar:
            car_net.set_session(sess)
        if settings.learn_init_car:
            init_car_net.set_session(sess)


        if settings.old:
            settings.run_2D = False

        print( "CARLA")
        print (" Car in RL.evaluate() "+str(trainableCars))
        self.evaluate_method_carla(trainablePedestrians, sess, settings, viz, trainableCars=trainableCars)




    def get_train_test_fileIndices(self, datasetFilenamesList, overfit = False, waymo=False, carla=False, new_carla=False, useRealTimeEnv=False,realtime_carla=False):
        all_dataset_size = len(datasetFilenamesList)
        spawn_points=list(datasetFilenamesList.keys())
        if useRealTimeEnv:
            dataset_size = len(datasetFilenamesList)
            train_set_percent = 0.7
            train_size = math.ceil(dataset_size * train_set_percent)
            test_size = dataset_size - train_size
            assert(test_size > 0 and train_size > 0), "Test data too small please fix the percents or something"
            train_set = spawn_points[1:1+train_size]
            train_set_asSet = set(train_set)
            val_set = [x for x in range(0, dataset_size) if x not in train_set_asSet]
            test_set = []
        elif realtime_carla:
            train_size = 25
            train_set = spawn_points[0:15]
            val_set = spawn_points[15:]
            print (datasetFilenamesList)
            print(" train set "+str(train_set))
            print(" val set " + str(val_set))
            test_set = []
        elif new_carla :
            train_size=25
            train_set = spawn_points[0:train_size]
            val_set = spawn_points[train_size:]
            test_set = []
        else:
            print("CARLA - regular data")
            train_size = 100#int(all_dataset_size * 0.667)
            train_set = list(range(train_size))
            val_set = list(range(train_size, all_dataset_size))
            test_set=list(range(0, 150))
        return train_set, val_set, test_set


    def evaluate_method_carla(self, trainablePedestrians, sess, settings, viz=False, supervised=False, trainableCars=None):
        if not os.path.exists(settings.evaluation_path):
            os.makedirs(settings.evaluation_path)

        filespath = settings.carla_path_test
        if viz:
            filespath = settings.carla_path_viz
            ending = "test_*"
        statics = []
        print("CARLA ")

        env = CARLAEnvironment(filespath, sess, None, None, None, settings)


        ending = "test_*"

        epoch = 0
        filename_list = {}
        saved_files_counter=0
        # Get files to run on.
        print(filespath + ending)
        for filepath in glob.glob(filespath + ending):
            parts = os.path.basename(filepath).split('_')
            pos = int(parts[-1])
            filename_list[pos] = filepath
        print (" Files ")
        print (filename_list)

        for  pos in sorted(filename_list.keys()):
            filepath = filename_list[pos]
            if pos % settings.TEST_JUMP_STEP_CARLA == 0 or viz:
                print(f"## Evaluating file {pos} out of 150")
                # TO DO: ADD settings.realTimeEnvOnline TO ENV.EVALUATE PARAMETERS
                prefix = get_carla_prefix_eval(supervised, settings.useRLToyCar or settings.useHeroCar,
                                               settings.new_carla, settings.realtime_carla, settings.realTimeEnvOnline,
                                               settings.ignore_external_cars_and_pedestrians)
                stats=[]
                try:
                    stats, saved_files_counter, initializer_stats= env.evaluate(prefix,filepath, trainablePedestrians, settings.evaluation_path, saved_files_counter,realtime_carla=settings.realtime_carla_only,trainableCars=trainableCars,viz=viz)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    logging.exception('Fatal error in main loop ' + settings.timestamp)
                    print("Unexpected error:", sys.exc_info()[0])
                    traceback.print_exception(*sys.exc_info())
                    if settings.stop_for_errors:
                        raise  # Always raise exception otherwise it will be masked and apperantly it would continue...

                if len(stats) > 0:
                    statics.extend(stats)
        np.save(settings.evaluation_path, statics)





import argparse

def main(setting):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-debugFullSceneRaycastRun', '--debugFullSceneRaycastRun',
        metavar='debugFullSceneRaycastRun',
        type=int,
        default=0,
        help='If Full scene raycast run only')

    argparser.add_argument(
        '-useSegmentationMapForOnlineLevel', '--useSegmentationMapForOnlineLevel',
        metavar='useSegmentationMapForOnlineLevel',
        type=int,
        default=1,
        help='If 0, open drive nice maps are used. Otherwise our segmented images are used instead')

    argparser.add_argument(
        '-renderSpeedLimits', '--renderSpeedLimits',
        metavar='renderSpeedLimits',
        type=int,
        default=0,
        help='If 1 will render speed limits in the vis output')

    argparser.add_argument(
        '-renderTrafficLights', '--renderTrafficLights',
        metavar='renderTrafficLights',
        type=int,
        default=0,
        help='If 1 will render traffic lights status in the vis output')

    args = argparser.parse_args()
    setting.debugFullSceneRaycastRun = args.debugFullSceneRaycastRun
    setting.onlineEnvSettings.useSegmentationMapForOnlineLevel = args.useSegmentationMapForOnlineLevel
    setting.onlineEnvSettings.renderSpeedLimits = args.renderSpeedLimits
    setting.onlineEnvSettings.renderTrafficLights = args.renderTrafficLights

    f=None
    if setting.timing:
        f=open('times.txt', 'w')
        start = time.time()
    rl=RL()

    if not f is None:
        end=time.time()
        f.write(str(end-start)+ " Setting up RL\n")
    rl.run(setting, time_file=f)

#import carlaEnv

if __name__ == "__main__":
    setup=run_settings()#evaluate=True)

    np.random.seed(RANDOM_SEED_NP)
    tf.random.set_seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


    if setup.profile:
        print("Run profiler----------------------------------------------------------------------")
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()
        main(setup)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
        print(" Save to "+str(os.path.join(setup.profiler_file_path, setup.name_movie+".psats")))
        stats.dump_stats(os.path.join(setup.profiler_file_path, setup.name_movie+".psats"))
    elif setup.memoryProfile:
        from memory_profiler import memory_usage
        main(setup)

    else:
        main(setup)

