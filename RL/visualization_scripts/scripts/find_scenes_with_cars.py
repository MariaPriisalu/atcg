import sys
sys.path.append("test/RL/") #
from RL.RLmain import RL,get_carla_prefix_train,get_carla_prefix_eval
import time
from RL.settings import run_settings, RANDOM_SEED
from RL.agent_car import CarAgent
import numpy  as np
from RL.visualization import view_2D, view_pedestrians, view_cars, view_prior_pos, view_agent,view_valid_pos,view_2D_rgb,view_pedestrians_nicer,view_cars_nicer,plot_car,colours
from dotmap import DotMap

from commonUtils.ReconstructionUtils import CreateDefaultDatasetOptions_CarlaRealTime, carla_labels,CreateDefaultDatasetOptions_CarlaOffline,CreateDefaultDatasetOptions_CarlaRealTimeOffline

import numpy as np
import glob
import tensorflow as tf

import os
external=False
validation=False
viz=False
real_time_carla=False

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Script to check that initialization prior makes sense
if __name__ == "__main__":
    settings=run_settings()
    settings.seq_len_train=450
    settings.seq_len_test = 450
    settings.seq_len_evaluate = 450
    settings.occlude_some_pedestrians=False
    settings.number_of_agents =1
    show_plots=False
    settings.realtime=real_time_carla
    show_cars=True

    f=None
    if settings.timing:
        f=open('times.txt', 'w')
        start = time.time()

    rl=RL()

    counter, filecounter, images_path, init, net, saver, tmp_net, init_net,goal_net,car_net,init_car_net = rl.pre_run_setup(settings)
    with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()) as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            tf.random.set_seed(RANDOM_SEED)
            sess.run(init)

            print((settings.load_weights))
            if "model.ckpt" in settings.load_weights:
                rl.load_net_weights(saver, sess, settings, latest=False)#, exact=True)
            else:
                rl.load_net_weights(saver, sess, settings, latest=True)

            sess.graph.finalize()
            trainableCars, trainablePedestrians = rl.get_cars_and_agents(net,car_net, init_net, goal_net, init_car_net, None, None, None,None,None,settings)
            init_net.set_session(sess)
            if init_car_net!=None:
                init_car_net.set_session(sess)
            if goal_net!=None:
                goal_net.set_session(sess)

            env, env_car, env_people = rl.get_environments_CARLA(None, images_path, None, None, settings, None)
            if validation:
                pos_show=[0,24,32,36,40,48,56,60,64,76,80,104,108,132,136,144]#4
                filespath = settings.carla_path_test
                if viz:
                    filespath = settings.carla_path_viz
                ending = "test_*"
                filename_list = {}
                saved_files_counter = 0
                # Get files to run on.
                epoch = 0
                for filepath in glob.glob(filespath + ending):
                    parts = os.path.basename(filepath).split('_')
                    pos = int(parts[-1])
                    filename_list[pos] = filepath
            else:
                pos_show=[0]
                epoch, filename_list, pos = rl.get_carla_files(settings, realtime=real_time_carla)
            train_set, val_set, test_set = rl.get_train_test_fileIndices(filename_list, carla=True,realtime_carla=real_time_carla)

            fig = plt.figure()

            if validation:
                use_set=test_set
            else:
                use_set = train_set
                pos_show=train_set
            pos_show=[0,1]
            counter=0
            for posIdx, position in enumerate(val_set):
                print("pos " + str(posIdx)+" pos "+str(position))
                if position==140:#22<posIdx:#True:#posIdx in pos_show :
                    print("pos " + str(posIdx))
                    filepath = filename_list[position]
                    print ("pos "+str(posIdx))
                    #pos_x, pos_y = env.default_camera_pos()
                    seq_len, seq_len_pfnn = env.default_seq_lengths(training=True, evaluate=False)
                    if settings.new_carla or settings.useRealTimeEnv or settings.realtime_carla:
                        if settings.new_carla:
                            file_name = filepath[0]
                        else:
                            file_name = filepath
                        if real_time_carla:
                            datasetOptions = CreateDefaultDatasetOptions_CarlaRealTimeOffline(settings)
                        elif settings.realTimeEnvOnline or settings.test_online_dataset:
                            datasetOptions = CreateDefaultDatasetOptions_CarlaRealTime(settings)
                        else:
                            datasetOptions = CreateDefaultDatasetOptions_CarlaOffline(settings)

                    else:
                        file_name = filepath
                        datasetOptions = CreateDefaultDatasetOptions_CarlaOffline(settings)


                    if validation:
                        prefix=get_carla_prefix_eval(False,  useToyEnv=True,new_carla=False, realtime_carla=real_time_carla,useOnlineEnv=settings.realTimeEnvOnline, no_external=settings.ignore_external_cars_and_pedestrians)
                    else:
                        prefix = get_carla_prefix_train(False, useToyEnv=True, new_carla=False, realtime_carla=real_time_carla, useOnlineEnv=settings.realTimeEnvOnline, no_external=settings.ignore_external_cars_and_pedestrians)



                    episode = env.set_up_episode(prefix, filepath, settings.camera_pos_x,settings.camera_pos_y,training=not validation, evaluate= validation,
                                                 useCaching=True,datasetOptions= datasetOptions,
                                                  time_file=None, seq_len_pfnn=seq_len_pfnn, trainableCars=trainableCars)

                    prit(" Ep use real time? "+str(episode.useRealTimeEnv))
                    num_repeats=1
                    for itr in range(num_repeats):
                        print(" Train on car? Environment ")
                        initParams = DotMap()
                        initParams.on_car = False
                        if settings.realTimeEnvOnline:
                            realTimeEnvObservation,observation_dict = env.realTimeEnv.reset(initParams)
                            env.parseRealTimeEnvObservation(realTimeEnvObservation, observation_dict, episode)

                        if settings.useRealTimeEnv:
                            alreadyAssignedCarKeys = set()
                            # for agentCar in env.all_car_agents:
                            env.all_car_agents.valid_car_keys_trainableagents = env.all_car_agents.valid_init.valid_car_keys
                            env.all_car_agents.reset(alreadyAssignedCarKeys, initParams)
                            heroAgentCars = env.all_car_agents
                            heroAgentPedestrians = trainablePedestrians

                            realTimeEnvObservation,observation_dict = env.environmentInteraction.reset(heroAgentCars=heroAgentCars, heroAgentPedestrians=heroAgentPedestrians, episode=episode)
                            env.parseRealTimeEnvObservation(realTimeEnvObservation,observation_dict, episode)




                        # Plot
                        img_from_above = np.zeros((episode.reconstruction.shape[1], episode.reconstruction.shape[2], 3), dtype=np.uint8)
                        img_above = view_2D(episode.reconstruction, img_from_above, 0)
                        img3 = view_2D_rgb(episode.reconstruction, np.ones((episode.reconstruction.shape[1], episode.reconstruction.shape[2], 3),  dtype=np.uint8) * 255, 0, white_background=False)
                        print("View PEDESTRIANS")
                        img3 = view_pedestrians_nicer(0, episode.people, img3, red=True)  # tensor=ep.reconstruction)
                        # current_frame, cars, frame,

                        img3 = view_cars_nicer(img3, episode.cars, 0)  # ,tensor=ep.reconstruction)


                        # For each car create prior and initialize pedestrian
                        if not show_cars:
                            img_above = view_pedestrians(0, episode.people, img_above, 0, trans=.15, no_tracks=True,
                                                         tensor=episode.reconstruction)

                            img_above = view_cars(img_above, episode.cars, img_above.shape[0], img_above.shape[1], 0,
                                                  episode.car_bbox, frame=0)  # tensor=ep.reconstruction)
                            img_above = view_cars(img_above, episode.cars, img_above.shape[0], img_above.shape[1], 0,
                                                  episode.car_bbox, frame=5)  # tensor=ep.reconstruction)

                            img_above_1=img_above.copy()
                            print (" not valid "+ str(img_above_1.shape[1] - 2 * seq_len)+ " "+str(img_above_1.shape[0] ))
                            for x in range(img_above_1.shape[1] ):
                                for y in range(img_above_1.shape[0] ):
                                    #print ("x "+str(x)+" y "+str(y)+" ep 1 "+str(episode.valid_positions[y,x])+" ep2 "+str(episode1.valid_positions[y,x]))
                                    if episode.valid_positions[y, x] != episode1.valid_positions[y,x]:  # episode.valid_position([0,y,x], no_height=True):
                                        img_above_1[img_above_1.shape[0] - 1  - y,  x,
                                        :] =np.zeros_like(img_above_1[img_above_1.shape[0] - 1  - y,  x,
                                        :])
                                        print (" Not valid "+str([x,y]))

                            # img2 = view_prior_pos(prior, img2, 0)
                            plt.imshow(img_above_1)

                            # plt.imshow(img2)
                            if show_plots:
                                plt.show()
                            fig.savefig(
                                "visualization-results/weimar_120/diff_in_in_valid_pos_" + str(
                                    position) + ".png")

                            print ("visualization-results/weimar_120/diff_in_in_valid_pos" + str(
                                position) + ".png")



                        else:
                            if settings.learn_init_car:
                                env.all_car_agents.agent_initialization(episode,training=True)
                            for agent_id, agent in enumerate(trainablePedestrians):
                                # print(" Ep use real time init agent? " + str(episode.useRealTimeEnv)
                                agent.init_agent(episode, training=False)
                                prior=episode.initializer_data[agent_id].prior
                                agents=[]
                                for id in range(agent_id+1):
                                    agents.append(episode.pedestrian_data[id].agent)
                                if episode.useRealTimeEnv:
                                    car_bbox=episode.initializer_data[agent_id].init_car_bbox

                                print (" Car bbox "+str(car_bbox))



                                img_from_above_loc_rgb = img3.copy()  # view_agent_nicer(agent_pos, img3, settings.agent_shape, settings.agent_shape_s,0)
                                transparency = 0.6
                                x_min = 0
                                y_min = 0
                                label = 26
                                col = colours[label]
                                col = (col[2], col[1], col[0])
                                cars_list =[]
                                for car_id, car in enumerate(episode.car_data):
                                    cars_list.append(car.car_bbox)
                                    plot_car(car.car_bbox[0], col, img_from_above_loc_rgb, 0, transparency, 0, 0)

                                img_above = view_cars(img_above, episode.cars, img_above.shape[0], img_above.shape[1], 0,
                                                      cars_list, frame=0)#, tensor=reconstrcution_local)#car_agent_predicted[0])

                                img_above = view_pedestrians(0, episode.people, img_above, 0, trans=.15, no_tracks=True)#,tensor=people_predicted[0])

                                img2 = view_agent(agents, img_above, 0, settings.agent_shape, [1, 2, 2], frame=0)



                                plt.imshow(img_from_above_loc_rgb)
                                plt.axis()
                                plt.tight_layout()
                                if show_plots:
                                    plt.show()
                                fig.savefig(
                                    "visualization-results/weimar_120/regular_" + str(
                                        position) +"_"+str(agent_id)+"_"+str(itr)+ ".png")

                                print ("visualization-results/weimar_120/regular_" + str(
                                    position) +"_"+str(agent_id)+"_"+str(itr)+ ".png")

                                # episode.calculate_goal_prior( episode.init_car_pos , episode.init_car_vel,agents[0][1:], car_dim[0],car_dim[1]).flatten()
                                ax = plt.subplot()
                                img2 = view_agent(agents, img_above, 0, settings.agent_shape, [1, 2, 2], frame=0)
                                ax.imshow(img_from_above_loc_rgb)
                                maxval = np.max(episode.heatmap[:])
                                print(" max val of goal " + str(maxval))
                                rescaled_prob = episode.heatmap / maxval
                                img_h = ax.imshow(np.flip(rescaled_prob, axis=0), alpha=0.6, cmap='plasma')
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="2%", pad=0.05)
                                if counter == 0:
                                    right_cb = plt.colorbar(img_h, cax=cax)
                                    counter = 1
                                plt.axis('off')
                                plt.tight_layout()
                                fig.savefig(
                                    "visualization-results/weimar_120/heatmap_" + str(
                                        position) + "_" + str(agent_id) + "_" + str(itr) + ".png")
                                if show_plots:
                                    plt.show()
                                if settings.learn_init_car and agent_id==0:
                                    for car_id in range(settings.number_of_car_agents):
                                        ax = plt.subplot()
                                        img2 = view_agent(agents, img_above, 0, settings.agent_shape, [1, 2, 2], frame=0)
                                        ax.imshow(img_from_above_loc_rgb)
                                        prior_car = episode.initializer_car_data[car_id].prior
                                        maxval = np.max(prior_car[:])
                                        print(" max val of goal " + str(maxval))
                                        rescaled_prob = prior_car / maxval
                                        img_h = ax.imshow(np.flip(rescaled_prob, axis=0), alpha=0.6, cmap='plasma')
                                        divider = make_axes_locatable(ax)
                                        cax = divider.append_axes("right", size="2%", pad=0.05)
                                        if counter == 0:
                                            right_cb = plt.colorbar(img_h, cax=cax)
                                            counter = 1
                                        plt.axis('off')
                                        plt.tight_layout()
                                        fig.savefig(
                                            "visualization-results/weimar_120/prior_car_" + str(
                                                position) + "_" + str(car_id) + "_" + str(itr) + ".png")
                                        if show_plots:
                                            plt.show()
                                        ax = plt.subplot()

                                        ax = plt.subplot()
                                        img2 = view_agent(agents, img_above, 0, settings.agent_shape, [1, 2, 2],
                                                          frame=0)
                                        ax.imshow(img_from_above_loc_rgb)
                                        probabilities = episode.initializer_car_data[car_id].init_distribution.reshape(
                                            prior.shape) * prior_car
                                        maxval = np.max(probabilities[:])
                                        print(" max val of goal " + str(maxval))
                                        rescaled_prob = probabilities / maxval
                                        img_h = ax.imshow(np.flip(rescaled_prob, axis=0), alpha=0.6, cmap='plasma')
                                        divider = make_axes_locatable(ax)
                                        cax = divider.append_axes("right", size="2%", pad=0.05)
                                        if counter == 0:
                                            right_cb = plt.colorbar(img_h, cax=cax)
                                            counter = 1
                                        plt.axis('off')
                                        plt.tight_layout()
                                        fig.savefig(
                                            "visualization-results/weimar_120/init_car_" + str(
                                                position) + "_" + str(car_id) + "_" + str(itr) + ".png")
                                        if show_plots:
                                            plt.show()
                                        ax = plt.subplot()
                                img2= view_agent(  agents, img_above, 0, settings.agent_shape, [1,2,2],frame=0)
                                ax.imshow(img_from_above_loc_rgb)
                                maxval=np.max(prior[:])
                                print(" max val of goal "+str(maxval))
                                rescaled_prob=prior/maxval
                                img_h = ax.imshow(np.flip(rescaled_prob, axis=0), alpha=0.6, cmap='plasma')
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="2%", pad=0.05)
                                if counter==0:
                                    right_cb = plt.colorbar(img_h, cax=cax)
                                    counter=1
                                plt.axis('off')
                                plt.tight_layout()
                                fig.savefig(
                                    "visualization-results/weimar_120/prior_"+str(position)+"_"+str(agent_id)+"_"+str(itr)+".png")
                                if show_plots:
                                    plt.show()
                                ax = plt.subplot()
                                probabilities = episode.initializer_data[agent_id].init_distribution.reshape(prior.shape)*prior
                                img2 = view_agent(agents, img_above, 0, settings.agent_shape, [1, 2, 2], frame=0)
                                ax.imshow(img_from_above_loc_rgb)
                                maxval = np.max(probabilities[:])
                                print(" max val of probabilities " + str(maxval))
                                rescaled_prob = probabilities / maxval
                                img_h = ax.imshow(np.flip(rescaled_prob, axis=0), alpha=0.6, cmap='plasma')
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="2%", pad=0.05)
                                if counter == 0:
                                    right_cb = plt.colorbar(img_h, cax=cax)
                                    counter = 1
                                plt.axis('off')
                                plt.tight_layout()
                                if show_plots:
                                    plt.show()
                                fig.savefig(
                                    "visualization-results/weimar_120/init_distr_" + str(
                                        position) + "_"+str(agent_id)+"_"+str(itr)+".png")

                                print ("visualization-results/weimar_120/init_distr_"+str(position)+"_"+str(agent_id)+"_"+str(itr)+".png")
                                ax = plt.subplot()
                                goal_prior=episode.calculate_goal_prior(id)
                                probabilities_goal = goal_prior.flatten()

                                probabilities_goal=probabilities_goal#episode.goal_prior
                                img3 = view_agent(agents, img_above, 0, settings.agent_shape, [1, 2, 2], frame=0)
                                ax.imshow(img_from_above_loc_rgb)
                                maxval = np.max(probabilities_goal[:])
                                print(" max val of goal " + str(maxval))
                                rescaled_prob = goal_prior / maxval
                                img_h = ax.imshow(np.flip(rescaled_prob, axis=0), alpha=0.6, cmap='GnBu')
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="2%", pad=0.05)
                                if counter == 0:
                                    right_cb = plt.colorbar(img_h, cax=cax)
                                    counter = 1
                                plt.axis('off')
                                plt.tight_layout()
                                if show_plots:
                                    plt.show()
                                fig.savefig(
                                    "visualization-results/weimar_120/prior_goal_" + str(
                                        position) +"_"+str(agent_id)+"_"+str(itr)+ ".png")

                                print ("visualization-results/weimar_120/prior_goal_" + str(
                                    position) + "_"+str(agent_id)+"_"+str(itr)+".png")
                                occluded_spaces=episode.initializer_data[agent_id].calculate_occlusions([1, 2, 2], settings.lidar_occlusion, settings.field_of_view_car)
                                ax = plt.subplot()
                                img4 = view_agent(agents, img_above, 0, settings.agent_shape, [1, 2, 2], frame=0)
                                ax.imshow(img_from_above_loc_rgb)
                                maxval = np.max(occluded_spaces[:])
                                print(" max val of goal " + str(maxval))
                                rescaled_prob = episode.initializer_data[agent_id].occlusions / maxval
                                img_h = ax.imshow(np.flip(rescaled_prob, axis=0), alpha=0.6, cmap='GnBu')
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="2%", pad=0.05)
                                if counter == 0:
                                    right_cb = plt.colorbar(img_h, cax=cax)
                                    counter = 1
                                plt.axis('off')
                                plt.tight_layout()
                                if show_plots:
                                    plt.show()
                                fig.savefig(
                                    "visualization-results/weimar_120/occlusion_" + str(
                                        position) + "_"+str(agent_id)+"_"+str(itr)+".png")

                                print ("visualization-results/weimar_120/occlusion_" + str(
                                    position) + "_"+str(agent_id)+"_"+str(itr)+".png")
                                valid_car = env.all_car_agents.valid_init.valid_positions_cars
                                ax = plt.subplot()
                                img5 = view_agent(agents, img_above, 0, settings.agent_shape, [1, 2, 2], frame=0)
                                ax.imshow(img_from_above_loc_rgb)

                                img_h = ax.imshow(np.flip(env.all_car_agents.valid_init.valid_positions_cars, axis=0), alpha=0.6, cmap='GnBu')
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="2%", pad=0.05)
                                if counter == 0:
                                    right_cb = plt.colorbar(img_h, cax=cax)
                                    counter = 1
                                plt.axis('off')
                                plt.tight_layout()
                                if show_plots:
                                    plt.show()
                                fig.savefig(
                                    "visualization-results/weimar_120/valid_carpos_" + str(
                                        position) + "_" + str(agent_id) + "_" + str(itr) + ".png")

                                print(
                                    "visualization-results/weimar_120/valid_carpos_" + str(
                                        position) + "_" + str(agent_id) + "_" + str(itr) + ".png")
                                env.environmentInteraction.onEpisodeEnd()
                env.entitiesRecordedDataSource=None
                env.environmentInteraction=None
