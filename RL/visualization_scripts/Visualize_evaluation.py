import numpy as np
import os.path

from RL.visualization_scripts.evaluate_stats_utils import evaluate
from RL.visualization_scripts.find_files_utils import sort_files, sort_files_eval, make_movies, find_gradients, \
    get_statistics_file_path, get_paths_for_evaluation, get_training_itr,get_paths
from RL.visualization_scripts.make_table_utils import make_table
from RL.visualization_scripts.plotting_functions import plot_separately, plot_goal, plot_loss
from RL.visualization_scripts.plotting_net_weights_functions import plot_weights_conv
from RL.visualization_scripts.plotting_gradient_functions import plot_gradient_by_sem_channel
from RL.visualization_scripts.plotting_car_functions import plot_gradients_car, plot_weights_car
from RL.visualization_scripts.read_files_utils import read_gradients, read_settings_file
from RL.visualization_scripts.stat_maps_utils import get_stats, initialize_stats_map
from RL.visualization_scripts.stat_maps_car_utils import get_cars_stats, initialize_car_map
from RL.visualization_scripts.stat_maps_init_utils import get_init_stats, initialize_init_map
from RL.visualization_scripts.visualization_utils import trendline,get_constants
from RL.visualization_scripts.save_mat import get_mat_plot_files
external=False
import sys

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
print(sys.path)

import matplotlib.pyplot as plt
from dotmap import DotMap



######################################################################## Change values here!!!
settings_map=DotMap()
settings_map.general_id = 0 # id for average initialization
settings_map.significance=5
np.set_printoptions(precision=settings_map.significance)
settings_map.significance='%.'+str(settings_map.significance)+'g'
# Save plots?
settings_map.save_plots=True # save gradient plots?
settings_map.save_regular_plots=True # save all plots?
settings_map.toy_case=False # Toy environment: if waymo, settings_map.carla or cityscapes then F

settings_map.make_movie=False # Save movie?
settings_map.save_mat=False # Save mat file with pedestrian prediction error?
settings_map.car_people=False # Plot cars and people together
settings_map.supervised=False # Is this a supervised agent i.e. trained with supervised loss
settings_map.print_values=False
settings_map.ending=''
settings_map.stats_of_individual_pedestrians=False
# insert timestamp here

timestamps=["2022-12-19-17-11-40.313120"]#"2022-12-17-14-41-38.249956", "2022-12-17-14-45-27.298069", "2022-12-17-14-48-00.001978", "2022-12-17-14-49-13.472273", "2022-12-17-14-52-22.122143", "2022-12-17-14-55-16.293453", "2022-12-17-14-57-29.435707", "2022-12-17-14-57-29.435707", "2022-12-17-14-58-42.293014", "2022-12-17-15-02-06.564007", "2022-12-17-15-05-00.541567"]
    #["2022-12-03-20-50-58.756565", "2022-12-06-11-33-45.886624", "2022-12-03-20-54-29.778857", "2022-12-03-20-55-25.636742"]#["2022-11-30-15-54-59.783619", "2022-11-30-15-55-20.470420", "2022-11-30-15-55-34.480497", "2022-11-30-15-55-55.250686","2022-11-24-09-47-48.504585", "2022-11-24-09-48-42.580725", "2022-11-24-09-49-21.071291", "2022-11-24-09-49-48.636519", "2022-11-24-09-50-02.960403", "2022-11-24-09-50-31.539385", "2022-11-24-09-50-45.762350", "2022-11-24-09-51-12.265705"]
#["2022-11-30-15-54-59.783619", "2022-11-30-15-55-20.470420", "2022-11-30-15-55-34.480497", "2022-11-30-15-55-55.250686", "2022-11-30-15-56-47.093918", "2022-11-30-15-57-37.668668", "2022-11-30-15-58-24.063186", "2022-11-30-16-01-06.859943", "2022-11-30-16-12-27.401990", "2022-11-30-16-11-51.338048", "2022-11-30-16-11-26.844636", "2022-11-30-16-15-25.018586"]#["2022-11-24-09-47-48.504585","2022-11-24-09-48-42.580725","2022-11-24-09-49-21.071291","2022-11-24-09-49-48.636519","2022-11-24-09-50-02.960403","2022-11-24-09-50-31.539385","2022-11-24-09-50-45.762350","2022-11-24-09-51-12.265705","2022-11-23-20-01-24.483975","2022-11-25-11-27-42.745404","2022-11-23-19-53-34.643471","2022-11-23-20-04-00.984762","2022-11-23-20-04-44.000732","2022-11-23-19-55-29.627147","2022-11-24-11-18-21.498222","2022-11-26-17-17-14.668177","2022-11-25-11-39-40.678113","2022-11-25-11-43-44.334914","2022-11-25-11-44-46.123917","2022-11-25-11-48-52.672087","2022-11-25-11-53-06.855716","2022-11-25-13-13-29.816707","2022-11-25-13-14-07.063191","2022-11-25-13-15-08.639198","2022-11-25-14-34-11.358926"]# Should distance moved be re-calculated?
calc_dist=True
#################




def read_files_and_plot(settings_map):
    settings_map=get_paths(settings_map)

    for timestamp in timestamps:
        try:
            plt.close("all")
            # For old files do not update plots!
            if "2018-" in timestamp or "2017-" in timestamp or "2019-01" in timestamp or "2019-02" in timestamp:
                settings_map.make_movie = False
            dt_object = datetime.strptime(timestamp, '%Y-%m-%d-%H-%M-%S.%f')
            if dt_object < datetime.strptime("2021-04-08-00-00-00.421640", '%Y-%m-%d-%H-%M-%S.%f'):
                settings_map.make_movie = False
            if dt_object > datetime.strptime("2020-11-20-13-55-01.877475",
                                             '%Y-%m-%d-%H-%M-%S.%f') and dt_object < datetime.strptime(
                    "2021-04-08-00-00-01.877475", '%Y-%m-%d-%H-%M-%S.%f'):
                break

            # find statistics files.
            files, path = get_statistics_file_path(settings_map, timestamp)

            # Read settings file
            settings_map= read_settings_file(settings_map, timestamp)
            if len(settings_map.labels_to_use) > 0 or settings_map.supervised:

                # Set paths
                settings_map.movie_path = settings_map.settings.statistics_dir + "/agent/"
                settings_map.name_movie_eval = settings_map.settings.name_movie_eval
                # Make movie
                make_movies(settings_map)

                if len(files) > 0 or settings_map.supervised:
                    # Plot car / people enviornment results
                    files_map= sort_files(files)

                    get_training_itr(files_map)

                    # Get statistics on validation set
                    stats_map_training = get_stats(files_map, settings_map, data_set="train")

                    stats_map_training= get_init_stats(stats_map_training, files_map, settings_map, data_set="train")

                    if len(files_map.init_maps_stats_goal) >= 1:
                        stats_map_training = get_init_stats(stats_map_training, files_map, settings_map, data_set="train",
                                                       goal_case=True)

                    if len(files_map.learn_car) > 0:
                        stats_map_car = get_stats(files_map, settings_map, car_case=True, data_set="train")
                    else:
                        stats_map_car = initialize_stats_map()

                    if len(files_map.init_maps_stats_car) >= 1:
                        stats_map_car = get_init_stats(stats_map_car, files_map, settings_map, data_set="train",
                                                       car_case=True)


                    # Validation set
                    if len(files_map.test_files) > 0:
                        stats_map_val= get_stats(files_map, settings_map, data_set="val")
                    else:
                        stats_map_val= initialize_stats_map()


                    if len(files_map.init_maps_test) >= 1 or len(files_map.init_map_stats_test) >= 1:
                        stats_map_val= get_init_stats(stats_map_val, files_map, settings_map, data_set="val")



                    if len(files_map.init_maps_stats_goal_test) >= 1:
                        stats_map_val = get_init_stats(stats_map_val, files_map, settings_map, data_set="val",
                                                       goal_case=True)


                    if len(files_map.learn_car_test) > 0:
                        stats_map_car_val= get_stats(files_map, settings_map, car_case=True, data_set="val")
                    else:
                        stats_map_car_val= initialize_stats_map()

                    if len(files_map.init_maps_stats_car_test) >= 1:
                        stats_map_car_val = get_init_stats(stats_map_car_val, files_map, settings_map, data_set="val",
                                                       car_case=True)
                    # Plot cars/ people stats
                    if len(files_map.filenames_itr_cars) > 0 and len(files_map.filenames_itr_cars[0]) > 0 and False:
                        sz_train = np.load(files_map.filenames_itr_cars[0][1]).shape
                        car_stats_map= get_cars_stats(files_map.filenames_itr_cars, files_map.test_files_cars, sz_train)

                        sz_train = np.load(files_map.filenames_itr_people[0][1]).shape
                        pedestrian_stats_map = get_cars_stats(files_map.filenames_itr_people, files_map.test_files_people, sz_train)
                    else:
                        car_stats_map= initialize_car_map(0, [])
                        pedestrian_stats_map = initialize_car_map(0, [])

                    ################################## Plot
                    if settings_map.save_regular_plots and len(files_map.filenames_itr) > 1:

                        do_plotting(stats_map_training,stats_map_val,stats_map_car,stats_map_car_val,settings_map, files_map)

                        ################################### Go through gradients and Plot!
                        if not settings_map.supervised and settings_map.save_regular_plots:
                            get_and_plot_gradients(files_map, path, settings_map)

                    # Make table with results
                    if len(files_map.test_files) >= 1:
                        make_table(stats_map_val, settings_map)

                    if len(files_map.learn_car_test) >= 1:
                        make_table(stats_map_car_val, settings_map)
                        if settings_map.save_mat:
                            get_mat_plot_files(stats_map_val, stats_map_car_val, settings_map)


            ############################################ Evaluate test set
            # find statistics files on test set.
            files_eval, path = get_paths_for_evaluation(settings_map.eval_path, timestamp)
            files_map_eval = sort_files_eval(files_eval)

            if len(files_map_eval.test_files) > 0:
                files_map_eval.train_itr = {}
                train_counter = 0
                files_map_eval.test_points = {0: 0}
            if len(files_map_eval.test_files) > 0:

                stats_map_eval= get_stats(files_map_eval, settings_map, data_set="test")

                if len(files_map_eval.init_maps) >= 1 or len(files_map_eval.init_map_stats) >= 1:
                    print(" Get test data init "+str(len(files_map_eval.init_maps)))
                    stats_map_eval = get_init_stats(stats_map_eval, files_map_eval, settings_map, data_set="test")



                if len(files_map_eval.init_maps_stats_goal_test) >= 1:
                    stats_map_eval = get_init_stats(stats_map_eval, files_map, settings_map, data_set="test",
                                                        goal_case=True)

                if len(files_map_eval.learn_car_test) > 0:
                    print(" Get test data for car"+str(len(files_map_eval.learn_car_test)))
                    stats_map_car_eval= get_stats(files_map_eval, settings_map,car_case=True, data_set="test")
                    # test_files, test_points,settings_map.num_measures
                if len(files_map_eval.init_maps_stats_car_test) >= 1:
                    stats_map_car_eval = get_init_stats(stats_map_car_eval, files_map_eval, settings_map, data_set="test",
                                                        car_case=True)
                # Make table
                make_table(stats_map_eval, settings_map)

                if len(files_map_eval.learn_car_test) > 0:
                    print("Car  ------------------------------")
                    make_table(stats_map_car_eval, settings_map)
        except:
            raise
            print("Something went wrong")


def get_and_plot_gradients(files_map, path, settings_map):
    gradient_struct = find_gradients(path, files_map)
    # try:
    grads, itr = read_gradients(gradient_struct.gradients)
    non_zero_channels = plot_gradient_by_sem_channel(grads, settings_map, init=False)
    if len(gradient_struct.weights) > 0:
        weights_holder = plot_weights_conv(gradient_struct.weights, non_zero_channels, settings_map, init=False)  #
    grads_init, itr = read_gradients(gradient_struct.gradients_init)
    non_zero_channels = plot_gradient_by_sem_channel(grads_init, settings_map, init=True)
    if len(gradient_struct.weights_init) > 0:
        weights_holder = plot_weights_conv(gradient_struct.weights_init, non_zero_channels, settings_map, init=True)  #
    grads_init, itr = read_gradients(gradient_struct.gradients_goal)
    non_zero_channels = plot_gradient_by_sem_channel(grads_init, settings_map, goal=True)
    if len(gradient_struct.weights_goal) > 0:
        weights_holder = plot_weights_conv(gradient_struct.weights_goal, non_zero_channels,
                                           settings_map, goal=True)  #
    # except ValueError as IndexError:
    #     print("Value Error or Index error")
    grads_init_car, itr = read_gradients(gradient_struct.gradients_init_car)
    non_zero_channels = plot_gradient_by_sem_channel(grads_init_car, settings_map, init=True, car=True)
    if len(gradient_struct.weights_init_car) > 0:
        weights_holder = plot_weights_conv(gradient_struct.weights_init_car, non_zero_channels, settings_map, init=True,
                                           car=True)  #
    grads_car, itr = read_gradients(gradient_struct.gradients_car)
    plot_gradients_car(grads_car, settings_map)
    if len(gradient_struct.weights_car) > 0:
        weights_c, itr = read_gradients(gradient_struct.weights_car)
        weights_holder = plot_weights_car(weights_c, settings_map)  #


def do_plotting(stats_map_training,stats_map_test, stats_map_car,stats_map_car_test,settings_map,files_map):

    print("Plot")
    plot_trains = True

    plots=[("rewards","Average reward","_avg_reward.png" ),
           ("initializer_cumulative_rewards", "Average initializer reward","_avg_reward_init.png"),
           ("hit_objs", "Number of hit objects", "_hit_objs.png"),
           ("ped_traj", "Frequency on ped traj", "_ped_traj.png"),
           ("people_hit","Number of people hit", "_people_hit.png"),
           ("dist_travelled","Distance travelled", "_dist_travelled.png"),
           ("car_hit","Number of times hit by car", "_car_hit.png"),
           ("people_heatmap","Average sum from people heatmap", "_avg_heatmap.png"),
           ("dist_left","Average min distance left", "_dist_left.png"),
           ("successes","Number of successes","_success.png"),
           ("norm_dist_travelled","Normalized distance travelled","_norm_dist.png"),
           ( "nbr_direction_switches","Number of direction switches", "_dir_switches.png"),
           ("loss","Loss","_loss.png"),
           ("initializer_loss","Initializer Loss","_loss_init.png"),
           ("goal_times","Goal times", "_goal_times.png"),
           ("goal_speed","Average Goal speed","_goal_speed.png"),
           ("initialization_variance", "Variance in different pedestrian initializations", "_init_var.png")]

    if settings_map.velocity:
        plots.append(("speed_mean", "Mean Speed","_speed_mean.png"))
        plots.append(("speed_var", "Standard deviation of average speeds exhibited in one batch", "_std_speed_batch.png"))
        plots.append(("variance","Average standard deviation of the speed exhibited during a trajectory, should be low","_std_speed_traj.png"))

    if settings_map.learn_init:
        plots.append(("dist_to_car", "Minimal distance to car", "_dist_to_car.png"))
        plots.append(("init_dist_to_car", "init distance to car","_init_to_car.png"))
        plots.append(("init_dist_to_goal","initial distance to goal","_init_dist_to_goal.png"))
        plots.append(("time_to_collision","Time to collision","_time_to_collision.png"))

    if settings_map.continous:
        plots.append(("","Agent mean step in x direction","_mean_x.png"))
        plots.append(("", "Mean of agent output y", "_mean_y.png"))
        plots.append(("", "Variance of agent output y","_var_y.png"))
        plots.append(("",  "Variance of agent output x","_var_x.png"))
        plots.append(("variance","Variance","_var_model.png"))
    else:
        plots.append(("entropy", "Entropy","_entropy.png"))
        plots.append(("prob_of_max_action", "Highest probability of an action","_max_prob.png"))
        plots.append(("most_freq_action","Mode action","_action.png"))



    for plot_name_trio in plots:
        metric=plot_name_trio[0]
        title=plot_name_trio[1]
        filename=plot_name_trio[2]
        print(metric)
        plot_separately(stats_map_training, stats_map_test, settings_map, metric, title, filename, plot_train=plot_trains)

    if len(files_map.learn_car_test):
        plots_car = [
            ("rewards","Car Average reward","_avg_reward_car.png"),
            ("initializer_cumulative_rewards", "Average initializer reward", "_avg_reward_init_car.png"),
            ("hit_objs","Car Number of hit objects", "_hit_objs_car.png"),
            ("hit_objs","Car Number of hit objects", "_hit_objs_car.png"),
            ("ped_traj", "Car Frequency on ped traj", "_ped_traj_car.png"),
            ("people_hit","Car Number of people hit", "_people_hit_car.png"),
            ("dist_travelled","Car Distance travelled", "_dist_travelled_car.png"),
            ("car_hit","Car Number of times hit by car", "_car_hit_car.png"),
            ( "people_heatmap","Car Average sum from people heatmap","_avg_heatmap_car.png"),
            ("dist_left", "Car Average min distance left","_dist_left_car.png"),
            ("successes", "Car Number of successes", "_success_car.png"),
            ("norm_dist_travelled","Car Normalized distance travelled","_norm_dist_car.png"),
            ("loss","Car loss","_loss_car.png"),
            ("initializer_loss", "Initializer Loss", "_loss_init_car.png"),
            ("entropy", "Car entropy", "_entropy_car.png"),
            ("prob_of_max_action","Car highest probability of an action", "_max_prob_car.png"),
            ("most_freq_action","Car mode action","_action_car.png"),
            ("speed_to_dist_correlation", "Correlation between distance to pedestrian and car's speed","_dist_to_pedestrian_correlation_to_car_speed.png"),
            ( "collision_between_car_and_agent", "Collisions_with_agent","_collision_car_with_agent.png"),
            ("initialization_variance", "Variance in different car initializations", "_init_var_car.png")
        ]
        for plot_name_trio in plots_car:
            metric = plot_name_trio[0]
            title = plot_name_trio[1]
            filename = plot_name_trio[2]
            plot_separately(stats_map_car, stats_map_car_test, settings_map, metric, title, filename,
                            plot_train=plot_trains)



    if len(files_map.init_maps) >= 1 or len(files_map.init_map_stats_test) >= 1:
        if settings_map.gaussian:
            if settings_map.learn_goal:
                plot_goal(stats_map_training, stats_map_test, settings_map, "goal_locations","goal_position_mode","goal_prior_mode", "Goal location distribution", "goal_location_distribution.png",
                          plot_train=False, axis=0, gaussian=True)
            plot_goal(stats_map_training, stats_map_test, settings_map, "init_locations","init_position_mode","init_prior_mode",
                      "Init location distribution", "init_location_distribution.png",
                      plot_train=False, axis=0)

        else:
            plot_goal(stats_map_training, stats_map_test, settings_map, "init_locations","position_mode_init","prior_mode_init",
                      "Init location distribution", "_location_distribution_init.png",
                      plot_train=False, axis=0)
            plot_loss(stats_map_training, stats_map_test, settings_map, "entropy_maps_init",
                      "Entropy of initialization distribution", "_entropy_init.png", plot_train=False)

            plot_loss(stats_map_training, stats_map_test, settings_map, "entropy_prior_maps_init",
                      "Entropy of prior distribution", "_prior_entropy_init.png", plot_train=False)
            plot_loss(stats_map_training, stats_map_test, settings_map, "diff_to_prior_maps_init",
                      "Difference to prior", "_diff_yp_prior_init.png", plot_train=False)
            if settings_map.learn_goal:
                plot_goal(stats_map_training, stats_map_test, settings_map, "goal_locations", "position_mode_goal",
                          "prior_mode_goal",
                          "Goal location distribution", "_location_distribution_goal.png",
                          plot_train=False, axis=0, gaussian=False)
                # Make scatter plot between car speed and number of collisions
                plot_loss(stats_map_training, stats_map_test, settings_map, "entropy_maps_goal",
                          "Entropy of initialization distribution", "_entropy_goal.png", plot_train=False)

                plot_loss(stats_map_training, stats_map_test, settings_map, "entropy_prior_maps_goal",
                          "Entropy of prior distribution", "_prior_entropy_goal.png", plot_train=False)
                plot_loss(stats_map_training, stats_map_test, settings_map, "diff_to_prior_maps_goal",
                          "Difference to prior", "_diff_yp_prior_goal.png", plot_train=False)
            if len(files_map.init_maps_stats_car)>0:
                plot_goal(stats_map_car, stats_map_car_test, settings_map, "init_locations", "position_mode_car",
                          "prior_mode_car",
                          "Car location distribution", "location_distribution_car.png",
                          plot_train=False, axis=0, gaussian=False)
                # Make scatter plot between car speed and number of collisions
                plot_loss(stats_map_car, stats_map_car_test, settings_map, "entropy_maps_car",
                          "Entropy of car initialization distribution", "_entropy_car.png", plot_train=False)

                plot_loss(stats_map_car, stats_map_car_test, settings_map, "entropy_prior_maps_car",
                          "Entropy of car prior distribution", "_prior_entropy_car.png", plot_train=False)
                plot_loss(stats_map_car, stats_map_car_test, settings_map, "diff_to_prior_maps_car",
                          "Difference to car prior", "_diff_yp_prior_car.png", plot_train=False)

from datetime import datetime

read_files_and_plot(settings_map)

