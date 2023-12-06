from RL.settings import STATISTICS_INDX_CAR_INIT, STATISTICS_INDX_MAP, STATISTICS_INDX_MAP_STAT, STATISTICS_INDX, \
    PEDESTRIAN_INITIALIZATION_CODE

from RL.visualization_scripts.evaluate_stats_utils import evaluate
import numpy as np

def initialize_stats_map():
    stats_maps={}
    for init_m in range(len(PEDESTRIAN_INITIALIZATION_CODE) + 2):
        stats_map = {}

        stats_map["rewards"]= []
        stats_map["hit_objs"] = []
        stats_map["ped_traj"] = []
        stats_map["people_hit"] = []
        stats_map["entropy"] = []
        stats_map["dist_left"] = []
        stats_map["dist_travelled"] = []
        stats_map["car_hit"] = []
        stats_map["pavement"] = []
        stats_map["plot_iterations"] = []
        stats_map["people_heatmap"] = []
        stats_map["successes"] = []
        stats_map["norm_dist_travelled"] = []
        stats_map["nbr_direction_switches"] = []
        stats_map["nbr_inits"] = []
        stats_map["loss"] = []
        stats_map["one_step_error"] = []
        stats_map["one_step_errors"] = []
        stats_map["one_step_errors_g"] = []
        stats_map["prob_of_max_action"] = []
        stats_map["most_freq_action"] = []
        stats_map["freq_most_freq_action"] = []
        stats_map["variance"] = []
        stats_map["speed_mean"] = []
        stats_map["speed_var"] = []
        stats_map["likelihood_actions"] = []
        stats_map["likelihood_full"]= []
        stats_map["prediction_error"] = []
        stats_map["dist_to_car"] = []
        stats_map["time_to_collision"] = []
        stats_map["init_dist_to_goal"] = []
        stats_map["goal_locations"] = []
        stats_map["init_locations"] = []
        stats_map["goal_times"] = []
        stats_map["goal_speed"] = []
        stats_map["init_dist_to_car"] = []
        stats_map["speed_to_dist_correlation"] = []
        stats_map["collision_between_car_and_agent"] = []
        stats_map["initializer_rewards"] = []
        stats_map["initializer_cumulative_rewards"] = []
        stats_map["initializer_loss"] = []
        stats_map["initialization_variance"]=[]
        stats_map["initialization_variance_collision"] = []
        stats_map["rewards_collision"] = []
        stats_map["rewards_not_collision"] = []
        stats_map["entropy_collison"] = []
        stats_map["entropy_not_collison"] = []
        stats_map["ped_traj_collision"]=[]
        stats_map["ped_traj_not_collision"] = []
        stats_map["dist_travelled_collision"] = []
        stats_map["dist_travelled_not_collision"] = []

        stats_maps[init_m]=stats_map
    return stats_maps

def get_stats(files_map,settings_map, car_case=False, data_set="train"):
    stats_map = initialize_stats_map()
    if (len(files_map.filenames_itr)==0 and car_case==False and data_set!="test"):
        return stats_map
    prev_test_pos = 0
    general_stats=[]
    stats_temp=[]
    files,iterations = choose_file_path_for_stats(car_case, data_set, files_map)


    for j, pair in enumerate(sorted(files)):
        if prev_test_pos != iterations[pair[0]] or (settings_map.temporal_case and j%2==0):
            for init_m, stats in enumerate(stats_temp):
                if len(stats)>0:
                    statistics = np.concatenate(stats, axis=0)
                    evaluate(statistics, settings_map, stats_map[init_m], car_case, stats_of_individual_pedestrians=settings_map.stats_of_individual_pedestrians)
                    if settings_map.temporal_case:
                        stats_map[init_m]["plot_iterations"].append(j/2)
                    else:
                        stats_map[init_m]["plot_iterations"].append(prev_test_pos)
            if len(general_stats)>0:
                if len(general_stats)>1:
                    statistics = np.concatenate(general_stats, axis=0)
                else:
                    statistics=general_stats[0]
                evaluate(statistics, settings_map, stats_map[settings_map.general_id], car_case, stats_of_individual_pedestrians=settings_map.stats_of_individual_pedestrians)
                if settings_map.temporal_case:
                    stats_map[settings_map.general_id]["plot_iterations"].append(j / 2)
                else:
                    stats_map[settings_map.general_id]["plot_iterations"].append(prev_test_pos)
        stats_temp = []
        for init_m in range(11):
            stats_temp.append([])
        general_stats = []
        try:
            cur_stat = np.load(pair[1],allow_pickle=True)
        except IOError:
            return stats_map
        for ep_nbr in range(cur_stat.shape[0]):
            if STATISTICS_INDX.init_method<cur_stat.shape[3]:
                init=int(cur_stat[ep_nbr,0, 0, STATISTICS_INDX.init_method])
            else:
                init=int(PEDESTRIAN_INITIALIZATION_CODE.learn_initialization)
            stats_map[settings_map.general_id]["nbr_inits"].append(init)
            stats_temp[init].append(np.expand_dims(cur_stat[ep_nbr,:, :, :], axis=0))
            general_stats.append(np.expand_dims(cur_stat[ep_nbr,:, :, :], axis=0))
        prev_test_pos = iterations[pair[0]]


    for init_m, stats in enumerate(stats_temp):
        if len(stats) > 0:
            statistics = np.concatenate(stats, axis=0)
            evaluate(statistics, settings_map, stats_map[init_m], car_case, stats_of_individual_pedestrians=settings_map.stats_of_individual_pedestrians)
            if settings_map.temporal_case:
                stats_map[init_m]["plot_iterations"].append(len(files) / 2)
            else:
                stats_map[init_m]["plot_iterations"].append(prev_test_pos)
    if len(general_stats) > 0:
        if len(general_stats) > 1:
            statistics = np.concatenate(general_stats, axis=0)
        else:
            statistics = general_stats[0]
        evaluate(statistics, settings_map, stats_map[settings_map.general_id], car_case, stats_of_individual_pedestrians=settings_map.stats_of_individual_pedestrians)
        if settings_map.temporal_case:
            stats_map[settings_map.general_id]["plot_iterations"].append(len(files) / 2)
        else:
            stats_map[settings_map.general_id]["plot_iterations"].append(prev_test_pos)
    return stats_map





def choose_file_path_for_stats(car_case, data_set, files_map):
    if data_set == "train":
        if car_case:
            files = files_map.learn_car
        else:
            files =files_map.filenames_itr
        iterations=files_map.train_itr
    else:
        if car_case:
            files = files_map.learn_car_test
        else:
            files = files_map.test_files
        iterations = files_map.test_points
    return files, iterations

