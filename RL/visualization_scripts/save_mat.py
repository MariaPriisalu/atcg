import scipy.io
from RL.visualization_scripts.make_table_utils import get_non_zero_inits
import numpy as np
def get_mat_plot_files(stats_map_base,stats_map_car,settings_map):

    metric_to_save=["initializer_cumulative_rewards", "collision_between_car_and_agent", "car_hit", "people_hit"]
    stats_maps= [stats_map_base, stats_map_car]

    mat_dict={}
    non_zero = get_non_zero_inits(stats_map_base)


    for init_nbr in non_zero:
        init_name=settings_map.init_names[init_nbr]
        for metric in metric_to_save:
            for stats_map in stats_maps:
                if metric in stats_map[init_nbr]:
                    values_mat=np.zeros(len(stats_map[init_nbr][metric]))
                    for i, val in enumerate(stats_map[init_nbr][metric]):
                        values_mat[i] = val[0]
                    variable_name=settings_map.name_movie
                    if "run_agent__carla_pfnn" in variable_name:
                        variable_name=variable_name[len("run_agent__carla_pfnn"):]
                    if stats_map==stats_map_car:
                        variable_name=variable_name+"_car"
                    mat_dict[variable_name]=values_mat
            print(mat_dict)
            filename=settings_map.timestamp
            if len(non_zero) > 0:
                filename = filename + "_" + init_name
            filename = filename + "_" + metric
            scipy.io.savemat(filename+".mat", mat_dict)
