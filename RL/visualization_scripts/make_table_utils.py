
import numpy as np
import scipy.stats as stats
def make_table(stats_map,settings_map,eval=False):
    if not eval:
        best_itr=-1
        best_reward=-100
        values=[]
        if len(stats_map[settings_map.general_id]["initializer_cumulative_rewards"])>0:
            for itr, pair in enumerate(stats_map[settings_map.general_id]["initializer_cumulative_rewards"]):
                if len(pair)>0:
                    if settings_map.stats_of_individual_pedestrians==False:
                        values.append(np.mean(pair))
                    else:
                        values.append(np.mean(pair[:,0]))
            print ("Best iter reward "+str(np.argmax(values))+ " len of file "+str(len(stats_map[settings_map.general_id]["initializer_cumulative_rewards"]))+" values "+str(values))
        best_itr = -1
        best_reward = -100
        values=[]
        jump_rate=1
        if len(stats_map[settings_map.general_id]["collision_between_car_and_agent"])>0:
            if len(stats_map[settings_map.general_id]["collision_between_car_and_agent"])==2*len(stats_map[settings_map.general_id]["rewards"]):
                jump_rate=2
            for itr, pair in enumerate(stats_map[settings_map.general_id]["collision_between_car_and_agent"]):
                if len(pair) > 0 and itr%jump_rate==0:
                    if settings_map.stats_of_individual_pedestrians==False:
                        values.append(pair)
                    else:
                        values.append(pair[:,0])
            print("Best iter collisions " + str(np.argmax(values))+ " len of file "+str(len(stats_map[settings_map.general_id]["collision_between_car_and_agent"]))+" len values "+str(len(values))+" values "+str(values))
    else:
        best_itr=0
    if best_itr<0:
        best_itr=len(stats_map[settings_map.general_id]["rewards"])-1
    final=False
    non_valid_keys={}#{-1,5,7}
    if final:
        non_valid_keys={-1,5,7,8,0}



    #print(non_valid_keys)
    print("Rewards: " +str(stats_map[settings_map.general_id]["rewards"]))
    print("Collisions with pedestrian : " + str(stats_map[settings_map.general_id]["collision_between_car_and_agent"]))

    init = "Metric"
    non_zero_nbrs=get_non_zero_inits(stats_map, non_valid_keys)

    if settings_map.stats_of_individual_pedestrians==False:
        for key in non_zero_nbrs:
            if key not in non_valid_keys:
                init = init + " & " + settings_map.init_names[key]
    else:
        for id in range(stats_map[non_zero_nbrs[0]]["rewards"][0].shape[1]):
            init = init + " &  agent" + str(id)

    init += " \\\\ \hline"
    print(init)
    init = "Cum. reward pedestrian"
    init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "rewards",settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "Cum. reward initializer"
    init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "initializer_cumulative_rewards",settings_map)
    init += " \\\\ \hline"
    print(init)

    init = "Hit objs"

    init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "hit_objs", settings_map)
    init += " \\\\ \hline"
    print(init)

    init = "Initialization variance"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map,"initialization_variance",settings_map)
    init += " \\\\ \hline"
    print(init)
    if not settings_map.continous:
        init = "Entropy"

        init = print_row(best_itr, final, init, non_zero_nbrs, stats_map,"entropy",settings_map)
        init += " \\\\ \hline"
    print(init)
    init = "Distance travelled (m)"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "dist_travelled",settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "Ped. Trajectory"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "ped_traj",settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "People hit"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "people_hit",settings_map)
    init += " \\\\ \hline"
    print(init)

    init = "Hit by car"

    init = print_row(best_itr, True, init, non_zero_nbrs,stats_map, "car_hit",settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "Pavement"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "pavement",settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "Distance to goal"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "dist_left",settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "No of sucesses"

    init = print_row(best_itr, True, init, non_zero_nbrs,stats_map,  "successes",settings_map)
    init += " \\\\ \hline"
    print(init)

    init = "Heatmap"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "people_heatmap",settings_map)
    init += " \\\\ \hline"
    print(init)


    init = "Nbr dir. switches"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "nbr_direction_switches",settings_map)
    init += " \\\\ \hline"
    print(init)

    init = "Norm. distance"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "norm_dist_travelled",settings_map)
    init += " \\\\ \hline"
    print(init)

    init = "NLL actions"
    for key in non_zero_nbrs:
        if best_itr < len(stats_map[key]["likelihood_actions"])and len(stats_map[key]["likelihood_actions"])>0:
            rew = stats_map[key]["likelihood_actions"][best_itr]
            val1 = '%s' % float(settings_map.significance % rew[0])
            val2 = '%s' % float(settings_map.significance % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    # init = print_row(best_itr, final, init, non_zero_nbrs, likelihood_actions)
    init += " \\\\ \hline"
    print(init)

    init = "NLL vel"
    for key in non_zero_nbrs:
        if best_itr < len(stats_map[key]["likelihood_full"])and len(stats_map[key]["likelihood_full"])>0:
            rew = stats_map[key]["likelihood_full"][best_itr]
            val1 = '%s' % float(settings_map.significance % rew[0])
            val2 = '%s' % float(settings_map.significance % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    #init = print_row(best_itr, final, init, non_zero_nbrs, likelihood_full)
    init += " \\\\ \hline"
    print(init)

    init = "ADE"
    for key in non_zero_nbrs:
        if best_itr < len(stats_map[key]["prediction_error"])and len(stats_map[key]["prediction_error"])>0:
            rew = stats_map[key]["prediction_error"][best_itr]
            val1 = '%s' % float(settings_map.significance % rew[0])
            val2 = '%s' % float(settings_map.significance % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    #init = print_row(best_itr, final, init, non_zero_nbrs, prediction_error)
    init += " \\\\ \hline"
    print(init)

    init = "Mean speed"

    init = print_row(best_itr, final, init, non_zero_nbrs, stats_map,"speed_mean",settings_map)
    init += " \\\\ \hline"
    print(init)

    init = "STD speed"

    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "speed_var",settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "Time to collision"
    init = print_row(best_itr, final, init, non_zero_nbrs,stats_map, "time_to_collision",settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "Dist to collision"
    init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "dist_travelled_collision", settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "Dist to not collision"
    init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "dist_travelled_not_collision", settings_map)
    init += " \\\\ \hline"
    print(init)

    can_print=False
    for key in non_zero_nbrs:
        if "ped_traj_not_collision" in stats_map[key]:
            can_print=True
    if can_print:
        init = "Freq on ped traj collision"
        init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "ped_traj_collision", settings_map)
        init += " \\\\ \hline"
        print(init)
        init = "Freq on ped traj no collision"
        init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "ped_traj_not_collision", settings_map)
        init += " \\\\ \hline"
        print(init)


    can_print = False
    for key in non_zero_nbrs:
        if "entropy_collison" in stats_map[key]:
            can_print = True
    if can_print:
        init = "Entropy collision"
        init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "entropy_collison", settings_map)
        init += " \\\\ \hline"
        print(init)

        init = "Entropy not collision"
        init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "entropy_not_collison", settings_map)
        init += " \\\\ \hline"
        print(init)


    init = "Reward collision"
    init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "rewards_collision", settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "Reward not collision"
    init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "rewards_not_collision", settings_map)
    init += " \\\\ \hline"
    print(init)
    init = "Collision initialization variance"
    init = print_row(best_itr, final, init, non_zero_nbrs, stats_map, "initialization_variance_collision", settings_map)
    init += " \\\\ \hline"
    print(init)

def get_non_zero_inits(stats_map, non_valid_keys=[]):
    non_zero_nbrs = []
    for key, val in stats_map.items():
        if len(val["rewards"]) > 0 and key not in non_valid_keys:
            non_zero_nbrs.append(key)
    if len(non_zero_nbrs) == 2:
        non_zero_nbrs.pop(1)
    return non_zero_nbrs


def print_row(best_itr, final, init, non_zero_nbrs,stats_map, metric_key,settings_map):
    for init_key in non_zero_nbrs:
        if best_itr < len(stats_map[init_key][metric_key]):

            rew = stats_map[init_key][metric_key][best_itr]
            if settings_map.stats_of_individual_pedestrians==False:

                if np.isscalar(rew):
                    init = print_values_for_plot(True, init, rew,0,settings_map.significance)
                else:
                    init = print_values_for_plot(final, init, rew[0], rew[1],settings_map.significance)

            else:
                for id in range(rew.shape[0]):
                    if len(rew.shape) == 2:
                        init = print_values_for_plot(final, init, rew[id, 0], rew[id, 1],settings_map.significance)
                    else:
                        init = print_values_for_plot(True, init, rew[id],0,settings_map.significance)
        else:
            init = init + " & -"
    return init


def print_values_for_plot(final, init, mean_val, std_val , sign ):
    val1 = '%s' % float(sign% mean_val)#np.mean(rew[id, 0]))

    if final:
        init = init + " & " + str(val1)
    else:
        val2 = '%s' % float(sign% std_val)  # np.mean(rew[id, 1]))
        init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
    return init
