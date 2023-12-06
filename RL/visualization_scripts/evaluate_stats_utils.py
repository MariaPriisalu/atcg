from RL.settings import STATISTICS_INDX, STATISTICS_INDX_CAR, PEDESTRIAN_MEASURES_INDX, CAR_MEASURES_INDX
import numpy as np
import scipy.stats as stats


def evaluate(statistics,settings_map, stats_map,  car_case=False, print_values=False, stats_of_individual_pedestrians=False, multiple_goals=True):


    if car_case:
        measures_indx = CAR_MEASURES_INDX
        stat_indx = STATISTICS_INDX_CAR
    else:
        stat_indx = STATISTICS_INDX
        measures_indx = PEDESTRIAN_MEASURES_INDX

        # Mask out dead or successful agents.
    agent_measures = statistics[:, :, :, stat_indx.measures[0]:stat_indx.measures[1]]
    valid_frames = agent_measures[:, :, :, measures_indx.agent_dead] == 0
    if car_case:
        for ep in range(valid_frames.shape[0]):
            for id in range(valid_frames.shape[1]):
                if np.any(agent_measures[ep, id, :, measures_indx.agent_dead]):
                    last_frame=np.argmax(agent_measures[ep, id, :, measures_indx.agent_dead])
                    valid_frames[ep,id,last_frame:]=False
    else:
        for ep in range(valid_frames.shape[0]):
            if np.any(agent_measures[ep, :, :, measures_indx.agent_dead]):
                last_frame=np.argmax(agent_measures[ep, :, :, measures_indx.agent_dead])
                for id in range(valid_frames.shape[1]):
                    valid_frames[ep,id,last_frame:]=False
    if car_case:
        hit_by_agent = np.sum(agent_measures[:, :, :, measures_indx.hit_by_agent] * valid_frames, axis=2) > 0
        not_hit_by_agent = hit_by_agent == 0
    else:
        hit_by_hero_car = np.sum(agent_measures[:, :, :, measures_indx.hit_by_hero_car] * valid_frames, axis=2) > 0
        not_hit_by_hero_car = hit_by_hero_car == 0

    agent_pos = statistics[:,:, :, stat_indx.agent_pos[0]:stat_indx.agent_pos[1]]
    init_pos = agent_pos[:,:,0,1:]
    variances=[]
    inits_collision_x=[]
    inits_collision_y = []
    for ep in range(init_pos.shape[0]):
        episode_init=init_pos[ep,:,:]
        variance_in_init_pos = np.var(init_pos[ep,:,:], axis=0) / (max(settings_map.settings.env_shape) ** 2)
        variances.append(variance_in_init_pos[0]+ variance_in_init_pos[1])
        if not car_case:
            if np.sum(hit_by_hero_car[ep,:])>0:
                collision_traj=episode_init[hit_by_hero_car[ep,:]]
                for pos in range(collision_traj.shape[0]):
                    inits_collision_x.append(collision_traj[pos,0])
                    inits_collision_y.append(collision_traj[pos, 1])
        else:
            if np.sum(hit_by_agent[ep, :]) > 0:

                collision_traj = episode_init[hit_by_agent[ep, :]]
                for pos in range(collision_traj.shape[0]):
                    inits_collision_x.append(collision_traj[pos,0])
                    inits_collision_y.append(collision_traj[pos,1])


    stats_map["initialization_variance"].append([np.mean(variances), np.std(variances)])

    var_x=np.var(inits_collision_x)
    var_y=np.var(inits_collision_y)
    var_sum=var_x+var_y
    stats_map["initialization_variance_collision"].append(var_sum/ (max(settings_map.settings.env_shape) ** 2))

    if len(stats_map["rewards"])==0:
        print(str(agent_pos.shape[1]))
    if car_case:
        agent_probabilities = statistics[:,:, :, stat_indx.probabilities]
    else:
        agent_probabilities = statistics[:,:, :, stat_indx.probabilities[0]:stat_indx.probabilities[1]]
    agent_reward = statistics[:,:, :, stat_indx.reward]
    agent_loss = statistics[:,:, :, stat_indx.loss]
    agent_action = statistics[:,:, :, stat_indx.action]
    agent_speed=statistics[:,:, :, stat_indx.speed]

    if stat_indx.reward_initializer<statistics.shape[-1] :
        initializer_reward=statistics[:,:, :, stat_indx.reward_initializer]
        initializer_reward_d = statistics[:, :, :, stat_indx.reward_initializer_d]

        stats_map["initializer_rewards"].append(np.zeros((initializer_reward.shape[1],2)))

        for id in range(initializer_reward.shape[1]):
            stats_map["initializer_rewards"][-1][id,:]=[np.mean(initializer_reward_d[:,id,0], axis=0), np.std(initializer_reward_d[:,id,0], axis=0)]

        initializer_cum_reward=np.sum(initializer_reward_d[:,:,0], axis=1)
        stats_map["initializer_cumulative_rewards"].append([np.mean(initializer_cum_reward), np.std(initializer_cum_reward)])

    if stat_indx.loss_initializer<statistics.shape[-1]:
        init_loss=statistics[:,:, 0, stat_indx.loss_initializer]
        stats_map["initializer_loss"].append([np.mean(init_loss), np.std(init_loss)])

    if not car_case:
        agent_likelihood=agent_probabilities[:,:,:,15]
        agent_likelihood_full = agent_probabilities[:,:,:,17]
        prediction_errors =agent_probabilities[:,:,:,18]*0.2
    if car_case:
        stats_map["goal_locations"].append(statistics[:, :, 3:5, stat_indx.goal])
    else:

        if statistics.shape[3]<stat_indx.goal[1]:
            stats_map["goal_locations"].append(statistics[:,:, 3:5, stat_indx.init_method])
        else:
            stats_map["goal_locations"].append(statistics[:, :, :, stat_indx.goal[0]:stat_indx.goal[1]])
    #if not car_case:
    stats_map["init_locations"].append(agent_pos[:,:,0,1:])

    if car_case:
        goal_time = statistics[:, :, 5, stat_indx.goal]
    else:
        if statistics.shape[3]<stat_indx.goal[1]:
            goal_time=statistics[:,:, 5, stat_indx.init_method]
        else:
            goal_time = statistics[:, :, :, stat_indx.goal_time]

    if stats_of_individual_pedestrians:
        stats_map["goal_times"].append(np.zeros((goal_time.shape[1], 2)))
        for id in range(goal_time.shape[1]):
            if car_case:
                stats_map["goal_times"][-1][id, :] = [np.mean(goal_time[:, id]), np.std(goal_time[:, id])]
            else:
                stats_map["goal_times"][-1][id,:]=[np.mean(goal_time[:,id,:]), np.std(goal_time[:,id,:])]
    else:
        if car_case:
            stats_map["goal_times"].append([np.mean(goal_time), np.std(goal_time)])
        else:
            stats_map["goal_times"].append([np.mean(goal_time), np.std(goal_time)])
    agent_reward_d = statistics[:, :,:, stat_indx.reward_d]


    base_divisor=np.sum(valid_frames, axis=2)
    number_of_valid_frames = np.maximum(base_divisor, np.ones_like(base_divisor))

    stats_map["loss"].append([np.mean(agent_loss), np.std(agent_loss)])

    one_error=agent_measures[:,:,:,measures_indx.one_step_prediction_error]#*mask
    stats_map["one_step_error"].append([np.mean(one_error*0.2), np.std(one_error*.2)])
    stats_map["one_step_errors"].append(one_error*0.2)

    likelihoods=[]
    likelihoods_full=[]
    prediction_error_local=[]
    cum_reward=np.zeros((valid_frames.shape[0], valid_frames.shape[1]))
    if not car_case:
        for ep_itr in range(valid_frames.shape[0]):
            likelihoods.append(0)
            likelihoods_full.append(0)
            #cum_reward.append(0)

            error_list=[]
            for frame in range(valid_frames.shape[2]):
                for ped_id in range(valid_frames.shape[1]):
                    if agent_likelihood[ep_itr,ped_id, frame]>0:
                        likelihoods[-1]-=np.log(agent_likelihood[ep_itr,ped_id, frame])
                        likelihoods_full[-1]-=agent_likelihood_full[ep_itr,ped_id, frame]
                    if valid_frames[ep_itr,ped_id, frame]:
                        error_list.append(prediction_errors[ep_itr,ped_id, frame])
                        cum_reward[ep_itr, ped_id]+=(agent_reward[ep_itr,ped_id ,frame])
            prediction_error_local.append(np.mean(error_list))

        stats_map["likelihood_actions"].append([np.mean(agent_likelihood)*(1/valid_frames.shape[1]), np.std(agent_likelihood)*(1/valid_frames.shape[1])])

        stats_map["likelihood_full"].append([np.mean(likelihoods_full)*(1/valid_frames.shape[1]), np.std(agent_likelihood_full)*(1/valid_frames.shape[1])])
        stats_map["prediction_error"].append([np.mean(prediction_error_local)*(1/valid_frames.shape[1]), np.std(agent_likelihood_full)*(1/valid_frames.shape[1])])


    goal_not_reached_mask = agent_measures[:,:, 0, measures_indx.dist_to_goal] > 0


    dist_global = np.zeros((agent_measures.shape[0],agent_measures.shape[1]))
    dist_travelled_tot = np.zeros((agent_measures.shape[0],agent_measures.shape[1]))
    seq_lens = np.zeros((agent_measures.shape[0],agent_measures.shape[1]))
    dist_left_to_goal=np.zeros((agent_measures.shape[0],agent_measures.shape[1]))
    initial_dist_to_goal=np.zeros((agent_measures.shape[0],agent_measures.shape[1]))
    for i in range(agent_measures.shape[0]):
        for ped_id in range(valid_frames.shape[1]):
            dist_global[i, ped_id] = 0
            seq_len = 0
            while seq_len + 1 < valid_frames.shape[1] and valid_frames[i, ped_id, seq_len + 1] == 1:
                seq_len = seq_len + 1
            seq_lens[i,ped_id] = seq_len
            dist_global[i,ped_id] = np.linalg.norm(agent_pos[i, ped_id, seq_len, 1:] - agent_pos[i,ped_id, 0, 1:]) * 0.2
            dist_travelled_tot[i,ped_id] = agent_measures[i,ped_id, seq_len, measures_indx.total_distance_travelled_from_init]
            dist_left_to_goal[i,ped_id]=np.min(agent_measures[i,ped_id, :seq_len+1, measures_indx.dist_to_goal])
            initial_dist_to_goal[i,ped_id]=agent_measures[i,ped_id, 0, measures_indx.dist_to_goal]

    non_zero_dist_left_to_goal = np.where( goal_not_reached_mask, dist_left_to_goal, 0)
    if agent_measures.shape[-1] >= 13:
        if stats_of_individual_pedestrians:
            stats_map["successes"].append(np.zeros(valid_frames.shape[1]))

            for ped_id in range(valid_frames.shape[1]):
                local_val=agent_measures[:, ped_id,:, measures_indx.goal_reached]*valid_frames[:,ped_id, :]
                if multiple_goals:
                    suc = np.sum(local_val, axis=-1)
                else:
                    suc = np.max(local_val, axis=-1)
                stats_map["successes"][-1][ped_id]=np.sum(suc) * 1.0 / len(suc)
        else:
            local_val = np.sum(agent_measures[:, :, :, measures_indx.goal_reached] * valid_frames[:, :, :],axis=1)
            if multiple_goals:
                suc = np.sum(local_val)
            else:
                suc = np.max(local_val)
            stats_map["successes"].append(suc)
    else:
        stats_map["successes"].append(np.sum(non_zero_dist_left_to_goal < np.sqrt(2)) * 1.0 / len(non_zero_dist_left_to_goal))
    goal_time_local=np.where(goal_time> 0, goal_time, 1)
    if stats_of_individual_pedestrians:
        stats_map["dist_left"].append(np.zeros((valid_frames.shape[1], 2)))
        stats_map["init_dist_to_goal"].append(np.zeros((valid_frames.shape[1], 2)))
        stats_map["goal_speed"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            stats_map["dist_left"][-1][ped_id,:]=[np.mean(non_zero_dist_left_to_goal[:, ped_id] * 0.2), np.std(non_zero_dist_left_to_goal[:,ped_id] * 0.2)]
            stats_map["init_dist_to_goal"][-1][ped_id,:]=[np.mean(initial_dist_to_goal[:, ped_id]* 0.2), np.std(initial_dist_to_goal[:, ped_id]* 0.2)]
            if car_case:
                local_speed = initial_dist_to_goal[:, ped_id] / goal_time_local[:, ped_id]
            else:
                local_speed=initial_dist_to_goal[:, ped_id]/goal_time_local[:, ped_id,0]
            stats_map["goal_speed"][-1][ped_id,:]=[np.mean(local_speed), np.std(local_speed)]
    else:
        stats_map["dist_left"].append([np.mean(non_zero_dist_left_to_goal[:, :] * 0.2), np.std(non_zero_dist_left_to_goal[:,:] * 0.2)])
        stats_map["init_dist_to_goal"].append([np.mean(initial_dist_to_goal[:, :]* 0.2), np.std(initial_dist_to_goal[:, :]* 0.2)])
        if car_case:
            local_speed = initial_dist_to_goal[:, :] / goal_time_local[:, :]
        else:
            local_speed = initial_dist_to_goal[:, :] / goal_time_local[:, :, 0]
        stats_map["goal_speed"].append([np.mean(local_speed), np.std(local_speed)])



    # Reward
    rew_sum = np.sum(agent_reward*valid_frames, axis=2)

    if stats_of_individual_pedestrians:

        stats_map["rewards"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            stats_map["rewards"][-1][ped_id,:]=[np.mean(rew_sum[:,ped_id]), np.std(rew_sum[:,ped_id])]
    else:

        stats_map["rewards"].append([np.mean(rew_sum.flatten()), np.std(rew_sum.flatten())])
        if car_case:
            stats_map["rewards_collision"].append([np.mean(rew_sum[hit_by_agent].flatten()), np.std(rew_sum[hit_by_agent].flatten())])
            stats_map["rewards_not_collision"].append(
                [np.mean(rew_sum[hit_by_agent==0].flatten()), np.std(rew_sum[hit_by_agent==0].flatten())])
        else:
            stats_map["rewards_collision"].append(
                [np.mean(rew_sum[hit_by_hero_car].flatten()), np.std(rew_sum[hit_by_hero_car].flatten())])
            stats_map["rewards_not_collision"].append(
                [np.mean(rew_sum[hit_by_hero_car==0].flatten()), np.std(rew_sum[hit_by_hero_car==0].flatten())])

    if not settings_map.continous:
        if stats_of_individual_pedestrians:
            stats_map["most_freq_action"].append(np.zeros((valid_frames.shape[1])))
            stats_map["freq_most_freq_action"].append(np.zeros((valid_frames.shape[1])))
            stats_map["prob_of_max_action"].append(np.zeros((valid_frames.shape[1])))
            for ped_id in range(valid_frames.shape[1]):
                most_freq_action_m = stats.mode(agent_action[:,ped_id], axis=None)
                stats_map["most_freq_action"][-1][ped_id]=most_freq_action_m[0]
                stats_map["freq_most_freq_action"][-1][ped_id]=most_freq_action_m[1] *1.0/ len(agent_action)
                if not car_case:
                    stats_map["prob_of_max_action"][-1][ped_id]=np.mean(np.max(agent_probabilities[:,ped_id], axis=2))
                else:
                    stats_map["prob_of_max_action"][-1][ped_id]=np.mean(agent_probabilities[:,ped_id])
        else:

            most_freq_action_m = stats.mode(agent_action[:, :], axis=None)
            stats_map["most_freq_action"].append(most_freq_action_m[0])
            stats_map["freq_most_freq_action"].append(most_freq_action_m[1] * 1.0 / len(agent_action))
            if not car_case:
                stats_map["prob_of_max_action"].append([np.mean(np.max(agent_probabilities, axis=2))])
            else:
                stats_map["prob_of_max_action"].append([np.mean(agent_probabilities)])
        if stats_of_individual_pedestrians:
            stats_map["variance"].append(np.zeros((valid_frames.shape[1], 2)))
            for ped_id in range(valid_frames.shape[1]):
                speed_variance = np.std(agent_speed[:, ped_id, :] * valid_frames[:, ped_id, :],axis=0) * 1.0  # np.sum(mask, axis=0)
                # print(" Std of speed shape "+str(tmp.shape)+" "+str(agent_speed.shape))
                stats_map["variance"][-1][ped_id, :] = [np.mean(speed_variance), np.std(speed_variance)]
        else:

            speed_variance = np.std(agent_speed * valid_frames,axis=0) * 1.0  # np.sum(mask, axis=0)

            # print(" Std of speed shape "+str(tmp.shape)+" "+str(agent_speed.shape))
            stats_map["variance"].append([np.mean(speed_variance), np.std(speed_variance)])
    else:

        stats_map["freq_most_freq_action"].append(np.zeros((valid_frames.shape[1])))
        stats_map["most_freq_action"].append(np.zeros((valid_frames.shape[1])))
        stats_map["entropy"].append(np.zeros((valid_frames.shape[1])))
        stats_map["prob_of_max_action"].append(np.zeros((valid_frames.shape[1])))
        stats_map["variance"].append(np.zeros((valid_frames.shape[1])))
        for ped_id in range(valid_frames.shape[1]):
            mean_x=agent_probabilities[:,ped_id,:,0]
            mean_y=agent_probabilities[:,ped_id,:,1]
            var=agent_probabilities[:,ped_id,:,3]
            stats_map["freq_most_freq_action"][-1][ped_id]=np.std(mean_x)
            stats_map["most_freq_action"][-1][ped_id]=np.std(mean_y)
            stats_map["entropy"][-1][ped_id]=np.mean(mean_x)
            stats_map["prob_of_max_action"][-1][ped_id]=np.mean(mean_y)
            stats_map["variance"][-1][ped_id]=np.mean(var)

    # Hit objects
    dist_global_temp=dist_global
    dist_global_temp[dist_global_temp==0]=1
    distance_travelled = np.sum(agent_measures[:,:, :, measures_indx.hit_obstacles]*valid_frames, axis=2)*1.0/(dist_global_temp*number_of_valid_frames)#*np.sum(mask, axis=1))
    if stats_of_individual_pedestrians:
        stats_map["hit_objs"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            stats_map["hit_objs"][-1][ped_id,:]=[np.mean(distance_travelled[:, ped_id]), np.std(distance_travelled[:, ped_id])]
    else:
        stats_map["hit_objs"].append([np.mean(distance_travelled), np.std(distance_travelled)])
    if car_case:
        # Entropy
        if stats_of_individual_pedestrians:
            stats_map["entropy"].append(np.zeros((valid_frames.shape[1], 2)))
            for ped_id in range(valid_frames.shape[1]):
                stats_map["entropy"][-1][ped_id,:]=[np.mean(agent_probabilities[:,ped_id,:]), np.std(agent_probabilities[:,ped_id,:])]
        else:
            stats_map["entropy"].append([np.mean(agent_probabilities),np.std(agent_probabilities)])
    else:

        hit_by_hero_car=np.sum(agent_measures[:,:,:, measures_indx.hit_by_hero_car]*valid_frames, axis=2)>0
        not_hit_by_hero_car=hit_by_hero_car==0

        calculate_entropy_of=agent_probabilities
        episode_mask=np.sum(valid_frames, axis=2)
        collision_mask=episode_mask*hit_by_hero_car
        non_collison_mask=episode_mask*not_hit_by_hero_car
        action_len = 9
        import scipy.stats
        #car_dist = agent_measures[:, 0, measures_indx.dist_to_agent]
        entropy_t = np.zeros(calculate_entropy_of.shape[0:2])
        for i, episode in enumerate(calculate_entropy_of):
            for ped_id, ped_episode in enumerate(episode):
                for f,frame in enumerate(ped_episode):
                    if valid_frames[i,ped_id,f]==1:
                        loca_entr=scipy.stats.entropy(frame[0:action_len])
                        if np.isnan(loca_entr)==False:
                            entropy_t[i,ped_id ] += loca_entr
                        #print(" Entropy " +str(entropy_t[i,ped_id ]))

        if print_values:
            for ped_id in range(valid_frames.shape[1]):
                print(" Entropy "+str(np.histogram(entropy_t[:, ped_id])))
                print("Entropy mean "+str(np.mean(entropy_t[:, ped_id]/np.sum(valid_frames[:, ped_id])))+" std "+str(np.std(entropy_t[:, ped_id]/np.sum(valid_frames[:, ped_id])))+ " previous mean "+str(np.mean(entropy_t[:, ped_id]/calculate_entropy_of.shape[1]))+" std "+str(np.std(entropy_t/calculate_entropy_of.shape[2])))
        if stats_of_individual_pedestrians:
            stats_map["entropy"].append(np.zeros((valid_frames.shape[1], 2)))
            for ped_id in range(valid_frames.shape[1]):

                stats_map["entropy"][-1][ped_id,:]=[np.mean(entropy_t[:, ped_id]/calculate_entropy_of.shape[2]), np.std(entropy_t[:, ped_id]/calculate_entropy_of.shape[2])]
        else:
            stats_map["entropy"].append([np.mean(entropy_t[:, :]/calculate_entropy_of.shape[2]), np.std(entropy_t[:, :]/calculate_entropy_of.shape[2])])
        collision_entropy=entropy_t[hit_by_hero_car]
        stats_map["entropy_collison"].append([np.mean(collision_entropy/calculate_entropy_of.shape[2]), np.std(collision_entropy/calculate_entropy_of.shape[2])])
        if print_values:
            for ped_id in range(valid_frames.shape[1]):
                print(" Collision  Entropy " + str(np.histogram(collision_entropy[:, ped_id])))
                print("Collision  Entropy mean " + str(np.mean(collision_entropy[:, ped_id] / np.sum(collision_mask[:, ped_id]))) + " std " + str(
                    np.std(collision_entropy[:, ped_id] / np.sum(collision_mask[:, ped_id]))) )
        #entropy.append([np.mean(entropy_t), np.std(entropy_t)])
        non_collision_entropy=entropy_t[not_hit_by_hero_car]
        stats_map["entropy_not_collison"].append([np.mean(non_collision_entropy / calculate_entropy_of.shape[2]),
                                              np.std(non_collision_entropy / calculate_entropy_of.shape[2])])

        if print_values:
            for ped_id in range(valid_frames.shape[1]):
                print("Non Collision  Entropy " + str(np.histogram(non_collision_entropy[:, ped_id])))
                print("Non Collision  Entropy mean " + str(np.mean(non_collision_entropy[:, ped_id] / np.sum(non_collison_mask[:, ped_id]))) + " std " + str(np.std(non_collision_entropy[:, ped_id] / np.sum(non_collison_mask[:, ped_id]))))

    import math

    distance_travelled = agent_measures[:,:, -1, measures_indx.distance_travelled_from_init] * 0.2

    if car_case:
        collision_with_pedestrian_mask = np.sum(agent_measures[:,:, :, measures_indx.hit_pedestrians] * valid_frames, axis=2) > 0
        if print_values:
            for ped_id in range(valid_frames.shape[1]):
                print("Distance travelled " + str(np.histogram(distance_travelled[:, ped_id])))
                print(" Distance travelled mean "+str(np.mean(distance_travelled[:, ped_id]))+" std "+str(np.std(distance_travelled[:, ped_id])))
        dist_travelled_by_collision=distance_travelled[hit_by_agent]#distance_travelled[collision_with_pedestrian_mask]
        if print_values:
            for ped_id in range(valid_frames.shape[1]):
                print("Distance travelled collisions: " + str(np.histogram(dist_travelled_by_collision[:, ped_id])))
                print(" Distance travelled collisions mean " + str(np.mean(dist_travelled_by_collision[:, ped_id])) + " std " + str(np.std(dist_travelled_by_collision[:, ped_id])))
        dist_travelled_not_by_collision=distance_travelled[hit_by_agent==0]#distance_travelled[collision_with_pedestrian_mask==0]
        if print_values:
            for ped_id in range(valid_frames.shape[1]):
                print("Distance travelled no collisions: " + str(np.histogram(dist_travelled_not_by_collision[:, ped_id])))
                print(" Distance travelled no collisions mean " + str(np.mean(dist_travelled_not_by_collision[:, ped_id])) + " std " + str(
                    np.std(dist_travelled_not_by_collision[:, ped_id])))
    else:
        dist_travelled_by_collision = distance_travelled[hit_by_hero_car]
        dist_travelled_not_by_collision = distance_travelled[hit_by_hero_car == 0]

    if stats_of_individual_pedestrians:
        stats_map["dist_travelled"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            stats_map["dist_travelled"][-1][ped_id, :]=[np.mean(dist_global[:, ped_id]), np.std(dist_global[:, ped_id])]
    else:
        stats_map["dist_travelled"].append( [np.mean(dist_global),np.std(dist_global)])
        stats_map["dist_travelled_collision"].append([np.mean(dist_travelled_by_collision), np.std(dist_travelled_by_collision)])
        stats_map["dist_travelled_not_collision"].append([np.mean(dist_travelled_not_by_collision), np.std(dist_travelled_not_by_collision)])

    # On pedestrian trajectory
    if stats_of_individual_pedestrians:
        stats_map["ped_traj"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            ped_traj = np.sum(agent_measures[:, ped_id,:, measures_indx.frequency_on_pedestrian_trajectory]*valid_frames[:,ped_id,:], axis=-1)/number_of_valid_frames[:,ped_id]#np.sum(mask, axis=1)
            stats_map["ped_traj"][-1][ped_id, :]=[np.mean(ped_traj), np.std(ped_traj)]
    else:

        ped_traj = np.sum(agent_measures[:, :, :,measures_indx.frequency_on_pedestrian_trajectory] * valid_frames[:, :, :]) / np.sum(number_of_valid_frames[:, :])  # np.sum(mask, axis=1)
        stats_map["ped_traj"].append([np.mean(ped_traj), np.std(ped_traj)])
        if not car_case:
            hit_by_hero_car_frame=np.tile(hit_by_hero_car[:,:,np.newaxis],(agent_measures.shape[2]))
            not_hit_by_hero_car_frame=hit_by_hero_car_frame==0
            ped_traj_collison = np.sum(
                agent_measures[:, :, :, measures_indx.frequency_on_pedestrian_trajectory] * valid_frames[:, :,
                                                                                            :])*hit_by_hero_car_frame / np.sum(valid_frames*hit_by_hero_car_frame)  # np.sum(mask, axis=1)

            ped_traj_not_collison = np.sum(
                agent_measures[:, :, :, measures_indx.frequency_on_pedestrian_trajectory] * valid_frames[:, :,
                                                                                            :]) * not_hit_by_hero_car_frame/ np.sum(valid_frames * not_hit_by_hero_car_frame)
            stats_map["ped_traj_collision"].append(
                [np.mean(ped_traj_collison), np.std(ped_traj_collison)])
            stats_map["ped_traj_not_collision"].append(
                [np.mean(ped_traj_not_collison), np.std(ped_traj_not_collison)])

    # Number of people hit
    if stats_of_individual_pedestrians:
        stats_map["people_hit"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            hit_pedestrians = np.sum(agent_measures[:, ped_id,:, measures_indx.hit_pedestrians]*valid_frames[:,ped_id,:], axis=-1)*1.0/number_of_valid_frames[:,ped_id]#np.sum(mask, axis=1)
            if car_case:
                collision_with_pedestrian_mask=np.sum(agent_measures[:, ped_id,:, measures_indx.hit_pedestrians]*valid_frames[:,ped_id,:], axis=-1)
                non_zeros=collision_with_pedestrian_mask>0
                only_non_zeros=collision_with_pedestrian_mask[non_zeros]
                if print_values:
                    print("Number of people hit "+str(sorted(hit_pedestrians[non_zeros]*number_of_valid_frames[non_zeros])))
                    print("Previous: Mean value "+str(np.mean(hit_pedestrians))+" std "+str(np.std(hit_pedestrians))+" New: non-zeros "+str(np.sum(non_zeros))+" of "+str(len(hit_pedestrians))+": "+str(np.sum(non_zeros)/len(hit_pedestrians))+" std of non-zeros "+str(np.std(only_non_zeros))+"  mean: "+str(np.mean(only_non_zeros)))
                #print(agent_measures[:, :, measures_indx.hit_pedestrians]*mask)
                non_zeros_new = (agent_measures[:, ped_id, :,measures_indx.hit_pedestrians]*valid_frames[:,ped_id,:]) > 0
                earliest_collisons=non_zeros_new.argmax(axis=-1)
                earliest_collisons_only=earliest_collisons[earliest_collisons>0]
                id_agent = agent_measures[:,ped_id, 0, measures_indx.id_closest_agent]

                if print_values:
                    print ("Earliest collision: mean "+str(np.mean(earliest_collisons_only))+" std "+str(np.std(earliest_collisons_only)))
                    print(" values "+str(sorted(earliest_collisons_only)))
                    print(" ID of closest pedestrian " + str(stats.mode(id_agent)) )
                stats_map["people_hit"][-1][ped_id, 0] = np.mean(np.sum(collision_with_pedestrian_mask))
                stats_map["people_hit"][-1][ped_id, 1] = np.std(np.sum(collision_with_pedestrian_mask))
            else:
                stats_map["people_hit"][-1][ped_id,0]=np.mean(hit_pedestrians)
                stats_map["people_hit"][-1][ped_id,1]=np.std(hit_pedestrians)
    else:
        number_of_episodes=valid_frames.shape[0]
        hit_pedestrians = np.sum(
            agent_measures[:, :, :, measures_indx.hit_pedestrians] * valid_frames[:, :, :],
            axis=-1) * 1.0 / number_of_valid_frames[:, :]  # np.sum(mask, axis=1)
        if car_case:
            collisions_before_mask=agent_measures[:, :, :, measures_indx.hit_pedestrians]
            collisions=collisions_before_mask * valid_frames[:, :, :]
            collision_with_pedestrian_per_agent_per_episode = np.sum(collisions,axis=-1)
            collision_with_pedestrian_total_all_agents_per_episode = np.sum(collision_with_pedestrian_per_agent_per_episode,axis=1)
            non_zeros = collision_with_pedestrian_total_all_agents_per_episode > 0
            only_non_zeros = collision_with_pedestrian_total_all_agents_per_episode[non_zeros]
            if print_values:
                print("Number of people hit " + str(
                    sorted(hit_pedestrians[non_zeros] * number_of_valid_frames[non_zeros])))
                print("Previous: Mean value " + str(np.mean(hit_pedestrians)) + " std " + str(
                    np.std(hit_pedestrians)) + " New: non-zeros " + str(np.sum(non_zeros)) + " of " + str(
                    len(hit_pedestrians)) + ": " + str(
                    np.sum(non_zeros) / len(hit_pedestrians)) + " std of non-zeros " + str(
                    np.std(only_non_zeros)) + "  mean: " + str(np.mean(only_non_zeros)))
            # print(agent_measures[:, :, measures_indx.hit_pedestrians]*mask)
            non_zeros_new = (agent_measures[:, :, :, measures_indx.hit_pedestrians] * valid_frames[
                                                                                                      :, :,
                                                                                                      :]) > 0
            earliest_collisons = non_zeros_new.argmax(axis=-1)
            earliest_collisons_only = earliest_collisons[earliest_collisons > 0]
            id_agent = agent_measures[:, :, 0, measures_indx.id_closest_agent]

            if print_values:
                print("Earliest collision: mean " + str(np.mean(earliest_collisons_only)) + " std " + str(
                    np.std(earliest_collisons_only)))
                print(" values " + str(sorted(earliest_collisons_only)))
                print(" ID of closest pedestrian " + str(stats.mode(id_agent)))
            stats_map["people_hit"].append([np.mean(collision_with_pedestrian_total_all_agents_per_episode), np.std(collision_with_pedestrian_total_all_agents_per_episode)])
        else:
            stats_map["people_hit"].append([np.mean(hit_pedestrians),np.std(hit_pedestrians)])

    if stats_of_individual_pedestrians:
        if car_case:
            stats_map["collision_between_car_and_agent"].append(np.zeros((valid_frames.shape[1], 2)))
            for ped_id in range(valid_frames.shape[1]):
                distance_travelled = np.sum(agent_measures[:, ped_id,:, measures_indx.hit_by_agent]*valid_frames[:,ped_id,:], axis=-1)#*1.0/number_of_valid_frames[:, ped_id]
                stats_map["collision_between_car_and_agent"][-1][ped_id, :]=[np.mean(distance_travelled), np.std(distance_travelled)]
        else:
            stats_map["collision_between_car_and_agent"].append(np.zeros((valid_frames.shape[1], 2)))
            for ped_id in range(valid_frames.shape[1]):
                distance_travelled = np.sum(agent_measures[:, ped_id, :, measures_indx.hit_by_hero_car] * valid_frames[:, ped_id, :],
                             axis=-1) #* 1.0 / number_of_valid_frames[:, ped_id]
                stats_map["collision_between_car_and_agent"][-1][ped_id, :] = [np.mean(distance_travelled), np.std(distance_travelled)]
    else:
        if car_case:
            collisions_between_agents=agent_measures[:, :,:, measures_indx.hit_by_agent]*valid_frames[:,:,:]
            collisions_between_agents_per_agent_per_episode=np.sum(collisions_between_agents, axis=-1)
            collisions_between_agents_total_agent_per_episode = np.sum(collisions_between_agents_per_agent_per_episode, axis=1)

            stats_map["collision_between_car_and_agent"].append([np.mean(collisions_between_agents_total_agent_per_episode), np.std(collisions_between_agents_total_agent_per_episode)])
        else:
            collisions_between_agents =agent_measures[:, :, :, measures_indx.hit_by_hero_car] * valid_frames[:, :, :]
            collisions_between_agents_per_agent_per_episode = np.sum(collisions_between_agents, axis=-1)
            collisions_between_agents_total_agent_per_episode = np.sum(collisions_between_agents_per_agent_per_episode, axis=1)
            stats_map["collision_between_car_and_agent"].append([np.mean(collisions_between_agents_total_agent_per_episode), np.std(collisions_between_agents_total_agent_per_episode)])
    # Normalized distance
    if agent_measures.shape[-1] >= 13:
        non_zero_distance = dist_global > 0
        norm_dist = np.where(non_zero_distance, np.divide(dist_travelled_tot, dist_global * 5), 0)
        # print norm_dist
        if stats_of_individual_pedestrians:
            stats_map["norm_dist_travelled"].append(np.zeros((valid_frames.shape[1], 2)))
            for ped_id in range(valid_frames.shape[1]):
                stats_map["norm_dist_travelled"][-1][ped_id, :]=[np.nanmean(norm_dist[:,ped_id]), np.nanstd(norm_dist[:,ped_id])]
        else:
            stats_map["norm_dist_travelled"].append([np.nanmean(norm_dist[:, :]),   np.nanstd(norm_dist[:, :])])
    # Actions
    if agent_measures.shape[-1] >= 12:
        if stats_of_individual_pedestrians:
            stats_map["nbr_direction_switches"].append(np.zeros((valid_frames.shape[1], 2)))
            for ped_id in range(valid_frames.shape[1]):
                num_switches = np.sum(agent_measures[:, ped_id, :,measures_indx.change_in_direction]*valid_frames[:,ped_id,:], axis=-1)*1/number_of_valid_frames[:,ped_id]#np.sum(mask, axis=1)
                stats_map["nbr_direction_switches"][-1][ped_id, :]=[np.mean(num_switches), np.std(num_switches)]
        else:
            num_switches = np.sum(
                agent_measures[:, :, :, measures_indx.change_in_direction] * valid_frames[:, :, :],
                axis=-1) * 1 / number_of_valid_frames[:, :]  # np.sum(mask, axis=1)
            stats_map["nbr_direction_switches"].append([np.mean(num_switches), np.std(num_switches)])
    # car hit
    car_hits = agent_measures[:, :, :,measures_indx.hit_by_car]*valid_frames
    car_hits=np.where(car_hits>0, 1, car_hits)

    number_of_episodes=valid_frames.shape[0]
    if stats_of_individual_pedestrians:
        stats_map["time_to_collision"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            timestep_to_collision = np.nonzero(car_hits[:,ped_id,:])[1]
            stats_map["time_to_collision"][-1][ped_id, :]=[np.mean(timestep_to_collision), np.std(timestep_to_collision)]

        stats_map["car_hit"].append(np.zeros(valid_frames.shape[1]))
        for ped_id in range(valid_frames.shape[1]):
            stats_map["car_hit"][-1][ped_id]=np.sum(car_hits[:,ped_id])*1.0/number_of_episodes
        if not car_case:
            stats_map["dist_to_car"].append(np.zeros((valid_frames.shape[1],2)))
            stats_map["init_dist_to_car"].append(np.zeros((valid_frames.shape[1],2)))
        for ped_id in range(valid_frames.shape[1]):
            if not car_case:
                car_dist = np.reciprocal(agent_measures[:, ped_id, :, measures_indx.inverse_dist_to_closest_car],
                                         where=agent_measures[:, ped_id, :,
                                               measures_indx.inverse_dist_to_closest_car] > 0) * valid_frames
            else:
                car_dist = agent_measures[:, ped_id, :, measures_indx.dist_to_closest_pedestrian]  * valid_frames
                cov = np.cov(np.concatenate([car_dist.reshape([1, -1]), agent_speed.reshape([1, -1])], axis=0))
                stats_map["speed_to_dist_correlation"].append(cov[1, 0])
            stats_map["dist_to_car"][-1][ped_id, :] = [np.mean(car_dist), np.std(car_dist)]
            stats_map["init_dist_to_car"][-1][ped_id, :] = [np.mean(car_dist[:, 0]), np.std(car_dist[:, 0])]
    else:
        if car_case:
            people_hit=agent_measures[:, :, :,measures_indx.hit_pedestrians]*valid_frames

            people_hit = np.where(people_hit > 0, 1, people_hit)
            timestep_to_collision = np.nonzero( people_hit)[1]
        else:
            timestep_to_collision = np.nonzero(car_hits)[1]
        stats_map["time_to_collision"].append([np.mean(timestep_to_collision), np.std(timestep_to_collision)])
        if not car_case:
            total_car_hits_per_agent_per_episode = np.sum(car_hits, axis=2)
            total_car_hits_for_all_agents_per_episode=np.sum(total_car_hits_per_agent_per_episode, axis=1)
            stats_map["car_hit"].append([np.mean(total_car_hits_for_all_agents_per_episode), np.std(total_car_hits_for_all_agents_per_episode)])
        else:
            total_car_hits_per_agent_per_episode = np.sum(car_hits, axis=2)*(1.0/np.sum(valid_frames, axis=-1))
            total_car_hits_for_all_agents_per_episode = np.sum(total_car_hits_per_agent_per_episode, axis=1)*(1.0/valid_frames.shape[1])
            stats_map["car_hit"].append(
                [np.mean(total_car_hits_for_all_agents_per_episode), np.std(total_car_hits_for_all_agents_per_episode)])
        if not car_case:
            car_dist = np.reciprocal(agent_measures[:, :, :, measures_indx.inverse_dist_to_closest_car], where=agent_measures[:, :, :, measures_indx.inverse_dist_to_closest_car]>0) * valid_frames
        else:
            car_dist = agent_measures[:, :, :,measures_indx.dist_to_closest_pedestrian] * valid_frames
            cov = np.cov(np.concatenate([car_dist.reshape([1, -1]), agent_speed.reshape([1, -1])], axis=0))
            stats_map["speed_to_dist_correlation"].append(cov[1, 0])
        stats_map["dist_to_car"].append([np.mean(car_dist), np.std(car_dist)])
        stats_map["init_dist_to_car"].append([np.mean(car_dist[:, 0]), np.std(car_dist[:, 0])])

    # Pavement
    pavement_t = agent_measures[:, :, :, measures_indx.iou_pavement] * valid_frames
    pavement_t[pavement_t > 0] = 1

    if stats_of_individual_pedestrians:
        stats_map["pavement"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            freq_on_pavement = np.sum(pavement_t[:,ped_id, :], axis=-1)*1.0/number_of_valid_frames[:,ped_id]#np.sum(mask, axis=1)
            stats_map["pavement"][-1][ped_id, :]=[np.mean(freq_on_pavement), np.std(freq_on_pavement)]
    else:
        freq_on_pavement = np.sum(pavement_t, axis=-1) * 1.0 / number_of_valid_frames  # np.sum(mask, axis=1)
        stats_map["pavement"].append( [np.mean(freq_on_pavement), np.std(freq_on_pavement)])
    # Heatmap
    if stats_of_individual_pedestrians:
        stats_map["people_heatmap"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            people_heatmap_local = agent_measures[:, ped_id,:, measures_indx.pedestrian_heatmap]*valid_frames[:,ped_id, :]
            people_heatmap_local[np.isnan(people_heatmap_local)] = 0
            freq_on_heatmap = np.sum(people_heatmap_local, axis=-1)*1.0/number_of_valid_frames[:,ped_id]#np.sum(mask, axis=1)
            stats_map["people_heatmap"][-1][ped_id, :]=[np.mean(freq_on_heatmap), np.std(freq_on_heatmap)]
    else:

        people_heatmap_local = agent_measures[:, :, :,measures_indx.pedestrian_heatmap] * valid_frames
        people_heatmap_local[np.isnan(people_heatmap_local)]=0
        freq_on_heatmap = np.sum(people_heatmap_local, axis=-1) * 1.0 / number_of_valid_frames  # np.sum(mask, axis=1)
        stats_map["people_heatmap"].append([np.mean(freq_on_heatmap), np.std(freq_on_heatmap)])
    # Average speed

    if stats_of_individual_pedestrians:
        stats_map["speed_mean"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            avg_speed = np.mean(agent_speed[:,ped_id, :] * valid_frames[:,ped_id, :], axis=-1) * 1.0 /number_of_valid_frames[:,ped_id]# np.sum(mask, axis=0)
            stats_map["speed_mean"][-1][ped_id, :]=[np.mean(avg_speed), np.std(avg_speed)]
    else:
        avg_speed = np.mean(agent_speed[:, :, :] * valid_frames[:, :, :],axis=-1) * 1.0 / number_of_valid_frames[:, :]  # np.sum(mask, axis=0)
        stats_map["speed_mean"].append([np.mean(avg_speed), np.std(avg_speed)])
    # Per trajectory std speed
    if stats_of_individual_pedestrians:
        stats_map["speed_var"].append(np.zeros((valid_frames.shape[1], 2)))
        for ped_id in range(valid_frames.shape[1]):
            speed_variance = np.std(agent_speed[:,ped_id, :] * valid_frames[:,ped_id, :], axis=-1) * 1.0 /number_of_valid_frames[:,ped_id]# np.sum(mask, axis=0)
            #print(" Std of speed shape "+str(tmp.shape)+" "+str(agent_speed.shape))
            stats_map["speed_var"][-1][ped_id, :]=[np.mean(speed_variance), np.std(speed_variance)]
    else:
        speed_variance = np.std(agent_speed[:, :, :] * valid_frames[:, :, :], axis=-1) * 1.0 / number_of_valid_frames[:, :]  # np.sum(mask, axis=0)
        # print(" Std of speed shape "+str(tmp.shape)+" "+str(agent_speed.shape))
        stats_map["speed_var"].append([np.mean(speed_variance), np.std(speed_variance)])