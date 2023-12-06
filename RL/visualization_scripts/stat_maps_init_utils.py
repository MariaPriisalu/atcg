from RL.settings import STATISTICS_INDX_CAR_INIT, STATISTICS_INDX_MAP, STATISTICS_INDX_MAP_STAT, PEDESTRIAN_INITIALIZATION_CODE

import numpy as np
import scipy


def initialize_init_map(stats_map):
    init_map=stats_map[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]

    init_map["cars_vel_x"] = []
    init_map["cars_vel_y"] = []
    init_map["cars_vel_speed"] = []
    init_map["cars_pos_x"] = []
    init_map["cars_pos_y"] = []
    init_map["iterations"] = []

    for add_on in [ "_init", "_car","_goal"]:
        init_map["position_mode" + add_on]=[]  # To Do: reshape to matrix index here!
        init_map["prior_mode" + add_on]=[]
        init_map["entropy_maps" + add_on]=[]
        init_map["entropy_prior_maps" + add_on]=[]
        init_map["kl_to_prior_maps" + add_on]=[]
        init_map["diff_to_prior_maps" + add_on]=[]
    stats_map[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]=init_map
    return stats_map

def choose_file_path_for_init_stats( data_set, files_map, goal_case=False, car_case=False):
    if data_set == "train":
        files =files_map.init_maps
        if car_case:
            files_stats = files_map.init_maps_stats_car
        elif goal_case:
            files_stats = files_map.init_maps_stats_goal
        else:
            files_stats = files_map.init_map_stats
        files_cars=  files_map.init_cars
        iterations=files_map.train_itr
    else:
        files_cars= files_map.init_cars_test
        if car_case:
            files_stats = files_map.init_maps_stats_car_test
        elif goal_case:
            files_stats = files_map.init_maps_stats_goal_test
        else:
            files_stats = files_map.init_map_stats_test
        files = files_map.init_maps_test
        iterations = files_map.test_points
    return files,files_stats,files_cars, iterations

def get_init_stats(stats_map, files_map, settings_map, data_set="train", goal_case=False, car_case=False):

    stats_map = initialize_init_map(stats_map)
    init_map = stats_map[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]

    files,files_stats,files_cars, iterations=choose_file_path_for_init_stats(data_set, files_map, goal_case,car_case)
    files=sorted(files)
    if goal_case==False and car_case==False:
        files_cars = sorted(files_cars)
        if (len(files) < 1 and len(files_stats) <= 0) or len(files_cars) <= 0:
            # print ("-----------------Return empty stat maps!" )
            return stats_map
        for j, pair in enumerate(files_cars):
            init_map["iterations"].append(iterations[pair[0]])
            car_test = np.load(pair[1])
            init_map["cars_vel_x"].append((np.mean(car_test[:,:,STATISTICS_INDX_CAR_INIT.car_vel[0]]), np.std(car_test[:,:, STATISTICS_INDX_CAR_INIT.car_vel[0]])))
            init_map["cars_vel_y"].append((np.mean(car_test[:,:, STATISTICS_INDX_CAR_INIT.car_vel[1]-1]), np.std(car_test[:,:, STATISTICS_INDX_CAR_INIT.car_vel[1]-1])))
            speed=np.linalg.norm(car_test[:, :, STATISTICS_INDX_CAR_INIT.car_vel[0]:STATISTICS_INDX_CAR_INIT.car_vel[1]], axis=2)*.2
            init_map["cars_vel_speed"].append((np.mean(speed), np.std(speed)))
            init_map["cars_pos_x"].append((np.mean(car_test[:,:, STATISTICS_INDX_CAR_INIT.car_pos[0]]), np.std(car_test[:,:, STATISTICS_INDX_CAR_INIT.car_pos[1]])))
            init_map["cars_pos_y"].append((np.mean(car_test[:, :, STATISTICS_INDX_CAR_INIT.car_pos[1]-1]), np.std(car_test[:,:,  STATISTICS_INDX_CAR_INIT.car_pos[1]-1])))
            manual_goal=car_test[:, :, STATISTICS_INDX_CAR_INIT.manual_goal[0]:STATISTICS_INDX_CAR_INIT.manual_goal[1]]
            if settings_map.gaussian:
                init_map["prior_mode_goal"].append(np.mean(manual_goal, axis=0))

    settings_map.gaussian=False

    iteration_list=files
    use_stats = False
    if len(files_stats)>0:
        print("Use stats!")
        use_stats=True
        iteration_list=sorted(files_stats)

    add_on="_init"
    if car_case:
        add_on="_car"
    elif goal_case:
        add_on = "_goal"
    for j, pair in enumerate(iteration_list):
        if not use_stats:
            #print (" Do not use stats")
            map_test = np.load(pair[1])
            distr=map_test[:,:,:, STATISTICS_INDX_MAP.init_distribution]
            prior = map_test[:, :,:,  STATISTICS_INDX_MAP.prior]
            entropys=[]
            entropys_prior = []
            kls=[]
            if settings_map.gaussian:
                init_map["goal_position_mode"].append(np.mean(distr_goal[:, :2], axis=0))
            else:
                for ep_itr in range(distr.shape[0]):
                    entropys.append([])
                    entropys_prior.append([])
                    kls.append([])
                    for id in range(distr.shape[1]):
                        entropys[-1].append(scipy.stats.entropy(distr[ep_itr,id, :]))
                        entropys_prior[-1].append(scipy.stats.entropy(prior[ep_itr,id,  :]))
                        kls[-1].append(scipy.stats.entropy(distr[ep_itr,:],qk=prior[ep_itr,id,:] ))
                        goal_mode[-1].append(np.unravel_index(np.argmax(product_goal[ep_itr,id, :]), settings_map.env_shape[1:]))
                        goal_prior[-1].append(np.unravel_index(np.argmax(prior_goal[ep_itr,id, :]), settings_map.env_shape[1:]))
            product=distr*prior
            init_mode = []
            init_prior = []
            for ep_itr in range(product.shape[0]):
                init_mode.append([])
                init_prior.append([])
                for id in range(distr.shape[1]):
                    init_mode[-1].append(np.unravel_index(np.argmax(product[ep_itr,id, :]), settings_map.env_shape[1:]))
                    init_prior[-1].append(np.unravel_index(np.argmax(prior[ep_itr,id, :]), settings_map.env_shape[1:]))

            diff_to_prior = np.sum(np.abs(distr - prior), axis=2)

        else:

            initialization_map_stats=np.load(pair[1])
            init_mode=initialization_map_stats[:,:,STATISTICS_INDX_MAP_STAT.init_position_mode[0]:STATISTICS_INDX_MAP_STAT.init_position_mode[1]]
            init_prior=initialization_map_stats[:,:,STATISTICS_INDX_MAP_STAT.init_prior_mode[0]:STATISTICS_INDX_MAP_STAT.init_prior_mode[1]]
            entropys=initialization_map_stats[:,:, STATISTICS_INDX_MAP_STAT.entropy]
            entropys_prior=initialization_map_stats[:,:, STATISTICS_INDX_MAP_STAT.entropy_prior]
            kls=initialization_map_stats[:,:, STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior]
            diff_to_prior=initialization_map_stats[:,:, STATISTICS_INDX_MAP_STAT.prior_init_difference]

        init_map["position_mode"+add_on].append(np.mean(init_mode, axis=0))  # To Do: reshape to matrix index here!
        init_map["prior_mode"+add_on].append(np.mean(init_prior, axis=0))
        init_map["entropy_maps"+add_on].append((np.mean(entropys), np.std(entropys)))
        init_map["entropy_prior_maps"+add_on].append((np.mean(entropys_prior), np.std(entropys_prior)))
        init_map["kl_to_prior_maps"+add_on].append((np.mean(kls), np.std(kls)))
        init_map["diff_to_prior_maps"+add_on].append((np.mean(diff_to_prior), np.std(diff_to_prior)))
    stats_map[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization] = init_map
    return stats_map#entropy_maps_test, entropy_prior_maps_test, kl_to_prior_maps_test, diff_to_prior_maps_test, cars_vel_test_x,cars_vel_test_y,cars_vel_test_speed, cars_pos_test_x,cars_pos_test_y, iterations,entropy_maps_test_goal, entropy_prior_maps_test_goal, kl_to_prior_maps_test_goal,diff_to_prior_maps_test_goal, goal_position_mode,goal_prior_mode, settings_map.gaussian, init_position_mode, init_prior_mode


