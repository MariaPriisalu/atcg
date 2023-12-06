import numpy as np
from dotmap import DotMap

def get_cars_stats(filenames_itr_cars, test_files_cars, sz_train):
    car_stats_map = initialize_car_map(len(filenames_itr_cars), sz_train)
    indx_test = 0
    indx_same = 0
    prev_itr = 0
    test_points_car = []
    if len(test_files_cars) > 0:
        sz_test_car = np.load(test_files_cars[0][1]).shape
        car_stats_map.statistics_test_car = np.zeros((len(test_files_cars) * sz_test_car[0], sz_test_car[1], sz_test_car[2]))
        for j, pair in enumerate(sorted(test_files_cars)):
            cur_statistics = np.load(pair[1])
            car_stats_map.statistics_test_car[indx_test * sz_test_car[0]:(indx_test + 1) * sz_test_car[0],:, :, :] = cur_statistics
            car_stats_map.test_itr_car.append(pair[0])
            if indx_same == pair[0]:
                prev_itr += 1
            else:
                prev_itr = 0
                test_points_car.append(j)
            indx_same = pair[0]

            car_stats_map.test_itr_car[indx_test * sz_test_car[0]:(indx_test + 1) * sz_test_car[0]] = list(range(
                pair[0] + prev_itr * sz_test_car[0], pair[0] + (prev_itr + 1) * sz_test_car[0]))
            indx_test += 1

    for i, pair in enumerate(sorted(filenames_itr_cars)):
        try:

            cur_statistics = np.load(pair[1])

            car_stats_map.statistics_car[car_stats_map.indx_train_car * sz_train[0]:(car_stats_map.indx_train_car + 1) * sz_train[0],:, :, :] = cur_statistics[:,:,
                                                                                                    0: sz_train[1], :]
            car_stats_map.indx_train_car += 1

        except IOError:
            print("Could not load " + pair[1])


    return car_stats_map


def initialize_car_map(number_of_datapoints, sz_train):
    car_stats_map = DotMap()
    car_stats_map.test_itr_car = []
    if number_of_datapoints>0:
        car_stats_map.statistics_car = np.zeros(( number_of_datapoints* sz_train[0], sz_train[1], sz_train[2], sz_train[3]))
    else:
        car_stats_map.statistics_car=[]
    car_stats_map.statistics_test_car = []
    car_stats_map.indx_train_car = 0
    return car_stats_map
