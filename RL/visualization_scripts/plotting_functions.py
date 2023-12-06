from RL.settings import PEDESTRIAN_INITIALIZATION_CODE
from RL.visualization_scripts.visualization_utils import trendline

import sys, os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_separately(stats_map, stats_map_test, settings_map, metric_name,title,filename,  plot_train=False, axis=0):
    print(metric_name)
    data_outer = []
    names_outer = []

    indx=1
    number_non_zero_inits=0

    for init_m in range(10):
        if len(stats_map[init_m][metric_name]) > 1:
            number_non_zero_inits=number_non_zero_inits+1

    number_of_plots=1
    allowed_inits=[{settings_map.general_id, 1,3, 5, 6,2,4,9, 10}]
    if number_non_zero_inits>4:
        number_of_plots=2
        allowed_inits = [{2,3,6,5}, {1,8,9,settings_map.general_id}]

    for plot_number in range(number_of_plots):
        fig = plt.figure(figsize=(8.0, 5.0))
        number_of_rows = 2
        number_of_columns = 2
        inits_in_plot = 0
        for init_m in allowed_inits[plot_number]:
            if len(stats_map[init_m][metric_name]) > 1:
                inits_in_plot = inits_in_plot + 1
        if inits_in_plot<4:
            number_of_rows = 1
            number_of_columns = inits_in_plot


        for init_m in stats_map.keys():
            data = []

            min_y = sys.maxsize
            max_y = -sys.maxsize - 1

            if len(stats_map[init_m][metric_name]) > 1  and (init_m in allowed_inits[plot_number] or settings_map.temporal_case) and indx<5:#10}
                if plot_train and len(stats_map[init_m][metric_name])>0:
                    ax = plt.subplot(number_of_rows, number_of_columns, indx)
                    stacked_reward_train = np.stack(stats_map[init_m][metric_name], axis=0)
                    if settings_map.stats_of_individual_pedestrians :
                        number_of_agents = stacked_reward_train.shape[1]

                    else:
                        number_of_agents = 1

                        if len(stacked_reward_train.shape) == 2 and stacked_reward_train.shape[-1] == 2:

                            stacked_reward_train = stacked_reward_train[:, axis]


                    if len(stacked_reward_train.shape) == 3 and stacked_reward_train.shape[-1] == 2:
                        stacked_reward_train = stacked_reward_train[:,:, axis]

                    if (min(stacked_reward_train.flatten()) < min_y):
                        min_y = min(stacked_reward_train.flatten())
                    if (max(stacked_reward_train.flatten()) > max_y):
                        max_y = max(stacked_reward_train.flatten())


                    for agent_id in range(number_of_agents):
                        local_data= stacked_reward_train
                        if settings_map.stats_of_individual_pedestrians:
                            local_data= stacked_reward_train[:,agent_id]


                        data1= ax.scatter(np.array(stats_map[init_m]["plot_iterations"])*0.01, local_data, alpha=0.4, label='training data agent'+str(agent_id))

                        data.append(data1)

                        if len(data_outer)<number_of_agents:
                            data_outer.append(data1)

                            names_outer.append("train "+str(agent_id))  # 'test data ' + str(init_m))
                    trend_data=stacked_reward_train
                    if settings_map.stats_of_individual_pedestrians:
                        trend_data = np.mean(stacked_reward_train, axis=1)
                    data3, = ax.plot(np.array(stats_map[init_m]["plot_iterations"]) * 0.01, trendline(trend_data), label='training avg agent' )
                    if len(data_outer) < number_of_agents+1:
                        data_outer.append(data3)
                        names_outer.append("train avg" )  # 'test avg.' + str(init_m))
                if len(stats_map_test[init_m][metric_name]) > 1:

                    stacked_reward = np.stack(stats_map_test[init_m][metric_name], axis=0)


                    if settings_map.stats_of_individual_pedestrians:
                        number_of_agents = stacked_reward.shape[1]
                    else:

                        number_of_agents = 1
                        if len(stacked_reward.shape) == 2 and stacked_reward.shape[-1] == 2:
                            stacked_reward = stacked_reward[:,axis]


                    if len(stacked_reward.shape) == 3 and stacked_reward.shape[-1] == 2:
                        stacked_reward = stacked_reward[:,:, axis]
                    if (min(stacked_reward.flatten()) < min_y):
                        min_y = min(stacked_reward.flatten())
                    if (max(stacked_reward.flatten()) > max_y):
                        max_y = max(stacked_reward.flatten())

                    for agent_id in range(number_of_agents):
                        local_data = stacked_reward
                        if settings_map.stats_of_individual_pedestrians:
                            local_data = stacked_reward[:,agent_id]
                        data2, = ax.plot(np.array(stats_map_test[init_m]["plot_iterations"])*0.01, local_data,marker="x", label='test data'+str(agent_id))
                        data.append(data2)
                        if len(data_outer)<number_of_agents*2+2 and plot_train and len(stats_map_test[init_m][metric_name]) > 1:
                            data_outer.append(data2)
                            names_outer.append("test "+str(agent_id))  # 'test data ' + str(init_m))
                    trend_data=stacked_reward
                    if settings_map.stats_of_individual_pedestrians:
                        trend_data =trendline(np.mean(stacked_reward[:], axis=1))
                    data4, = ax.plot(np.array(stats_map_test[init_m]["plot_iterations"]) * 0.01, trendline(trend_data), label='test avg' + str(agent_id))
                    if len(data_outer) < number_of_agents*2+2 and plot_train and len(stats_map_test[init_m][metric_name]) > 1:
                        data_outer.append(data4)
                        names_outer.append("test avg")  # 'test avg.' + str(init_m))

                ax.set_title(settings_map.init_names[init_m])

                indx=indx+1
                if number_of_plots ==2 and indx==5:
                    indx=1
        plt.subplots_adjust(left=0.15, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=0.25)


        fig.text(0.5, 0.04, 'Epoch', ha='center', fontdict={'size':'large'})
        fig.text(0.04, 0.5, title, va='center', rotation='vertical', fontdict={'size':'large'})
        fig.legend(data_outer, names_outer, loc='lower right', shadow=True)#, ncol=2)

        if settings_map.save_plots and settings_map.save_regular_plots:
            #print( plot_number)
            print( "Saved  " + settings_map.timestamp + filename)
            fig.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp+"_"+str(plot_number)+"_"+ filename))
            plt.close('all')
        else:
            plt.show()


def plot_goal(stats_map_training, stats_map_test, settings_map, key_name,avg_key_name, prior_key_name,title, filename, plot_train=False, axis=0, gaussian=False):
    data_outer = []
    names_outer = []

    for id in range(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name][-1].shape[1]):
        fig = plt.figure(figsize=(8.0, 5.0))
        number_of_rows = 1
        number_of_columns =2

        data = []

        indx=0
        if len(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name]) > 0:

            nbr_train_itrs=stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name][-1].shape[0]
            stacked_pos_train = np.stack(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name], axis=0)
            stacked_pos_train = np.reshape(stacked_pos_train[:,:,id,:], (-1, 2))
            avg_goal_location = np.stack(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][avg_key_name], axis=0)
            prior_goal_location=np.stack(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][prior_key_name], axis=0)

            if settings_map.gaussian:
                avg_goal_location =avg_goal_location +prior_goal_location
        if len(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name]) > 1:
            stacked_pos = np.vstack(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name])#, axis=0)
            stacked_pos = np.reshape(stacked_pos[:,id,:], (-1, 2))
            nbr_test_itrs = stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name][-1].shape[
                0]  # stacked_pos.shape[0] / len(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][avg_key_name])

            nbr_test_repeat =len(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"]) #stacked_pos.shape[0] / len(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][avg_key_name])
            avg_goal_location_test = np.stack(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][avg_key_name], axis=0)
            prior_goal_location_test = np.stack(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][prior_key_name], axis=0)
            if avg_goal_location_test.shape[0]!=nbr_test_repeat:
                avg_goal_location_test_copy=np.zeros((nbr_test_repeat,avg_goal_location_test.shape[1] ,avg_goal_location_test.shape[2]))
                number_of_points_in_batch= int(avg_goal_location_test.shape[0]/nbr_test_repeat)
                for i in range(nbr_test_repeat):
                    avg_goal_location_test_copy[i,:,:]=np.mean(avg_goal_location_test[i*number_of_points_in_batch:(i+1)*number_of_points_in_batch,:,:],axis=0)
                avg_goal_location_test=avg_goal_location_test_copy
            if prior_goal_location_test.shape[0] != nbr_test_repeat:
                prior_goal_location_test_copy = np.zeros(
                    (nbr_test_repeat, prior_goal_location_test.shape[1], prior_goal_location_test.shape[2]))
                number_of_points_in_batch = int(prior_goal_location_test.shape[0] / nbr_test_repeat)
                for i in range(nbr_test_repeat):
                    prior_goal_location_test_copy[i, :, :] = np.mean(
                        prior_goal_location_test[i * number_of_points_in_batch:i * number_of_points_in_batch, :, :],
                        axis=0)
                prior_goal_location_test = prior_goal_location_test_copy

            if settings_map.gaussian:
                avg_goal_location_test = avg_goal_location_test + prior_goal_location_test



        #print ("Inits  " +str(allowed_inits[plot_number]))
        for indx in range(2):
            if len(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name]) > 1:
                ax = plt.subplot(number_of_rows, number_of_columns, indx+1)
                if  len(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name])>0:

                    data1= ax.scatter(np.repeat(np.array(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"])*0.01,nbr_train_itrs), stacked_pos_train[:,indx], alpha=0.4, label='goal locations')
                    data3= ax.scatter(np.array(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"])*0.01, avg_goal_location[:,id,indx], alpha=0.4, label='average goal')
                    data5= ax.scatter(np.array(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"]) * 0.01, prior_goal_location[:,id,indx], alpha=0.4, label='prior')
                    data.append(data1)
                    data.append(data3)
                    data.append(data5)
                if len(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name]) > 1:
                    data2= ax.scatter(np.repeat(np.array(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"])*0.01, nbr_test_itrs), stacked_pos[:,indx], alpha=0.4, label='test data')

                    data4, = ax.plot(np.array(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"]) * 0.01, avg_goal_location_test[:,id,indx], label='test average goal')
                    data6= ax.scatter(np.array(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"]) * 0.01, prior_goal_location_test[:,id,indx], alpha=0.4, label='test prior')
                    data.append(data2)
                    data.append(data4)
                    data.append(data6)


                if len(data_outer)==0:
                    data_outer.append(data1)
                    data_outer.append(data3)
                    data_outer.append(data5)
                    names_outer.append("train")  # 'test data ' + str(init_m))
                    names_outer.append("train mean")  # 'test avg.' + str(init_m))
                    names_outer.append("train prior")  # 'test avg.' + str(init_m))

                if len(data_outer)==3 and len(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key_name]) > 1:
                    #print ("Added tag")
                    data_outer.append(data2)
                    data_outer.append(data4)
                    data_outer.append(data6)
                    names_outer.append("test")  # 'test data ' + str(init_m))
                    names_outer.append("test mean")  # 'test avg.' + str(init_m))
                    names_outer.append("test prior")  # 'test avg.' + str(init_m))
        plt.subplots_adjust(left=0.15, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=0.25)


        fig.text(0.5, 0.04, 'Epoch', ha='center', fontdict={'size':'large'})
        fig.text(0.04, 0.5, title, va='center', rotation='vertical', fontdict={'size':'large'})
        fig.legend(data_outer, names_outer, loc='lower right', shadow=True, ncol=2)

        if settings_map.save_plots and settings_map.save_regular_plots:

            print( "Saved  " + settings_map.timestamp + filename)
            fig.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp+"_"+ filename[0: len(".png")]+str(id)+ ".png"))
        else:
            plt.show()
        #plot_separately_cars(train_itr, avg_reward, test_itr, avg_reward_test, title, "cars_"+filename, plot_train=plot_train)


def plot_loss(stats_map_training, stats_map_test, settings_map, key,title, filename, plot_train=False):
    data_outer = []
    names_outer = []
    fig = plt.figure(figsize=(8.0, 5.0))
    indx=1


    data = []

    min_y = sys.maxsize
    max_y = -sys.maxsize - 1

    if len(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key]) > 1:
        stacked_reward = np.stack(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key], axis=0)
        nbr_test_itrs=len(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"])
        if stacked_reward.shape[0] != nbr_test_itrs:
            print(stacked_reward.shape)
            stacked_reward_copy = np.zeros((nbr_test_itrs, stacked_reward.shape[1]))
            number_of_points_in_batch = int(stacked_reward.shape[0] / nbr_test_itrs)
            for i in range(nbr_test_itrs):
                stacked_reward_copy[i, :] = np.mean(stacked_reward[i * number_of_points_in_batch:(i + 1) * number_of_points_in_batch, :],
                    axis=0)
            stacked_reward = stacked_reward_copy

        if len(stacked_reward.shape) == 2 and stacked_reward.shape[1] == 2:
            stacked_reward = stacked_reward[:, 0]
        if (min(stacked_reward) < min_y):
            min_y = min(stacked_reward)
        if (max(stacked_reward) > max_y):
            max_y = max(stacked_reward)

        data2, = plt.plot(np.array(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"]), stacked_reward, label='test data')
        data4, = plt.plot(np.array(stats_map_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"]), trendline(stacked_reward), label='test avg')


        data.append(data2)
        data.append(data4)
        if plot_train and len(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key])>0:
            stacked_reward_train = np.stack(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key], axis=0)
            if len(stacked_reward_train.shape) == 2 and stacked_reward_train.shape[1] == 2:
                stacked_reward_train = stacked_reward_train[:, 0]
            if (min(stacked_reward_train) < min_y):
                min_y = min(stacked_reward_train)
            if (max(stacked_reward_train) > max_y):
                max_y = max(stacked_reward_train)
            data1= plt.scatter(np.array(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"]), stacked_reward_train, alpha=0.4, label='training data')
            data3, = plt.plot(np.array(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]["plot_iterations"]), trendline(stacked_reward_train), label='training avg')
            data.append(data1)
            data.append(data3)


        if len(data_outer)==0:
            data_outer.append(data2)
            data_outer.append(data4)
            names_outer.append("test" )  # 'test data ' + str(init_m))
            names_outer.append("test avg")  # 'test avg.' + str(init_m))

        if len(data_outer)==2 and plot_train and len(stats_map_training[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][key])>0:
                data_outer.append(data1)
                data_outer.append(data3)
                names_outer.append("train")  # 'test data ' + str(init_m))
                names_outer.append("train avg")  # 'test avg.' + str(init_m))



        plt.subplots_adjust(left=0.15, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=0.25)

        fig.text(0.5, 0.04, 'Epoch', ha='center', fontdict={'size':'large'})
        fig.text(0.04, 0.5, title, va='center', rotation='vertical', fontdict={'size':'large'})
        fig.legend(data_outer, names_outer, loc='lower right', shadow=True, ncol=2)

        if settings_map.save_plots and settings_map.save_regular_plots:
            print("Saved  " + settings_map.timestamp + filename)


            fig.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + filename))
        else:
            plt.show()


