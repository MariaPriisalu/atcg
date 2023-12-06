from RL.settings import PEDESTRIAN_INITIALIZATION_CODE
from RL.visualization_scripts.visualization_utils import trendline
import matplotlib.pyplot as plt
import numpy as np
import sys,os

def plot_separately_cars(train_itr, avg_reward, test_itr, avg_reward_test, title, filename,plot_train,settings_map):

    data_outer = []
    names_outer = []
    fig = plt.figure(figsize=(8.0, 5.0))
    indx=1


    for init_m, reward_init_m in enumerate(avg_reward_test):
        data = []
        min_y = sys.maxsize
        max_y = -sys.maxsize - 1
        if len(avg_reward_test[init_m]) > 1 and init_m in {5,PEDESTRIAN_INITIALIZATION_CODE.learn_initialization, -1, 9}:
            ax = plt.subplot(2 ,2,indx )
            stacked_reward = np.stack(avg_reward_test[init_m], axis=0)
            if len(stacked_reward.shape) == 2 and stacked_reward.shape[1] == 2:
                stacked_reward = stacked_reward[:, 0]
            if (min(stacked_reward) < min_y):
                min_y = min(stacked_reward)
            if (max(stacked_reward) > max_y):
                max_y = max(stacked_reward)
            data2, = ax.plot(np.array(test_itr[init_m]) * 0.01, stacked_reward, label='test data')
            data4, = ax.plot(np.array(test_itr[init_m]) * 0.01, trendline(stacked_reward), label='test avg')

            data.append(data2)
            data.append(data4)

            if plot_train and len(avg_reward[init_m])>1:

                stacked_reward_train = np.stack(avg_reward[init_m], axis=0)

                if len(stacked_reward_train.shape) == 2 and stacked_reward_train.shape[1] == 2:
                    stacked_reward_train = stacked_reward_train[:, 0]
                if (min(stacked_reward_train) < min_y):
                    min_y = min(stacked_reward_train)
                if (max(stacked_reward_train) > max_y):
                    max_y = max(stacked_reward_train)
                data1= ax.scatter(np.array(train_itr[init_m]) * 0.01, stacked_reward_train,  alpha=0.4,label='training data')
                data3, = ax.plot(np.array(train_itr[init_m]) * 0.01, trendline(stacked_reward_train), label='training avg')
                data.append(data1)
                data.append(data3)


            ax.set_title(settings_map.init_names[init_m])
            if len(data_outer) == 0:
                data_outer.append(data2)
                data_outer.append(data4)
                names_outer.append("test")  # 'test data ' + str(init_m))
                names_outer.append("test avg")  # 'test avg.' + str(init_m))
                if plot_train and len(avg_reward[init_m])>1:
                    data_outer.append(data1)
                    data_outer.append(data3)
                    names_outer.append("train")  # 'test data ' + str(init_m))
                    names_outer.append("train avg")  # 'test avg.' + str(init_m))
            indx = indx + 1
    plt.subplots_adjust(left=0.15, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=0.25)

    fig.text(0.5, 0.04, 'Epoch', ha='center', fontdict={'size':'large'})
    fig.text(0.04, 0.5, title, va='center', rotation='vertical', fontdict={'size':'large'})
    fig.legend(data_outer, names_outer, loc='lower right', shadow=True, ncol=2)

    if settings_map.save_plots and settings_map.save_regular_plots:
        #print "Saved  " + timestamp + filename


        fig.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + filename))
    else:
        plt.show()


def plot_gradients_car(gradients,settings_map):
    if len(gradients)>0:
        fig1, axarr = plt.subplots(len(gradients),1)
        plt.suptitle("Gradients car")
        i=0
        for grad in gradients:
            for j in range(grad.shape[1]):
                if len(gradients)==1:
                    axarr.plot(grad[:,j].flatten(), label=str(j))
                    axarr.set_title("Weight " +str(i))
                    axarr.set_xlabel('train_itr')
                    axarr.set_ylabel('Gradient car')
                else:

                    axarr[i].plot(grad[:,j].flatten(), label=str(j))
                    axarr[i].set_title("Weight " +str(i))
                    axarr[i].set_xlabel('train_itr')
                    axarr[i].set_ylabel('Gradient car')
            axarr[i].legend(loc='upper right',ncol=int(np.ceil(grad.shape[1]/3)))
            i=i+1

        if settings_map.save_plots:
            print("Saved  " + settings_map.timestamp + 'gradient_car.png')
            fig1.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + 'gradient_car.png'))
        else:
            plt.show()


def plot_weights_car(weights_holder,settings_map):

    fig1, axarr = plt.subplots(len(weights_holder))
    plt.suptitle("Weights car")
    i=0
    max_j=0
    for weight in weights_holder:
        for j in range(weight.shape[1]):
            max_j=max(max_j, j)
            if len(weights_holder)==1:
                axarr.plot(weight[:,j].flatten(), label=str(j))
                axarr.set_title("Weight" +str(i))
                axarr.set_xlabel('train_itr')

            else:
                axarr[i].plot(weight[:,j].flatten(), label=str(j))
                axarr[i].set_title("Weight" +str(i))
                axarr[i].set_xlabel('train_itr')
        axarr[i].legend(loc='upper right',ncol=int(np.ceil(weight.shape[1]/3)))
        i=i+1
    if settings_map.save_plots:
        print("Saved  " + settings_map.timestamp + 'weights_car.png')
        fig1.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + 'weights_car.png'))
    else:
        plt.show()
