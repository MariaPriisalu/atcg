import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_gradient_by_sem_channel(grads, settings_map,init=False, car=False, goal=False):
    non_zero_channels = []
    if len(grads) > 0:
        sz = grads[0].shape
        print("gradient shape "+str(sz))
        if len(sz) > 5:
            settings_map.in_2D = False
        if len(sz)>3:
            for channel in range(sz[-2]):
                print("Grads "+str(np.sum(np.abs(grads[0][:, :, :, channel])))+" "+str(grads[0][:, :, :, channel].shape))
                if not settings_map.in_2D:
                    if np.any(grads[0][:, :, :, :, channel]) > 0:
                        non_zero_channels.append(channel)
                        print("NON_zero channel" + str(channel))
                else:
                    if np.any(grads[0][:, :, :, channel]) > 0:
                        non_zero_channels.append(channel)
                        print("NON_zero channel" + str(channel))
            fig1, axarr = plt.subplots(len(non_zero_channels) // 3 + 1, 3)

            plt.suptitle("Gradients conv, channel wise")

            for channel in range(len(non_zero_channels)):
                #print "Channel "+str(channel)

                if not settings_map.in_2D:
                    if sz[-1] > 1:
                        local = grads[0][:, :, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2] * sz[3]*sz[-1])
                    else:
                        local = grads[0][:, :, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2] * sz[3])
                else:
                    if sz[-1]>1:
                        local = grads[0][:, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2]*sz[-1])
                    else:
                        local = grads[0][:, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2])

                local = np.mean(np.absolute(local), axis=1)
                if len(non_zero_channels) // 3==0:
                    axarr[channel % 3].plot(np.squeeze(local))
                else:
                    axarr[channel // 3, channel % 3].plot(np.squeeze(local))

                if non_zero_channels[channel] >= len(settings_map.labels_to_use):
                    axarr[channel // 3, channel % 3].set_title(
                        settings_map.labels_to_use[non_zero_channels[channel] % len(settings_map.labels_to_use)] + " t=" + str(
                            non_zero_channels[channel] // len(settings_map.labels_to_use)))
                else:
                    if len(non_zero_channels) // 3 == 0:
                        axarr[ channel % 3].set_title(settings_map.labels_to_use[non_zero_channels[channel]])
                    else:
                        axarr[channel // 3, channel % 3].set_title(settings_map.labels_to_use[non_zero_channels[channel]])
                if len(non_zero_channels) // 3 == 0:
                    axarr[ channel % 3].set_xlabel('train_itr')
                else:
                    axarr[channel // 3, channel % 3].set_xlabel('train_itr')
            if settings_map.save_plots:
                name = settings_map.timestamp + 'gradient_conv2.png'

                if goal:
                    name = settings_map.timestamp + 'gradient_conv_goal.png'
                elif init:
                    if car:
                        name = settings_map.timestamp + 'gradient_conv_init_car.png'
                    else:
                        name = settings_map.timestamp + 'gradient_conv_init.png'
                print("Saved  " +name)
                fig1.savefig(os.path.join(settings_map.target_dir, name))
            else:
                plt.show()
    return non_zero_channels


def plot_gradient_softmax(gradients,grads,settings_map,non_zero_channels):
    if len(gradients) > 0:
        fig1, axarr = plt.subplots(3, 3)
        action_label = [ 'downL', 'down', 'downR', 'left', 'stand', 'right','upL', 'up', 'upR']

        for channel in range(len(grads[-1][0,:])):

            local = np.absolute(grads[-1][:, channel])
            #print "Local: "+str( local.shape)
            axarr[channel // 3, channel % 3].plot(np.squeeze(local))
            if non_zero_channels[channel] >= len(action_label):
                axarr[channel // 3, channel % 3].set_title(
                    action_label[non_zero_channels[channel] % len(action_label)] + " t=" + str(
                        non_zero_channels[channel] // len(action_label)))
            else:
                axarr[channel // 3, channel % 3].set_title(action_label[non_zero_channels[channel]])
            axarr[channel // 3, channel % 3].set_xlabel('train_itr')

        if settings_map.save_plots:
            print("Saved  " + settings_map.timestamp + 'gradient_biases_fc2.png')
            fig1.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + 'gradient_biases_fc2.png'))
        else:
            plt.show()
        for counter, grad in enumerate(grads):
            print(str(grad.shape) + " " + str(counter))


def plot_gradients_fully_connected(grads, settings_map):
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Gradients fc")
    for action in range(grads[-1].shape[1]):
        axarr[action // 3, action % 3].plot(np.sum(np.abs(grads[-2][:, action]), axis=1))
        axarr[action // 3, action % 3].set_title(settings_map.actions_names[action])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
    if settings_map.save_plots:
        print("Saved  " + settings_map.timestamp + 'gradient_fc2.png')
        fig1.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + 'gradient_fc2.png'))
    else:
        plt.show()


def plot_gradients_softmax(gradients, grads,settings_map):
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Gradients softmax")
    for action in range(gradients[-1].shape[1]):
        axarr[action // 3, action % 3].plot(np.sum(np.abs(grads[-1][:, action]), axis=1))
        axarr[action // 3, action % 3].set_title(settings_map.actions_names[gradients])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
        axarr[action // 3, action % 3].set_ylabel('Gradient Softmax')
    if settings_map.save_plots:
        print("Saved  " + settings_map.timestamp + 'gradient_softmax2.png')
        fig1.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + 'gradient_softmax2.png'))
    else:
        plt.show()


