import pickle, os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_weights_conv(weights, non_zero_channels, settings_map, init=False, car=False, goal=False):
    with open(weights[0][1], 'rb') as handle:
        b = pickle.load(handle, encoding="latin1", fix_imports=True)
    weights_holder = []
    for indx, var in enumerate(b):
        shape = [len(weights)]

        for dim in var.shape:
            shape.append(dim)
        # print(str(indx)+" "+str(var.shape))
        weights_holder.append(np.zeros(shape))
    itr = 0
    for pair in sorted(weights):
        with open(pair[1], 'rb') as handle:
            try:
                b = pickle.load(handle, encoding="latin1", fix_imports=True)
            except EOFError:
                break
            for indx, var in enumerate(b):
                weights_holder[indx][itr, :] = var
            #print
        itr += 1
    sz = weights_holder[0].shape
    # print("Weights holder : "+str(sz))
    fig1, axarr = plt.subplots(len(non_zero_channels) // 3 + 1, 3)
    plt.suptitle("Weights conv, channel wise")
    weights_abs_value = []
    names_labels = []

    for channel in range(len(non_zero_channels)):
        #print channel

        if not settings_map.in_2D:

            local = weights_holder[0][:, :, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2] * sz[3]*sz[-1])
        else:

            local = weights_holder[0][:, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2]*sz[-1])

        weights_abs_value.append(np.mean(np.abs(np.squeeze(local))))
        local = np.mean(local, axis=1)
        if len(non_zero_channels) // 3 == 0:
            axarr[channel % 3].plot(np.squeeze(local))
        else:
            axarr[channel // 3, channel % 3].plot(np.squeeze(local))
        names_labels.append(settings_map.labels_to_use[non_zero_channels[channel] % len(settings_map.labels_to_use)])
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
        name=settings_map.timestamp + 'weights_conv2.png'
        if goal:
            name = settings_map.timestamp + 'weights_conv_goal.png'
        elif init:
            if car:
                name = settings_map.timestamp + 'weights_conv_init_car.png'
            else:
                name = settings_map.timestamp + 'weights_conv_init.png'
        print("Saved  " + name)
        fig1.savefig(os.path.join(settings_map.target_dir, name))
    else:
        plt.show()

    for pair in sorted(zip(weights_abs_value, names_labels), key=lambda tup: tup[0], reverse=True):
        print("%s & %.3e \\" % (pair[1], pair[0]))
    return weights_holder


def plot_weights_direction(weights_holder,settings_map):
    fig1, axarr = plt.subplots(2, 3)
    plt.suptitle("Segmentation weights conv, direction wise")
    if settings_map.in_2D:
        local = weights_holder[0][:, :, :, 3]
    else:
        local = weights_holder[0][:, :, :, :, 3]
    for direction in range(3):
        cur = np.mean(local, axis=direction + 1)
        for direction2 in range(2):
            axarr[direction2, direction].plot(np.squeeze(np.mean(cur, axis=direction2 + 1)))
            axarr[direction2, direction].set_title("direction " + str(direction + 1))
            axarr[direction2, direction].set_xlabel('train_itr')
    if settings_map.save_plots:
        print("Saved  " + settings_map.timestamp + 'weights_seg_conv2.png')
        fig1.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + 'weights_seg_conv2.png'))
    else:
        plt.show()


def plot_weights_fully_connected(weights_holder,settings_map):
    global fig1, axarr, action
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Weights fc")
    for action in range(weights_holder[-1].shape[1]):
        axarr[action // 3, action % 3].plot(weights_holder[-2][:, action])
        axarr[action // 3, action % 3].set_title(settings_map.actions_names[action])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
    if settings_map.save_plots:
        print("Saved  " + settings_map.timestamp + 'weights_fc2.png')
        fig1.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + 'weights_fc2.png'))
    else:
        plt.show()


def plot_weights_fc(weights_holder,settings_map):
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Weights fc")
    for action in range(weights_holder[-1].shape[1]):
        axarr[action // 3, action % 3].plot(np.mean(weights_holder[-2][:, action], axis=1))
        axarr[action // 3, action % 3].set_title(settings_map.actions_names[action])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
    if settings_map.save_plots:
        print("Saved  " + settings_map.timestamp + 'weights_fc2.png')
        fig1.savefig(os.path.join(settings_map.target_dir, settings_map.timestamp + 'weights_fc2.png'))
    else:
        plt.show()


def plot_weights_softmax(weights_holder,settings_map):
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Weights biases fc")
    print(weights_holder[-1])
    for action in range(weights_holder[-1].shape[1]):
        axarr[action // 3, action % 3].plot(weights_holder[-1][:, action])
        axarr[action // 3, action % 3].set_title(settings_map.actions_names[action])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
    if settings_map.save_plots:
        print("Saved  " + settings_map.timestamp + 'weights_biases_fc2.png')
        fig1.savefig(os.path.join(settings_map.settings_map.target_dir, settings_map.timestamp + 'weights_biases2.png'))
    else:
        plt.show()


