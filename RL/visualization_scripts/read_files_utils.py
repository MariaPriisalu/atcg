
import pickle, glob, os, subprocess

import numpy as np

from RL.visualization_scripts.visualization_utils import get_constants
from RL.visualization_scripts.find_files_utils import get_paths
def read_gradients(gradients):
    itr = 0
    grads = []
    if len(gradients) > 0:
        with open(gradients[0][1], 'rb') as handle:
            try:
                b = pickle.load(handle, encoding="latin1", fix_imports=True)
            except EOFError:
                return grads, itr

        grads = []
        K = 0
        for var in b:
            # print var.shape
            shape = [len(gradients)]
            for dim in var.shape:
                shape.append(dim)
            grads.append(np.zeros(shape))
            #print (str(K) + " " + str(shape))


        for pair in sorted(gradients):

            with open(pair[1], 'rb') as handle:
                try:
                    b = pickle.load(handle, encoding="latin1", fix_imports=True)
                except EOFError:
                    return grads, itr
                for indx, var in enumerate(b):
                    # print("Gradients " + str(indx) + " " + pair[1] + " " + str(var.shape) + " " + str(grads[indx].shape))

                    grads[indx][itr, :] = var.copy()
                    # print ("Gradients "+str(indx)+" " + pair[1] + " " + str(np.sum(np.absolute(var[:])))+" "+str(np.sum(np.absolute(grads[indx][itr, :])))+" "+str(np.sum(np.absolute(grads[indx][ :]))))

            itr += 1
    return grads, itr


def read_settings_file(settings_map, timestamp):
    actions, actions_names, labels, labels_mini, train_nbrs, test_nbrs, init_names = get_constants()

    settings_map.timestamp=timestamp

    settings_map.semantic_channels_separated = False
    settings_map.in_2D=False
    settings_map.temporal_case=False
    settings_map.carla=False
    settings_map.num_measures = 6
    settings_map.num_measures_car=0
    settings_map.continous=False
    settings_map.gaussian=False
    settings_map.velocity = False
    settings_map.learn_init=False
    settings_map.env_shape=settings_map.settings.env_shape
    settings_map.init_names=init_names

    # Find settings file
    settings_file = glob.glob(settings_map.settings_path + "*/" + timestamp + "*" + settings_map.settings_ending)

    if len(settings_file) == 0:
        settings_file = glob.glob(settings_map.settings_path + timestamp + "*" + settings_map.settings_ending)
    if len(settings_file) == 0:
        return settings_map

    # Get movie name
    settings_map.name_movie = os.path.basename(settings_file[0])[len(timestamp) + 1:-len("_settings.txt")]
    settings_map.target_dir = settings_map.settings_path + settings_map.name_movie
    subprocess.call("cp " + settings_file[0] + " " + settings_map.target_dir + "/", shell=True)

    settings_map.semantic_channels_separated=False
    settings_map.labels_to_use=labels
    settings_map.in_2D =False
    settings_map.temporal_case =False
    settings_map.carla =False
    settings_map.continous =False
    settings_map.gaussian=False
    settings_map.learn_init =False
    settings_map.learn_goal = False
    settings_map.velocity =False
    settings_map.toy_case =False

    with open(settings_file[0]) as s:
        for line in s:
            if "Semantic channels separated" in line:
                if line[len("Semantic channels separated: "):].strip()=="1":
                    settings_map.semantic_channels_separated = True
            if "Minimal semantic channels : " in line:
                if line[len("Minimal semantic channels : "):].strip() == "True":
                    mini_labels = True
                    if mini_labels:
                        settings_map.labels_to_use = labels_mini
            if "Number of measures" in line:
                if " car" in line:
                    settings_map.num_measures_car = int(line[len("Number of measures car : "):])
                else:
                    settings_map.num_measures = int(line[len("Number of measures: "):])
            if "2D input to network:"in line:
                if line[len("2D input to network: "):].strip()== "True":
                    settings_map.in_2D=True
                print(line[len("2D input to network: "):].strip() +" "+str(bool(line[len("2D input to network: "):].strip()))+" "+str(settings_map.in_2D))
            if "Temporal case : " in line :
                if line[len("Temporal case : " ):].strip()== "True":
                    settings_map.temporal_case=True
            if "CARLA : "  in line and line[len("CARLA : " ):].strip()== "True":
                settings_map.carla=True
            if "Continous: " in line and line[len("Continous: " ):].strip()== "True":
                settings_map.continous=True
            if "Gaussian init net: " in line and line[len("Gaussian init net: " ):].strip()== "True":
                settings_map.gaussian=True
            if "Learn initialization:  " in line and line[len("Learn initialization:  " ):].strip()== "True":
                settings_map.learn_init=True
            if "Velocity : " in line and line[len("Velocity : " ):].strip() == "True":
                settings_map.velocity=True
            if "Toy case : "in line and line[len("Toy case : " ):].strip() == "True":
                settings_map.toy_case=True
            if "Learn goal : " in line and line[len("Learn goal : "):].strip() == "True":
                settings_map.learn_goal = True
    if settings_map.carla:
        settings_map.temporal_case=False

    return settings_map
