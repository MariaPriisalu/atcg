import glob, subprocess, os,traceback
from dotmap import DotMap
from RL.settings import run_settings

def sort_files(files):
    files_map=DotMap()
    files_map.filenames_itr_cars = []
    files_map.test_files_cars = []
    files_map.filenames_itr_people = []
    files_map.test_files_people = []
    files_map.filenames_itr = []
    files_map.test_files = []
    files_map.reconstructions_test = []
    files_map.iterations = {}
    files_map.iterations_cars = []
    files_map.special_cases = []
    files_map.init_maps = []
    files_map.init_maps_stats_car = []
    files_map.init_maps_stats_goal = []
    files_map.init_cars = []
    files_map.init_maps_test = []
    files_map.init_maps_stats_car_test = []
    files_map.init_maps_stats_goal_test=[]
    files_map.init_cars_test = []
    files_map.learn_car = []
    files_map.learn_car_test = []
    files_map.init_map_stats = []
    files_map.init_map_stats_test = []
    files_map.reconstructions = []

    nbr_files = 0
    files_map.numbers = []
    files_map.numbers_people = []


    for agent_file in sorted(files):

        basename = os.path.basename(agent_file)
        #print(basename)
        if not "pose" in basename:
            nbrs = basename.strip()[:-len('.npy')]
            vals = nbrs.split('_')
            #print(vals)
            if vals[1] == 'weimar' or 'test' in vals[-4] or 'test' in vals[-6] or 'test' in vals[3]:  # Testing data

                if "learn" in vals[-2]:
                    files_map.learn_car_test.append((int(vals[-3]), agent_file, int(vals[-5])))
                elif "init" in vals[-3]:
                    files_map.init_map_stats_test.append((int(vals[-4]), agent_file, int(vals[-8])))

                elif "init" in vals[-2]:
                    if "car" in vals[-1]:

                        files_map.init_cars_test.append((int(vals[-3]), agent_file, int(vals[-7])))

                    else:
                        files_map.init_maps_test.append((int(vals[-3]), agent_file, int(vals[-7])))

                elif "people" in vals[-4]:
                    if not "reconstruction" in vals[-1]:
                        files_map.test_files_people.append((int(vals[-1]), agent_file))

                        files_map.numbers_people.append(int(vals[-1]))

                elif "car" in vals[-4]:
                    if not "reconstruction" in vals[-1]:
                        files_map.test_files_cars.append((int(vals[-1]), agent_file))

                        files_map.numbers_people.append(int(vals[-1]))

                elif "init" in vals[-4]:
                    if "car" in vals[-1]:

                        files_map.init_maps_stats_car_test.append((int(vals[-5]), agent_file, int(vals[-9])))
                    elif "goal"in vals[-1]:
                        files_map.init_maps_stats_goal_test.append((int(vals[-5]), agent_file, int(vals[-9])))
                elif not "reconstruction" in vals[-1]:
                    try:
                        files_map.test_files.append((int(vals[-1]), agent_file, int(vals[2])))
                    except ValueError:
                        files_map.test_files.append((int(vals[-1]), agent_file, int(vals[4])))
                else:
                    try:
                        files_map.reconstructions_test.append((int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[2])))
                    except ValueError:
                        files_map.reconstructions_test.append(
                            (int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[4])))
                if  not "reconstruction" in vals[-1] and not  "init" in vals and not  "learn" in vals:
                    if int(vals[-1]) not in list(files_map.iterations.keys()):
                        files_map.iterations[int(vals[-1])]= False
                    elif files_map.iterations[int(vals[-1])]:
                        files_map.special_cases.append(int(vals[-1]))
            else:
                if "learn" in vals[-2]:
                    files_map.learn_car.append((int(vals[-3]), agent_file, int(vals[-6])))
                elif "init" in vals[-3]:
                    files_map.init_map_stats.append((int(vals[-4]), agent_file, int(vals[-7])))
                    # print(" Init stats test" + str(len(files_map.init_map_stats_test)))
                elif "init" in vals[-2]:
                    if "car" in vals[-1]:
                        files_map.init_cars.append((int(vals[-3]), agent_file, int(vals[-6])))
                        # print(" Init cars " + str(len(files_map.init_cars_test)))
                    else:
                        files_map.init_maps.append((int(vals[-3]), agent_file, int(vals[-6])))
                        # print(" Init " + str(len(files_map.init_maps_test)))
                elif "people" in vals[-4]:
                    if not "reconstruction" in vals[-1]:
                        files_map.filenames_itr_people.append((int(vals[-1]), agent_file))
                        files_map.numbers_people.append(int(vals[-1]))

                elif "car" in vals[-4]:
                    if not "reconstruction" in vals[-1]:
                        files_map.filenames_itr_cars.append((int(vals[-1]), agent_file))
                        files_map.numbers_people.append(int(vals[-1]))
                elif "init" in vals[-4]:
                    if "car" in vals[-1]:

                        files_map.init_maps_stats_car.append((int(vals[-5]), agent_file, int(vals[-8])))
                        #files_map.init_maps_cars.append((int(vals[-3]), agent_file, int(vals[-6])))
                    elif "goal" in vals[-1]:

                        files_map.init_maps_stats_goal.append((int(vals[-5]), agent_file, int(vals[-8])))
                        #files_map.init_maps_cars.append((int(vals[-3]), agent_file, int(vals[-6])))

                elif not "reconstruction" in vals[-1]:
                    try:
                        files_map.filenames_itr.append((int(vals[-1]), agent_file, int(vals[2])))
                        files_map.numbers.append(int(vals[-1]))
                    except ValueError:
                        files_map.filenames_itr.append((int(vals[-1]), agent_file, int(vals[3])))
                else:
                    try:
                        files_map.reconstructions.append((int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[2])))
                    except ValueError:
                        files_map.reconstructions.append((int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[3])))

                if  not "reconstruction" in vals[-1] and not  "init" in vals and not  "learn" in vals:
                    if "people" in vals[-4] or "car" in vals[-4]:

                        files_map.iterations_cars.append(int(vals[-1]))
                    if int(vals[-1]) not in list(files_map.iterations.keys()):
                        files_map.iterations[int(vals[-1])] = True
                    elif not files_map.iterations[int(vals[-1])] :

                        files_map.special_cases.append(int(vals[-1]))
            nbr_files += 1
    files_map.filenames_itr.sort(key=lambda x: x[2])
    if files_map.filenames_itr[-1][2]==24: # Realtime dataset
        train_set=[21, 20, 13, 16, 6, 18, 15, 14, 24, 19, 23, 12, 8, 9, 10]
        val_set=[7, 17, 25, 22, 11]
        for tuple in files_map.filenames_itr:
            print (tuple[2])
            if tuple[2] in val_set:
                files_map.test_files.append(tuple)
        filenames_itr_new = [x for x in files_map.filenames_itr if not x in val_set]
        files_map.filenames_itr=filenames_itr_new
        print ("Realtime dataset")
        print(" Training files")
        print(files_map.filenames_itr)
        print(" Test files ")
        print(files_map.test_files)

    else:
        files_map.filenames_itr.sort(key=lambda x: x[0])
    return files_map


def sort_files_eval(files):

    files_map_eval = DotMap()
    files_map_eval.test_files_poses=[]
    files_map_eval.test_files = []

    files_map_eval.reconstructions_test=[]
    nbr_files = 0


    files_map_eval.init_maps = []
    files_map_eval.init_map_stats = []
    files_map_eval.init_maps_stats_car_test=[]
    files_map_eval.init_maps_stats_goal_test=[]
    files_map_eval.init_cars = []
    files_map_eval.learn_car_test = []

    for agent_file in sorted(files):
        basename = os.path.basename(agent_file)
        nbrs = basename.strip()[:-len('.npy')]
        vals = nbrs.split('_')
        print(vals)
        if "learn" in vals[-2]:

            files_map_eval.learn_car_test.append((int(vals[-5]), agent_file, int(vals[-3])))
        elif "init" in vals[-4] and not "random" in vals[-5] and not "car" in vals[-5]:
            if "car" in vals[-1]:
                files_map_eval.init_maps_stats_car_test.append((int(vals[-5]), agent_file, int(vals[-9])))

            elif "goal" in vals[-1]:
                files_map_eval.init_maps_stats_goal_test.append((int(vals[-5]), agent_file, int(vals[-9])))
                # files_map.init_maps_cars.append((int(vals[-3]), agent_file, int(vals[-6])))
        elif "init" in vals[-3] and not "random" in vals[-4] and not  "car" in vals[-4]:
            #print("init file")
            files_map_eval.init_map_stats.append((int(vals[-4]), agent_file, int(vals[-8])))
        elif "init" in vals[-2] and not "random" in vals[-3]  and not "car" in vals[-3]:
            if "car" in vals[-1]:

                files_map_eval.init_cars.append((int(vals[-5]), agent_file, int(vals[-3])))
            else:
                #print("init file")
                files_map_eval.init_maps.append((int(vals[-3]), agent_file, int(vals[-7])))
        elif "poses" in vals[-1]:
            #print("poses")
            files_map_eval.test_files_poses.append((int(vals[-2]), agent_file, int(vals[-6])))
        else:
            try:
                pos=int(vals[-1])
            except ValueError:
                try:
                    pos = int(vals[-1][:-len("reconstruction")])
                except ValueError:
                    print(basename)
                    pos=-1

            if pos>=0:
                if not "reconstruction" in vals[-1]:
                    #print("pedestrian")
                    files_map_eval.test_files.append((int(vals[-3]), agent_file, int(vals[-1])))
                else:
                    if int(vals[-1][:-len("reconstruction")]) <42:
                        files_map_eval.reconstructions_test.append((int(vals[-3]), agent_file, int(vals[-1][:-len("reconstruction")])))

                nbr_files += 1
    return files_map_eval


def make_movies( settings_map):
    command = "ffmpeg -framerate 25 -i " + settings_map.movie_path + settings_map.name_movie + "frame_%06d.jpg -c:v libx264  -pix_fmt yuv420p -y " + settings_map.target_dir + "/" + settings_map.timestamp + "_" + settings_map.name_movie + ".mp4"
    if settings_map.save_regular_plots and settings_map.make_movie:
        print(command)
        subprocess.call(command, shell=True)
        try:
            command = "ffmpeg -framerate 10 -i " + settings_map.movie_path + settings_map.name_movie + '_cars_frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + settings_map.target_dir + "/" + settings_map.timestamp + "_" + settings_map.name_movie + '_car.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()

        print(command)
        subprocess.call(command, shell=True)
        try:
            command = "ffmpeg -framerate 10 -i " + settings_map.movie_path + settings_map.name_movie + '_people_frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + settings_map.target_dir + "/" + settings_map.timestamp + "_" + settings_map.name_movie + '_people.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()
        try:
            command = "ffmpeg -framerate 10 -i " + settings_map.movie_path + settings_map.name_movie + '_train__cars_frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + settings_map.target_dir + "/" + settings_map.timestamp + "_" + settings_map.name_movie + '_train_car.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()

        print(command)
        subprocess.call(command, shell=True)
        try:
            command = "ffmpeg -framerate 10 -i " + settings_map.name_movie_eval + settings_map.name_movie + 'frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + settings_map.target_dir + "/" + settings_map.timestamp + "_" + settings_map.name_movie + '_eval.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()

        print(command)
        subprocess.call(command, shell=True)
        try:
            command = "ffmpeg -framerate 10 -i " + settings_map.movie_path + settings_map.name_movie + '_train__people_frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + settings_map.target_dir + "/" + settings_map.timestamp + "_" + settings_map.name_movie + '_train_people.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()


def find_gradients(path,settings_map):

    match = path + "*.pkl"
    gradient_struct=DotMap()
    gradient_struct.weights = []
    gradient_struct.weights_init = []
    gradient_struct.weights_init_car = []
    gradient_struct.weights_car=[]
    gradient_struct.weights_goal = []
    gradient_struct.gradients = []
    gradient_struct.gradients_init = []
    gradient_struct.gradients_init_car = []
    gradient_struct.gradients_car=[]
    gradient_struct.gradients_goal = []
    files = glob.glob(match)

    for file in files:
        basename = os.path.basename(file)
        nbrs = basename.strip()[len(settings_map.timestamp):-len('.pkl')]
        if settings_map.toy_case:
            vals = nbrs.split('_')
            if len(vals) == 3:
                gradient_struct.weights.append((int(vals[1]), file, int(vals[0])))
            else:
                gradient_struct.gradients.append((int(vals[1]), file, int(vals[0])))
        else:
            vals = basename.split('_')
            if "weights" in basename:
                if "goal" in vals[-1]:
                    print(vals)
                    gradient_struct.weights_goal.append(int(vals[-6]), file, int(vals[3]))
                elif "car" in vals[-2]:
                    gradient_struct.weights_car.append((int(vals[-4]), file, int(vals[2])))
                elif "init" in vals[-2]:
                    if "car" in vals[-3]:
                        try:
                            gradient_struct.weights_init_car.append((int(vals[-5]), file, int(vals[2])))
                        except ValueError:
                            gradient_struct.weights_init_car.append((int(vals[-5]), file, int(vals[3])))
                    else:

                        try:
                            gradient_struct.weights_init.append((int(vals[-4]), file, int(vals[2])))
                        except ValueError:
                            gradient_struct.weights_init.append((int(vals[-4]), file, int(vals[3])))
                else:
                    try:

                        gradient_struct.weights.append((int(vals[-2]), file, int(vals[2])))
                    except ValueError:
                        gradient_struct.weights.append((int(vals[-2]), file, int(vals[3])))
            else:
                if "goal" in vals[-1]:
                    print(vals)
                    gradient_struct.gradients_goal.append((int(vals[-4]), file, int(vals[2])))
                if "car" in vals[-2]:
                    gradient_struct.gradients_car.append((int(vals[-3]), file, int(vals[2])))
                elif "init" in vals[-2]:
                    if "car" in vals[-3]:
                        try:
                            gradient_struct.gradients_init_car.append((int(vals[-4]), file, int(vals[2])))
                        except ValueError:
                            gradient_struct.gradients_init_car.append((int(vals[-4]), file, int(vals[3])))
                    else:
                        try:
                            gradient_struct.gradients_init.append((int(vals[-3]), file, int(vals[2])))
                        except ValueError:
                            gradient_struct.gradients_init.append((int(vals[-3]), file, int(vals[3])))
                else:
                    try:
                        gradient_struct.gradients.append((int(vals[-2]), file, int(vals[2])))
                    except ValueError:
                        gradient_struct.gradients.append((int(vals[-2]), file, int(vals[3])))
    return gradient_struct


def get_statistics_file_path(settings_map, timestamp):
    path = settings_map.stat_path + timestamp + settings_map.ending + '*'
    match = path + "/*.npy"
    match2 = path + "*.npy"
    print("Pattern " + str(match))
    files = glob.glob(match)
    print("Pattern " + str(match2))
    files = files + glob.glob(match2)
    print("Number of files: " + str(len(files)))
    return files, path


def get_paths_for_evaluation(eval_path, timestamp):
    print("Evaluation ------------------------------")
    path = eval_path + '*' + timestamp

    match = path + "*/*.npy"
    match2 = path + "*.npy"
    files_eval = glob.glob(match)
    files_eval = files_eval + glob.glob(match2)
    print(match)
    print(match2)
    return files_eval, path

def get_paths(settings_map):
    # Set some paths
    settings_map.settings = run_settings()
    settings_map.settings_path = settings_map.settings.path_settings_file

    settings_map.settings_ending = "settings.txt"

    settings_map.stat_path = "localUserData/statistics/train"
    settings_map.eval_path = settings_map.settings.eval_main
    return settings_map

def get_training_itr(files_map):

    files_map.train_itr = {}
    train_counter = 0
    files_map.test_points = {}
    keys = list(files_map.iterations.keys())
    for itr in sorted(keys):
        training = files_map.iterations[itr]
        if training:
            files_map.train_itr[itr] = train_counter
            if itr in files_map.special_cases:
                files_map.test_points[itr] = train_counter
            if itr not in files_map.iterations_cars:  # or toy_case:
                train_counter = train_counter + 1
        else:
            files_map.test_points[itr] = train_counter
            if itr in files_map.special_cases:
                files_map.train_itr[itr] = train_counter
