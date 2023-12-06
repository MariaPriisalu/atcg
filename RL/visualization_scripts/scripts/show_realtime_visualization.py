import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plyfile import PlyData, PlyElement

#from RL.visualization import view_pedestrians, view_cars, view_2D

# BEcause py 2.7 and no enums yet..
CARLA_DATASET = 1
WAYMO_DATASET = 2
OTHER_DATASET = 3

DATASET_TO_USE = WAYMO_DATASET

LIMIT_FRAME_NUMBER = 199 # We put 2 here for debug purposes to make it load faster
PEDESTRIAN_SEG_LABEL = 26


# Requires from datetime and os
from RL.settings import run_settings
from RL.extract_tensor import objects_in_range, extract_tensor, objects_in_range_map
from RL.episode import SimpleEpisode

if DATASET_TO_USE == OTHER_DATASET:
    from colmap.reconstruct import reconstruct3D_ply
else:
    from commonUtils.ReconstructionUtils import reconstruct3D_ply, cityscapes_colours, cityscapes_labels, CreateDefaultDatasetOptions_Waymo, CreateDefaultDatasetOptions_CarlaRealTime

# Top view of 3D world.
def view_2D( tensor, img_from_above_loc, seq_len, people=False, white_background=False):
    background_color = 192
    if white_background:
        background_color = 255
    if len(tensor.shape) == 3:
        segmentation = (tensor[ :, :, 3] * 33.0).astype(np.int)
        dim=segmentation.shape
    else:
        segmentation = (tensor[:, :, :, 3] * 33.0).astype(np.int)
        dim = segmentation.shape[1:]
    for x in range(min(img_from_above_loc.shape[1]-2*seq_len, dim[1])):
        for y in range(min(img_from_above_loc.shape[0]-2*seq_len, dim[0])):
            if len(tensor.shape) == 3:
                values = segmentation[ y, x]  # v ==22 or (5<v and 11>v)]#v>0
                if values > 0:
                    img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] = cityscapes_colours[values]
                else:
                    img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] = [
                        background_color, background_color, background_color]

            else:
                values = [i for i, v in enumerate(segmentation[:, y, x]) if v>1] #v ==22 or (5<v and 11>v)]#v>0
                if len(values) > 0:
                    z = np.max(values)
                    img_from_above_loc[img_from_above_loc.shape[0]-1-seq_len-y, seq_len+x, :] = cityscapes_colours[segmentation[z, y, x]]
                else:
                    img_from_above_loc[img_from_above_loc.shape[0]-1-seq_len-y, seq_len+x, :] = [background_color,background_color,background_color]

    img_from_above_loc =img_from_above_loc[...,::-1] #cv2.cvtColor(img_from_above_loc, cv2.COLOR_BGR2RGB)

    return img_from_above_loc

# Visualization of pedestrians.
# If tensor is not empty => view the trajectory densities for pedestrians only
# If tensor is empty and no_tracks = False => View trajectories of the pedestrians over time + on the given frame (parameter) the location for each pedestrian with an orange box
def view_pedestrians(frame, people, current_frame, seq_len, trans=.15, tensor=[], no_tracks=False):

    if tensor is None or len(tensor)>0:
        area_to_study=tensor
        if len(tensor.shape)==4:
            area_to_study=tensor[:,:, :, 4]
        inverted = np.flip(np.amax(area_to_study, axis=0), axis=0)

        for i in range(3):
            current_frame[:,:,i]=current_frame[:,:,i]*(np.ones(np.shape(inverted))-inverted) + current_frame[:,:,i]*(inverted * cityscapes_colours[PEDESTRIAN_SEG_LABEL][i])
    else:
        if not no_tracks:
            # Draw the trajectories of all pedestrians and frames
            people_frame = np.zeros((current_frame.shape[0],current_frame.shape[1],1))
            for i, person_list in enumerate(people):
                 for person in person_list:
                    #print person
                    person_bbox = [ max(current_frame.shape[0]-1-int(seq_len+person[1, 1]),0),min(current_frame.shape[0]-1-int(seq_len+person[1, 0]), current_frame.shape[0]-1),
                                   max(int(seq_len+person[2, 0]),0), min(int(seq_len+person[2, 1]), current_frame.shape[1]-1)]
                    s=current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :].shape

                    if 0 not in s and person_bbox[0]<current_frame.shape[0] and person_bbox[2]<current_frame.shape[1] and person_bbox[1]>=0 and person_bbox[3]>=0:

                        people_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1] = np.ones((s[0], s[1],1))

                        current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :] =\
                            current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :]*(1 - trans) \
                                + trans *  np.tile(cityscapes_colours[PEDESTRIAN_SEG_LABEL], (s[0], s[1], 1)) # np.tile([0,0,0], (s[0], s[1], 1))

        # Draw an orange rectangle where each pedestrian is currently located
        if frame<len(people):
            for person in people[frame]:
                person_bbox = [current_frame.shape[0] - 1 - int(seq_len + max(person[1, :])),
                               current_frame.shape[0] - 1 - int(seq_len + min(person[1, :])),
                               int(seq_len + min(person[2, :])), int(seq_len + max(person[2, :]))]
                #print person_bbox
                s = current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :].shape
                current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :] = np.tile([230, 92, 0],
                                                                                                      (s[0], s[1], 1))
        #mask=(1 - trans)*people_frame+(1-people_frame)
        #current_frame =int(current_frame *np.tile(mask,(1,1,3))+ trans *np.tile(people_frame, (1,1,3))* np.tile(colours[26], (current_frame.shape[0],current_frame.shape[1], 1)))  # np.tile([0,0,0], (s[0], s[1], 1))



    return current_frame

# Visualization of cars.
def view_cars( current_frame, cars, width, depth, seq_len, frame=-1, tensor=np.array((0)),  hit_by_car=-1 , transparency=0.1):

    transparency = 0.5
    if not tensor.shape:
        frames = []
        if frame>=0:
            if frame<len(cars):
                frames.append(frame)
        else:
            frames=list(range(0, len(cars)))

        for f in frames:

            for car in cars[f]:
                label=26
                col=cityscapes_colours[label]
                col=(col[2],col[1], col[0])
                # if len(car)==7:
                #     label=car[6]
                car_bbox = [max(current_frame.shape[0] - 1 - int(seq_len + car[3]-1),0),
                            min(current_frame.shape[0] - 1 - int(seq_len + car[2])+1,current_frame.shape[0]-1 ),
                            max(seq_len+car[4],0), min(seq_len+car[5], current_frame.shape[1]-1)]

                if car_bbox[0]<current_frame.shape[0]  and  car_bbox[2]<current_frame.shape[1] and car_bbox[1]>=0 and car_bbox[3]>=0 :
                    sh = current_frame[car_bbox[0]:car_bbox[1],car_bbox[2]:car_bbox[3], :]
                    # if hit_by_car:

                    #     print "Car viz "+str(f)+" "+str(car)
                    #current_frame[ car_bbox[0]:car_bbox[1],car_bbox[2]: car_bbox[3], :] = ((1-transparency)*current_frame[ car_bbox[0]:car_bbox[1],car_bbox[2]: car_bbox[3], :]+transparency*256*np.ones,((sh.shape[0],sh.shape[1], 3))).astype(int)
                    current_frame[car_bbox[0]:car_bbox[1],car_bbox[2]:car_bbox[3], :] =(current_frame[car_bbox[0]:car_bbox[1],car_bbox[2]:car_bbox[3], :]*(1-transparency)).astype(int)+ (np.tile(col,(sh.shape[0],sh.shape[1], 1))*transparency).astype(int)
    else:

        selected_tensor_area=tensor
        if len(tensor.shape)==4:
            selected_tensor_area=tensor[:, :, :, 5]


        inverted = np.flip(np.amax(selected_tensor_area, axis=0), axis=0)

        max_val=max(np.max(inverted),1)
        inverted=inverted/max_val

        for i in range(3):
            current_frame[:,:,i]=current_frame[:,:,i]*(1-transparency)*(np.ones(np.shape(inverted)[0:2])-inverted) +current_frame[:,:,i]*(transparency)*(inverted*0)#current_frame[:,:,i]*(np.ones(np.shape(inverted)[0:2])-inverted) +current_frame[:,:,i]*(inverted*colours[26][i])
    return current_frame

def init_episode(settings, cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample,
                 people_sample, pos_x, pos_y, seq_len_pfnn, tensor, training):
    if training:

        episode = SimpleEpisode(tensor, people_sample, cars_sample, pos_x, pos_y, settings.gamma,
                                settings.seq_len_train, settings.reward_weights_pedestrian,settings.reward_weights_initializer,agent_size=settings.agent_shape,
                                people_dict=people_dict_sample, cars_dict=cars_dict_sample,
                                init_frames=init_frames,
                                follow_goal=settings.goal_dir, action_reorder=settings.reorder_actions,
                                threshold_dist=settings.threshold_dist, init_frames_cars=init_frames_cars,
                                temporal=settings.temporal, predict_future=settings.predict_future,
                                run_2D=settings.run_2D, velocity_actions=settings.velocity or settings.continous,
                                seq_len_pfnn=seq_len_pfnn, end_collide_ped=settings.end_on_bit_by_pedestrians, defaultSettings=settings)
    else:
        while len(cars_sample) < settings.seq_len_test:
            if settings.carla:
                cars_sample.append([])
                people_sample.append([])
            else:
                cars_sample.append(cars_sample[-1])
                people_sample.append(people_sample[-1])
        episode = SimpleEpisode(tensor, people_sample, cars_sample, pos_x, pos_y, settings.gamma, settings.seq_len_test,
                                settings.reward_weights_pedestrian,settings.reward_weights_initializer, agent_size=settings.agent_shape,
                                people_dict=people_dict_sample, cars_dict=cars_dict_sample,
                                init_frames=init_frames,
                                follow_goal=settings.goal_dir, action_reorder=settings.reorder_actions,
                                threshold_dist=settings.threshold_dist, init_frames_cars=init_frames_cars,
                                temporal=settings.temporal, predict_future=settings.predict_future,
                                run_2D=settings.run_2D, velocity_actions=settings.velocity or settings.continous,
                                seq_len_pfnn=seq_len_pfnn, end_collide_ped=settings.end_on_bit_by_pedestrians, defaultSettings=settings)
    return episode

def set_up_episode(settings, cars, people, pos_x, pos_y, tensor, training, road_width=0, time_file=None, people_dict={},
                   car_dict={}, init_frames={}, init_frames_cars={}, seq_len_pfnn=-1):

    cars_sample = objects_in_range(cars, pos_x, pos_y, settings.depth, settings.width,
                                   carla=settings.carla)

    people_sample = objects_in_range(people, pos_x, pos_y, settings.depth, settings.width,
                                     carla=settings.carla)
    # cars_sample=cars
    # people_sample=people

    people_dict_sample = {}
    if len(people_dict) > 0:
        people_dict_sample, init_frames = objects_in_range_map(people_dict, pos_x, pos_y, settings.depth,
                                                               settings.width, init_frames=init_frames)
    cars_dict_sample = {}
    if len(car_dict) > 0:
        cars_dict_sample, init_frames_cars = objects_in_range_map(car_dict, pos_x, pos_y, settings.depth,
                                                                  settings.width, init_frames=init_frames_cars)

    episode = init_episode(settings,cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample,
                                people_sample, pos_x, pos_y, seq_len_pfnn, tensor, training)
    print(("seq len: " + str(episode.seq_len)))
    return episode

def main():
    outputFolder = None
    filepath = None

    if DATASET_TO_USE == OTHER_DATASET:
        file_name="Datasets/colmap/colmap/tubingen_000136"

        reconstruction, people_rec, cars_rec, scale ,camera_locations_colmap, middle = reconstruct3D_ply(file_name, setup, False)
    elif DATASET_TO_USE == CARLA_DATASET:# Read CARLA
        setup = run_settings()
        setup.seq_len_train = 1

        filepath = "Datasets/CarlaDataset/test_0"
        # In carla:
        reconstruction, people_rec, cars_rec, scale, ped_dict, cars_2D, people_2D, valid_ids, cars_dict, init_frames, init_frames_cars = reconstruct3D_ply(
            filepath, setup.scale_x)
    elif DATASET_TO_USE == WAYMO_DATASET: # Same for Carla-Real time actualy
        filepath = "Datasets/carla-realtime/train/test_1"
        outputFolder = "Datasets/tempoutvis"
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        centeringFilePath = os.path.join(filepath, "centering.p")
        peopleFilePath = os.path.join(filepath, "people.p")

        # If the dataset wasn't post-processed at all, do it now
        if not os.path.exists(centeringFilePath):
            dummy_metadata = {}
            dummy_metadata["frame_min"] = 0 # TODO: fix these by reading cars and people frames len!
            dummy_metadata["frame_max"] = 200

            setup = run_settings()
            if setup.useRealTimeEnv:
                options = CreateDefaultDatasetOptions_CarlaRealTime(setup)
            else:
                options = CreateDefaultDatasetOptions_Waymo(dummy_metadata)

            reconstruction, people_rec, cars_rec, scale, ped_dict, cars_2D, people_2D, valid_ids, cars_dict, init_frames, init_frames_cars, heroCarDetails = reconstruct3D_ply(
                filepath, datasetOptions=options, recalculate=False)

        metadata = None
        with open(centeringFilePath, 'rb') as metadataFileHandle:
            metadata = pickle.load(metadataFileHandle, encoding="latin1", fix_imports=True)

        setup = run_settings()
        if setup.useRealTimeEnv:
            options = CreateDefaultDatasetOptions_CarlaRealTime(setup)
        else:
            options = CreateDefaultDatasetOptions_Waymo(metadata)

        # Set the parameters of the env
        min_bbox = metadata["min_bbox"]
        max_bbox = metadata["max_bbox"]
        env_depth =max_bbox[0] - min_bbox[0]
        env_height = max_bbox[2] - min_bbox[2]
        env_width = max_bbox[1] - min_bbox[1]
        setup = run_settings(env_depth=env_depth, env_width=env_width, env_height=env_height)
        setup.seq_len_train = 1

        reconstruction, people_rec, cars_rec, scale, ped_dict, cars_2D, people_2D, valid_ids, cars_dict, init_frames, init_frames_cars, heroCarDetails = reconstruct3D_ply(
            filepath, setup.scale_x, datasetOptions=options, recalculate=False)

    pos_x = 0
    pos_y = -setup.width // 2


    # View first frame, no cars
    tensor, density = extract_tensor(pos_x, pos_y, reconstruction, setup.height, setup.width, setup.depth)
    print("Episode set up")
    ep = set_up_episode(setup, cars_rec, people_rec, pos_x, pos_y, tensor, True, 0)

    print("View 2D")
    img_from_above = np.zeros((ep.reconstruction.shape[1], ep.reconstruction.shape[2], 3), dtype=np.uint8)
    img_above = view_2D(ep.reconstruction, img_from_above, 0)

    fig = plt.figure()
    plt.imshow(img_above)
    plt.show()
    print (" Save figure "+str(os.path.join(outputFolder, "_environment.png")))
    fig.savefig(os.path.join(outputFolder, "_environment.png"))
    width_bar = min(setup.seq_len_train, 10)
    if env_width > 128 or env_depth > 256:
        middle = np.mean(people_rec[0][0], axis=1)
        middle = middle.astype(int)
        x_min = middle[1] - 128 - width_bar
        y_min = img_from_above.shape[0] - 1 - (middle[0] + 64) - width_bar
        img_from_above_local = img_from_above[
                               img_from_above.shape[0] - 1 - (middle[0] + 64) - width_bar:img_from_above.shape[
                                                                                              0] - 1 - (
                                                                                          middle[0] - 64) + width_bar,
                               middle[1] - 128 - width_bar:middle[1] + 128 + width_bar]

    for frame in range(LIMIT_FRAME_NUMBER):        #
        #print "View PEDESTRIANS"
        img = img_above.copy()
        img = view_pedestrians(frame, ep.people, img, 0, trans=.15) #tensor=ep.reconstruction)
        #print "View cars"

        img = view_cars(img, ep.cars, img.shape[0], img.shape[1], setup.seq_len_train, frame=frame)# tensor=ep.reconstruction)

        fig = plt.figure()
        plt.imshow(img)
        plt.show()
        print(" Save figure " + os.path.join(outputFolder, ('{0}.png'.format(frame))))
        fig.savefig(os.path.join(outputFolder, ('{0}.png'.format(frame)))) #("Datasets/CarlaDataset/outvis/Valid_pavement_weimar_120_frames_ppl.png")

if __name__ == "__main__":
    main()

