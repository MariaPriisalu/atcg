import numpy as np
import copy, sys, os
from commonUtils.systemUtils import deep_getsizeof, timing
from commonUtils.ReconstructionUtils import AGENT_MAX_HEIGHT_VOXELS, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS
from commonUtils.ReconstructionUtils import isLabelIgnoredInReconstruction


# TODO: Test
# Takes coordinates in cityscapes coordinate system (in voxels). Returns in tensor [z, y, x]
def extract_tensor(pos_x, pos_y, reconstruction, height, width, depth):
    begin_z = 0
    tensor = np.zeros((height, width, depth, 6))
    density = 0



    #x,y,z in cityscapes
    for z in range(begin_z, height + begin_z):# x
        for y in range(pos_y, pos_y + width):#y
            for x in range(pos_x, pos_x + depth):  # z
                xyzInReconstruction = reconstruction.get((x,y,z), None)
                if xyzInReconstruction != None:
                    tensor[int(z - begin_z), int(y - pos_y),int(x - pos_x) , CHANNELS.rgb[0]:CHANNELS.rgb[1]] = [
                        xyzInReconstruction[CHANNELS.rgb[0]] * (1 / 255.0),
                        xyzInReconstruction[CHANNELS.rgb[0]+1] * (1 / 255.0),
                        xyzInReconstruction[CHANNELS.rgb[0]+2] * (1 / 255.0)]

                    labelInRec = xyzInReconstruction[3]
                    if not isLabelIgnoredInReconstruction(labelInRec, isCarlaLabel=False): # The input data is in cityscapes !
                        tensor[int(z - begin_z), int(y - pos_y),int(x - pos_x),  CHANNELS.semantic] = \
                            xyzInReconstruction[CHANNELS.semantic] * (1 / NUM_SEM_CLASSES)
                    density += 1

    return tensor, density


def return_highest_object( tens):
    z = 0
    while z < tens.shape[0] - 1 and np.linalg.norm(tens[z]) == 0:
        z = z + 1
    return z

def return_highest_col(tens):
    z = 0
    while z < AGENT_MAX_HEIGHT_VOXELS and  z < tens.shape[0] - 1 and np.linalg.norm(tens[z, :]) == 0:
        z = z + 1
    return z
#

def return_highest_object_v1( tens):
    z = 0
    while z < AGENT_MAX_HEIGHT_VOXELS and z < tens.shape[0] - 1 and tens[z] == 0:
        z = z + 1
    return z

def return_highest_col_v1(tens):
    z = 0
    while z < AGENT_MAX_HEIGHT_VOXELS and  z < tens.shape[0] - 1 and (np.all(tens[z,:] == 0)):
        z = z + 1
    return z



# This updates the tensor created previously with the function above to add 0.1 density to each voxel occupied by an agent or car
# reconstruction_2D is a 2D projection of the 3D space given as input in tensor
def frame_reconstruction(tensor, cars_a, people_a, no_dict=True, temporal=False, predict_future=False, run_2D=False, reconstruction_2D=[], number_of_frames_eval=-1):
    reconstruction, reconstruction_2D = reconstruct_static_objects(tensor, run_2D)


    if number_of_frames_eval>=0:
        number_of_frames=number_of_frames_eval
    else:
        number_of_frames = len(cars_a)


    #print(("Frame reconstruction "+str(temporal)))
    cars_predicted=[]
    people_predicted=[]
    for frame in range(number_of_frames):
        cars_map = np.zeros(tensor.shape[:3])#.shape[1:3])
        if run_2D:
            cars_map = np.zeros(tensor.shape[1:3])

        for car in cars_a[frame]:
            car_a = []

            car_middle_exact = np.array([np.mean(car[0:1]), np.mean(car[2:3]), np.mean(car[4:5])])
            for count in range(3):
                p = car[count*2]
                car_a.append(max(p, 0))
                car_a.append(min(car[count*2+1]+1, reconstruction.shape[count]))
            car_middle=np.array([np.mean(car_a[0:1]),np.mean(car_a[2:3]), np.mean(car_a[4:5])])
            if predict_future:
                closest_car=[]
                cc_voxel=[0,0,0]
                vel=[0,0,0]
                min_dist=50
                if frame>0 and no_dict:
                    for car_2 in cars_a[frame-1]:
                        car_middle2 = np.array([np.mean(car_2[0:1]), np.mean(car_2[2:3]), np.mean(car_2[4:5])])
                        dist=np.linalg.norm(car_middle_exact[1:]-car_middle2[1:])
                        if dist< min_dist:
                            closest_car = car_middle2
                            cc_voxel=np.array(car_2)
                            vel=car_middle_exact-car_middle2
                            min_dist=dist.copy()

                    vel[0]=0

                    if np.linalg.norm(vel)>0:

                        loc_frame=1
                        while np.all(closest_car[1:]>1) and np.all(reconstruction.shape[1:3]-closest_car[1:]>1):
                            cc_voxel=cc_voxel+np.array([0,0, vel[1], vel[1], vel[2], vel[2]])
                            if run_2D:
                                cars_map[
                                max(cc_voxel[2].astype(int), 0):min(cc_voxel[3].astype(int), reconstruction.shape[1]),
                                max(cc_voxel[4].astype(int), 0):min(cc_voxel[5].astype(int),
                                                                    reconstruction.shape[2])] = cars_map[
                                                                                                max(cc_voxel[2].astype(
                                                                                                    int), 0):min(
                                                                                                    cc_voxel[3].astype(
                                                                                                        int),
                                                                                                    reconstruction.shape[
                                                                                                        1]),
                                                                                                max(cc_voxel[4].astype(
                                                                                                    int), 0):min(
                                                                                                    cc_voxel[5].astype(
                                                                                                        int),
                                                                                                    reconstruction.shape[
                                                                                                        2])] + 0.1
                            else:
                                cars_map[max(cc_voxel[0].astype(int),0):min(cc_voxel[1].astype(int),reconstruction.shape[0]),
                                max(cc_voxel[2].astype(int),0):min(cc_voxel[3].astype(int),reconstruction.shape[1]),
                                max(cc_voxel[4].astype(int),0):min(cc_voxel[5].astype(int),reconstruction.shape[2]) ] = cars_map[max(cc_voxel[0].astype(int),0):min(cc_voxel[1].astype(int),reconstruction.shape[0]),
                                max(cc_voxel[2].astype(int),0):min(cc_voxel[3].astype(int),reconstruction.shape[1]),
                                max(cc_voxel[4].astype(int),0):min(cc_voxel[5].astype(int),reconstruction.shape[2]) ]+0.1
                            if temporal:
                                if run_2D:
                                    cars_map[max(cc_voxel[2].astype(int), 0):min(cc_voxel[3].astype(int), reconstruction.shape[1]),
                                    max(cc_voxel[4].astype(int), 0):min(cc_voxel[5].astype(int),
                                                                        reconstruction.shape[2])] = loc_frame*np.ones_like(cars_map[
                                                                                                                 max(cc_voxel[
                                                                                                                         2].astype(
                                                                                                                     int),
                                                                                                                     0):min(
                                                                                                                     cc_voxel[
                                                                                                                         3].astype(
                                                                                                                         int),
                                                                                                                     reconstruction.shape[
                                                                                                                         1]),
                                                                                                                 max(cc_voxel[
                                                                                                                         4].astype(
                                                                                                                     int),
                                                                                                                     0):min(
                                                                                                                     cc_voxel[
                                                                                                                         5].astype(
                                                                                                                         int),
                                                                                                                     reconstruction.shape[
                                                                                                                         2])])
                                else:
                                    cars_map[
                                    max(cc_voxel[0].astype(int), 0):min(cc_voxel[1].astype(int), reconstruction.shape[0]),
                                    max(cc_voxel[2].astype(int), 0):min(cc_voxel[3].astype(int), reconstruction.shape[1]),
                                    max(cc_voxel[4].astype(int), 0):min(cc_voxel[5].astype(int),
                                                                        reconstruction.shape[
                                                                            2])] = loc_frame * np.ones_like(cars_map[max(
                                        cc_voxel[0].astype(int), 0):min(cc_voxel[1].astype(int), reconstruction.shape[0]),
                                                                                                            max(cc_voxel[
                                                                                                                2].astype(
                                                                                                                int),
                                                                                                                0):min(
                                                                                                                cc_voxel[
                                                                                                                    3].astype(
                                                                                                                    int),
                                                                                                                reconstruction.shape[
                                                                                                                    1]),
                                                                                                            max(cc_voxel[
                                                                                                                4].astype(
                                                                                                                int),
                                                                                                                0):min(
                                                                                                                cc_voxel[
                                                                                                                    5].astype(
                                                                                                                    int),
                                                                                                                reconstruction.shape[
                                                                                                                    2])])

                            loc_frame=loc_frame+1
                            closest_car=closest_car+vel


            reconstruction[car_a[0]:car_a[1], car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory] =reconstruction[car_a[0]:car_a[1], car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory]+0.1
            if run_2D:
                reconstruction_2D[ car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory] = reconstruction_2D[car_a[2]:car_a[3],car_a[4]:car_a[5], CHANNELS.cars_trajectory] + 0.1

            if temporal and predict_future:

                reconstruction[car_a[0]:car_a[1], car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory] =frame* np.ones_like(reconstruction[
                                                                                             car_a[0]:car_a[1],
                                                                                             car_a[2]:car_a[3],
                                                                                             car_a[4]:car_a[5], CHANNELS.cars_trajectory] )
                if run_2D:
                    reconstruction_2D[ car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory] = frame * np.ones_like(
                        reconstruction_2D[car_a[2]:car_a[3],car_a[4]:car_a[5], CHANNELS.cars_trajectory])
        #if no_dict:
        if run_2D:
            cars_predicted.append(reconstruction_2D[ :, :, CHANNELS.cars_trajectory].copy())
        else:
            cars_predicted.append(reconstruction[:,:,:,CHANNELS.cars_trajectory].copy())

        if predict_future:
            if no_dict:
                if temporal:
                    temp_array =cars_predicted[-1]>0
                    cars_predicted[-1] =cars_predicted[-1]-(frame* temp_array)
                    temp_2 = cars_map > 0
                    cars_predicted[-1][temp_2] = cars_map[temp_2].copy()

                else:
                    temp_2 = cars_map > 0
                    cars_predicted[-1][temp_2] =cars_predicted[-1][temp_2]+ cars_map[temp_2].copy()
            else:
                if temporal:
                    temp_array = cars_predicted[-1] > 0
                    cars_predicted[-1] = cars_predicted[-1] - (frame * temp_array)

        # print "Predicted "+ str(np.sum(cars_predicted[-1][:]))
    for frame in range(len(people_a)):
        if run_2D:
            people_map = np.zeros(tensor.shape[1:3])
        else:
            people_map = np.zeros(tensor.shape[:3])  # .shape[1:3])
        for person in people_a[frame]:
            x_pers=[]
            person_middle_exact=np.array([np.mean(person[0]), np.mean(person[1]), np.mean(person[2])])
            for count in range(3):
                p=person[count, :].copy()
                if len(p)>0:
                    x_pers.append((max(min(p),0), min(max(p)+1,reconstruction.shape[count])))
            if len(x_pers)==3:
                if run_2D:
                    reconstruction_2D[ int(x_pers[1][0]):int( x_pers[1][1]),
                    int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] =reconstruction_2D[int(x_pers[1][0]):int( x_pers[1][1]),
                    int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] +0.1

                reconstruction[int(x_pers[0][0]):int(x_pers[0][1]), int(x_pers[1][0]):int(x_pers[1][1]),
                int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] = reconstruction[int(x_pers[0][0]):int(x_pers[0][1]),
                                                              int(x_pers[1][0]):int(x_pers[1][1]),
                                                              int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] + 0.1
            if temporal and predict_future:
                if run_2D:
                    reconstruction_2D[ int(x_pers[1][0]):int(x_pers[1][1]),
                    int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] = frame * np.ones_like(
                        reconstruction_2D[
                        int(x_pers[1][0]):int(x_pers[1][1]),
                        int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory])

                reconstruction[int(x_pers[0][0]):int(x_pers[0][1]), int(x_pers[1][0]):int(x_pers[1][1]),
                int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] = frame*np.ones_like(reconstruction[int(x_pers[0][0]):int(x_pers[0][1]),
                                                              int(x_pers[1][0]):int(x_pers[1][1]),
                                                              int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] )
            if predict_future:

                closest_person = []
                cc_voxel = [0, 0, 0]
                vel = [0, 0, 0]
                min_dist = 50
                if frame > 0 and no_dict:
                    for person_2 in people_a[frame - 1]:
                        person_middle2 = np.array([np.mean(person_2[0]), np.mean(person_2[1]), np.mean(person_2[2])])
                        dist = np.linalg.norm(person_middle_exact[1:] - person_middle2[1:])
                        if dist < min_dist:
                            closest_person = person_middle2
                            cc_voxel = np.array(person_2)
                            vel = person_middle_exact - person_middle2
                            min_dist = dist.copy()


                    vel[0] = 0

                    if np.linalg.norm(vel) > 0:
                        loc_frame=1
                        while np.all(closest_person[1:] > 1) and np.all(reconstruction.shape[1:3] - closest_person[1:] > 1):

                            cc_voxel = cc_voxel + np.tile(vel,[2,1]).T

                            if run_2D:
                                people_map[
                                max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int),
                                                                       reconstruction.shape[1]),
                                max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                       reconstruction.shape[2])] = people_map[max(cc_voxel[1][
                                                                                                           0].astype(
                                                                                                       int), 0):min(
                                                                                                       cc_voxel[1][
                                                                                                           1].astype(
                                                                                                           int),
                                                                                                       reconstruction.shape[
                                                                                                           1]),
                                                                                                   max(cc_voxel[2][
                                                                                                           0].astype(
                                                                                                       int), 0):min(
                                                                                                       cc_voxel[2][
                                                                                                           1].astype(
                                                                                                           int),
                                                                                                       reconstruction.shape[
                                                                                                           2])] + 0.1
                            else:

                                people_map[max(cc_voxel[0][0].astype(int), 0):min(cc_voxel[0][1].astype(int), reconstruction.shape[0]),
                                max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int), reconstruction.shape[1]),
                                max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                    reconstruction.shape[2])] =  people_map[max(cc_voxel[0][0].astype(int), 0):min(cc_voxel[0][1].astype(int), reconstruction.shape[0]),
                                max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int), reconstruction.shape[1]),
                                max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                    reconstruction.shape[2])] +0.1
                            if temporal:
                                if run_2D:
                                    people_map[
                                    max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int), reconstruction.shape[1]),
                                    max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                           reconstruction.shape[2])] = loc_frame*np.ones_like(people_map[
                                                                                                                    max(
                                                                                                                        cc_voxel[
                                                                                                                            1][
                                                                                                                            0].astype(
                                                                                                                            int),
                                                                                                                        0):min(
                                                                                                                        cc_voxel[
                                                                                                                            1][
                                                                                                                            1].astype(
                                                                                                                            int),
                                                                                                                        reconstruction.shape[
                                                                                                                            1]),
                                                                                                                    max(
                                                                                                                        cc_voxel[
                                                                                                                            2][
                                                                                                                            0].astype(
                                                                                                                            int),
                                                                                                                        0):min(
                                                                                                                        cc_voxel[
                                                                                                                            2][
                                                                                                                            1].astype(
                                                                                                                            int),
                                                                                                                        reconstruction.shape[
                                                                                                                            2])])
                                else:
                                    people_map[
                                    max(cc_voxel[0][0].astype(int), 0):min(cc_voxel[0][1].astype(int),
                                                                           reconstruction.shape[0]),
                                    max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int),
                                                                           reconstruction.shape[1]),
                                    max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                           reconstruction.shape[
                                                                               2])] = loc_frame * np.ones_like(
                                        people_map[
                                        max(
                                            cc_voxel[
                                                0][
                                                0].astype(
                                                int),
                                            0):min(
                                            cc_voxel[
                                                0][
                                                1].astype(
                                                int),
                                            reconstruction.shape[
                                                0]),
                                        max(
                                            cc_voxel[
                                                1][
                                                0].astype(
                                                int),
                                            0):min(
                                            cc_voxel[
                                                1][
                                                1].astype(
                                                int),
                                            reconstruction.shape[
                                                1]),
                                        max(
                                            cc_voxel[
                                                2][
                                                0].astype(
                                                int),
                                            0):min(
                                            cc_voxel[
                                                2][
                                                1].astype(
                                                int),
                                            reconstruction.shape[
                                                2])])

                            loc_frame=loc_frame+1
                            closest_person = closest_person + vel
        if run_2D:
            people_predicted.append(reconstruction_2D[ :, :, CHANNELS.pedestrian_trajectory].copy())
            # print("Creating people predicted: "+str(np.sum(np.abs( people_predicted[-1]))))
        else:
            people_predicted.append(reconstruction[:, :, :, CHANNELS.pedestrian_trajectory].copy())
        # print ("No dict? "+str(no_dict)+" temporal? "+str(temporal)+" predict_future "+str(predict_future))
        if no_dict:
            if temporal and predict_future:
                temp_array=people_predicted[-1]>0
                temp_2=people_map>0

                people_predicted[-1] = people_predicted[-1]-(frame*temp_array)
                people_predicted[-1][temp_2]=people_map[temp_2].copy()

            else:
                temp_2 = people_map > 0
                people_predicted[-1][temp_2] =people_predicted[-1][temp_2]+ people_map[temp_2].copy()
        else:
            if temporal:
                temp_array = people_predicted[-1] > 0
                people_predicted[-1] = people_predicted[-1] - (frame * temp_array)
        # print("After processing people : " + str(np.sum(np.abs(people_predicted[-1]))))

    return reconstruction, cars_predicted,people_predicted, reconstruction_2D


#@timing
def reconstruct_static_objects_old( tensor, run_2D):
    reconstruction = tensor.copy()
    #print("REC 1")
    if run_2D:
        reconstruction_2D = np.zeros(reconstruction.shape[1:])
        for x in range(reconstruction_2D.shape[0]):
            #if x % 100 == 0:
            #    print(f"x {x} / {reconstruction_2D.shape[0]}")

            for y in range(reconstruction_2D.shape[1]):
                z = return_highest_col(tensor[:, x, y, :3])
                reconstruction_2D[x, y, CHANNELS.rgb[0]] = tensor[z, x, y, CHANNELS.rgb[0]]
                reconstruction_2D[x, y, CHANNELS.rgb[0]+1] = tensor[z, x, y, CHANNELS.rgb[0]+1]
                reconstruction_2D[x, y, CHANNELS.rgb[0]+2] = tensor[z, x, y, CHANNELS.rgb[0]+2]
                z1 = return_highest_object(tensor[:, x, y, CHANNELS.semantic])
                reconstruction_2D[x, y, CHANNELS.semantic] = tensor[z1, x, y, CHANNELS.semantic]
    else:
        reconstruction_2D=[]
    return reconstruction, reconstruction_2D

#@timing
def reconstruct_static_objects(tensor, run_2D):
    #print("REC ALT")
    rgb_index_s = CHANNELS.rgb[0]
    rgb_index_e = CHANNELS.rgb[1]
    sem_index = CHANNELS.semantic
    reconstruction = tensor.copy()
    if run_2D:
        reconstruction_2D = np.zeros(reconstruction.shape[1:])
        for x in range(reconstruction_2D.shape[0]):
            for y in range(reconstruction_2D.shape[1]):
                z1 = return_highest_object_v1(tensor[:, x, y, sem_index])
                reconstruction_2D[x, y, sem_index] = tensor[z1, x, y, sem_index]

                reconstruction_2D[x, y, rgb_index_s:rgb_index_e] = tensor[z1, x, y, rgb_index_s:rgb_index_e]

    else:
        reconstruction_2D=[]
    return reconstruction, reconstruction_2D
