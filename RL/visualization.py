import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.utils_functions import get_goal_frames
import multiprocessing

import scipy.stats as stats

from PFNNPythonLiveVisualizer import PFNNLiveVisualizer as PFNNViz
from commonUtils.ReconstructionUtils import CHANNELS
import sys
print(sys.path)
from RL.settings import STATISTICS_INDX,STATISTICS_INDX_POSE, STATISTICS_INDX_MAP, STATISTICS_INDX_CAR, STATISTICS_INDX_CAR_INIT,PEDESTRIAN_MEASURES_INDX, PEDESTRIAN_REWARD_INDX, CAR_REWARD_INDX, CAR_MEASURES_INDX, PEDESTRIAN_INITIALIZATION_CODE
from commonUtils.ReconstructionUtils import cityscapes_colours as colours
from commonUtils.ReconstructionUtils import cityscapes_labels as labels
from commonUtils.ReconstructionUtils import  NUM_SEM_CLASSES
#from RL.settings import STATISTICS_INDX,STATISTICS_INDX_POSE, STATISTICS_INDX_MAP, STATISTICS_INDX_CAR, STATISTICS_INDX_CAR_INIT,PEDESTRIAN_MEASURES_INDX, PEDESTRIAN_REWARD_INDX, CAR_REWARD_INDX, CAR_MEASURES_INDX, PEDESTRIAN_INITIALIZATION_CODE, NUM_SEM_CLASSES



actions = []

v = [-1, 0, 1]
j = 0
for y in range(3):
    for x in range(3):
        actions.append([ v[y], v[x]])
        j += 1
actions = [actions[k] for k in [4, 1, 0, 3, 6, 7, 8, 5, 2]]

# Ciprian's code:

def getBonePos(poseData, boneId):
    x = poseData[boneId*3 + 0]
    y = poseData[boneId*3 + 1]
    z = poseData[boneId*3 + 2]
    return x,y,z

def distance2D(x0, z0, x1, z1):
    return math.sqrt((x0-x1)**2 + (z0-z1)**2)

def prettyPrintBonePos(poseData, boneID):
    print(("Bone {0} has position: ({1:.2f},{2:.2f},{3:.2f})".format(boneID, *getBonePos(poseData, boneID))))


# Visualize what the agent sees.
def view_agent_input(id, frame, agent,agent_shape,agent_sight ,episode):
    #id, pos, breadth,frame_in,vel=[]
    neigh ,_,_,_,_= episode.get_agent_neighbourhood(id, agent, [agent_shape[0] + agent_sight[0], agent_shape[1] + agent_sight[1],
                                            agent_shape[2] + agent_sight[2]],0, pedestrian_view_occluded=self.settings.pedestrian_view_occluded, field_of_view=self.settings.field_of_view)
    trans=0.9
    if episode.run_2D:
        img_from_above_neigh = np.ones((neigh.shape[1],neigh.shape[0], 3),dtype=np.uint8)*255
    else:
        img_from_above_neigh = np.ones((neigh.shape[2], neigh.shape[1], 3), dtype=np.uint8) * 255

    im_2d_agent=view_2D(neigh, img_from_above_neigh,0, people=True)

    im_2d_agent[0:agent_sight[0],:,:]=trans*im_2d_agent[0:agent_sight[0],:,:]+(1-trans)*255.0*np.ones(im_2d_agent[0:agent_sight[0],:,:].shape)
    im_2d_agent[-agent_sight[0], :, :] = trans * im_2d_agent[-agent_sight[0] , :, :] + (1 - trans) * 255.0*np.ones(
        im_2d_agent[-agent_sight[0], :, :].shape)

    im_2d_agent[agent_sight[0] :-agent_sight[0],0:agent_sight[1], :] = trans * im_2d_agent[agent_sight[0] :-agent_sight[0],0:agent_sight[1], :] + (1 - trans) * 255.0*np.ones(
        im_2d_agent[agent_sight[0] :-agent_sight[0],0:agent_sight[1], :].shape)

    im_2d_agent[agent_sight[0] :-agent_sight[0],-agent_sight[1], :] = trans * im_2d_agent[agent_sight[0] :-agent_sight[0],-agent_sight[1], :] + (1 - trans) *255.0* np.ones(
        im_2d_agent[agent_sight[0] :-agent_sight[0],-agent_sight[1], :].shape)
    return im_2d_agent.astype(np.uint8)

def view_agent_input_fast( pos,agent_shape,agent_sight ,img_from_above):
    start_pos = np.zeros(2, np.int)
    min_pos = np.zeros(2, np.int)
    max_pos = np.zeros(2, np.int)
    breadth=[ agent_shape[1]+agent_sight[1], agent_shape[2]+agent_sight[2]]
    mini_ten = np.zeros((breadth[0] * 2 + 1, breadth[1] * 2 + 1, 6), dtype=np.float)
    for i in range(2):
        min_pos[i] = pos[i] - breadth[i]
        max_pos[i] = pos[i] + breadth[i] + 1
        if min_pos[i] < 0:  # agent below rectangle of visible.
            start_pos[i] = -min_pos[i]
            min_pos[i] = 0
            if max_pos[i] < 0:
                return mini_ten
        if max_pos[i] > img_from_above.shape[i]:
            max_pos[i] = img_from_above.shape[i]
            if min_pos[i] > img_from_above.shape[i]:
                return mini_ten
    tmp=img_from_above[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], :].copy()
    mini_ten[start_pos[0]:start_pos[0] + tmp.shape[0], start_pos[1]:start_pos[1] + tmp.shape[1], :] = tmp

    trans=0.9
    img_from_above_neigh = np.ones((mini_ten.shape[2],mini_ten.shape[1], 3),dtype=np.uint8)*255
    im_2d_agent=view_2D(mini_ten, img_from_above_neigh,0, people=True)
    im_2d_agent[0:agent_sight[0],:,:]=trans*im_2d_agent[0:agent_sight[0],:,:]+(1-trans)*255.0*np.ones(im_2d_agent[0:agent_sight[0],:,:].shape)
    im_2d_agent[-agent_sight[0], :, :] = trans * im_2d_agent[-agent_sight[0] , :, :] + (1 - trans) * 255.0*np.ones(
        im_2d_agent[-agent_sight[0], :, :].shape)

    im_2d_agent[agent_sight[0] :-agent_sight[0],0:agent_sight[1], :] = trans * im_2d_agent[agent_sight[0] :-agent_sight[0],0:agent_sight[1], :] + (1 - trans) * 255.0*np.ones(
        im_2d_agent[agent_sight[0] :-agent_sight[0],0:agent_sight[1], :].shape)
    im_2d_agent[agent_sight[0] :-agent_sight[0],-agent_sight[1], :] = trans * im_2d_agent[agent_sight[0] :-agent_sight[0],-agent_sight[1], :] + (1 - trans) *255.0* np.ones(
        im_2d_agent[agent_sight[0] :-agent_sight[0],-agent_sight[1], :].shape)
    return im_2d_agent.astype(np.uint8)

# Insert agent, any pedestrians and any cars.
def view_agent(  agent, img_from_above,seq_len, size_a, agent_sight, car=[], people=[], cars=[], frame=-1, valid_action=True, goal=[],no_tracks=False, agent_frame=-1 , hit_by_car=-1, x_min=0, y_min=0, car_goal=[], colors=[], supress=False):
    current_frame = img_from_above.copy() # ( frame, people, current_frame, width, depth)
    #print("Agent goal in view_agent: "+str(goal))
    if agent_frame<0:
        agent_frame=frame

    if len(people)>0 or len(cars)>0:
        current_frame = view_pedestrians(agent_frame, people, current_frame, seq_len, trans=0.1, no_tracks=no_tracks,  x_min=x_min, y_min=y_min)
        current_frame = view_cars(current_frame, cars, current_frame.shape[1], current_frame.shape[2],seq_len,car, frame=agent_frame, hit_by_car=hit_by_car,  x_min=x_min, y_min=y_min) #frame=frame)
    else:
        if not supress:
            print("No cars or pedestrian in frame "+str(agent_frame))
    # Current position.

    transparency_sight = 0.9
    transparency = 0.1
    dif = []
    if len(car_goal) > 0 and not np.linalg.norm(car_goal) == 0:
        for cgoal in car_goal:
            plot_car_goal(current_frame, cgoal[1:], seq_len, 1)
    if isinstance(agent, list): # To do. Should be ndarray or array?

        for agent_id in range(len(agent)):
            color = []
            if len(colors)>agent_id:
                color=colors[agent_id]
            local_goal=[]
            if len(goal)>0:
                local_goal=goal[agent_id]
            add_pedestrian_agent(agent_sight, current_frame, dif,local_goal, img_from_above,
                                 agent[agent_id], seq_len, size_a, transparency,
                                 x_min, y_min,frame=agent_frame, color=color, id=agent_id)
    else:
        for agent_id in range(agent.shape[0]):
            color = []
            if len(colors) > agent_id:
                color = colors[agent_id]
            # if len(car_goal) > 0 and not np.linalg.norm(car_goal) == 0:
            #     plot_car_goal(current_frame, car_goal[1:], seq_len, 1)
            add_pedestrian_agent(agent_sight, current_frame, dif, goal[agent_id,:,:], img_from_above, agent[agent_id,:,:], seq_len, size_a, transparency,
                                 x_min, y_min,frame=agent_frame,color=color, id=agent_id)
    return current_frame


def add_pedestrian_agent(agent_sight, current_frame, dif, goal, img_from_above, agent, seq_len, size_a, transparency,
                         x_min, y_min, frame=-1, color=[], id=-1):
    pos=[]


    if frame>=0:
        pos = agent[frame]
    else:
        pos=agent
    if np.sum(np.abs(pos))==0:
        print("Agent pos is 0 frame "+str(frame)+" "+str(id))
    if x_min > 0 or y_min > 0:
        seq_len = 0
    for a_dim, s_dim in zip(agent_sight, size_a):
        dif.append(a_dim + s_dim)
    if len(pos) == 3:
        person_bbox = [max(current_frame.shape[0] - 1 - int(seq_len + pos[1] + size_a[1] - y_min), 0),
                       min(current_frame.shape[0] - 1 - int(seq_len + pos[1] - size_a[1] - y_min),
                           current_frame.shape[0] - 1),
                       max(int(seq_len + pos[2] - size_a[2] - x_min), 0),
                       min(int(seq_len + pos[2] + size_a[2] - x_min), current_frame.shape[1] - 1)]

    else:
        person_bbox = [max(current_frame.shape[0] - 1 - int(seq_len + pos[0] + size_a[1] - y_min), 0),
                       min(current_frame.shape[0] - 1 - int(seq_len + pos[0] - size_a[1] - y_min),
                           current_frame.shape[0] - 1),
                       max(int(seq_len + pos[1] - size_a[2] - x_min), 0),
                       min(int(seq_len + pos[1] + size_a[2] - x_min), current_frame.shape[1] - 1)]

    if person_bbox[0] < person_bbox[1] and person_bbox[2] < person_bbox[3]:
        if min(person_bbox) > 0 and max(person_bbox[0:2]) < img_from_above.shape[0] and max(person_bbox[2:]) < \
                img_from_above.shape[1]:
            if len(goal) > 0 and not np.linalg.norm(goal) == 0:

                plot_car_goal(current_frame, goal[frame,:], seq_len,2)

            if color:
                color_local = np.array([int(color[0] * 256), int(color[1] * 256), int(color[2] * 256)]).reshape((1, 1, 3))

            if len(current_frame.shape)==3:


                if color:

                    bbox_size =  current_frame[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3], :].shape
                    current_frame[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3], :] = np.tile(color_local, (bbox_size[0], bbox_size[1], 1))
                else:
                    current_frame[person_bbox[0]:person_bbox[1] + 1, person_bbox[2]:person_bbox[3] + 1, :] = (
                            transparency * current_frame[person_bbox[0]:person_bbox[1] + 1,
                                           person_bbox[2]:person_bbox[3] + 1,
                                           :]).astype(int)
            else:

                if color:
                    bbox_size =current_frame[person_bbox[0]:person_bbox[1] , person_bbox[2]:person_bbox[3],:].shape
                    current_frame[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3], :] =np.tile(color_local, (bbox_size[0], bbox_size[1], 1))
                else:
                    current_frame[person_bbox[0]:person_bbox[1] + 1, person_bbox[2]:person_bbox[3] + 1] = (
                            transparency * current_frame[person_bbox[0]:person_bbox[1] + 1,
                                           person_bbox[2]:person_bbox[3] + 1]).astype(int)
            # else:
            #    current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :]=np.zeros(current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :].shape)


def plot_car_goal(current_frame, goal, seq_len, channel_ones, color=[]):
    goal_bbox = [current_frame.shape[0] - 1 - int(seq_len + goal[0] + 1),
                 current_frame.shape[0] - 1 - int(seq_len + goal[0] - 1),
                 int(seq_len + goal[1] - 1), int(seq_len + goal[1] + 1)]
    if color:
        bbox_size=current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3],:].shape
        color_local=np.array([int(color[0]*256),int(color[1]*256),int(color[2]*256)]).reshape((1,1,3))
        current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3],:] = np.tile(color_local, (bbox_size[0], bbox_size[1],1))

    else:
        if len(current_frame.shape)==3:
            current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3], channel_ones:3] = np.zeros(
                current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3], channel_ones:3].shape)
            current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3], channel_ones] = np.ones(
                current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3], channel_ones].shape) * 256
        else:
            current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3]] = np.zeros(
                current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3]].shape)
            current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3]] = np.ones(
                current_frame[goal_bbox[0]:goal_bbox[1], goal_bbox[2]:goal_bbox[3]].shape) * 256


def view_car_agent_nicer(agent, img_from_above, frame,  colors, transparency = 0.6):
    current_frame = img_from_above.copy() # ( frame, people, current_frame, width, depth)
    pixels = set()
    if len(agent.shape) == 2:
        frames=0
        ids=agent.shape[0]
    elif len(agent.shape) == 3:
        frames = agent.shape[1]
        ids = agent.shape[0]
    for f in range(frames):
        for id in range(ids):
            if len(agent.shape)==2:
                bbox = agent[id, :]
            else:
                bbox = agent[id,f,:]
            dif = []
            if frame==f:
                transparency_loc=2*transparency
            else:
                transparency_loc = transparency
            person_bbox=[current_frame.shape[0] - 1 - int(bbox[3]),
             current_frame.shape[0] - 1 - int(bbox[2]),
             int(bbox[4]), int(bbox[5])]
            s = current_frame[person_bbox[0]:person_bbox[1] + 1, person_bbox[2]:person_bbox[3] + 1, :].shape
            if person_bbox[0]>0 and person_bbox[2]>0 and person_bbox[1]>0 and person_bbox[3]>0 and person_bbox[1]<img_from_above.shape[0] and person_bbox[2]<img_from_above.shape[1] and person_bbox[0]<img_from_above.shape[0] and person_bbox[3]<img_from_above.shape[1]:

               for x in range(person_bbox[0], person_bbox[1]):
                    for y in range(person_bbox[2], person_bbox[3]):
                        print ("str "+str(id) )
                        print(colors)
                        pixels.add((x, y,  colors[id]))
    for pixel in pixels:
        col=pixel[2]
        current_frame[pixel[0], pixel[1], :] = (current_frame[pixel[0], pixel[1], :] * (1 - transparency)).astype(
            int) + (np.array(col[0:3])*256 * transparency).astype(int)

        #print pixel

    return current_frame

def view_agent_nicer(agent, img_from_above, size_a, agent_sight, frame,colours_in, transparency = 0.6):
    current_frame = img_from_above.copy() # ( frame, people, current_frame, width, depth)
    pixels = set()
    if len(agent.shape) == 2:
        frames=1
        ids=agent.shape[0]
    else:
        frames =agent.shape[1]
        ids =agent.shape[0]
    for f in range(frames):
        for id in range(ids):
            if len(agent.shape)==2:
                pos = agent[id,  :]
            else:
                pos = agent[id,f,:]
            dif = []
            if frame==f:
                transparency_loc=2*transparency
                for a_dim, s_dim in zip(agent_sight, size_a):
                    dif.append(a_dim + s_dim)
                if len(pos)==3:
                    person_bbox = [current_frame.shape[0] - 1 - int( pos[1]+ size_a[1]),
                                   current_frame.shape[0] - 1 - int( pos[1]- size_a[1]),
                                   int( pos[2]- size_a[2]), int( pos[2]+ size_a[2])]
                else:
                    person_bbox = [current_frame.shape[0] - 1 - int( pos[0] + size_a[1]),
                                   current_frame.shape[0] - 1 - int( pos[0] - size_a[1]),
                                   int( pos[1] - size_a[2]), int( pos[1] + size_a[2])]

            else:
                transparency_loc = transparency
                if len(pos)==3:
                    person_bbox = [current_frame.shape[0] - 1 - int( pos[1]+ 1),
                                   current_frame.shape[0] - 1 - int( pos[1]- 2),
                                   int( pos[2]- 1), int( pos[2]+ 2)]
                else:

                    person_bbox = [current_frame.shape[0] - 1 - int( pos[0] + 1),
                                   current_frame.shape[0] - 1 - int( pos[0] - 2),
                                   int( pos[1] - 1), int( pos[1] + 2)]
            s = current_frame[person_bbox[0]:person_bbox[1] + 1, person_bbox[2]:person_bbox[3] + 1, :].shape

            if person_bbox[0]>0 and person_bbox[2]>0 and person_bbox[1]>0 and person_bbox[3]>0 and person_bbox[1]<img_from_above.shape[0] and person_bbox[2]<img_from_above.shape[1] and person_bbox[0]<img_from_above.shape[0] and person_bbox[3]<img_from_above.shape[1]:

                for x in range(person_bbox[0], person_bbox[1]):
                    for y in range(person_bbox[2], person_bbox[3]):
                        pixels.add((x, y, f, colours_in[id]))
    for pixel in pixels:
        trans=transparency

        col=pixel[3]

        color=np.array(col[0:3])*256

        current_frame[pixel[0], pixel[1], :] = (current_frame[pixel[0], pixel[1], :] * (1 - trans)).astype(
            int) + (np.array(color) * trans).astype(int)

        #print pixel

    return current_frame

# Visualization of cars.
def view_cars_nicer( current_frame, cars, frame,transparency=0.6):

    pixels=set()
    frames=list(range(0, len(cars)))
    col = (0, 0, 255)
    for f in frames:
        for car in cars[f]:
            if f==frame:
                #transparency_loc=3*transparency
                car_bbox = [max(current_frame.shape[0] - 1 - int( car[3] - 1), 0),
                            min(current_frame.shape[0] - 1 - int( car[2]) + 1, current_frame.shape[0] - 1),
                            max( car[4], 0), min( car[5], current_frame.shape[1] - 1)]
            else:
                #transparency_loc=transparency
                middle = [int(np.mean(car[0:2])), int(np.mean(car[2:4])), int(np.mean(car[4:]))]
                car_bbox = [max(current_frame.shape[0] - 1 - int(middle[1]+2),0),
                            min(current_frame.shape[0] - 1 - int(middle[1]-2)+1,current_frame.shape[0]-1 ),
                            max(middle[2]-2,0), min(middle[2]+2, current_frame.shape[1]-1)]

            if car_bbox[0]<current_frame.shape[0]  and  car_bbox[2]<current_frame.shape[1] and car_bbox[1]>=0 and car_bbox[3]>=0 :
                for x in range(int(car_bbox[0]), int(car_bbox[1])+1):
                    for y in range(int(car_bbox[2]), int(car_bbox[3])+1):
                        pixels.add((x,y))
                # sh = current_frame[car_bbox[0]:car_bbox[1],car_bbox[2]:car_bbox[3], :]
                # current_frame[car_bbox[0]:car_bbox[1],car_bbox[2]:car_bbox[3], :] =(current_frame[car_bbox[0]:car_bbox[1],car_bbox[2]:car_bbox[3], :]*(1-transparency_loc)).astype(int)+ (np.tile(col,(sh.shape[0],sh.shape[1], 1))*transparency_loc).astype(int)
    for pixel in pixels:
        current_frame[pixel[0], pixel[1], :]=(current_frame[pixel[0], pixel[1], :]*(1-transparency)).astype(int)+(np.array(col)*transparency).astype(int)
    return current_frame

def view_pedestrians_nicer(frame, people, current_frame, trans=.6, red=False):
    pixels = set()
    for i, person_list in enumerate(people):
         for person in person_list:
            if frame == i:
                transparency_loc=trans*2
                person_bbox = [ max(current_frame.shape[0]-1-int(person[1, 1]),0),min(current_frame.shape[0]-1-int(person[1, 0]), current_frame.shape[0]-1),
                               max(int(person[2, 0]),0), min(int(person[2, 1]), current_frame.shape[1]-1)]
                #print "Pedestrian size" + str(person_bbox)
            else:
                transparency_loc = trans
                middle=np.mean(person, axis=1).astype(int)
                person_bbox = [max(current_frame.shape[0] - 1 - int(middle[1]+1), 0),
                               min(current_frame.shape[0] - 1 - int(middle[1]-1),current_frame.shape[0] - 1),
                               max(int( middle[2]-1), 0),
                               min(int( middle[2]+1), current_frame.shape[1] - 1)]
            s=current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :].shape

            if 0 not in s and person_bbox[0]<current_frame.shape[0] and person_bbox[2]<current_frame.shape[1] and person_bbox[1]>=0 and person_bbox[3]>=0:
                # current_frame[person_bbox[0]:person_bbox[1] + 1, person_bbox[2]:person_bbox[3] + 1, :] = \
                #     current_frame[person_bbox[0]:person_bbox[1] + 1, person_bbox[2]:person_bbox[3] + 1, :] * (
                #     1 - transparency_loc) \
                #     + transparency_loc * np.tile(( 0, 255, 0), (s[0], s[1], 1))  # np.tile([0,0,0], (s[0], s[1], 1))
                for x in range(person_bbox[0], person_bbox[1] + 1):
                    for y in range(person_bbox[2], person_bbox[3] + 1):
                        pixels.add((x, y))
    if red:
        col=[255, 0, 0]
    else:
        col=[0, 0, 255]
    for pixel in pixels:
        current_frame[pixel[0], pixel[1], :] = (current_frame[pixel[0], pixel[1], :] * (1 - trans)).astype(int) + (np.array(col) * trans).astype(int)

    return current_frame



# Visualization of cars.
def  view_cars( current_frame, cars, width, depth, seq_len, car, frame=-1, tensor=np.array((0)),  hit_by_car=-1 , transparency=0.1, x_min=0, y_min=0,show_tensor_intentsity=False):


    transparency = 0.5
    if not tensor.shape:
        frames = []
        if frame>=0:
            if frame<len(cars):
                frames.append(frame)
        else:
            frames=list(range(0, len(cars)))

        for f in frames:

            for car_cur in cars[f]:

                label=26
                col=colours[label]
                col=(col[2],col[1], col[0])
                plot_car(car_cur, col, current_frame, seq_len, transparency, x_min, y_min)
                # print (car_cur)
            label = 25
            col = colours[label]
            col = (col[2], col[1], col[0])
            # print ("Car "+str(car)+" frame "+str(f))
            # print ("Conditions: "+str(len(car)>0 ) + " "+str( len(car[f])>1))
            if isinstance(car, list):
                for car_id in range(len(car)):

                    plot_car_list_obj(car, car_id, col, current_frame, f, seq_len, transparency, x_min, y_min)
            else:
                for car_id in range(car.shape[0]):

                    plot_car_agent(car, car_id, col, current_frame, f, seq_len, transparency, x_min, y_min)

        if frame >= len(cars):

            print("Want to visualize frame " + str(frame) + " but length of ped list is " + str(len(cars)))
    else:

        selected_tensor_area=tensor
        if len(tensor.shape)==4:
            selected_tensor_area=tensor[:, :, :, CHANNELS.cars_trajectory]

        if show_tensor_intentsity:

            inverted = np.flip(np.amax(selected_tensor_area, axis=0), axis=0)

            max_val=max(np.max(inverted),1)
            inverted=inverted/max_val

            for i in range(3):
                current_frame[:,:,i]=current_frame[:,:,i]*(1-transparency)*(np.ones(np.shape(inverted)[0:2])-inverted) +current_frame[:,:,i]*(transparency)*(inverted*0)#current_frame[:,:,i]*(np.ones(np.shape(inverted)[0:2])-inverted) +current_frame[:,:,i]*(inverted*colours[26][i])
        else:
            inverted=np.flip(np.amax(selected_tensor_area, axis=0), axis=0)
            inverted =inverted!=0
            inverted=np.concatenate(( inverted[:,:,np.newaxis],np.zeros_like(inverted[:,:,np.newaxis]),np.zeros_like(inverted[:,:,np.newaxis]) ), axis=2)
            current_frame[inverted] = 1  # current_frame[:,:,i]*(np.ones(np.shape(inverted)[0:2])-inverted) +current_frame[:,:,i]*(inverted*colours[26][i])

    return current_frame


def plot_car_agent(car, car_id, col, current_frame, f, seq_len, transparency, x_min, y_min):
    isCarInstanceValidOnFrame = isinstance(car[car_id, :, :], list) and car[f] != None and len(car[car_id, f, :]) > 1
    isCarInstanceValidOnFrame = isCarInstanceValidOnFrame or (car.any() and car.size > f and len(car[car_id, f, :]) > 1)
    if isCarInstanceValidOnFrame:
        plot_car(car[car_id, f, :], col, current_frame, seq_len, transparency, x_min, y_min)


def plot_car_list_obj(car, car_id, col, current_frame, f, seq_len, transparency, x_min, y_min):
    isCarInstanceValidOnFrame = isinstance(car[car_id], list) and car[car_id][f] != None and len(
        car[car_id][f]) > 1
    isCarInstanceValidOnFrame = isCarInstanceValidOnFrame or (
            car[car_id].any() and car[car_id].size > f and len(car[car_id][f]) > 1)
    if isCarInstanceValidOnFrame:
        plot_car(car[car_id][f], col, current_frame, seq_len, transparency, x_min, y_min)


def plot_car(car, col, current_frame, seq_len, transparency, x_min, y_min):
    car_bbox = [max(current_frame.shape[0] - 1 - int(seq_len + int(car[3]) - 1) - y_min, 0),
                min(current_frame.shape[0] - 1 - int(seq_len + int(car[2])) + 1 - y_min, current_frame.shape[0] - 1),
                max(seq_len + int(car[4]) - x_min, 0), min(seq_len + int(car[5]) - x_min, current_frame.shape[1] - 1)]
    if car_bbox[0] < current_frame.shape[0] and car_bbox[2] < current_frame.shape[1] and car_bbox[1] >= 0 and car_bbox[3] >= 0:
        if len(current_frame.shape)==3:
            sh = current_frame[car_bbox[0]:car_bbox[1], car_bbox[2]:car_bbox[3], :]
            current_frame[car_bbox[0]:car_bbox[1], car_bbox[2]:car_bbox[3], :] = (current_frame[car_bbox[0]:car_bbox[1],
                                                                                  car_bbox[2]:car_bbox[3], :] * (
                                                                                  1 - transparency)).astype(int) + (
                                                                                 np.tile(col, (sh.shape[0], sh.shape[1],
                                                                                               1)) * transparency).astype(int)
        else:
            sh = current_frame[car_bbox[0]:car_bbox[1], car_bbox[2]:car_bbox[3]]
            current_frame[car_bbox[0]:car_bbox[1], car_bbox[2]:car_bbox[3]] = (current_frame[car_bbox[0]:car_bbox[1],car_bbox[2]:car_bbox[3]] * (1 - transparency)) + (np.ones( (sh.shape[0], sh.shape[1])) * transparency)

# Visualization of pedestrians.
def view_pedestrians(frame, people, current_frame, seq_len, trans=.15, tensor=[], no_tracks=False, x_min=0, y_min=0,view_density=False):

    if len(tensor)>0:
        area_to_study=tensor
        if len(tensor.shape)==4:
            area_to_study=tensor[:,:, :, CHANNELS.pedestrian_trajectory]
        if view_density:
            print ("Sum of tensor "+str(np.sum(area_to_study)))
            inverted = np.flip(np.amax(area_to_study, axis=0), axis=0)
            print("Sum of inverted tensor " + str(np.sum(inverted)))
            for i in range(3):
                current_frame[:,:,i]=current_frame[:,:,i]*(np.ones(np.shape(inverted))-inverted) + current_frame[:,:,i]*(inverted*colours[26][i])
        else:
            inverted = np.flip(np.amax(area_to_study, axis=0), axis=0)
            inverted = inverted != 0
            inverted = np.concatenate(( np.zeros_like(inverted[:, :, np.newaxis]),
                                       np.zeros_like(inverted[:, :, np.newaxis]),inverted[:, :, np.newaxis]), axis=2)
            current_frame[inverted] = 1
    else:
        if not no_tracks:
            people_frame = np.zeros((current_frame.shape[0],current_frame.shape[1],1))
            for i, person_list in enumerate(people):
                 for person in person_list:

                    person_bbox = [ max(current_frame.shape[0]-1-int(seq_len+person[1, 1]-y_min),0),min(current_frame.shape[0]-1-int(seq_len+person[1, 0]-y_min), current_frame.shape[0]-1),
                                   max(int(seq_len+person[2, 0]-x_min),0), min(int(seq_len+person[2, 1]-x_min), current_frame.shape[1]-1)]

                    s=current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :].shape

                    if 0 not in s and person_bbox[0]<current_frame.shape[0] and person_bbox[2]<current_frame.shape[1] and person_bbox[1]>=0 and person_bbox[3]>=0:

                        people_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1] = np.ones((s[0], s[1],1))

                        current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :] =\
                            current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :]*(1 - trans) \
                                + trans *  np.tile(colours[26], (s[0], s[1], 1)) # np.tile([0,0,0], (s[0], s[1], 1))

        if frame<len(people):

            for person in people[frame]:

                person_bbox = [current_frame.shape[0] - 1 - int(seq_len + max(person[1, :]-y_min)),
                               current_frame.shape[0] - 1 - int(seq_len + min(person[1, :]-y_min)),
                               int(seq_len + min(person[2, :])-x_min), int(seq_len + max(person[2, :])-x_min)]
                #print person_bbox
                s = current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :].shape
                current_frame[person_bbox[0]:person_bbox[1]+1, person_bbox[2]:person_bbox[3]+1, :] = np.tile([230, 92, 0],
                                                                                                      (s[0], s[1], 1))
        else:
            print("Want to visualize frame "+str(frame)+" but length of ped list is "+str(len(people)))



    return current_frame

def make_movie_paralell(people, tensor, statistics_all, width, depth,seq_len,counter,agent_shape, agent_sight, episode , pool=None,training=True,
               ep_nbr=-1, nbr_episodes_to_vis=2, name_movie='', path_results='Datasets/cityscapes/agent/'):
    if pool is None:
        pool = multiprocessing.Pool(nbr_episodes_to_vis)
    inputs = []
    for statistics_loc in statistics_all:
        if ep_nbr == -1:
            ep_nbr = counter #/statistics_loc.shape[0]
        agent_pos = statistics_loc[:, :, STATISTICS_INDX.agent_pos[0]:STATISTICS_INDX.agent_pos[1]]
        agent_velocity = statistics_loc[:, :, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]]
        agent_reward = statistics_loc[:, :, STATISTICS_INDX.reward]
        agent_probabilities = statistics_loc[:, :, STATISTICS_INDX.probabilities[0]:STATISTICS_INDX.probabilities[1]]
        agent_action=statistics_loc[:, :, STATISTICS_INDX.action]
        agent_measures=statistics_loc[:, :, STATISTICS_INDX.measures[0]:STATISTICS_INDX.measures[1]]

        img_from_above = np.ones((width+2*seq_len, depth+2*seq_len, 3), dtype=np.uint8)*255
        img_from_above = view_2D(tensor, img_from_above, seq_len)
        img_from_above = view_cars(img_from_above, episode.cars, img_from_above.shape[0], img_from_above.shape[1], seq_len)
        img_from_above = view_pedestrians(img_from_above, people, img_from_above, seq_len, trans=0.1)

        for ep_itr in range(statistics_loc.shape[0]- nbr_episodes_to_vis, statistics_loc.shape[0]):

            #print "Episode"+str(ep_itr+ep_nbr)
            agent = agent_pos[ep_itr, :, ]
            #velocity = agent_velocity[ep_itr, :, :]
            action=agent_action[ep_itr,:]
            reward = agent_reward[ep_itr, :]
            probabilities=agent_probabilities[ep_itr, :, :]
            measures=agent_measures[ep_itr,:,:]

            for frame in range(seq_len - 1):
                inputs.append((action[frame], agent[frame,:], agent_shape, agent_sight, counter, ep_itr, ep_nbr,
                               frame, img_from_above, measures, name_movie, path_results,  probabilities[frame, 0:9].copy(), reward.copy(),
                               seq_len, training))
                #plot(inputs[counter])
                counter+=1
    print((path_results+name_movie+'frame'))
    return counter, inputs, pool


def plot(args):
    q,arg=args
    action, agent, agent_shape, agent_sight, counter, ep_itr, ep_nbr, frame, img_from_above,measures, name_movie, path_results, probs, reward, seq_len, training=arg
    action_names = [ 'downL', 'down', 'downR', 'left', 'stand', 'right','upL', 'up', 'upR']
    grid = plt.GridSpec(4, 5, wspace=0.3, hspace=0.4)
    fig = plt.figure(figsize=(18.5, 9))
    axarray = []
    axarray.append(fig.add_subplot(grid[:-1, :]))
    axarray.append(fig.add_subplot(grid[-1, 0:2]))
    axarray.append(fig.add_subplot(grid[-1, 2:4]))
    axarray.append(fig.add_subplot(grid[-1, -1]))
    axarray[0].set_xticks([])
    axarray[0].set_yticks([])
    axarray[1].set_ylim([0, 1])
    axarray[2].set_ylim([-10, 10])
    axarray[1].set_title('Policy')
    axarray[2].set_title('Reward')
    axarray[2].set_xlabel('iterations')
    axarray[2].set_ylabel('immediate reward')
    axarray[3].set_title("Agent's input")
    axarray[3].set_xticks([])
    axarray[3].set_yticks([])
    axarray[2].set_xlim([0, seq_len])
    axarray[1].cla()
    axarray[1].set_ylim([0, 1])
    axarray[1].set_title('Policy')
    axarray[2].cla()
    axarray[2].set_ylim([-1, 5])
    axarray[2].set_xlim([0, seq_len])
    axarray[2].set_title('Reward')
    name = " Episode " + str(ep_itr + ep_nbr) + " " + str(
        action_names[int(action)])
    if not training:
        name = "Validation " + name
        axarray[0].set_title(name, color='red')
    else:
        axarray[0].set_title(name)

    current_frame = img_from_above.copy()  # ( frame, people, current_frame, width, depth)


    pos=agent[1:3].copy()
    pos[0] = pos[0] + seq_len
    pos[1] =pos[1] + seq_len


    lower_lim=[]
    lower_lim.append(int(pos[0] - agent_shape[1]-agent_sight[1]))
    lower_lim.append(int(pos[1] - agent_shape[2] - agent_sight[2]))

    upper_lim=[]
    upper_lim.append(int(pos[0] +agent_shape[1]+agent_sight[1]+1))
    upper_lim.append(int(pos[1]+agent_shape[2] + agent_sight[2]+1))
    transparency = 0.3

    im_2d_agent=current_frame[lower_lim[0]:upper_lim[0],lower_lim[1]:upper_lim[1], :].copy()

    lower_lim[0] = int(pos[0] - agent_shape[1])
    lower_lim[1] = int(pos[1] - agent_shape[2])

    upper_lim[0] = int(pos[0] + agent_shape[1] + 1)
    upper_lim[1] = int(pos[1] + agent_shape[2] + 1)

    current_frame[lower_lim[0]:upper_lim[0],lower_lim[1]:upper_lim[1], :] = (
        transparency * current_frame[lower_lim[0]:upper_lim[0],lower_lim[1]:upper_lim[1], :]).astype(int)

    axarray[0].imshow(current_frame)

    axarray[1].bar(action_names,probs, color='#80aaff')
    axarray[2].plot(list(range(frame + 1)), measures[0:frame + 1, 0], label="car coll.", color='#ff6600')  # cars
    axarray[2].plot(list(range(frame + 1)), measures[0:frame + 1, 1], label="inter. people", color='#33cc33')  # people
    axarray[2].plot(list(range(frame + 1)), measures[0:frame + 1, 2]*5, label="IOU pavement.",color='#ff0066')  # Static objs
    axarray[2].plot(list(range(frame + 1)), measures[0:frame + 1, 3], label="inter. objs.", color='#4d88ff')  # Out of axis
    axarray[2].plot(list(range(frame + 1)), reward[0:frame + 1], label="reward", color='#000000')

    box = axarray[2].get_position()
    axarray[2].set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
    axarray[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

    # frame, agent,agent_shape,agent_sight ,episode
    #input = view_agent_input_fast( agent[ 1:3].astype(int), agent_shape, agent_sight,img_from_above)


    trans = 0.9
    im_2d_agent[0:agent_sight[0], :, :] = trans * im_2d_agent[0:agent_sight[0], :, :] + (1 - trans) * 255.0 * np.ones(
        im_2d_agent[0:agent_sight[0], :, :].shape)

    im_2d_agent[-agent_sight[0], :, :] = trans * im_2d_agent[-agent_sight[0], :, :] + (1 - trans) * 255.0 * np.ones(
        im_2d_agent[-agent_sight[0], :, :].shape)

    im_2d_agent[agent_sight[0]:-agent_sight[0], 0:agent_sight[1], :] = trans * im_2d_agent[
                                                                               agent_sight[0]:-agent_sight[0],
                                                                               0:agent_sight[1], :] + (
                                                                                                      1 - trans) * 255.0 * np.ones(
        im_2d_agent[agent_sight[0]:-agent_sight[0], 0:agent_sight[1], :].shape)

    im_2d_agent[agent_sight[0]:-agent_sight[0], -agent_sight[1], :] = trans * im_2d_agent[
                                                                              agent_sight[0]:-agent_sight[0],
                                                                              -agent_sight[1], :] + (
                                                                                                    1 - trans) * 255.0 * np.ones(
        im_2d_agent[agent_sight[0]:-agent_sight[0], -agent_sight[1], :].shape)

    axarray[3].imshow(im_2d_agent.astype(np.uint8))


    print(("Saving "+path_results + name_movie + 'frame_%06d.jpg' % counter))
    fig.savefig(path_results + name_movie + 'frame_%06d.jpg' % counter)
    plt.close(fig)
    q.put(arg, False)

def autolabel(rects,ax, avg_p, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{:.0f}'.format((height-avg_p)*100), ha=ha[xpos], va='bottom')


from multiprocessing import Process, Pipe
# Visualization!

def make_movie(tensor, people, cars, statistics_all, width, depth,seq_len,counter,agent_shape, agent_sight ,poses,
               prior_map, prior_car,statistics_car,initialization_goal, initialization_car_map, learn_init=False,car_goal=[], training=True,number_of_agents=1,number_of_car_agents=0,
               ep_nbr=-1, nbr_episodes_to_vis=2, name_movie='', path_results='Datasets/cityscapes/agent/',
               episode_name='',constrained_weights=False, action_reorder=False, inits_frames=False,
               velocity_show=False, velocity_sigma=False, jointsParents=None, continous=False, CLA_DOESNT_REMOVE_META=True, goal=False, gaussian_goal=False):
    plt.close("all")
    print("Start visualixation len of people "+str(len(people)))
    mini_labels_mapping = {11: 0, 13: 1, 14: 2, 4: 2, 5: 2, 15: 2, 16: 2, 17: 3, 18: 3, 7: 4, 9: 4, 6: 4, 10: 4, 8: 5,
                           21: 6, 22: 6, 12: 7, 20: 8, 19: 8}
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # if velocity_sigma:
    #     action_names = ['upR', 'up', 'upL', 'left', 'downL','down','downR','right']

    colors = [plt.cm.Dark2(i) for i in range(8)]


    width_ratios=2*np.ones(max(number_of_agents+1,3))
    width_ratios[0]=1
    # height_ratios=2*np.ones(number_of_agents+1)
    # height_ratios[-1]=1
    axarray_distr_car,axarray_distr_agent, fig_distr= get_init_frame_set_up(colors, number_of_agents, number_of_car_agents)

    axarray, fig = get_frame_set_up(colors, number_of_agents, width_ratios)

    for statistics_loc in statistics_all:
        if ep_nbr == -1:
            ep_nbr = counter #/statistics_loc.shape[0]
        agent_pos = statistics_loc[:,:, :, STATISTICS_INDX.agent_pos[0]:STATISTICS_INDX.agent_pos[1]]

        agent_measures=statistics_loc[:,:, :, STATISTICS_INDX.measures[0]:STATISTICS_INDX.measures[1]]

        agent_goals=statistics_loc[:,:, :, STATISTICS_INDX.goal[0]:STATISTICS_INDX.goal[1]]
        speeds=statistics_loc[:,:, :, STATISTICS_INDX.speed]

        if inits_frames:
            agent_init_frames = statistics_loc[:,:, 5, STATISTICS_INDX.init_method]

        # Get image from above

        width_bar=min(seq_len,10)
        # if width >128 or depth >256:
        #     width_bar=0
        img_from_above = np.ones((width+2*width_bar, depth+2*width_bar, 3), dtype=np.uint8)*255

        img_from_above = view_2D(tensor, img_from_above, width_bar)


        ### View weights
        # axarray[4].cla()
        # axarray[4].bar(range(len(reward_weights)),reward_weights)
        # axarray[4].axes.set_xticklabels(['Cars', 'People', 'Pavement', 'Objects', 'Distance', 'Out of axis'],rotation='vertical')
        #names_d = {11:"building", 12:"wall", 13:"fence", 17:"pole", 20:"traffic sign", 21:"vegetation"}
        for ep_itr in range(max(statistics_loc.shape[0]- nbr_episodes_to_vis,0), statistics_loc.shape[0]):

            print(("Episode"+str(ep_itr+ep_nbr)))
            # Get statistics for this episode.
            agent = np.round(agent_pos[ep_itr,:, :, 1:3]).astype(int)

            agent_goal=agent_goals[ep_itr,:,:]
            goal_frames = get_goal_frames(statistics_loc, ep_itr)
            if len(statistics_car)>0:
                car=statistics_car[ep_itr,:,:,STATISTICS_INDX_CAR.bbox[0]:STATISTICS_INDX_CAR.bbox[1]]
                car_goal=statistics_car[ep_itr,:, :3, STATISTICS_INDX_CAR.goal]
            else:
                car=[]
                car_goal=[]

            speed=speeds[ep_itr]

            #print ("width "+str(width)+" depth "+str(depth))
            if width > 128 or depth > 256:
                middle = agent[0]
                x_min = max(middle[1] - 128 -width_bar,0)
                y_min = max((middle[0] -64)-width_bar,0)
                lower_lim=min(max(img_from_above.shape[0]-1-(middle[0] + 64)-width_bar,0),img_from_above.shape[0] )
                upper_lim = max( min(img_from_above.shape[0]-1-(middle[0] - 64)+width_bar,img_from_above.shape[0]),0)
                lower_lim2=max(middle[1] - 128-width_bar,0)
                upper_lim2=min(middle[1] + 128+width_bar, img_from_above.shape[1])
                img_from_above_local = img_from_above[ lower_lim:upper_lim,lower_lim2:upper_lim2,:]

            else:
                x_min = 0
                y_min = 0
                img_from_above_local = img_from_above

            if np.linalg.norm(agent_goal)==0:
                agent_goal=[]
            # Set title
            txt=fig.text(90, .025,'', color='g')
            # Reward

            agentVisual = []
            if jointsParents!=None:

                for agent_id in range(number_of_agents):
                    ax_id=3*agent_id+1
                    axarray[ax_id].cla()
                    useTkAGG = False
                    if CLA_DOESNT_REMOVE_META:
                        axarray[ax_id].set_title("Agent's pose", color=colors[agent_id])
                    agentVisual.append(PFNNViz(jointsParents[agent_id], getBonePos, useTkAGG, ax=axarray[ax_id], skeletonColor=colors[agent_id]))


            if len(prior_map)>0:
                for ped_id in range(number_of_agents):
                    initialization_distr = np.reshape(prior_map[ep_itr, ped_id, :, STATISTICS_INDX_MAP.init_distribution],
                                                      (width, depth))
                    initialization_prior = np.reshape(prior_map[ep_itr, ped_id, :, STATISTICS_INDX_MAP.prior], (width, depth))
                    if len(prior_map) and np.sum(initialization_distr[:,:])>0:
                        # (agent, axarray,axarray_distr_agent, initialization_distr, initialization_prior, ped_id,agent_sight, agent_goal, seq_len, colors, x_min=0, y_min=0):
                        plot_init_distribution(agent, axarray,axarray_distr_agent, initialization_distr, initialization_prior, ped_id, agent_sight, agent_goal, seq_len, colors, x_min=x_min, y_min=y_min)

            if len(initialization_car_map)>0:
                for car_id in range(number_of_car_agents):
                    initialization_distr_car = np.reshape(initialization_car_map[ep_itr, car_id, :, STATISTICS_INDX_MAP.init_distribution],
                                                      (width, depth))
                    initialization_prior_car = np.reshape(initialization_car_map[ep_itr, car_id, :, STATISTICS_INDX_MAP.prior], (width, depth))
                    # (agent, axarray,axarray_distr_agent, initialization_distr, initialization_prior, ped_id,agent_sight, agent_goal, seq_len, colors, x_min=0, y_min=0):
                    plot_init_distribution_car(car, car_goal, axarray_distr_car, initialization_distr_car,initialization_prior_car, car_id, seq_len, colors, x_min=0, y_min=0)

            if len(prior_map)>0 or len(initialization_car_map)>0:
                fig_distr.savefig(path_results + name_movie + 'frame_%06d.jpg' % counter)
                counter = counter + 1
            # Go through all frames
            frame=0
            while frame<(seq_len - 1) and  np.all(agent_measures[ep_itr,:, max(frame, 0), PEDESTRIAN_MEASURES_INDX.agent_dead] == 0):

                # Set names
                name = " Episode " + str(ep_itr+ep_nbr)#+ sem_class_name
                if len(episode_name)>0:
                    name=name +" "+episode_name
                #name=name+" "+str(agent_pos[ep_itr, frame, 1:3])
                if not training:
                    name = "Validation " + name
                    axarray[0].set_title(name, color='red')
                else:
                    axarray[0].set_title(name)

                # TODO: maybe keeping a buffer instead of imshow and recreating the buffer each time ?
                ##### View agent!
                agent_frame=-1
                if inits_frames:
                    agent_frame=int(agent_init_frames[ep_itr])+frame
                if len(initialization_goal) > 0 and np.sum(initialization_distr) > 0:
                    plot_goal_distribution(agent, agent_goal, agent_id, agent_measures, agent_shape, agent_sight, axarray, colors,
                                           depth, ep_itr, frame, gaussian_goal, goal_frames, prior_car, prior_map, seq_len, width,
                                           x_min, y_min)


                tmp=view_agent(agent, img_from_above_local, width_bar,agent_shape, agent_sight,car=car,
                               people=people,cars=cars,frame=frame,
                               goal=agent_goal,car_goal=car_goal,agent_frame=agent_frame, x_min=x_min, y_min=y_min, colors=colors )

                im = axarray[0].imshow(tmp)
                for agent_id in range(number_of_agents):
                    if agentVisual:
                        itr=int(poses[ep_itr,agent_id,frame, STATISTICS_INDX_POSE.agent_pose_frames])
                        next_itr = int(poses[ep_itr,agent_id, frame+1, STATISTICS_INDX_POSE.agent_pose_frames])
                        agentVisual[agent_id].updateSkeletonGraph(poses[ep_itr,agent_id, itr, STATISTICS_INDX_POSE.agent_high_frq_pos[0]],poses[ep_itr,agent_id, itr, STATISTICS_INDX_POSE.agent_high_frq_pos[1]-1], poses[ep_itr,agent_id, itr, STATISTICS_INDX_POSE.pose[0]:STATISTICS_INDX_POSE.pose[1]])


                fig.savefig(path_results+name_movie+'frame_%06d.jpg' %counter)

                counter = counter + 1
                frame=frame+1
    print(("Saved frame " + path_results + name_movie + 'frame_%06d.jpg' % counter))
    return counter


def get_init_frame_set_up(colors, number_of_agents,number_of_car_agents):
    fig_distr = plt.figure(figsize=(18.5, 9))
    axarray_distr_agent = []
    grid_distr = plt.GridSpec(4, max(number_of_agents, number_of_car_agents))
    for agent_id in range(number_of_agents):
        axarray_distr_agent.append(fig_distr.add_subplot(grid_distr[0, agent_id]))
        axarray_distr_agent[-1].set_title("Initial distribution of agent " + str(agent_id))  # , color=colors[agent_id])
        axarray_distr_agent.append(fig_distr.add_subplot(grid_distr[1, agent_id]))  # initialization distribution
        axarray_distr_agent[-1].set_title("Prior distribution of agent " + str(agent_id), color=colors[agent_id])
    axarray_distr_car = []
    for car_id in range(number_of_car_agents):
        axarray_distr_car.append(fig_distr.add_subplot(grid_distr[2, car_id]))  # prior
        axarray_distr_car[-1].set_title("Initial distribution of car " + str(car_id), color=colors[car_id])
        axarray_distr_car.append(fig_distr.add_subplot(grid_distr[3, car_id]))  # prior
        axarray_distr_car[-1].set_title("Prior distribution of car " + str(car_id), color=colors[car_id])
    return axarray_distr_car,axarray_distr_agent, fig_distr

def get_frame_set_up(colors, number_of_agents, width_ratios):
    grid = plt.GridSpec(max(number_of_agents + 1, 3), max(number_of_agents + 1, 3),
                        width_ratios=width_ratios)  # , height_ratios=height_ratios)#, wspace=0.3, hspace=0.4) # number_of_agents, number_of_agents+1
    fig = plt.figure(figsize=(18.5, 9))
    axarray = []
    axarray.append(fig.add_subplot(grid[:-1, 1:-1]))  # 0 -env
    for agent_id in range(number_of_agents):
        axarray.append(fig.add_subplot(grid[agent_id, 0], projection='3d'))
        axarray[-1].set_title("Pose agent " + str(agent_id))  # , color=colors[agent_id])
        axarray.append(fig.add_subplot(grid[agent_id, -1]))  # initialization distribution
        axarray[-1].set_title("Initialization distribution " + str(agent_id), color=colors[agent_id])
        axarray.append(fig.add_subplot(grid[-1, agent_id + 1]))  # prior

        axarray[-1].set_title("Prior distribution " + str(agent_id), color=colors[agent_id])
    axarray[0].set_xticks([])
    axarray[0].set_yticks([])
    return  axarray, fig


def plot_goal_distribution(agent, agent_goal, agent_id, agent_measures, agent_shape, agent_sight, axarray, colors, depth, ep_itr,
                           frame, gaussian_goal, goal_frames, prior_car, prior_map, seq_len, width, x_min, y_min):
    for ped_id in range(agent_measures.shape[1]):
        if frame in goal_frames[ped_id]:
            goal_distr, goal_prior = get_goal_distr(depth, width, ep_itr, gaussian_goal, ped_id, prior_car, prior_map,
                                                    frame)
            axarray[ped_id * 3 + 1].set_title("Goal distribution " + str(agent_id) + " frame " + str(frame),
                                              color=colors[agent_id])
            product=goal_distr * max_value
            max_value = 1.0 / np.max(product)
            transparency = 0.1
            heatmap = np.flip(product * max_value, axis=0)
            # add_pedestrian_agent(agent_sight, heatmap, [], agent_goal[ped_id, :], heatmap, agent[ped_id, frame, :],
            #                      seq_len, agent_shape, transparency, x_min=x_min, y_min=y_min,frame=frame,color=colors[agent_id], id=ped_id)
            img_h = axarray[ped_id * 3 + 1].imshow(heatmap, cmap='plasma')

            max_value_prior = 1.0 / np.max(goal_prior)
            img_h = axarray[ped_id * 3 + 2].imshow(np.flip(goal_prior * max_value_prior, axis=0), cmap='plasma')

            axarray[ped_id * 3 + 2].set_title("Goal prior  " + str(agent_id) + " frame " + str(frame),
                                              color=colors[agent_id])



def plot_init_distribution(agent, axarray,axarray_distr_agent, initialization_distr, initialization_prior, ped_id,agent_sight, agent_goal, seq_len, colors, x_min=0, y_min=0):
    product=initialization_distr*initialization_prior
    max_value = 1.0 / np.max(product)
    max_value_prior = 1.0 / np.max(initialization_prior)
    heatmap = np.flip(product * max_value, axis=0)
    pedestrian_pos = agent[ped_id, 0, :]
    # TO DO: add plot of pedestrian!"
    transparency_sight = 0.9
    transparency = 0.1
    dif = []
    if len(axarray)>0:
        img_h = axarray[ped_id * 3 + 2].imshow(heatmap, cmap='plasma')
        img_h = axarray[ped_id * 3 + 3].imshow(np.flip(initialization_prior * max_value_prior, axis=0), cmap='plasma')
    #agent_sight, current_frame, dif, goal, img_from_above, agent, seq_len, size_a, transparency,x_min, y_min, frame=-1, color=[], id=-1)
    # add_pedestrian_agent(agent_sight, heatmap, [], agent_goal[ped_id, :], heatmap, agent[ped_id, :, :],seq_len, agent_sight, transparency,
    #                      x_min=x_min, y_min=y_min, frame=0, id=ped_id)
    axarray_distr_agent[ped_id * 2].imshow(heatmap, cmap='plasma')
    axarray_distr_agent[ped_id * 2 + 1].imshow(np.flip(initialization_prior * max_value_prior, axis=0), cmap='plasma')

def plot_init_distribution_car(car,cargoal, axarray_distr_agent, initialization_distr, initialization_prior, car_id, seq_len, colors, x_min=0, y_min=0):
    product=initialization_distr*initialization_prior
    max_value = 1.0 / np.max(product)
    max_value_prior = 1.0 / np.max(initialization_prior)
    heatmap = np.flip(product * max_value, axis=0)

    # TO DO: add plot of pedestrian!"
    transparency_sight = 0.9
    transparency = 0.1
    dif = []
    #agent_sight, current_frame, dif, goal, img_from_above, agent, seq_len, size_a, transparency,x_min, y_min, frame=-1, color=[], id=-1)
    # plot_car_agent(car, car_id, colors[car_id], heatmap, 0, seq_len, transparency, x_min, y_min)
    # plot_car_goal(heatmap, cargoal[car_id][1:], seq_len, 1)
    axarray_distr_agent[car_id * 2].imshow(heatmap, cmap='plasma')
    axarray_distr_agent[car_id * 2 + 1].imshow(np.flip(initialization_prior * max_value_prior, axis=0), cmap='plasma')



def get_goal_distr(width, depth, ep_itr, gaussian_goal, ped_id, prior_car, prior_map, frame):
    if not gaussian_goal:
        goal_distr = np.reshape(prior_map[ep_itr][ ped_id][frame][ :, STATISTICS_INDX_MAP.goal_distribution],
                                (width, depth))
        goal_prior = np.reshape(prior_map[ep_itr][ ped_id][frame][ :, STATISTICS_INDX_MAP.goal_prior], (width, depth))
        return goal_distr,goal_prior
    else:
        goal_distr = np.zeros((width, depth), dtype=np.uint8)
        goal_prior = np.zeros((width, depth), dtype=np.uint8)
        manual_goal = prior_car[ep_itr,
                      STATISTICS_INDX_CAR_INIT.manual_goal[0]:STATISTICS_INDX_CAR_INIT.manual_goal[
                          1]].copy().astype(int)
        goal_agent = manual_goal.copy()

        goal_agent[0] = goal_agent[0] + (
                prior_map[ep_itr, 0, STATISTICS_INDX_MAP.goal_distribution] * goal_prior.shape[0])
        goal_agent[1] = goal_agent[1] + (
                prior_map[ep_itr, 1, STATISTICS_INDX_MAP.goal_distribution] * goal_prior.shape[1])
        # print ("Goal " + str(manual_goal) + " goal agent " + str(goal_agent))
        manual_goal = manual_goal.astype(int)
        goal_agent = goal_agent.astype(int)
        prior_box = [min(max(manual_goal[0] - 4, 0), goal_prior.shape[0] - 4),
                     max(min(manual_goal[0] + 5, goal_prior.shape[0]), 0),
                     min(max(manual_goal[1] - 4, 0), goal_prior.shape[1] - 4),
                     max(min(manual_goal[1] + 5, goal_prior.shape[1]), 0)]

        goal_prior[prior_box[0]:prior_box[1], prior_box[2]:prior_box[3]] = np.ones(
            goal_prior[prior_box[0]:prior_box[1], prior_box[2]:prior_box[3]].shape)
        # print ("Prior val " + str([max(manual_goal[0]-4,0),min(manual_goal[0]+5,goal_prior.shape[0]), max(manual_goal[1]-4,0),min(manual_goal[1]+5,goal_prior.shape[1])]))
        # print ("Goal prior valus "+str(goal_prior[max(manual_goal[0]-4,0):min(manual_goal[0]+5,goal_prior.shape[0]), max(manual_goal[1]-4,0):min(manual_goal[1]+5,goal_prior.shape[1])]))
        goal_box = [min(max(goal_agent[0] - 4, 0), goal_prior.shape[0] - 4),
                    max(min(goal_agent[0] + 5, goal_prior.shape[0]), 0),
                    min(max(goal_agent[1] - 4, 0), goal_prior.shape[1] - 4),
                    max(min(goal_agent[1] + 5, goal_prior.shape[1]), 0)]
        goal_distr[goal_box[0]:goal_box[1], goal_box[2]:goal_box[3]] = np.ones(
            goal_distr[goal_box[0]:goal_box[1], goal_box[2]:goal_box[3]].shape)
        # print ("Goal distr "+str(goal_box))
        # print ("Goal prior valus after if " + str(
        return goal_distr, goal_prior

def make_movie_manual(people, tensor,width, depth,seq_len,agent_shape, agent_sight, episode,frame, img_from_above,car_goal=[], name_movie='', path_results='',  jointsParents=None):
    plt.close("all")


    fig = plt.figure(figsize=(18.5, 9))
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.5, 9))
    agent=[]
    agent_goal=[]
    for id in range(len(episode.pedestrian_data)):
        agent.append(episode.pedestrian_data[id].agent)#[frame]#statistics_loc[:, :, 0:3]
        agent_goal.append(episode.pedestrian_data[id].goal[:,1:])#statistics_loc[:, 3:5, -1]
    #car_id=episode.initializer_data[id].init_car_id
    if episode.use_car_agent:
        car=[]
        car_goal=[]
        for car_id in range(len( episode.car_data)):
            car.append(episode.car_data[car_id].car_bbox)#statistics_car[ep_itr, :, 14:20]
            car_goal.append(episode.car_data[car_id].car_goal)
    else:
        car=[]

    # Get image from above
    width_bar = min(seq_len, 10)
    print("Image size "+str(img_from_above.shape))
    if np.sum(img_from_above)==0:
        img_from_above = np.ones((width+2*width_bar, depth+2*width_bar, 3), dtype=np.uint8)*255
        img_from_above = view_2D(tensor, img_from_above, width_bar, white_background=True)
        #img_from_above =view_valid_pos(episode, img_from_above, 0)
    print (" Make movie car goal "+str(car_goal))
    tmp=view_agent(agent, img_from_above, width_bar,agent_shape, agent_sight,car=car, people=people,cars=episode.cars,
                   frame=frame, goal=agent_goal, car_goal=car_goal)
    plt.imshow(tmp) # Is this ok? Animate if slow!
    external=True

    print("View agent")


    #cv2.waitKey(0)

    plt.axis('off')
    if not os.path.exists(path_results):
        os.makedirs(path_results, exist_ok=True)
    fig.savefig(os.path.join(path_results,name_movie+'frame_%06d.jpg' %frame) ,bbox_inches='tight')
    plt.show()  # block=False)
    print(("Saved manual img " + path_results + name_movie + 'frame'+str(frame)))
    for id in range(len(episode.pedestrian_data)):
        if jointsParents[id] and frame>0:
            # print("Visualize pose")
            # if DO_VISUALIZATION:
            # Init the visualization engine if needed
            useTkAGG = False  # Use this on True only if there are compatibility issues. For example in Pycharm you need it
            # Unfortunately it will show down rendering and processing a lot. The only option in case you need it with True
            # would be to render at a slower fps
            itr=episode.pedestrian_data[id].agent_pose_frames[frame]
            itr_prev=episode.pedestrian_data[id].agent_pose_frames[max(frame-1,0)]
            previous_pose=episode.pedestrian_data[id].agent_pose[itr_prev,:]
            current_pose=episode.pedestrian_data[id].agent_pose[itr,:]
            diff=current_pose-previous_pose
            # print(diff)
            # print(("Difference in pose "+str(np.mean(np.abs(diff)))))
            # print((" squared "+str(np.mean(diff**2))))
            # print((" min "+str(min(abs(diff)))))
            # print((" max "+str(max(abs(diff)))))

            #print itr
            agentVisual = PFNNViz(jointsParents[id], getBonePos, useTkAGG)
            # print "Frame "+str(frame)+"  pose_itr"+str(itr)
            # print "Agent pose: "+str(episode.agent_pose[itr, :])
            name=name_movie+"_pose_"+str(id)+'frame_%06d.jpg'
            agentVisual.updateSkeletonGraphManual(episode.pedestrian_data[id].agent_high_frq_pos[itr,0],episode.pedestrian_data[id].agent_high_frq_pos[itr,1], episode.pedestrian_data[id].agent_pose[itr,:], os.path.join(path_results,name%frame))
            #ax2 = agentVisual.fig.gca(projection='3d')



def make_movie_eval(people, tensor, statistics_all, width, depth,seq_len,counter,agent_shape, agent_sight, episode,
               ep_nbr=-1, nbr_episodes_to_vis=2, name_movie='', path_results=''):
    plt.close("all")

    fig = plt.figure(figsize=(18.5, 9))
    for statistics_loc in statistics_all:
        agent_pos = statistics_loc[:, :, STATISTICS_INDX.agent_pos[0]:STATISTICS_INDX.agent_pos[1]]
        #agent_goals=statistics_loc[:, 0:3, -1]
        agent_goals = statistics_loc[:,:, STATISTICS_INDX.goal[0]:STATISTICS_INDX.goal[1]]
        agent_measures = statistics_loc[:, :, STATISTICS_INDX.measures[0]:STATISTICS_INDX.measures[1]]

        # Get image from above
        width_bar = min(seq_len, 10)
        img_from_above = np.ones((width+2*width_bar, depth+2*width_bar, 3), dtype=np.uint8)*255
        img_from_above = view_2D(tensor, img_from_above, width_bar, white_background=True)

        for ep_itr in range(max(statistics_loc.shape[0]- nbr_episodes_to_vis,0), statistics_loc.shape[0]):
            print(("Episode"+str(ep_itr)))#+ep_nbr)
            # Get statistics for this episode.
            agent = agent_pos[ep_itr, :, 1:3].astype(int)
            agent_goal=agent_goals[ep_itr,:,:]


            # Go through all frames
            for frame in range(seq_len - 1):
                #print "Visualize agent pos " + str(agent[frame]) + " " + str(agent_goal)
                if (not agent_measures[ep_itr, frame, PEDESTRIAN_MEASURES_INDX.agent_dead]):
                    ##### View agent!
                #if agent_pos
                    tmp=view_agent(agent, img_from_above, width_bar,agent_shape, agent_sight, people=people,cars=episode.cars,
                                   frame=frame, goal=agent_goal[:,frame,:], hit_by_car=agent_measures[ep_itr, max(frame-1,0), PEDESTRIAN_MEASURES_INDX.goal_reached])
                    plt.imshow(tmp) # Is this ok? Animate if slow!
                    if False:
                        plt.show()
                        #cv2.waitKey(0)
                    else:
                        plt.axis('off')
                        fig.savefig(path_results+name_movie+'frame_%06d.jpg' %counter ,bbox_inches='tight')
                    counter = counter + 1
    print(("Saved eval img "+path_results+name_movie+'frame'))
    return counter


def make_movie_cars( tensor, statistics_all, seq_len,car_counter,agent_shape, agent_sight, episode,people_list, car_list, nbr_episodes_to_vis=2, name_movie='', path_results=''):
    plt.close("all")

    movie_type = "people"
    if len(episode.cars[0])>0:
        movie_type = "cars"


    fig = plt.figure(figsize=(18.5, 9))

    for statistics_loc in statistics_all:

        agent_pos = statistics_loc[:, :, 0:3]
        #agent_goals=statistics_loc[:, 0:3, -1]
        agent_goals = statistics_loc[:, 3:5, -1]

        # Get image from above
        # for ep_itr in range(max(statistics_loc.shape[0]- nbr_episodes_to_vis,0), statistics_loc.shape[0]):
        for ep_itr in [0,2,4]:#range(max(statistics_loc.shape[0]- nbr_episodes_to_vis,0), statistics_loc.shape[0]):
            if ep_itr< statistics_loc.shape[0]:
                print(("Episode "+str(ep_itr)))

                # Get statistics for this episode.
                agent = agent_pos[ep_itr, :, 1:3].astype(int)
                agent_goal=agent_goals[ep_itr,:]
                hit_by_car=statistics_loc[ep_itr, :, 38]
                reward=statistics_loc[ep_itr, :, 34]
                hit_by_pedestrian = statistics_loc[ep_itr, :, 38+8]
                on_ped_trajectory = statistics_loc[ep_itr, :, 38 + 1]
                people_hist= statistics_loc[ep_itr, :, 38 + 10]
                out_of_axis=statistics_loc[ep_itr, :, 38+5]
                people=episode.people
                cars=episode.cars
                if len(cars[0])>0:
                    for frame in range(len(cars)):
                        if np.linalg.norm(car_list[ep_itr, frame, :])>0:
                            cars[frame]=[car_list[ep_itr, frame, :]]
                        else:
                            cars[frame]=[]
                else:
                    for frame in range(len(people)):
                        if np.linalg.norm(people_list[ep_itr, frame, :])>0:
                            people[frame]=[np.array([people_list[ep_itr, frame, 0:2], people_list[ep_itr, frame, 2:4], people_list[ep_itr, frame, 4:]])]
                        else:
                            people[frame]=[]


                frame=0
                while frame <seq_len - 1:
                    img_from_above = np.ones((tensor.shape[1], tensor.shape[2], 3), dtype=np.uint8) * 255
                    ##### View agent!

                    tmp=view_agent(agent, img_from_above, 0,agent_shape, agent_sight, people=people,cars=cars,
                                   frame=frame, goal=agent_goal)
                    #tmp =tmp[...,[2,0,1]] #cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

                    plt.imshow(tmp) # Is this ok? Animate if slow!
                    plt.axis('off')
                    fig.savefig(path_results+name_movie+'_'+movie_type+'_frame_%06d.jpg' %car_counter ,bbox_inches='tight')

                    car_counter = car_counter + 1
                    name=path_results+name_movie+'_'+movie_type+'_frame_%06d.jpg' %car_counter

                    # if (out_of_axis[frame]==1 ):#or hit_by_car[frame]==1 or hit_by_pedestrian[frame]==1):
                    #     print str(out_of_axis[frame])+" "+str(hit_by_car[frame])+" "+str(hit_by_pedestrian[frame])
                    #
                    #     break
                    frame=frame+1
    print(("Saved cars frame " + path_results+name_movie+'_'+movie_type+'_frame'))
    return car_counter
#[[[ 0 10] [13 17] [ 0  3]]]

def make_image_cars( tensor, statistics_all, seq_len,car_counter,agent_shape, agent_sight, episode,people_list, car_list, nbr_episodes_to_vis=2, name_movie='', path_results=''):
    plt.close("all")

    # Create figure and axes
    fig, ax = plt.subplots(1,figsize=(18.5, 9))
    for statistics_loc in statistics_all:
        agent_pos = statistics_loc[:, :, 0:3]
        #agent_goals=statistics_loc[:, 0:3, -1]
        agent_goals = statistics_loc[:, 3:5, -1]

        # Get image from above

        for ep_itr in range(max(statistics_loc.shape[0]- nbr_episodes_to_vis,0), statistics_loc.shape[0]):
            print(("Episode"+str(ep_itr)))#+ep_nbr)
            # Get statistics for this episode.
            agent = agent_pos[ep_itr, :, 0:3].astype(int)
            agent_goal=agent_goals[ep_itr,:]
            hit_by_car=statistics_loc[ep_itr, :, 38]
            out_of_axis=statistics_loc[ep_itr, :, 38+5]
            img_from_above = np.ones((tensor.shape[1], tensor.shape[2], 3), dtype=np.uint8) * 255
            # Go through all frames
            frame=0

            people = episode.people
            cars = episode.cars
            if len(cars[0]) > 0:
                for frame in range(len(cars)):
                    if np.linalg.norm(car_list[ep_itr, frame, :]) > 0:
                        cars[frame] = [car_list[ep_itr, frame, :]]
                    else:
                        cars[frame] = []
            else:
                for frame in range(len(people)):
                    if np.linalg.norm(people_list[ep_itr, frame, :]) > 0:
                        people[frame] = [np.array([people_list[ep_itr, frame, 0:2], people_list[ep_itr, frame, 2:4],
                                                   people_list[ep_itr, frame, 4:]])]
                    else:
                        people[frame] = []

            list_type = people
            movie_type = "people"

            if len(episode.cars[0]) > 0:
                list_type = cars
                movie_type = "cars"

            img_from_above = np.ones((tensor.shape[1], tensor.shape[2], 3), dtype=np.uint8) * 255
            ##### View agent!

            tmp = view_agent(agent, img_from_above, 0, agent_shape, agent_sight, people=people,
                             cars=cars,
                             frame=0, goal=agent_goal)
            car_centers_x=[]
            car_centers_y = []

            for car_l in list_type:
                for car in car_l:
                    if len(car)>3:
                        car_centers=[img_from_above.shape[0] - 1 - int(car[3]),img_from_above.shape[0] - 1 - int(car[2]),car[4], car[5]]
                    else:
                        car_centers = [img_from_above.shape[0] - 1 - int(car[1][1]),
                                       img_from_above.shape[0] - 1 - int(car[1][0]), car[2][0], car[2][1]]
                    car_centers_y.append(int(np.mean(car_centers[0:2])))
                    car_centers_x.append(int(np.mean(car_centers[2:4])))
                    if car_centers_y[-1]>0 and car_centers_x[-1]>0 and car_centers_y[-1]< tmp.shape[0] and car_centers_x[-1]< tmp.shape[0]:
                        tmp[int(np.mean(car_centers[0:2])),int(np.mean(car_centers[2:4])),: ]=[0,0,0]


            for car in agent:
                car_centers = [img_from_above.shape[0] - 1 - int(car[1]), car[2]]

                if car_centers[0] > 0 and car_centers[1] > 0 and car_centers[0] < tmp.shape[0] and \
                                car_centers[1] < tmp.shape[0]:
                    tmp[int(car_centers[0]), int(car_centers[1]), :] = [255, 0, 0]

            #ax.plot(car_centers_x, car_centers_y)
            # agent_centers=agent
            # agent_centers[:,1]=img_from_above.shape[0] - 1-agent_centers[:,1]
            # ax.plot(agent_centers[:,1], agent_centers[:,0])



            ##### View agent!

            im=ax.imshow(tmp)  # Is this ok? Animate if slow!
            plt.axis('off')
            fig.savefig(path_results + name_movie + '_'+movie_type+'_image_%06d.jpg' % car_counter, bbox_inches='tight')



    print(("Saved img "+path_results + name_movie + '_'+movie_type+'_image'))
    return car_counter






import os, glob

def make_mock_movie( statistics_all,seq_len,counter, f,successful,training=True, ep_nbr=-1, nbr_episodes_to_vis=2, video_name=''):
    for statistics_loc in statistics_all:
        if ep_nbr == -1:
            ep_nbr = counter
        agent_velocity = statistics_loc[:, :, 3:6]
        for ep_itr in range(statistics_loc.shape[0]- nbr_episodes_to_vis, statistics_loc.shape[0]):
            print(("Episode"+str(ep_itr+ep_nbr)))
            velocity = agent_velocity[ep_itr, :, :]
            for frame in range(seq_len - 1):

                name = " Episode " + str(ep_itr+ep_nbr) + " " + str(
                    velocity[frame])
                if not training:
                    name = "Validation " + name
                vel_string=str(velocity[frame])

                path='Datasets/cityscapes/agent/*[[]*'+vel_string[1:-1]+'[]]frame_%06d.jpg' %counter
                wanted_path='Datasets/cityscapes/agent/'+video_name+'frame_%06d.jpg' %successful
                local_files=glob.glob(path)
                if local_files:
                    if len(local_files)==1:
                        #os.rename(local_files[0], wanted_path )
                        f.write(local_files[0]+'\n')
                        f.write(wanted_path+ '\n')
                    successful+=1
                else:
                    print("Not file")
                    print(path)
                counter = counter + 1
    return counter, successful


# Top view of 3D world.
def view_2D( tensor, img_from_above_loc, seq_len, people=False, white_background=False):
    background_color = 192
    if white_background:
        background_color = 255
    if len(tensor.shape) == 3:
        segmentation = (tensor[ :, :, 3] * NUM_SEM_CLASSES).astype(np.int)
        dim=segmentation.shape
    else:
        segmentation = (tensor[:, :, :, 3] * NUM_SEM_CLASSES).astype(np.int)
        dim = segmentation.shape[1:]
    for x in range(min(img_from_above_loc.shape[1]-2*seq_len, dim[1])):
        for y in range(min(img_from_above_loc.shape[0]-2*seq_len, dim[0])):
            if len(tensor.shape) == 3:
                values = segmentation[ y, x]  # v ==22 or (5<v and 11>v)]#v>0
                if values > 0:
                    img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] = colours[values]
                else:
                    img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] = [
                        background_color, background_color, background_color]

            else:
                values = [i for i, v in enumerate(segmentation[:, y, x]) if v>1] #v ==22 or (5<v and 11>v)]#v>0
                if len(values) > 0:
                    z = np.max(values)
                    img_from_above_loc[img_from_above_loc.shape[0]-1-seq_len-y, seq_len+x, :] = colours[segmentation[z, y, x]]
                else:
                    img_from_above_loc[img_from_above_loc.shape[0]-1-seq_len-y, seq_len+x, :] = [background_color,background_color,background_color]

    img_from_above_loc = cv2.cvtColor(img_from_above_loc, cv2.COLOR_BGR2RGB)

    return img_from_above_loc


def return_highest_col(tens):
    z = 0
    while z < tens.shape[0] - 1 and np.linalg.norm(tens[z, :]) == 0:
        z = z + 1
    return z


# Top view of 3D world.
def view_2D_rgb( tensor, img_from_above_loc, seq_len, people=False, white_background=False):
    background_color = 192
    if white_background:
        background_color = 255

    if len(tensor.shape) == 3:
        segmentation = (tensor[ :, :, 3] * NUM_SEM_CLASSES).astype(np.int)
        dim=segmentation.shape
    else:
        segmentation = (tensor[:, :, :, 3] * NUM_SEM_CLASSES).astype(np.int)
        dim = segmentation.shape[1:]

    for x in range(min(img_from_above_loc.shape[1]-2*seq_len, dim[1])):
        for y in range(min(img_from_above_loc.shape[0]-2*seq_len, dim[0])):
            if len(tensor.shape) > 3:
                z = return_highest_col(tensor[:,y,x,0:3])
                if np.linalg.norm(tensor[z, y, x,:3])==0:
                    img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] = background_color
                else:
                    img_from_above_loc[img_from_above_loc.shape[0]-1-seq_len-y, seq_len+x, :] = tensor[z, y, x,:3]*255
            else:
                if np.linalg.norm(tensor[ y, x, :3]) == 0:
                    img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] = background_color
                else:
                    img_from_above_loc[img_from_above_loc.shape[0]-1-seq_len-y, seq_len+x, :] = tensor[ y, x,:3]*255

    #img_from_above_loc = cv2.cvtColor(img_from_above_loc, cv2.COLOR_BGR2RGB)

    return img_from_above_loc


def view_2D_sem( tensor, img_from_above_loc, seq_len, sem_class):
    segmentation = (tensor[:, :, :, 3] * NUM_SEM_CLASSES).astype(np.int)

    for x in range(min(img_from_above_loc.shape[1]-2*seq_len, segmentation.shape[2])):
        for y in range(min(img_from_above_loc.shape[0]-2*seq_len, segmentation.shape[1])):
            if sem_class in segmentation[:, y, x]:
                img_from_above_loc[img_from_above_loc.shape[0]-1-seq_len-y, seq_len+x, :] = (255,0,0)#img_from_above_loc[img_from_above_loc.shape[0]-1-seq_len-y, seq_len+x, :]


    #img_from_above_loc = cv2.cvtColor(img_from_above_loc, cv2.COLOR_BGR2RGB)

    return img_from_above_loc

def view_valid_pos( episode, img_from_above_loc, seq_len):
    for x in range(img_from_above_loc.shape[1]-2*seq_len-1):
        for y in range(img_from_above_loc.shape[0]-2*seq_len-1):
            if episode.valid_positions[y,x]:#episode.valid_position([0,y,x], no_height=True):
                img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] = img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] /2
    return img_from_above_loc

def view_prior_pos( prior, img_from_above_loc, seq_len):
    max_prior=np.max(prior)
    for x in range(img_from_above_loc.shape[1]-2*seq_len):
        for y in range(img_from_above_loc.shape[0]-2*seq_len):
            if prior[y,x]>0:#episode.valid_position([0,y,x], no_height=True):
                img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] = np.rint(img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - y, seq_len + x, :] *(1-(prior[y,x]/max_prior))).astype(int)
    return img_from_above_loc

def view_test_pos(test_loc, img_from_above_loc, seq_len):
    for pos in test_loc:

        img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - pos[0], seq_len + pos[1], :] = np.zeros(img_from_above_loc[img_from_above_loc.shape[0] - 1 - seq_len - pos[0], seq_len + pos[1], :].shape )
    return img_from_above_loc

def view_valid_init_pos( episode, img_from_above_loc, seq_len):
    #mask=np.zeros(img_from_above_loc.shape[0:2])
    for i,pos in enumerate(episode.valid_pavement[0]):
        x=seq_len + episode.valid_pavement[2][i]
        y=img_from_above_loc.shape[0] - 1 - seq_len - episode.valid_pavement[1][i]

        #if mask[y,x]==0:
        img_from_above_loc[y, x, :] = img_from_above_loc[y,x,:] /2
    return img_from_above_loc

def view_valid_pedestrians(episode,current_frame, seq_len, trans=.3):
    for person in episode.valid_people:


        person_bbox = [ current_frame.shape[0]-1-int(seq_len+max(person[1, :])),current_frame.shape[0]-1-int(seq_len+min(person[1, :])),int(seq_len+min(person[2, :])), int(seq_len+max(person[2, :]))]
        s=current_frame[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3], :].shape

        current_frame[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3], :] =\
            current_frame[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3], :]*(1 - trans) \
            + trans *  np.tile([255,0,0], (s[0], s[1], 1)) # np.tile([0,0,0], (s[0], s[1], 1))
    return current_frame