
from agent_net import NetAgent
import numpy as np

from RL.car_measures import end_of_episode_measures_car,  find_closest_pedestrian_in_list, \
    iou_sidewalk_car, find_closest_controllable_pedestrian,find_closest_car
from settings import  CAR_MEASURES_INDX,CAR_REWARD_INDX,NBR_MEASURES_CAR, PEDESTRIAN_MEASURES_INDX
from utils.utils_functions import overlap

from memory_profiler import profile as profilemem
from settings import run_settings

memoryLogFP_decisionsCar = None
if run_settings.memoryProfile:
    memoryLogFP_decisionsCar = open("memoryLogFP_CarDecisions.log", "w+")

import copy

class CarAgent(NetAgent):
    def __init__(self, settings, net, grad_buffer, init_net=None, init_grad_buffer=None):
        super(CarAgent, self).__init__(settings, net, grad_buffer, init_net=init_net, init_grad_buffer=init_grad_buffer)
        self.reset_agent_history(1)
        self.init_dir=[]
        self.isPedestrian = False


    def reset_agent_history(self, seq_len):
        self.measures=np.zeros((seq_len, NBR_MEASURES_CAR))
        self.car = [None] * seq_len
        self.bbox=[None] * seq_len
        self.velocities =  [None] * seq_len
        self.speed =  [None] * seq_len
        self.probabilities =  np.zeros(( seq_len,2))
        self.reward= [None] * seq_len
        self.action=[None]*seq_len
        self.angle=[None]*seq_len
        self.closest_car = [None] * seq_len
        self.external_car_vel = [None] * seq_len
        self.alive = True
        self.distracted = False
        self.car_id=-1

        self.on_car=False

    def getOnlineRealtimeAgentId(self):
        return super().getOnlineRealtimeAgentId()

    def initial_position(self, pos,goal, seq_len, current_frame=0, vel=0, init_dir=[], car_id=-1, on_car=False):
        self.reset_agent_history(seq_len)
        super(CarAgent, self).initial_position(pos,goal, current_frame, vel, init_dir)
        self.car[current_frame]=copy.copy(self.pos_exact)
        self.init_dir=copy.copy(init_dir)
        if len(self.position)>0:
            self.set_bbox(0)
        self.car_id=car_id
        self.on_car=on_car
        #print ("Init car agent car "+str(self.car[current_frame])+" car vel "+str(self.init_dir)+" car goal "+str(goal)+" current frame "+str(current_frame))
        # Is the planned position a valid position, or will aent collide?

    def is_distracted(self):
        add_noise = False
        # print (" Is car distracted? "+str(self.distracted)+" distracted len "+str(self.distracted_len))
        if not self.distracted and self.settings.add_noise_to_car_input:

            add_noise = np.random.binomial(1, self.settings.car_noise_probability)
            # print (" Add noise to car input? Draw random variable with prob "+str(self.settings.car_noise_probability)+" value "+str(add_noise))
            if add_noise:
                self.distracted = True
                self.distracted_len = np.random.poisson(self.settings.avg_distraction_length)
                # print (" Set car distracted for "+str(self.distracted_len))
        elif self.distracted_len > 0 and self.distracted:
            # print (" Agent is still distracted ")
            return True
        else:
            # print (" Agent is not distracted ")
            self.distracted = False
            return False
        return add_noise

        # Give the coordinates of the bounding box
    def set_bbox(self, frame, next_pos=[]):
        #print("Add car bbox frame "+str(frame))
        min_dim =min(self.settings.car_dim[1:])
        max_dim = max(self.settings.car_dim[1:])
        car_pos=[]
        if len(next_pos)>0:
            car_pos=next_pos
        else:
            car_pos=self.car[frame]
        if frame==0:
            vel=self.init_dir
            if np.linalg.norm(vel[1:]) < 1e-5:
                vel=np.array([0,1,1])
        else:
            vel=car_pos-self.car[frame-1]
        # print (" find bbox " + str(frame)+" vel "+str(vel))
        if np.linalg.norm(vel[1:])>1e-5:
            if abs(vel[1])>abs(vel[2]):
                bbox=[car_pos[0]-self.settings.car_dim[0], car_pos[0]+self.settings.car_dim[0],
                                  car_pos[1] - max_dim, car_pos[1]+max_dim,
                                  car_pos[2] - min_dim, car_pos[2]+min_dim]
            else:
                bbox = [car_pos[0] - self.settings.car_dim[0],
                                    car_pos[0] + self.settings.car_dim[0],
                                    car_pos[1] - min_dim, car_pos[1] + min_dim,
                                    car_pos[2] - max_dim, car_pos[2] + max_dim]
        else:
            bbox =self.bbox[frame-1]
        if len(next_pos)==0:
            self.bbox[frame]=bbox
        return bbox


    def is_next_pos_valid(self, episode, next_pos):
        bbox=self.set_bbox(self.frame+1, next_pos=next_pos)
        valid = episode.valid_position(next_pos, bbox=bbox)
        if self.settings.allow_car_to_live_through_collisions:
            valid = True

        alive = self.is_agent_alive(episode)
        # print ("Car "+str(self.id)+" is next pos valid " + str(valid)+" is alive "+str(alive))
        return valid, alive



    # @profilemem(stream=memoryLogFP_decisionsCar)
    def next_action(self, episode, training=True, viz=False, manual=False):
        self.is_distracted()
        agent_frame = self.frame
        self.get_car_net_input_to_episode( episode,agent_frame)

        #===================================================================

        # print (" Car net input car velocity "+str(episode.pedestrian_data[self.id].velocity_car)+" car pos "+str(episode.car))
        # print (" Car position before feed forward  "+str( episode.car[self.frame])+" frame "+str(self.frame))

        # Note: the apply_net below call will fill in data in episode which is read and updated in this instance of car agent
        self.velocities[self.frame]=self.net.feed_forward(self.id, episode, self.current_init_frame + self.frame, training,
                                         agent_frame, distracted=self.distracted)
        # print (" Returned from net "+str(self.velocities[self.frame]))
        self.angle[self.frame]=episode.car_data[self.id].action_car_angle
        # print (" Save in car agent angle "+str(self.angle[self.frame])+" frame "+str(self.frame))
        self.speed[self.frame] = episode.car_data[self.id].speed_car
        self.probabilities[self.frame,:] = episode.car_data[self.id].probabilities_car
        # print (" Probailities in  car agent "+str(self.probabilities[self.frame,:])+" input from episode "+str(episode.probabilities_car))
        self.action[self.frame] = episode.car_data[self.id].action_car

        if self.settings.supervised_car:
            # print (" Supervised net ")
            # print(" Car original pos " + str(self.car[self.frame])+" step "+str(self.init_dir)+" frame "+str(self.frame))
            car_pos_after_step=self.car[self.frame]+self.init_dir
            # print(" Car planned position "+str(car_pos_after_step))
            min_dim = min(self.settings.car_dim[1:])
            max_dim = max(self.settings.car_dim[1:])
            bbox_car=[]

            if abs(self.init_dir[1]) > abs(self.init_dir[2]):
                bbox_car= [car_pos_after_step[0] - self.settings.car_dim[0],
                       car_pos_after_step[0] + self.settings.car_dim[0],
                       car_pos_after_step[1] - max_dim,
                       car_pos_after_step[1] + max_dim,
                       car_pos_after_step[2] - min_dim,
                       car_pos_after_step[2] + min_dim]
            else:
                bbox_car = [car_pos_after_step[0] - self.settings.car_dim[0],
                            car_pos_after_step[0] + self.settings.car_dim[0],
                            car_pos_after_step[1] - min_dim,
                            car_pos_after_step[1] + min_dim,
                            car_pos_after_step[2] - max_dim,
                            car_pos_after_step[2] + max_dim]
            # print(" Car bbox at planned position " + str(bbox_car))
            # print(" Agent original pos " + str(episode.agent[self.frame]) + " step " + str(episode.pedestrian_data[self.id].velocity[self.frame])+" frame "+str(self.frame))
            overlap=False
            for pedestrian in episode.pedestrian_data:
                agent_pos_after_step =pedestrian.agent[self.frame]+pedestrian.velocity[self.frame]
                # print(" Agent pos after step" + str(agent_pos_after_step)+" agent shape "+str(self.settings.agent_shape))
                bbox_agent = [agent_pos_after_step[0] - self.settings.agent_shape[0],
                              agent_pos_after_step[0] + self.settings.agent_shape[0],
                              agent_pos_after_step[1] - self.settings.agent_shape[1],
                              agent_pos_after_step[1] + self.settings.agent_shape[1],
                              agent_pos_after_step[2] - self.settings.agent_shape[2],
                              agent_pos_after_step[2] + self.settings.agent_shape[2]]
                # print(" Agent bbox at planned position " + str(bbox_agent))
                if overlap(bbox_agent[2:], bbox_car[2:], 1):
                    overlap=True
            if overlap:
                # print("overlap "+str(bbox_agent[2:])+" "+str(bbox_car[2:])+" do not take step")
                self.velocities[self.frame]=np.zeros_like(self.velocities[self.frame])
                self.speed[self.frame] = 0

                self.action[self.frame] = 0
            else:
                # print("no overlap " + str(bbox_agent[2:]) + " " + str(bbox_car[2:]) + " take step "+str(self.init_dir))
                self.velocities[self.frame] = self.init_dir
                self.speed[self.frame] = np.linalg.norm(self.init_dir[1:])
                self.action[self.frame] =1

        # print ("After saving in vector  Velocity car " + str(episode.pedestrian_data[self.id].velocity_car)+" frame "+str(self.frame))

        #print(" Save in agent action "+str(self.action[self.frame])+" frame "+str(self.frame))
        #print ( " Car velocity after feed forward  " + str(episode.pedestrian_data[self.id].velocity_car)+" speed"+str(self.speed[self.frame])+" probabilities "+str(self.probabilities[self.frame]) + " frame " + str(self.frame)+" ")
        if manual:
            print ( " Car velocity after feed forward  " + str(episode.velocity_car)+" speed"+str(self.speed[self.frame])+" probabilities "+str(self.probabilities[self.frame]) + " frame " + str(self.frame)+" ")

            if self.is_agent_alive(episode):
                value_y = eval(input("Next action: y "))
                value_z = eval(input("Next action: z "))
                self.velocities[self.frame] =np.array([0,value_y, value_z ])
                episode.car_data[self.id].speed_car = np.linalg.norm(self.velocities[self.frame] [1:])
                self.speed[self.frame] =episode.car_data[self.id].speed_car
                print(" car's speed "+str(self.speed[self.frame]  )+" frame "+str(self.frame))
                episode.car_data[self.id].velocity_car= self.velocities[self.frame]
                if episode.car_data[self.id].speed_car==0:
                    episode.car_data[self.id].action_car=0
                else:
                    episode.car_data[self.id].speed_car=1
            else:
                print("Car is dead")
        # print (" Done with next action " + str(self.velocities[self.frame]))
        return self.velocities[self.frame]

    def get_car_net_input_to_episode(self,  episode, agent_frame=-1):

        # IMPORTANTE NOTE: the pattern is to put the metrices inside update_metrics
        # This is really an exception, since getting feedback from environment in a correct way will not change the result of this mainly.
        # And because there are used until (inside the net forwarding model, for instance) update_metrics is called
        # ============================================
        self.external_car_vel[self.frame]=episode.car_data[self.id].external_car_vel
        car_vel=self.init_dir
        if self.frame>0:
            car_vel=self.velocities[self.frame-1]


        episode.car_data[self.id].intersection_with_pavement = iou_sidewalk_car(self.bbox[self.frame],
                                                                                episode.reconstruction)
        self.measures[self.frame, CAR_MEASURES_INDX.iou_pavement] = copy.copy(episode.car_data[self.id].intersection_with_pavement)

        # TO DO: Add occlusion here
        _, episode.car_data[self.id].dist_to_closest_ped = find_closest_pedestrian_in_list(self.frame, self.car[self.frame],episode, self.settings.agent_shape, car_vel, self.settings.field_of_view_car, is_fake_episode=True)
            # self.frame, self.car[self.frame], episode.people)
        self.measures[self.frame, CAR_MEASURES_INDX.dist_to_closest_pedestrian] = copy.copy(episode.car_data[
            self.id].dist_to_closest_ped)
        # TO DO: Add occlusion here
        #frame, pos,episode.pedestrian_data)
        min_dist,  episode.car_data[self.id].id_closest_agent=find_closest_controllable_pedestrian(self.frame, self.car[self.frame],episode, self.settings.agent_shape, car_vel, self.settings.field_of_view_car, is_fake_episode=True)
        self.measures[self.frame, CAR_MEASURES_INDX.id_closest_agent] = episode.car_data[self.id].id_closest_agent
        # TO DO: Add occlusion here
        self.closest_car[self.frame], episode.car_data[self.id].dist_to_closest_car = find_closest_car(self.frame, self.car[self.frame],episode, self.settings.agent_shape, car_vel, self.settings.field_of_view_car, is_fake_episode=True)
        # print(" Add closest car frame " + str(self.frame) + " val " + str(episode.closest_car))
        self.measures[self.frame, CAR_MEASURES_INDX.dist_to_closest_car] = copy.copy(
            episode.car_data[self.id].dist_to_closest_car)
        #self.frame, self.car[self.frame], episode.cars, car_vel)
        episode.car_data[self.id].closest_car = self.closest_car

    def update_episode_with_car_data(self,  episode):
        # episode.agent, episode.car, episode.pedestrian_data[self.id].velocity, episode.pedestrian_data[self.id].velocity_car, agent_frame
        episode.car_data[self.id].car_id = self.car_id
        episode.car_data[self.id].car = self.car
        if self.frame > 0:
            episode.car_data[self.id].velocity_car = self.velocities
        # print (" Add velocity car "+str(episode.pedestrian_data[self.id].velocity_car))
        episode.car_data[self.id].car_dir = self.init_dir
        episode.car_data[self.id].car_goal = self.goal
        episode.car_data[self.id].car_bbox= self.bbox

    def carry_out_step(self, next_pos, step, episode):
        # print("Carry out step: "+str(next_pos)+" frame "+str(self.frame+1))
        if len(self.car) <= self.frame+1:
            self.car.append(self.pos_exact)
        else:
            self.car[self.frame+1] = self.pos_exact

        self.set_bbox(self.frame+1)

    def update_agent_pos_in_episode(self, episode, updatedFrame):
        assert self.frame == updatedFrame - 1, "We are not targeting the correct frame !"

        episode.car_data[self.id].car[updatedFrame] = self.pos_exact
        #print (" Added car frame "+str(updatedFrame)+"to episode, value  exact "+str( episode.car[updatedFrame] )+" "+str(episode.car))

    def on_post_tick(self, episode):

        self.frame += 1

    def update_metrics(self, episode):
        carLastFrameData = episode.environmentInteraction.getAgentLastFrameData(self)

        updatedFrame = carLastFrameData.frame - 1
        assert updatedFrame == self.frame - 1, "Expecting to compute metrics for previous frame! but got "+str(updatedFrame)+" previous: "+str(self.frame - 1)
        
        self.measures[updatedFrame, CAR_MEASURES_INDX.distracted] = self.distracted
        #print ("Evaluate car list "+str(episode.cars)+" episode.people "+str(episode.people))



        # NOTE: TODO Ciprian: this should be ideally taken from environment interaction, both the position of the pedestrian
        # Plus the collisions test inside
        # However, it might not even worth the effort since being very close to the car (< 20 cm) really could mean a collision in the end


        self.set_bbox(self.frame)
        self.measures=end_of_episode_measures_car(self.id,updatedFrame, self.measures, episode, self.bbox[self.frame],
                                                  end_on_hit_by_pedestrians=True, goal=self.goal,
                                                  allow_car_to_live_through_collisions=self.settings.allow_car_to_live_through_collisions,
                                                  agent_size=self.settings.agent_shape, carla_new=self.settings.new_carla)

        #if not self.settings.allow_car_to_live_through_collisions:

        self.measures[max(self.frame - 1, 0), CAR_MEASURES_INDX.agent_dead] = max(self.measures[0:self.frame, CAR_MEASURES_INDX.agent_dead] )

        #self.evaluate_car_reward(max(self.frame - 1, 0))

        #print (" Car measures "+str(self.measures[max(self.frame - 1, 0),:])+" frame "+str(self.frame))
        #episode.end_of_episode_measures(max(self.frame - 1, 0)) # To do: make independent of variables.

    def mark_agent_hit_obstacle_in_episode(self, episode,updatedFrame):
        if self.settings.allow_car_to_live_through_collisions:
            pass
        self.measures[updatedFrame, CAR_MEASURES_INDX.hit_obstacles] = 1
        #pass
        #self.measures[self.frame, 3] = 1

    def mark_agent_dead_in_episode(self, episode,updatedFrame):

        self.measures[updatedFrame, CAR_MEASURES_INDX.agent_dead] = 1
        # print("Mark agent dead in local episode " + str(self.measures[updatedFrame, CAR_MEASURES_INDX.agent_dead])+" frame "+str(self.frame))

    def mark_valid_move_in_episode(self, episode,updatedFrame):
        if self.settings.allow_car_to_live_through_collisions:
            pass
        self.measures[updatedFrame, CAR_MEASURES_INDX.hit_obstacles] = 0
        pass
        #self.measures[self.frame, 3] = 0

    # def evaluate_car_reward(self,frame):
    #     #max_speed=70000/3600*5/17 # 70 km/h
    #     self.reward[frame]=0
    #     if self.measures[frame, CAR_MEASURES_INDX.agent_dead]:
    #         return
    #     self.measures[frame,CAR_MEASURES_INDX.distance_travelled_from_init] = np.linalg.norm(self.car[frame + 1] - self.car[0])
    #     if np.abs(self.settings.reward_weights_car[CAR_REWARD_INDX.distance_travelled])>0:
    #         self.reward[frame]=self.settings.reward_weights_car[CAR_REWARD_INDX.distance_travelled]*self.measures[frame, CAR_MEASURES_INDX.distance_travelled_from_init]/(self.settings.car_max_speed_voxelperframe*(frame+1))
    #
    #     if self.settings.allow_car_to_live_through_collisions:
    #         # print (" Penalty for collision ?"+str(self.measures[frame, CAR_MEASURES_INDX.hit_by_agent]))
    #         self.reward[frame]+= self.settings.reward_weights_car[CAR_REWARD_INDX.collision_pedestrian_with_car] *self.measures[frame, CAR_MEASURES_INDX.hit_by_agent]
    #     else:
    #         self.reward[frame] +=self.settings.reward_weights_car[CAR_REWARD_INDX.collision_pedestrian_with_car]*self.measures[frame,CAR_MEASURES_INDX.hit_pedestrians]
    #         self.reward[frame] += self.settings.reward_weights_car[CAR_REWARD_INDX.collision_car_with_car] * \
    #                               self.measures[frame, CAR_MEASURES_INDX.hit_by_car]
    #
    #
    #         self.reward[frame] += self.settings.reward_weights_car[CAR_REWARD_INDX.collision_car_with_objects] * \
    #                               self.measures[frame, CAR_MEASURES_INDX.hit_obstacles]
    #
    #
    #         self.reward[frame] += self.settings.reward_weights_car[CAR_REWARD_INDX.penalty_for_intersection_with_sidewalk] * \
    #                               self.measures[frame, CAR_MEASURES_INDX.iou_pavement]
    #     self.reward[frame] += self.settings.reward_weights_car[CAR_REWARD_INDX.penalty_for_speeding] * max(self.speed[frame]-self.settings.car_reference_speed, 0)
    #     # print(" Car speed "+str(self.speed[frame])+" refernce "+str(self.settings.car_reference_speed)+" diffrenece to reference "+str(self.speed[frame]-self.settings.car_reference_speed)+" reward "+str(self.settings.reward_weights_car[CAR_REWARD_INDX.penalty_for_speeding] * max(self.speed[frame]-self.settings.car_reference_speed, 0)))
    #     if abs(self.settings.reward_weights_car[CAR_REWARD_INDX.reached_goal] )>0:
    #         if self.measures[frame, CAR_MEASURES_INDX.goal_reached] == 0:
    #             local_measure = 0
    #             if frame == 0:
    #                 orig_dist = np.linalg.norm(np.array(self.goal[1:]) - self.car[0][1:])
    #                 local_measure = ((orig_dist - self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) / self.settings.car_max_speed_voxelperframe)
    #                 # print (" Frame is zero. initial dist to goal "+str(orig_dist)+" current dist "+str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal])+" diff "+str((orig_dist - self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) )+" normalized by speed "+str(local_measure) +" normalized by "+str(max_speed))
    #             if frame > 0 and self.measures[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] > 0:
    #                 local_measure = ((self.measures[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] - self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) / self.settings.car_max_speed_voxelperframe)
    #                 # print ("Previous dist to goal " + str(self.measures[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] ) + " current dist " + str(
    #                 #     self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) + " diff " + str((self.measures[frame - 1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] - self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal])) + " normalized by speed " + str(
    #                 #     local_measure) +" normalized by "+str(max_speed))
    #
    #             self.reward[frame] += self.settings.reward_weights_car[ CAR_REWARD_INDX.distance_travelled_towards_goal] * local_measure
    #
    #             # print (" Linear reward for getting closer to goal " + str(self.reward[frame]) + " local measure " + str(local_measure))
    #
    #         else:
    #             self.reward[frame] += self.settings.reward_weights_car[CAR_REWARD_INDX.reached_goal]
    #             # print (" Reached goal " + str(self.reward[frame]) + " car  " +str(self.car[frame + 1])+ "goal"+str(self.goal)+" frame "+str(frame))
    #     if frame>0:
    #         collision = (
    #         self.measures[frame, CAR_MEASURES_INDX.agent_dead] and self.measures[frame - 1, CAR_MEASURES_INDX.agent_dead])
    #         # if collision:
    #         #     print ("Car Frame "+str(frame)+" collisons. "+str( self.measures[frame, CAR_MEASURES_INDX.agent_dead])+" collisons. "+str( self.measures[frame-1, CAR_MEASURES_INDX.agent_dead]))
    #         reached_goal = self.measures[frame, CAR_MEASURES_INDX.goal_reached] and self.measures[
    #             frame - 1, CAR_MEASURES_INDX.goal_reached]
    #         #print("Car collided "+str(collision)+" goal reached "+str(reached_goal))
    #         if (collision or reached_goal):
    #             self.reward[frame] =0
    #             # print (" Set reward to 0: collision "+str(collision)+" reached goal "+str(reached_goal)+" Frame "+str(frame))
    #     # print("Car distance travelled original pos:"+str( self.car[0])+" new pos "+str(self.car[frame + 1])+" diff "+str(self.car[frame + 1] - self.car[0])+" dist "+str(self.measures[frame,CAR_MEASURES_INDX.distance_travelled_from_init]))
    #     # print("Car reward:" + str( self.reward[frame]) + " mutiplier: "+str(self.settings.reward_weights_car[CAR_REWARD_INDX.distance_travelled]) )
    #     # print(" Collision reward "+str(self.measures[frame,CAR_MEASURES_INDX.hit_pedestrians])+" mulitiplier "+str(self.measures[frame, CAR_MEASURES_INDX.hit_by_agent]))

    # Check if agent is alive
    def is_agent_alive(self, episode):
        if self.frame==0:
            return True
        frame = max(0, self.frame - 1)
        if self.measures[frame, CAR_MEASURES_INDX.agent_dead]:
            return False

        # print("Frame in car alive " + str(frame))
        no_collision_with_cars=self.measures[frame, CAR_MEASURES_INDX.hit_by_car] <= 0
        # print("No collision with cars?  " + str(no_collision_with_cars))
        if self.settings.stop_on_goal_car:
            goal_not_reached=self.measures[frame, CAR_MEASURES_INDX.goal_reached] == 0
        else:
            goal_not_reached =True
        # print("Goal not reached?  " + str(goal_not_reached))
        #no_collision_in_distance=self.measures[max(0, self.frame - 1), CAR_MEASURES_INDX.inverse_dist_to_closest_car] <= 1
        no_collision_with_pedestrian=self.measures[frame, CAR_MEASURES_INDX.hit_pedestrians]<= 0
        # print("No collision with pedestrian?  " + str(no_collision_with_pedestrian))
        if self.settings.allow_car_to_live_through_collisions:
            # print (" Did car hit agent? "+str(self.measures[frame,CAR_MEASURES_INDX.hit_by_agent]))
            if self.settings.car_input == "difference":
                return goal_not_reached and self.measures[frame,CAR_MEASURES_INDX.hit_by_agent]==0
            return self.measures[frame,CAR_MEASURES_INDX.hit_by_agent]==0
        if frame>0:
            no_collision_with_obstacle=max(self.measures[0:frame+1, CAR_MEASURES_INDX.hit_obstacles])<= 0
        else:
            no_collision_with_obstacle = self.measures[0, CAR_MEASURES_INDX.hit_obstacles] <= 0
        # print("No collision with obstacle "+str(no_collision_with_obstacle))
        # print("Is agent alive? "+str(no_collision_with_cars and goal_not_reached and no_collision_with_pedestrian and no_collision_with_obstacle))
        return no_collision_with_cars  and no_collision_with_pedestrian and no_collision_with_obstacle and  goal_not_reached


    def update_episode(self, episode, speed, value, velocity, probabilities):
        self.velocity= velocity
        self.velocities[self.frame] = velocity
        self.probabilities[self.frame] = probabilities
        self.speed[self.frame] = speed

    # statistics, episode, filename, filename_weights,poses,priors,statistics_car, seq_len=-1
    # statistics, episode, filename, filename_weights, poses, priors, statistics_car
    def train(self, ep_itr, statistics, episode, filename, filename_weights, poses, priors, statistics_car,  last_frame):

        episode.car_data[self.id].closest_car = self.closest_car
        return self.net.train(self.id, ep_itr, statistics, episode, filename, filename_weights, poses, priors,statistics_car ,seq_len=last_frame)



    def evaluate(self,ep_itr,  statistics, episode, poses, priors, statistics_car,  last_frame):
        episode.car_data[self.id].closest_car = self.closest_car
        return self.net.evaluate(self.id, ep_itr, statistics, episode, poses, priors, statistics_car ,seq_len=last_frame)


    # These two functions are used to avoid serializing in the cached version the networks !
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't include states that shouldn't be saved
        if "net" in state:
            del state["net"]
        if "goal_net" in state:
            del state["goal_net"]
        if "init_net" in state:
            del state["init_net"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # TODO: add the things back if needed !
        self.baz = 0

