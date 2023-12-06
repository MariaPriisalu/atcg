# Official Repository of _Varied Realistic Autonomous Vehicle Collision Scenario Generation_ #


Code for the paper [_Varied Realistic Autonomous Vehicle Collision Scenario Generation_](https://link.springer.com/chapter/10.1007/978-3-031-31438-4_24) published at the Scandinavian Conference on Image Analysis.

**Authors:** [Maria Priisalu](http://www.maths.lth.se/sminchisescu/research/profile/7/maria-priisalu), [Ciprian Paduraru](https://scholar.google.com/citations?user=EaAekU4AAAAJ&hl=en),  and [Cristian Sminchisescu](http://www.maths.lth.se/sminchisescu/)

### Overview
This repository contains code for training the Semantic Pedestrian Locomotion agent and the Adversarial Test Case Generator.
The reinforcement learning logic and agents are in the folder `RL`.
The Semantic Pedestrian Locomotion policy gradient network with two 3D convolutional layers can be found in the class `Seg_2d_min_softmax` in `net_sem_2d.py`, and the Adversarial Test Synthesizer in the class `InitializerNet` in `initializer_net.py`.
The pedestrian agent's logic (moving after an action) can be found in the abstract class `agent.py`.


The class `Episode` is a container class. It contains the agent's actions, positions, and reward for the length of an episode.
The class `Environment` goes through different episode setups (different environments) and applies the agent in the method `work`.
Currently this class also contains the visualization of the agent.

The class `test_episode.py` contains unit tests for the initialization of an episode, and the different parts of the reward.
The class `test_tensor` contains unit tests for the 3D reconstruction needed for environment.

Other folders:
 `CARLA_simulation_client`: CARLA Client code for gathering a CARLA dataset.
 `colmap`: scripts utilizing a modified version of colmap.
 `commonUtils`: The adapted PFNN code base and some environment descriptions.
 `Datasets`: empty folder for gathered datasets. 
 `licences`: the licences of some of the libraries the repository uses.
 `localData`: contains trained models in `Models` and experimental results in `come_results`.
 `localUserData`: a directory structure to gather data from experiments.
 `tests`: various unit tests.
 `triangulate_cityscapes`: Reconstruction of the 3D RGB and Segmentation using cityscapes depthmaps and triangulation- not used in the paper.
 `utils`: various utility functions

### Running the code

To train a new model: from the main directory, type:
```
python RL/RLmain.py
```
To evaluate a model: from the main directory, type:
```
python RL/evaluate.py
```

To visualize results insert the timestamp of the run into RL/visualization_scripts/Visualize_evaluation.py
```
timestamps=[ "2021-10-27-21-14-35.842985"]
```
and run the file with 
```
python RL/visualization_scripts/Visualize_evaluation.py
```
from spl/.

### Licence
This work utilizes [CARLA](https://github.com/carla-simulator/carla) (MIT Licence), [Waymo](https://github.com/waymo-research/waymo-open-dataset) (Apache Licence), [PFNN](https://github.com/sreyafrancis/PFNN) (free for academic use), [COLMAP](https://colmap.github.io/license.html) (new BSD licence), [Cityscapes](https://github.com/mcordts/cityscapesScripts) (the Cityscapes licence - free for non-profit use), [GRFP](https://github.com/D-Nilsson/GRFP) (free for academic use), and [PANet](https://github.com/ShuLiu1993/PANet) as well as the list of libraries in the yml file. 
The requirements of the licences that the work builds upon apply. Note that different licences may apply to different directories. We wish to allow the use of our repository in academia freely or as much as allowed by the licences of our dependencies. We provide no warranties on the code.


If you use this code in your academic work please cite one of the works,
```
@InProceedings{10.1007/978-3-031-31438-4_24,
author="Priisalu, Maria
and Paduraru, Ciprian
and Smichisescu, Cristian",
editor="Gade, Rikke
and Felsberg, Michael
and K{\"a}m{\"a}r{\"a}inen, Joni-Kristian",
title="Varied Realistic Autonomous Vehicle Collision Scenario Generation",
booktitle="Image Analysis",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="354--372",
isbn="978-3-031-31438-4"
}

```
### Prerequisites
The models have been trained on Ubuntu 20.04, Tensorflow 2.6.0 (see official prerequists), python 3.8.12 on Nvidia Titan Xp.
Please see the yml file for the exact anaconda environment. 
Note that `pfnncharacter` is a special library and must be installed by hand. See below.

### Installation
Install the conda environment according to `environment.yml`
To install pfnncharacter enter `commonUtils/PFNNBaseCode` and follow the README.

# Datasets
## CARLA
In CARLA 0.8.2 or 0.9.4 you can gather the equivalent dataset with the scripts in `CARLA_simulation_client`.

The original CARLA dataset has been gathered with CARLA 0.8.2. You can find the client script for gathering the data in `CARLA_simulation_client/gather_dataset.py`. This script gathers 3D reconstructions of scenes (i.e. depth+images+ semantic segmentation) every 50th frame and images and bounding boxes of moving objects at every frame. 
Gather a training set (path to folder in `settings.py` `self.carla_path`) with the script on `CARLA_simulation_client/gather_dataset.py` Town 1 and all of the test data is created on Town 2, which should be placed in `self.carla_path_test`.
You can change `output_folder` as necessary.

Next to combine the static objects in the `.ply` (viewable in Meshlab) files run `carla_utils/combine_reconstructions.py`.

### Caching mechanism
The caching of episodes data happens in <code>environment_abstract.py</code>, inside <code>set_up_episode</code> function. This means that the first time around may be slow, but once the dataset is cached everything will run faster.

```
for each epoch E:
   for each scenerio in dataset S:
      episode = set_up_episode(S)
      .....
```
# Models
## Adversarial Test Case Generator
Models mentioned in the CoRL paper can be found in `localData/Models/BehaviorVariedTestCaseGeneration`.


To instead train the Adversarial Test Case Generator make sure that the following settings are True:
```    
    self.learn_init =True # Do we need to learn initialization
    self.learn_goal=False # Do we need to learn where to place the goal of the pedestrian at initialization
    self.learn_time=False
    ...
    self.useRealTimeEnv = True
    self.useHeroCar = True
    self.useRLToyCar = False
```


# Real time interaction with Carla simulation engine

There are two paths to use the Carla package this, either the SIMPLE one using a released package (Ubuntu 18 and 20), or a DEVELOPMENT one for debugging and dev.

## Carla installation
When installed run the CARLA server using: 

./CarlaUE4.sh -carla-server-timeout=5000000ms -carla-streaming-port=0 (huge constant just to be sure that you get not timeouted while raycasting).

First, take the modified branch of Carla from: https://github.com/paduraru2009/carla 
It contains some new actors for real time communication, raycasting utils, interaction, PFNNN animation model, etc - to detail more soon.
Install it using the official instructions.

Build https://github.com/paduraru2009/carla as explained, run  ```make launch``` to get the editor.  
     A1. OPTIONAL if you don't want to touch the original Engine/Carla code
     On Linux, you need CLion: go to Editor Settings and modify source to CLion, hit Generate solution from File and it will generate the solution. Put it back to Null editor. Now open CLion and open folder Unreal/CarlaUE4 where the CMakeFile is located. Now you can modify and generate everything. run CarlaUE4Editor-Linux-Debug solution configuration. 
     On Windows, development is much more stable. Use the same as above, but use Visual Studio 2017.
     
     A2. To package the project and run it without modifying the source code: File-> Package Project -> OS -> choose your output folder. Then you can use the executable without building again.
     
To make the release, after you do ```make PythonAPI & make launch``` also do ```make plugins```, then ```make package``` each time when you change the code.
     
### CLient / server on different machines/OS.
The fastest development mode is to use the CLient on Ubuntu (Python, ML libraries etc), Server on Windows. Just run the server on a machine, and for each script that you run from our side (or tutorials) give the ```--host IPaddress``` of the target machine.

## Gather new dataset with real ground truth:   
   Use the gather_dataset3.py (and even better the gatherDataset_Record configuration if you are using Pycharm !) to build a new dataset. A few options that you will observer there:
```
--simulationReplayMode
0
--outputDataBasePath
"DatasetCustom/Data1"

--scenesConfigFile
"DatasetCustom/Data1/scenesConfig.json"
--no_server_rendering
0
--no_client_rendering

0
--forceExistingRaycastActor
0
```
These self explanatory parameters decide if you want to use rendering or not, if you want to force a raycast actor location on the map (not recommended)the map to use, the path of the output training data resulted, and most important: the scene configuration file as explained below.

There is a template folder left in this repository named DatasetCustom_template. To gather a new sample dataset do the followings:
* A. Copy the DatasetCustom_template to a path you wish
* B. Run the gather_dataset3.py (recommended using the above named pycharm config), by pointing --outputDataBasePath = path/to/DatasetCustom_template and --scenesConfigFile = path/to/DatasetCustom_template/scenesConfig.json

   It will run and gather the scenes data according the definition inside scenesConfig.json (see details below about the config parameters and what they represent).
   
* C. When it finishes, run in the DatasetCustom_template folder the DatasetCopyUtil.py script which copies all recoreded data to DataSetOut subfolder. you can directly use this sample dataset inside the train/val datasets of RlAgent.


### Details about the sceneConfig json file and parameters
You can collect data from a different number of scenes and episodes at the same time. As you can see in the json file you can in this worder: define a town name, number of episodes per scene, number of frames to capture, a fixed seed, how many pedestrians and cars would you like, the STARTING position of the car (```X,Y,Z```), the TARGET position of the car(```TX, TY, TZ```), if the view is from hero perspective (recommended as yes), the raycasting voxel resolution size, if it should use lidar or not, the parameters for how noisy the lidar is, at how many steps you want to get visual output saved when gathering data (```frame_step```), at how many steps do you want to save visual output when you do replay (```frame_step_replay```):

If randomSpawnLocation is 0, then the car will be spawned at location X,Y,Z. Else if 1, a random spawn point on the map will be used. If staticCar, then the hero car will not move at all. The TX,TY,TZ represent the target location and used if not a static car, but they can miss - meaning that car will not stop at that position, same with lidarData if heroView = 0. frame_step default is 10, this means at each 10 frames you will have a new lidar and images output coming out.

```
"scenes":{		
			"scene101":{
			"map":"Town03",
			"framesPerEpisodes":200,
			"numEpisodesPerScene":6,
			"numCarlaVehicles":2,
			"numCarlaPedestrians":8,
			"numTrainableVehicles": 1,
			"numTrainablePedestrians": 1,
			"simFixedSeed":0,
			"randomObserverSpawnLocation" : 1,
			"staticCar" : 1,
			"X":-9.40,
			"Y":136.90,
			"Z":0.0,
			"pitch":0.0,
			"roll":0.0,
			"yaw":90.0,
			"TX":-9.80,
			"TY":179.85,
			"TZ":0.0,
			"heroView":1,
			"rerouteAllowed":0,
			"voxelRes":5.0,
			"voxelsX":1024,
			"voxelsY":512,
			"voxelsZ":700,

				"lidarData":{
				"useLidar":1,
				"noise_dropoff_general_rate":0.0,
				"noise_dropoff_intensity_limit":1.0,
				"noise_dropoff_zero_intensity":0.0,
				"upperFOV":30.0,
				"lowerFOV":-25.0,
				"channels":64,
				"range":100.0,
				"pointsPerSecond":1000000.0
				},
			"frame_step":10,
			"frame_step_replay":1
			}
   .... other scenes following
```

Thus, the agent will travel from ```X,Y,Z => TX, TY, TZ``` using a default scripted Carla agent that take nice decisions in general. If ```rerouteAllowed is 1```, when getting at the destination point, the car will choose another target destination. If 0, the car will stop at the destination. The captures are made from a kind of drone camera attached above the car, then a series of matrix transformations are computed in the code to get the world location of pedestrians, cars and lidar detected pointcloud. To control the drone camera perspective and parameters you can use the static variables in ```DataGatherParams``` object.

Each scene has a name, After you run the script, you'll see that at the ```--outputDataBasePath``` a folder with ply and scene setup data will be created with the same folder name as scene name. This is for **caching** purposes. Thus, if you want next time to capture a huge number of episodes, you'll benefit from previous raycasting process !. 95% of the time usually is spent on raycasting process to get the scene point cloud through raycasting. From a performance perspective, the raycasting will happen once for each scene setup then all episodes are played. The output folder will contain the episodes data in ```--outputDataBasePath/MapName/EpisodeIndex/SpawnIndex/..``` . The ```SpawnIndex``` is obtained after the code checks which is the closest spawn point position of a hero car to the observer position given in the scene setup. If you run the script you will see an output like ```$$$ For scene name {dataGatherParams.sceneName} we found that the closest spawn point index is {self.closestSpawnPointIndex}```. Probably this is just for legacy purposes...

Note: This is also very valuable for real time interaction with the purposes of training RL agents. the setup could then be this:
 * Step 1. you run the same script to create some specific scene data
 * Step 2. you load from those folders the scenes once during your training episodes.


There are two ways to get the scene observer position and transforms from the environment:
* A. If you use the our branch custom release package you will get a text containing the observer location and rotation. Just navigate through the world using WASD keys and mouse, find a nice position then put it in your config file, as in the figure below
![](https://github.com/muumintroll/RLAgent/blob/new_master/RayCastActor_release.png)
* B. If you use the UE4 Editor, you can drag an object to a location as in the figure below.
![](https://github.com/muumintroll/RLAgent/blob/new_master/RayCastActor.png)

As Output you will see something like in the figure below, containing the Images captured from a hero car perspective (**ONLY** if --hero parameter was used and if ENABLE_SAVING is set to True in the code), the people.p/cars.p containing the trajectories for that episode, and the ply files for the scene.
![](https://github.com/muumintroll/RLAgent/blob/new_master/GatherDatasetOutput.png)
Note: **this output folder can be used as before without modifying the RLAgent code. You can use the ```visulizationdemo``` Pycharm environment or the ```visualization_scripts/show_real_reconstruction_small.py```, by pointing the output folder**.

## Data recoded replay, trained agent statistics visualization

Use the same script ```gather_dataset3.py```, and if you are using Pycharm there is a preconfigured parmaeters profile named ```gatherDataset_Replay``` in the repository. The parameters are as given below. Sketch of params in order: ```simulationReplayMode``` must be 1, next a few parameters similar to the one used above. What is different from the previous explanation: you could simulate both cars and/or pedestrians, you can also OPTIONALLY use a trained timestamp value which will try to import a trained agent (car or pedestrian) from the stats file given (```simulationReplayTimestamp```) - this will also replay the episodes from that stats file (```timeStampEpisodeIndex``` contains the list of episodes to be replayed), the name of the scene to be replayed must be given in ```sceneName```.
If stats are used, you can check the individual per episode results in the same output data folder, suffixed this time with the index of the episode. To control the camera perspective and parameters for visualization of replay you can use the static variables in ```DataGatherParams``` object.

```
--simulationReplayMode
1
--outputDataBasePath
"DatasetCustom/Data1"
--scenesConfigFile
"DatasetCustom/Data1/scenesConfig.json"
--no_server_rendering
0
--no_client_rendering
0
--forceExistingRaycastActor
0
--simulationReplayPath
DatasetCustom/Data1/test_0
--simulationReplayUseTrainedStats_car
1
--simulationReplayUseTrainedStats_ped
1
--simulationReplayTimestamp
2021-06-17-20-22-34.493714agent_test_174_test_0_-64_0
--sceneNameToReplay
scene100
--timeStampEpisodeIndex
"17,22,44,62,83,88"
```


## A few notes for the rendering on server/client side.
Usually on real-time interaction you want as fast as possible progress. So there are many ways of doing rendering for server / client that can be set using the ```--no_server_rendering``` and ```--no_client_rendering```, but strongly suggesting you to use one of the two:

- `No rendering on server (1) side and Minimal Rendering on client (0) `: it would look like in the picture below, observe that you can see the pedestrians, cars, simplified scene at runtime with minimal performance loss: ![](https://github.com/muumintroll/RLAgent/blob/new_master/SimplifiedRendering.png). The scene will be zoomed in around the current observer location but you can modify the view to other sides with mouse control.
  
-  `No rendering on server (1) side and No Rendering on client (1)`: both sides are black, you can't see anything real time with you still get API feedback and output folders with episode data !':. This is the fastest method for simulation.

### Getting a top down view image for 2D debugger 
- Step 1: Open from Unreal Engine 4 the level you want.
- Step 2: In the levels blueprint make sure you see the F12 connection with the raycastactor and 
the full scene enabled.
- Step 3: Play the level, hit F12. Wait a bit.... it will output the ply files and others for reconstruction
- Step 4: run the reconstruction process with RLmain.py and parameter -debugFullSceneRaycastRun 1
This will reconstruct the scene and put in the same folder the centering.p files and other needed in the later process
- Step 5: run visualizationdemo pycharm config which produces a file with a giving name (see params). Copy that file onto RLAgent\cache\no_rendering_mode\ .

Step 6: When you first run again RLmain, you'll get the tga of this top view segmentation that you obtained in Step 5 + road marks and everything else needed for the debugger


