import numpy as np
#from RL.settings import DEFAULT_SKIP_RATE_ON_EVALUATION_CARLA, DEFAULT_SKIP_RATE_ON_EVALUATION_WAYMO, PEDESTRIAN_MEASURES_INDX, CAR_MEASURES_INDX,PEDESTRIAN_INITIALIZATION_CODE,PEDESTRIAN_REWARD_INDX, NBR_MEASURES, NBR_MEASURES_CAR,STATISTICS_INDX, STATISTICS_INDX_POSE, STATISTICS_INDX_CAR,STATISTICS_INDX_CAR_INIT, STATISTICS_INDX_MAP, RANDOM_SEED_NP, RANDOM_SEED
from RL.reward_data_holder import RewardDataHolder

class PedestrianDataHolder(RewardDataHolder):

    def __init__(self, seq_len, seq_len_pfnn,use_pfnn_agent, DTYPE):
        vector_len = max(seq_len - 1, 1)
        super(PedestrianDataHolder, self).__init__(seq_len, DTYPE, valid_positions=None)
        # Variables for agent initialization -- constants
        self.speed_init = 0
        self.vel_init = np.zeros((1, 3), dtype=DTYPE)


        # Depends on run specific variables

        #self.measures = np.zeros((vector_len, NBR_MEASURES), dtype=DTYPE)

        # Variables for gathering statistics/ agent movement
        self.agent = [[] for _ in range(seq_len)]
        self.agent[0] = np.zeros(1 * 3, dtype=DTYPE)
        self.turning_point = np.ones(seq_len)
        self.velocity = [[]] * vector_len
        self.velocity[0] = np.zeros(3, dtype=DTYPE)
        self.action = [None] * vector_len  # For softmax
        self.action[0] = [4]
        self.speed = [None] * vector_len
        self.angle = np.zeros(vector_len, dtype=DTYPE)

        self.init_method = 0

        self.loss = np.zeros(vector_len, dtype=DTYPE)
        self.probabilities = np.zeros((vector_len, 27), dtype=DTYPE)

        self.goal_person_id_val = ""

        self.manual_goal = []
        self.agent_prediction_people = []


        # Pose variables- depend on if pfnn is used
        if use_pfnn_agent:
            self.avg_speed = np.zeros((seq_len_pfnn + seq_len))
            self.agent_high_frq_pos = np.zeros((seq_len_pfnn + seq_len, 2))
            self.agent_pose_frames = np.zeros((seq_len_pfnn + seq_len), dtype=np.int)
            self.agent_pose = np.zeros((seq_len_pfnn + seq_len, 93))
            self.agent_pose_hidden = np.zeros((seq_len_pfnn + seq_len, 512))
        self.seq_len=seq_len


    def get_original_dist_to_goal(self, frame):
        self.goal_to_agent_init_dist=np.linalg.norm(self.agent[frame][1:]-self.goal[frame,1:])
        return self.goal_to_agent_init_dist
