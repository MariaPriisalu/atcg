
import os
import tensorflow as tf


from settings import RANDOM_SEED
tf.random.set_seed(RANDOM_SEED)


import numpy as np
import os,sys

from settings import run_settings
sys.path.insert(0,os.path.join(os.path.dirname( os.path.abspath(__file__)), '..'))
from RL.RLmain import RL


##
# Main script- train an RL agent on real or toy case.
#







def main(setting):

    RL().evaluate(setting, viz=False)

if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    setup=run_settings()
    setup.realtime_carla=False
    setup.update_frequency_test = 10
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    main(setup)
