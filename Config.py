# this file is for the collection of parameters

# this is for log variance noise parameter
import numpy as np
from numpy import pi
import datetime


class Config:
    def __init__(self):
        self.SEED_NUMBER = 0

        self.WORLD_SIZE = 1.0  # 2.5
        self.ACTION_DIM = 2
        self.STATE_DIM = 29 #4  # 20

        self.TERMINAL_VEL = 0.1  # norm(action) that you believe as a signal to stop 0.1

        # all times are in second
        self.DELTA_T = 0.1 #0.01  # time to perform one action
        self.EPISODE_TIME = 1  # # maximum length of time for one episode. if monkey can't firefly within this time period, new firefly comes
        self.EPISODE_LEN = int(self.EPISODE_TIME / self.DELTA_T)  # number of time steps(actions) for one episode

        self.TOT_T = 2000000000  # total number of time steps for this code

        self.BATCH_SIZE = 64  # for replay memory (default:64)
        self.REWARD = 10  # for max reward
        self.NUM_EPOCHS = 2# for replay memory
        self.DISCOUNT_FACTOR = 0.99

        self.BOX_STEP_SIZE = 5e-1 # not used
        self.STD_STEP_SIZE = 2e-5  # 1e-4 action space noise (default: 2e-3)

        self.filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_path = '../new-firefly-forward_v2-data/'

        self.goal_radius_range = [0.2* self.WORLD_SIZE, 0.5* self.WORLD_SIZE] #0.375: best radius
        self.GOAL_RADIUS_STEP_SIZE = 1e-5

        self.gains_range = [0.8, 1.2, pi/5, 3*pi/10] # [vel min, vel max, ang min, ang max]
        self.noise_range = [np.log(0.01), np.log(1), np.log(pi/4/100), np.log(pi/4)]# ln(noise_var): SNR=[100 easy, 1 hard] [vel min, vel max, ang min, ang max]

        #self.gains_range = [8, 12, 8, 12] # [vel min, vel max, ang min, ang max]
        #self.std_range = [1e-2, 2, 1e-2, 2]# [vel min, vel max, ang min, ang max]



