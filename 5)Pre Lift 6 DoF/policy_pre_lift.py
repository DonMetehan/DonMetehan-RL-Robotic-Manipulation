#policy.py

import numpy as np

class LinearPolicyPreLift:
    def __init__(self, theta_1, theta_2):
        assert theta_1.shape == (7, 6)
        assert len(theta_2) == 7
        self.theta_1 = np.array(theta_1)
        self.theta_2 = np.array(theta_2)

    def act_pre_lift(self, s, orientation_error):
        state = np.concatenate((s, orientation_error)) 
        u = self.theta_1 @ state + self.theta_2
        torque = np.append(u, 1)
        return torque

