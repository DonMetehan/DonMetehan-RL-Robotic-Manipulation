
#policy.py

import numpy as np

class LinearPolicy:
    def __init__(self, theta_1, theta_2):
        assert theta_1.shape == (7, 3)
        assert len(theta_2) == 7
        self.theta_1 = np.array(theta_1)
        self.theta_2 = np.array(theta_2)

    def act(self, s):
        u = self.theta_1 @ s + self.theta_2
        torque = np.append(u, 1)
        return torque
