import math
import numpy as np

def compute_reward_converge(current_distance,w=1, v=1, alpha=1e-5, orientation_weight=1):
    
    # Compute positional distance reward
    distance_reward = - v * np.log(current_distance**2 + alpha)
    
    # Final reward
    reward_converge = distance_reward
    
    return reward_converge






