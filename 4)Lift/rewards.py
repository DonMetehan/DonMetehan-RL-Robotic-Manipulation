#rewards.py

import numpy as np
import math

    
def compute_reward_lift(current_distance_lift,current_gripper_to_bread,is_contact_gripper_with_bread,v=1, alpha=1e-5):

    distance_bread_lift_target_rewards = - v * np.log(current_distance_lift**2 + alpha)
    
    reward_lift = distance_bread_lift_target_rewards
    
    reward_lift +=  - np.log(current_gripper_to_bread**2 + alpha)
    
    if is_contact_gripper_with_bread:
        reward_lift += 20
    else :
        reward_lift -= 10

    return reward_lift
