#reward.py

import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def quaternion_orientation_error(q_current, q_desired):
    """
    Computes the orientation error between two quaternions as a scalar penalty.
    
    Args:
        q_current (array-like): Current quaternion [x, y, z, w].
        q_desired (array-like): Desired quaternion [x, y, z, w].
        
    Returns:
        float: Scalar orientation error (radians).
    """
    r_current = R.from_quat(q_current)
    r_desired = R.from_quat(q_desired)
    
    q_error = r_desired * r_current.inv()
    q_error = q_error.as_quat()  
    
    theta = 2 * np.arccos(np.clip(abs(q_error[3]), -1.0, 1.0)) 
    return theta  # In radians

                        
def compute_reward_push(current_gripper_bread_distance, current_distance, is_contact_gripper_with_bread,
                        gripper_orientation, desired_orientation, gripper_position, bread_position, push_target_close,
                        v=1, alpha=1e-5):    
    
    distance_wall_bread_rewards = - v * np.log(current_distance**2 + alpha)
    distance_reward = -0.5 * np.log(current_gripper_bread_distance**2 + alpha)
    
    reward_push = distance_wall_bread_rewards + distance_reward
    
    if is_contact_gripper_with_bread:
        reward_push += 40
    else:
        reward_push -= 15
        
    # Orientation penalty
    orientation_error = quaternion_orientation_error(gripper_orientation, desired_orientation)

    # Encourage Lower Gripper Position
    desired_gripper_z = bread_position[2] - 0.05  
    position_error = gripper_position[2] - desired_gripper_z  

    # **Penalize higher gripper positions more aggressively**
    if position_error > 0.1:  
        reward_push -= 200 * position_error  

    # **Reward lower gripper positions aggressively**
    if position_error < 0:  
        reward_push += 500 * abs(position_error) 

    return reward_push, orientation_error, position_error

