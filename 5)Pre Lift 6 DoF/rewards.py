#rewards.py

import numpy as np
import math
def quaternion_orientation_error(actual_quat, target_quat):
    #ORIENTATION ERROR
    #The unit quaternions are defined as q = np.array([x, y, z, w])
    #Compute the Skew-Symm. matrix for the desired orientation
    skew_des = np.array([[int(0), -target_quat[2], target_quat[1]], [target_quat[2], int(0) , -target_quat[0]], [-target_quat[1], target_quat[0], int(0)]])
    #Compute the orientation error
    orientation_err = (actual_quat[3]*np.array([target_quat[0], target_quat[1], target_quat[2]])) - (target_quat[3]*np.array([actual_quat[0], actual_quat[1], actual_quat[2]])) - np.matmul(skew_des,np.array([actual_quat[0], actual_quat[1], actual_quat[2]])) 
    
    return orientation_err
    

def compute_reward_pre_lift(gripper_orientation, desired_orientation, best_ori, prev_error, pre_lift_target, gripper_position, prev_position, is_contact_gripper_with_bread, alpha=1e-5):
    orientation_error = quaternion_orientation_error(gripper_orientation, desired_orientation)
    
    error_x, error_y, error_z = map(abs, orientation_error)
    best_error_x, best_error_y, best_error_z = map(abs, best_ori)
    prev_error_x, prev_error_y, prev_error_z = map(abs, prev_error)

    orientation_error_norm = np.linalg.norm(orientation_error)

    if error_x < best_error_x:
        reward_x = -np.log(error_x + alpha) + 2 
        best_error_x = error_x 
    else:
        reward_x = - (1.2 * error_x**2) 

    if error_y < best_error_y:  
        reward_y = -np.log(error_y + alpha) + 2  
        best_error_y = error_y 
    else:
        reward_y = - (1.2 * error_y**2)

    if error_z < best_error_z:
        reward_z = -np.log(error_z + alpha) + 2  
        best_error_z = error_z
    else:
        reward_z = - (1.2 * error_z**2)
    
    
    distance_to_target = np.linalg.norm(gripper_position - pre_lift_target)
    distance_to_target_reward = - np.log(distance_to_target + alpha)
    
    orientation_error_norm_reward =  - np.log(orientation_error_norm + alpha)
    
    total_reward = reward_x + reward_y + reward_z + distance_to_target_reward + orientation_error_norm_reward

    if is_contact_gripper_with_bread:
        total_reward += 5
    else:
        total_reward -= 1

    distance_to_target = np.linalg.norm(gripper_position - pre_lift_target)



    best_ori = [best_error_x, best_error_y, best_error_z]
    prev_error = [error_x, error_y, error_z] 
    prev_position = gripper_position
    
    return total_reward, orientation_error, best_ori, prev_error, error_x, error_y, error_z, orientation_error_norm, prev_position


