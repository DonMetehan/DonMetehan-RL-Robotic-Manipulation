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
    
def compute_reward_pre_lift(gripper_orientation,desired_orientation,best_ori,v=1, alpha=1e-5):

    orientation_error = quaternion_orientation_error(gripper_orientation,desired_orientation)
    error_x, error_y, error_z = map(abs, orientation_error)
    best_error_x, best_error_y, best_error_z = map(abs, best_ori)


    improvement_x = best_error_x - error_x
    improvement_y = best_error_y - error_y
    improvement_z = best_error_z - error_z

    orientation_error_norm = np.sqrt(orientation_error[0]**2 +orientation_error[1]**2 +orientation_error[2]**2)
    
    # Compute separate orientation rewards for each axis
    if improvement_x > 0 and error_x < 0.05:
        best_error_x = error_x
        reward_x = - 10 * np.log(error_x**2 + alpha) 
    else:
        reward_x = - 60 * np.exp(error_x)

    if improvement_y > 0:
        best_error_y = error_y
        reward_y = - 30 * np.log(error_y**2 + alpha) 
    else:
        reward_y = - 30 * np.exp(error_y)

    if improvement_z > 0 and error_z < 0.1 :
        best_error_z = error_z
        reward_z = - 15 * np.log(error_z**2 + alpha) 
    else:
        reward_z = - 40 * np.exp(error_z)


    # Final ödülü
    total_reward = reward_x + reward_y + reward_z

    best_ori = [best_error_x,best_error_y,best_error_z]

    reward_pre_lift = total_reward 

    return reward_pre_lift, orientation_error , best_ori, error_x ,error_y, error_z, orientation_error_norm
