#utils.py

import numpy as np
from rewards import compute_reward_pre_lift
from rewards import quaternion_orientation_error
import math
from policy import LinearPolicy
import os

def Sample(env, noisy_theta_1_converge, noisy_theta_2_converge, noisy_theta_1_push, noisy_theta_2_push,noisy_theta_1_pre_lift,noisy_theta_2_pre_lift,rollout_num, T):
    print("----------------------------------------------------")   
    plot = (rollout_num + 1) % 10 == 0
    plot = True
    obs = env.reset()
    
    s = obs['robot0_eef_pos']


    obs = env.reset()
    gripper_quat = obs['robot0_eef_quat']

    alpha=1e-5
    push_target = (0.274,0,0.822)
    converge_target = (-0.04, 0, 0.823)
    desired_orientation = np.array([0.701,0.092,0.701,-0.092])
    
    policy_converge = LinearPolicy(theta_1=noisy_theta_1_converge, theta_2=noisy_theta_2_converge)    
    policy_push = LinearPolicy(theta_1=noisy_theta_1_push, theta_2=noisy_theta_2_push)
    policy_pre_lift= LinearPolicy(theta_1=noisy_theta_1_pre_lift, theta_2=noisy_theta_2_pre_lift)
    
    S_pre_lift, A_pre_lift  = [], []
    torques, velocities = [], []
    frames = [] if plot else None

    
    total_reward_pre_lift  = 0

    is_successful_converge = False
    is_successful_push = False
    is_successful_pre_lift = False
    
    pre_lift_phase_inf = False

    orientation_thresholds = {
        (round(threshold, 2), axis): False
        for threshold in np.arange(0.40, 0.01, -0.01)  
        for axis in range(3)  
    }

    orientation_reward_values = {
        round(threshold, 2): int(5000 * (0.40 - threshold))  # Reward increases as error decreases
        for threshold in np.arange(0.40, 0.01, -0.01)
    }
    
    for t in range(T):
        
        if plot:
            frames.append(env.sim.render(camera_name="sideview", width=512, height=512))
        
        # Converge phase
        if not is_successful_converge:
            action_converge = policy_converge.act(s)
            next_obs, _, _, _ = env.step(action_converge)
            
            next_s = next_obs['robot0_eef_pos']
            bread_pos = next_obs['bread_pos']
            
            
            current_distance = math.dist(next_s, converge_target)
            if current_distance < 0.004:
                is_successful_converge = True
            s = next_s
        
        
        # Push phase
        if is_successful_converge and not is_successful_push:
                           
            action_push = policy_push.act(s)
            next_obs_push, _, _, _ = env.step(action_push)
            
            next_s = next_obs_push['robot0_eef_pos']
            
            bread_pos = next_obs_push['bread_pos']
           
            current_distance = math.dist(push_target, bread_pos)
                          
            if current_distance < 0.004:
                is_contact_bread_with_wall = (env.check_contact("bread_g0", "wall_collision"))
            if current_distance < 0.004:    
                if is_contact_bread_with_wall and current_distance < 0.002:
                    is_successful_push = True
                    print("Push Phase is done.")
                
            s = next_s

        # pre_lift phase
        if is_successful_converge and is_successful_push and not is_successful_pre_lift:
            if not pre_lift_phase_inf:
                print("pre_lift Phase is Starting!")
                pre_lift_phase_inf = True
                previous_gripper_orientation = next_obs_push['robot0_eef_quat']
                previous_orientation_error = quaternion_orientation_error(previous_gripper_orientation,desired_orientation)
                best_orientation_error_x = abs(previous_orientation_error[0])
                best_orientation_error_y = abs(previous_orientation_error[1])
                best_orientation_error_z = abs(previous_orientation_error[2])
                best_ori = [best_orientation_error_x,best_orientation_error_y,best_orientation_error_z]
                print("Pre Lift Initial Error",previous_orientation_error)
            
            action_pre_lift = policy_pre_lift.act(s)
            next_obs_pre_lift, _, _, _ = env.step(action_pre_lift)
            
            next_s = next_obs_pre_lift['robot0_eef_pos']

            gripper_orientation = next_obs_pre_lift['robot0_eef_quat']

            
            reward_pre_lift, current_orientation_error, best_ori, error_x ,error_y,error_z, orientation_error_norm= compute_reward_pre_lift(gripper_orientation,desired_orientation, best_ori)
            total_reward_pre_lift += reward_pre_lift               


            if  error_x < 0.067 and error_y < 0.356 and error_z  < 0.021:

                is_successful_pre_lift = True
                print("pre_lift Phase Completed!")
                print("Gripper Orientation Error", current_orientation_error)
                print("Error Norm",orientation_error_norm)
                print("EEF Position",next_s)
                break
                
            if t == T-1:
                print("Gripper Orientation Error", current_orientation_error)
                print("Error Norm",orientation_error_norm)
                print("EEF Position",next_s)
                
            s = next_s
            
    total_reward_pre_lift /= 100
    path = {
        'frames': frames,
    }
    
    path_pre_lift = {
        'total_reward_pre_lift': total_reward_pre_lift,
        'is_successful_pre_lift' : is_successful_pre_lift,
        'theta_pre_lift': {'theta_1': noisy_theta_1_pre_lift, 'theta_2': noisy_theta_2_pre_lift}
    }    
    
    return path, path_pre_lift , True
    
