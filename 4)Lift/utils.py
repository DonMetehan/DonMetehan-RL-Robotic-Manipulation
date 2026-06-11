#utils.py

import numpy as np
from rewards import compute_reward_lift
import math
from policy import LinearPolicy
import os

def Sample(env, noisy_theta_1_converge, noisy_theta_2_converge, noisy_theta_1_push, noisy_theta_2_push,noisy_theta_1_lift,noisy_theta_2_lift,rollout_num, T):
    print("----------------------------------------------------")   
    plot = True
    obs = env.reset()
    
    s = obs['robot0_eef_pos'] 
    push_target = (0.274,0,0.822)
    converge_target = (-0.04, 0, 0.823)
    lift_target = (0.274,0,0.850)
    
    policy_converge = LinearPolicy(theta_1=noisy_theta_1_converge, theta_2=noisy_theta_2_converge)    
    policy_push = LinearPolicy(theta_1=noisy_theta_1_push, theta_2=noisy_theta_2_push)
    policy_lift= LinearPolicy(theta_1=noisy_theta_1_lift, theta_2=noisy_theta_2_lift)
    
    S_lift, A_lift  = [], []
    bread_to_lift_target = []
    torques, velocities = [], []
    frames = []
    
    total_reward_lift  = 0

    is_successful_converge = False
    is_successful_push = False
    is_successful_lift = False
    
    lift_phase_inf = False
    save_flag = False
    # Initialize distance tracking only once at the start of the lift phase

    distance_thresholds = {
        0.025: False, 0.024: False, 0.023: False,
        0.022: False, 0.021: False, 0.020: False, 0.019: False,
        0.018: False, 0.017: False, 0.016: False, 0.015: False,
        0.014: False, 0.013: False, 0.012: False, 0.011: False, 0.010: False,
        0.009: False, 0.008: False, 0.007: False, 0.006: False, 0.005: False,
        0.004: False, 0.003: False
    }
    reward_values = {
        0.025: 100, 0.024: 200, 0.023: 300,
        0.022: 400, 0.021: 500, 0.020: 600, 0.019: 700,
        0.018: 800, 0.017: 900, 0.016: 1000, 0.015: 1100,
        0.014: 1200, 0.013: 1300, 0.012: 1400, 0.011: 1500, 
        0.010: 1600,0.009: 1700, 0.008: 1800,0.007: 1900, 
        0.006: 2000, 0.005: 2100,0.004: 2200, 0.003: 2300
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
            joint_torques= action_converge[:-1]
            joint_velocities = next_obs['robot0_joint_vel']

            torques.append(joint_torques)
            velocities.append(joint_velocities)
            
            bread_to_lift_target.append(math.dist(lift_target, bread_pos))          
            
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
           
            joint_torques= action_push[:-1]
            joint_velocities = next_obs_push['robot0_joint_vel']
            
            torques.append(joint_torques)
            velocities.append(joint_velocities)
            current_distance = math.dist(push_target, bread_pos)
            bread_to_lift_target.append(math.dist(lift_target, bread_pos))
                          
            if current_distance < 0.004:
                is_contact_bread_with_wall = (env.check_contact("bread_g0", "wall_collision"))
            if current_distance < 0.004:    
                if is_contact_bread_with_wall and current_distance < 0.002:
                    is_successful_push = True
                    print("Push Phase is done.")
                
            s = next_s
        
        previous_bread = bread_pos
        previous_distance_lift = math.dist(bread_pos, lift_target)
        previous_gripper_to_bread = math.dist(bread_pos, s)

        # Lift phase
        if is_successful_converge and is_successful_push and not is_successful_lift:
            if not lift_phase_inf:
                print("Lift Phase is Starting!")
                lift_phase_inf = True
                        
            is_contact_gripper_with_bread= (
                env.check_contact("gripper0_finger1_collision", "bread_g0") or
                env.check_contact("gripper0_finger2_collision", "bread_g0")
            )
            
            
            is_contact_bread_with_wall = (env.check_contact("bread_g0", "wall_collision"))
            action_lift = policy_lift.act(s)
            next_obs_lift, _, _, _ = env.step(action_lift)

            next_s = next_obs_lift['robot0_eef_pos']
            bread_pos = next_obs_lift['bread_pos']
            joint_torques = action_lift[:-1]
            joint_velocities = next_obs_lift['robot0_joint_vel']
            
            current_distance_lift = math.dist(bread_pos, lift_target)
            current_gripper_to_bread = math.dist(next_s, bread_pos)
            torques.append(joint_torques)
            velocities.append(joint_velocities)
            bread_to_lift_target.append(current_distance_lift)
            
            reward_lift = compute_reward_lift(current_distance_lift,current_gripper_to_bread,is_contact_gripper_with_bread)
            total_reward_lift += reward_lift
            
            if current_distance_lift < previous_distance_lift:
                total_reward_lift += 50            
            else :
                total_reward_lift -= 10
                
            if bread_pos[2] > previous_bread[2]:
                total_reward_lift += 50            
            else :
                total_reward_lift -= 10
            
            
            if current_distance_lift < previous_distance_lift: 
                for threshold in sorted(distance_thresholds.keys(), reverse=True):
                    if current_distance_lift <= threshold and not distance_thresholds[threshold]:
                        print(f"Bread reached {threshold}m up to table for the first time! Reward applied.")
                        total_reward_lift += reward_values[threshold]
                        distance_thresholds[threshold] = True  
                        
                        
            if current_distance_lift < 0.002:
                total_reward_lift += 5000
                is_successful_lift = True
                print("Lift Phase Completed!")
                print(f"Bread position : {bread_pos}")
                print(f"Current Distance : {current_distance_lift}")
                break

            previous_gripper_to_bread = current_gripper_to_bread
            previous_bread = bread_pos
            previous_distance_lift = current_distance_lift
            s = next_s
            
 
    
    path = {
        'frames': frames,
        'Velocity' : velocities,
        'Torques' : torques
    }
    
    path_lift = {
        'total_reward_lift': total_reward_lift,
        'bread_to_lift_target' : bread_to_lift_target,
        'is_successful_lift' : is_successful_lift,
        'theta_lift': {'theta_1': noisy_theta_1_lift, 'theta_2': noisy_theta_2_lift},
        'final_distance' : current_distance_lift
    }    
    
    return path, path_lift , plot
    
