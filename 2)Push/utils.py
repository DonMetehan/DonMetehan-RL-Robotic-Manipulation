#utils.py

import numpy as np
from rewards import compute_reward_push
from rewards import quaternion_orientation_error
import math
from policy import LinearPolicy
import os



def Sample(env, theta_1_converge, theta_2_converge, noisy_theta_1_push, noisy_theta_2_push, T, rollout_num):

    
    print("----------------------------------------------------")
    plot = True
    obs = env.reset()
    s = obs['robot0_eef_pos'] 
    bread_position = obs['bread_pos']
    gripper_quat = obs['robot0_eef_quat']
    print("Gripper initial Quaternion", gripper_quat)

    push_target = (0.274,0,0.822)
    converge_target = (-0.04, 0, 0.823)
    desired_orientation = np.array([0.7506,0.0002,-0.6607,-0.0002])
    
    policy_converge = LinearPolicy(theta_1=theta_1_converge, theta_2=theta_2_converge)  
    policy_push = LinearPolicy(theta_1=noisy_theta_1_push, theta_2=noisy_theta_2_push)

    
    distance_gripper_to_bread  = []
    torques, velocities = [], []
    frames = [] if plot else None  # Only store frames when needed
    S_push, A_push  = [], []
    bread_to_wall_distance = []
    distance_wall_bread_list = []

    total_reward_push  = 0
    is_successful_converge = False
    is_successful_push = False
    push_phase_inf = False
    save_flag = False
    
    distance_thresholds = {
        0.25: False, 0.20: False, 0.17: False, 0.14: False, 0.12: False, 0.10: False,
        0.09: False, 0.08: False, 0.07: False, 0.06: False, 0.05: False, 0.04: False,
        0.03: False, 0.025: False, 0.020: False, 0.015: False, 0.010: False,
        0.009: False, 0.008: False, 0.007: False, 0.006: False, 0.005: False, 0.004: False, 0.003: False
    }

    reward_values = {
        0.25: 50, 0.20: 75, 0.17: 100, 0.14: 1500, 0.12: 2000, 0.10: 2500, 
        0.09: 3000, 0.08: 4000, 0.07: 5000, 0.06: 7500, 0.05: 10000, 0.04: 15000,
        0.03: 20000, 0.025: 25000, 0.020: 30000, 0.015: 35000, 0.010: 40000, 
        0.009: 45000, 0.008: 50000, 0.007: 55000, 0.006: 60000, 0.005: 65000, 0.004: 75000, 0.003: 85000
    }
    
    orientation_thresholds = {
        2.00: False, 1.95: False, 1.90: False, 1.85: False, 1.80: False, 1.75: False, 
        1.70: False, 1.65: False, 1.60: False, 1.55: False, 1.50: False, 1.45: False, 
        1.40: False, 1.35: False, 1.30: False, 1.25: False, 1.20: False, 1.15: False, 
        1.10: False, 1.05: False, 1.00: False, 0.95: False, 0.90: False, 0.85: False, 
        0.80: False, 0.75: False, 0.70: False, 0.65: False, 0.60: False, 0.55: False, 
        0.50: False, 0.45: False, 0.40: False, 0.35: False, 0.30: False, 0.25: False, 
        0.20: False, 0.17: False, 0.14: False, 0.12: False, 0.10: False, 
        0.09: False, 0.08: False, 0.07: False, 0.06: False, 0.05: False, 
        0.04: False, 0.03: False, 0.025: False, 0.020: False, 0.015: False, 
        0.010: False, 0.009: False, 0.008: False, 0.007: False, 0.006: False, 
        0.005: False, 0.004: False, 0.003: False
    }


    orientation_reward_values = {
        2.00: 500, 1.95: 750, 1.90: 1000, 1.85: 1500, 1.80: 2000, 1.75: 2500, 
        1.70: 3000, 1.65: 3500, 1.60: 4000, 1.55: 4500, 1.50: 5000, 1.45: 6000, 
        1.40: 7000, 1.35: 8000, 1.30: 9000, 1.25: 10000, 1.20: 11000, 1.15: 12000, 
        1.10: 13000, 1.05: 14000, 1.00: 15000, 0.95: 16000, 0.90: 17000, 0.85: 18000, 
        0.80: 19000, 0.75: 20000, 0.70: 22500, 0.65: 25000, 0.60: 27500, 0.55: 30000, 
        0.50: 35000, 0.45: 40000, 0.40: 45000, 0.35: 50000, 0.30: 55000, 0.25: 60000, 
        0.20: 70000, 0.17: 80000, 0.14: 90000, 0.12: 100000, 0.10: 120000, 
        0.09: 135000, 0.08: 150000, 0.07: 170000, 0.06: 200000, 0.05: 250000, 
        0.04: 300000, 0.03: 350000, 0.025: 400000, 0.020: 450000, 0.015: 500000, 
        0.010: 550000, 0.009: 600000, 0.008: 650000, 0.007: 700000, 0.006: 750000, 
        0.005: 800000, 0.004: 900000, 0.003: 1000000
    }
    


    
    for t in range(T):
    

        # ✅ Always append frames when `plot=True`, capturing every timestep
        if plot:

            try:
                frame = env.sim.render(camera_name="sideview", width=1024, height=1024)
                frames.append(frame)
            except Exception as e:
                print(f"[ERROR] Failed to capture frame at timestep {t}: {e}")

            
        if not is_successful_converge:
            action_converge = policy_converge.act(s)
            next_obs, _, _, _ = env.step(action_converge)
            
            next_s = next_obs['robot0_eef_pos']
            joint_torques= action_converge[:-1]
            joint_velocities = next_obs['robot0_joint_vel']
        
            
            torques.append(joint_torques)
            velocities.append(joint_velocities)
            distance_gripper_to_bread.append(math.dist(next_s,bread_position))
            current_distance = math.dist(next_s, converge_target)  
            bread_to_wall_distance.append(math.dist(bread_position, push_target))
            
            if current_distance < 0.004:
                is_successful_converge = True
                bread_pos = next_obs['bread_pos']
            s = next_s
  
        # Push phase
        if is_successful_converge and not is_successful_push:
            if not push_phase_inf:
                print("Push Phase is Starting!")
                previous_gripper_orientation = next_obs['robot0_eef_quat']
                previous_distance = math.dist(bread_position, push_target)
                previous_gripper_bread_distance = math.dist(s,bread_pos)
                previous_orientation_error = quaternion_orientation_error(previous_gripper_orientation,desired_orientation)
                
                desired_gripper_z = bread_pos[2] - 0.05 
                previous_position_error = next_s[2] - desired_gripper_z

                push_phase_inf=True
                
            action_push = policy_push.act(s)
            next_obs_push, _, _, _ = env.step(action_push)
            
            is_contact_gripper_with_bread = (
                env.check_contact("gripper0_finger1_collision", "bread_g0") or
                env.check_contact("gripper0_finger2_collision", "bread_g0") 
            )         
  
            next_s = next_obs_push['robot0_eef_pos']
            bread_pos = next_obs_push['bread_pos']
            joint_torques= action_push[:-1]
            joint_velocities = next_obs_push['robot0_joint_vel']
            gripper_orientation = next_obs_push['robot0_eef_quat']
            
            current_distance = math.dist(bread_pos, push_target)
            current_gripper_bread_distance = math.dist(next_s,bread_pos)
            
            reward_push,orientation_error,position_error = compute_reward_push(current_gripper_bread_distance, current_distance, is_contact_gripper_with_bread,gripper_orientation,desired_orientation,next_s,bread_pos,distance_thresholds[0.006])
            
            total_reward_push += reward_push
                   
            if previous_distance > current_distance:
                total_reward_push += 40
            else :
                total_reward_push -= 5
                
            if previous_gripper_bread_distance > current_gripper_bread_distance:
                total_reward_push += 4
            else :
                total_reward_push -= 2
        
            distance_gripper_to_bread.append(current_gripper_bread_distance)
            torques.append(joint_torques)
            velocities.append(joint_velocities)    
            S_push.append(next_s)
            A_push.append(action_push)
            bread_to_wall_distance.append(current_distance)
            
            if orientation_error < previous_orientation_error:
                total_reward_push += 100
            else:
                total_reward_push -= 100 * (orientation_error - previous_orientation_error)

            if position_error < previous_position_error:
                total_reward_push += 100       
            else:
                total_reward_push -= 50
            
            if orientation_error < previous_orientation_error:
                for threshold_orientation in sorted(orientation_thresholds.keys(), reverse=True):
                    if orientation_error <= threshold_orientation and not orientation_thresholds[threshold_orientation]:
                        print(f"Gripper reached {threshold_orientation} orientation error for the first time! Reward applied.")
                        total_reward_push += orientation_reward_values[threshold_orientation] 
                        orientation_thresholds[threshold_orientation] = True 

            if previous_distance > current_distance:
                for threshold in sorted(distance_thresholds.keys(), reverse=True):
                    if current_distance <= threshold and not distance_thresholds[threshold]:
                        print(f"Breaed reached {threshold}m for the first time! Reward applied.")
                        total_reward_push += reward_values[threshold] 
                        distance_thresholds[threshold] = True 

            if distance_thresholds[0.003]:      
                
                is_contact_bread_with_wall = (
                env.check_contact("bread_g0", "wall_collision"))
                
            if distance_thresholds[0.003]: 
                if is_contact_bread_with_wall and current_distance < 0.002:
                    print(f"Bread to Push Target Distance : {current_distance}")
                    print(f"Gripper Position : {next_s}!")
                    print(f"Orientation Error : {orientation_error}")
                    print(f"Position Error : {position_error}")
                    total_reward_push += 20000    
                    if position_error < 0.06:
                        total_reward_push += 10000
                        print("OMG!Gripper is also VERY CLOSE to the table")  
                    else:
                        total_reward_push -= 100 * position_error
                    is_successful_push = True
                    print("PERFECT!Push Phase is done")
                    gripper_orientation_push = next_obs_push['robot0_eef_quat']
                    print("Push Last Gripper Orientation",gripper_orientation_push)
                    break
                    
            if t== T-1:
                    print(f"Gripper Position : {next_s}!")
                    print(f"Bread Position : {bread_pos}")
                    print(f"Push Target : {push_target}")
                    print(f"Orientation Error : {orientation_error}")
                    print(f"Position Error : {position_error}")
            s = next_s
            previous_distance = current_distance  
            previous_gripper_bread_distance = current_gripper_bread_distance
            previous_orientation_error = orientation_error
            previous_position_error = position_error

    total_reward_push = (total_reward_push / 100)
    path = {
        'frames': frames,
        'distance_gripper_to_bread': distance_gripper_to_bread,
        'Velocity': velocities,
        'Torques': torques
    }

    path_push = {
        'observations': np.array(S_push),
        'actions': np.array(A_push),
        "is_successful_push" :is_successful_push,
        'total_reward_push': total_reward_push,
        'bread_to_wall_distance' : bread_to_wall_distance,
        'theta_push': {'theta_1': noisy_theta_1_push, 'theta_2': noisy_theta_2_push},
        'final_distance': current_distance 
    }    
    
    return path, path_push , plot
