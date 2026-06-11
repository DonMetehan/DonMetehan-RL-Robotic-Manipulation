import numpy as np
from rewards import compute_reward_converge
import math
from policy import LinearPolicy
import os


def Sample(env, noisy_theta_1_converge, noisy_theta_2_converge, T, rollout_num, plot=True):
    print("----------------------------------------------------")
    obs = env.reset()
    plot = True
    s = obs['robot0_eef_pos'] 
    converge_target = (-0.04, 0, 0.823)
    
    policy_converge = LinearPolicy(theta_1=noisy_theta_1_converge, theta_2=noisy_theta_2_converge)

    S_converge, A_converge, distances_converge  = [], [], []
    torques, velocities = [], []
    total_reward_converge  = 0
    frames = []
    is_successful_converge = False
    save_flag = False
    
    if 'distance_thresholds' not in locals():
        distance_thresholds = {0.20: False, 0.15: False, 0.10: False, 0.05: False, 0.025: False, 0.010: False,0.009: False,0.008: False,0.007: False,0.006: False,0.005: False,0.004: False}
    
    previous_distance = math.dist(s, converge_target)
    
    for t in range(T):
        
        frames.append(env.sim.render(camera_name="sideview", width=1024, height=1024))
        
        is_contact_gripper_with_table = (
            env.check_contact("gripper0_finger1_collision", "table_collision") or
            env.check_contact("gripper0_finger2_collision", "table_collision") or
            env.check_contact("gripper0_finger1_pad_collision", "table_collision") or
            env.check_contact("gripper0_finger2_pad_collision", "table_collision")
        )
        is_contact_gripper_with_bread = (
            env.check_contact("gripper0_finger1_collision", "bread_g0") or
            env.check_contact("gripper0_finger2_collision", "bread_g0") or
            env.check_contact("gripper0_finger1_pad_collision", "bread_g0") or
            env.check_contact("gripper0_finger2_pad_collision", "bread_g0")
        )

        action_converge = policy_converge.act(s)
        next_obs, _, _, _ = env.step(action_converge)
        
        next_s = next_obs['robot0_eef_pos']
        joint_torques= action_converge[:-1]
        joint_velocities = next_obs['robot0_joint_vel']
        
        torques.append(joint_torques)
        velocities.append(joint_velocities)
        current_distance = math.dist(next_s, converge_target)

        reward_converge = compute_reward_converge(current_distance)
        
        total_reward_converge += reward_converge
        improvement = previous_distance - current_distance
        
        if improvement > 0:
            # Reward for reaching distance thresholds
            for threshold in sorted(distance_thresholds.keys(), reverse=True):
                if current_distance <= threshold and not distance_thresholds[threshold]:
                    print(f"Gripper reached {threshold}m for the first time! Reward applied.")
                    reward_values = {0.20: 50, 0.15: 100, 0.10: 250, 0.05: 500, 0.025: 1000, 0.010: 1500, 0.009: 2000, 0.008: 3000, 0.007: 4000, 0.006: 5000,0.005: 6000,0.004: 7000}
                    total_reward_converge += reward_values[threshold]  
                    distance_thresholds[threshold] = True 
                
        

        if improvement > 0.001: 
            total_reward_converge += 1000 * improvement  
        elif improvement > 0.0005:
            total_reward_converge += 500 * improvement 
        elif improvement == 0:
            total_reward_converge -= 1
        else :
            total_reward_converge -= 1000 * improvement
        
        S_converge.append(next_s)
        A_converge.append(action_converge)
        distances_converge.append(current_distance)
        
        if is_contact_gripper_with_table:
            print("Gripper touched to the table")
            print(f"Gripper Position  {next_s}!")
            print(f"Distance to Converge Target is  {current_distance}!") 
            total_reward_converge -=1000
            break
        if is_contact_gripper_with_bread:
            print("Gripper touched to the bread")
            print(f"Gripper Position  {next_s}!")
            print(f"Distance to Converge Target is  {current_distance}!")
            total_reward_converge -=1000
            break         

        if current_distance < 0.005:
            print(f"Gripper Position  {next_s}!")
            print(f"Distance to Converge Target is  {current_distance}!")
            print("PERFECT : Gripper is on target!, Converge Phase is Completed")
            total_reward_converge += 5000
            is_successful_converge = True
            break
        if t==T-1:           
            print(f"Gripper Position  {next_s}!")
            print(f"Distance to Converge Target is  {current_distance}!")
            print(f"Converge Target is  {converge_target}!")

                    
        s = next_s
        previous_distance = current_distance
        

    path = {
        'frames': frames,
        'distances': distances_converge,
        'Velocity': velocities,
        'Torques': torques
    }
    path_converge = {
        'observations': np.array(S_converge),
        'actions': np.array(A_converge),
        'total_reward_converge': total_reward_converge,
        'is_successful_converge': is_successful_converge,
        'theta_converge': {'theta_1': noisy_theta_1_converge, 'theta_2': noisy_theta_2_converge},
        'final_distance': current_distance 
    }

    
    return path, path_converge, plot
    


