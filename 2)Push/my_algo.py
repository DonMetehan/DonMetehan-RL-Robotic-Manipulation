#my_algo.py

import numpy as np
import robosuite as suite
from utils import Sample
from my_cem import CEM
import imageio
import os
import matplotlib.pyplot as plt
import time
import logging
import warnings

# RoboSuite ve tüm logları tamamen kapat
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("robosuite").setLevel(logging.ERROR)
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
logging.getLogger("robosuite.controllers").setLevel(logging.ERROR)

# Genel INFO ve WARNING mesajlarını kapat
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
# Tüm uyarıları (WARNING) tamamen kapat
warnings.simplefilter("ignore")


start_time = time.time()

# ✅ Load last saved episode number and noise level
episode_log_path = "last_episode.txt"
default_noise_scale = 0.0 # Set your default noise level

if os.path.exists(episode_log_path):
    with open(episode_log_path, "r") as f:
        lines = f.readlines()

    start_episode = int(lines[0].strip()) + 1  # Always exists

    # Check if noise level exists, otherwise use default
    if len(lines) > 1:
        noise_scale_push = float(lines[1].strip())
    else:
        noise_scale_push = default_noise_scale
else:
    start_episode = 1
    noise_scale_push = default_noise_scale


print(f"Resuming training from Episode {start_episode}")


# ✅ Create environment
env = suite.make(
    env_name="Lift",  
    robots="Panda",   
    has_renderer=False,  
    has_offscreen_renderer=True,  
    use_camera_obs=False,
    control_freq=20,
    render_gpu_device_id=0
)



data_converge = np.load("model_converge.npz")
base_theta_1_converge = data_converge["theta_1_converge"]
base_theta_2_converge = data_converge["theta_2_converge"]
print("Converge parameters loaded from model_converge.npz")

# ✅ Define best_values file
best_values_file = "best_values.npz"
# ✅ Load previous best values if they exist
if os.path.exists(best_values_file):
    data_best = np.load(best_values_file)

    # ✅ Eksik anahtarları kontrol et
    required_keys = [
        "best_reward", "best_reward_episode", "best_reward_rollout",
        "lowest_distance", "lowest_distance_episode", "lowest_distance_rollout",
        "total_reward_push_reward",  "total_reward_push_distance"
    ]

    missing_keys = [key for key in required_keys if key not in data_best]
    if missing_keys:
        raise ValueError(f"ERROR: Missing keys in best_values.npz: {missing_keys}")

    # ✅ En iyi ödüle sahip değerleri yükle
    best_reward_ever = data_best["best_reward"]
    best_reward_episode = data_best["best_reward_episode"]
    best_reward_rollout = data_best["best_reward_rollout"]


    # ✅ En düşük mesafeye sahip değerleri yükle
    lowest_distance_ever = data_best["lowest_distance"]
    lowest_distance_episode = data_best["lowest_distance_episode"]
    lowest_distance_rollout = data_best["lowest_distance_rollout"]

    # ✅ Theta değerleri SADECE REWARD bazlı güncellenecek
    base_theta_1_push = data_best["best_theta_1_reward"]  # 🔥 SADECE best_reward için değişecek
    base_theta_2_push = data_best["best_theta_2_reward"]  # 🔥 SADECE best_reward için değişecek

    # ✅ Total Reward Push değerlerini her metrik için ayrı al
    total_reward_push_reward = data_best["total_reward_push_reward"]
    total_reward_push_distance = data_best["total_reward_push_distance"]


    
    # ✅ Best Reward Data (En yüksek ödül bazlı path)
    best_reward_data = {
        'total_reward_push': data_best["total_reward_push_reward"],
        'final_distance': data_best["best_final_distance_reward"],
        'theta_push': {
            'theta_1': data_best["best_theta_1_reward"],
            'theta_2': data_best["best_theta_2_reward"]
        }
    }


    # ✅ Lowest Distance Data (En düşük mesafe bazlı path)
    lowest_distance_data = {
        'total_reward_push': data_best["total_reward_push_distance"],
        'final_distance': data_best["best_final_distance_distance"],
        'theta_push': {
            'theta_1': data_best["best_theta_1_distance"],
            'theta_2': data_best["best_theta_2_distance"]
        }
    }
    print("✅ Loaded previous best values from best_values.npz")

else:
    best_reward_ever = -float("inf")
    best_reward_episode = None
    best_reward_rollout = None


    lowest_distance_ever = float("inf")
    lowest_distance_episode = None
    lowest_distance_rollout = None
    
    best_reward_data = None
    lowest_distance_data = None
    print("No previous best values found. Initializing new ones.")
    
    if os.path.exists("model_push.npz"):
        data_push = np.load("model_push.npz")  # ✅ Load the file first
        base_theta_1_push = data_push["theta_1_push"]
        base_theta_2_push = data_push["theta_2_push"]
        print("Push Thetas loaded from model_push.npz")
    else:
        base_theta_1_push = np.random.randn(7, 3)
        base_theta_2_push = np.random.randn(7)
        print("No previous Thetas found. Initializing new ones.")

# ✅ Ensure best_theta is always defined before saving
best_theta_1_push = base_theta_1_push
best_theta_2_push = base_theta_2_push

T = 90
n_episodes = 1
n_rollouts = 1
noise_decay = 0.99
final_noise = 0.000

elite_percentage = 0.1
cem = CEM(elite_percentage=elite_percentage)


distance_plot_dir = "./Distances"
os.makedirs(distance_plot_dir, exist_ok=True)

video_dir = "./Videos"
os.makedirs(video_dir, exist_ok=True)

info_dir = "./Log"
os.makedirs(info_dir, exist_ok=True)

best_info_dir = "./Best Info"
os.makedirs(best_info_dir, exist_ok=True)

velocity_dir = "./Velocities"
os.makedirs(info_dir, exist_ok=True)

torque_dir = "./Torques"
os.makedirs(info_dir, exist_ok=True)


frame_dir = "./Frames"
os.makedirs(frame_dir, exist_ok=True)

# ✅ Track success rate across all episodes
total_success_count = 0
total_rollouts_count = 0

for episode in range(start_episode, start_episode + n_episodes):
    
    print(f"\nEpisode {episode}/{start_episode + n_episodes - 1}")
    print(f"Noise Level : {noise_scale_push}")
    paths=[]
    paths_push = []
    
    rollout_push_rewards = []
    rollout_gripper_to_bread = []
    rollout_distances_wall_bread = []
    
    
    info_path = os.path.join(info_dir, f"Episode {episode} Info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Episode {episode} Log:\n\n")

    episode_video_dir = os.path.join(video_dir, f"Episode {episode}")
    os.makedirs(episode_video_dir, exist_ok=True)
    
    distance_plot_episode_dir = os.path.join(distance_plot_dir, f"Episode {episode}")
    os.makedirs(distance_plot_episode_dir, exist_ok=True)
    
    velocity_plot_episode_dir = os.path.join(velocity_dir, f"Episode {episode}")
    os.makedirs(velocity_plot_episode_dir, exist_ok=True)
    
    torque_plot_episode_dir = os.path.join(torque_dir, f"Episode {episode}")
    os.makedirs(torque_plot_episode_dir, exist_ok=True)
    
    for rollout in range(n_rollouts):

        
        noisy_theta_1_push = base_theta_1_push + noise_scale_push * np.random.randn(7, 3)
        noisy_theta_2_push = base_theta_2_push + noise_scale_push * np.random.randn(7)
        
        path, path_push, save_flag = Sample(env, base_theta_1_converge, base_theta_2_converge, noisy_theta_1_push, noisy_theta_2_push, T, rollout)
        
        
        paths.append(path)
        paths_push.append(path_push)
        
        rollout_push_rewards.append(path_push['total_reward_push'])
        rollout_gripper_to_bread.append(path['distance_gripper_to_bread'])
        rollout_distances_wall_bread.append(path_push['bread_to_wall_distance'])
        

        best_updated = False

        # ✅ Check and update best reward
        if path_push['total_reward_push'] > best_reward_ever:
            best_reward_ever = path_push['total_reward_push']
            best_reward_episode = episode
            best_reward_rollout = rollout + 1
            best_theta_1_push = noisy_theta_1_push
            best_theta_2_push = noisy_theta_2_push
            best_reward_data = path_push.copy()  # ✅ This ensures the dictionary does not lose keys

            best_updated = True
            print("Highest reward achieved. Thetas and best reward saved.")

        

        # ✅ Check and update lowest distance
        if path_push['final_distance'] < lowest_distance_ever:
            lowest_distance_ever = path_push['final_distance']
            lowest_distance_episode = episode
            lowest_distance_rollout = rollout + 1
            lowest_distance_data = path_push.copy()  
            best_updated = True
            print("Lowest distance achieved.")

        if best_updated:
            np.savez("best_values.npz",
                     # 🔹 En iyi toplam ödül verileri (reward bazlı path)
                     best_reward=best_reward_ever, 
                     best_reward_episode=best_reward_episode, 
                     best_reward_rollout=best_reward_rollout,
                     total_reward_push_reward=best_reward_data["total_reward_push"],
                     best_theta_1_reward=best_reward_data["theta_push"]["theta_1"],
                     best_theta_2_reward=best_reward_data["theta_push"]["theta_2"],
                     best_final_distance_reward=best_reward_data["final_distance"],

                    

                     # 🔹 En iyi mesafe verileri (distance bazlı path)
                     lowest_distance=lowest_distance_ever, 
                     lowest_distance_episode=lowest_distance_episode,
                     lowest_distance_rollout=lowest_distance_rollout,
                     total_reward_push_distance=lowest_distance_data["total_reward_push"],
                     best_theta_1_distance=lowest_distance_data["theta_push"]["theta_1"],
                     best_theta_2_distance=lowest_distance_data["theta_push"]["theta_2"],
                     best_final_distance_distance=lowest_distance_data["final_distance"])
            
            print("✅ Updated best values saved.")

        print(f"Episode {episode} - Rollout {rollout + 1} Push Reward: {path_push['total_reward_push']}")
        
        if save_flag:
            
            for i, frame in enumerate(path['frames']):
                frame_path = os.path.join(frame_dir, f"push_after_{i}.png")
                imageio.imwrite(frame_path, np.flipud(frame))
            plt.figure()
            plt.plot(
                range(len(path_push['bread_to_wall_distance'])),
                path['distance_gripper_to_bread'],
                marker='x',
                color='blue',
                label='Gripper to Bread Distance'
            )
            plt.plot(
                range(len(path_push['bread_to_wall_distance'])),
                path_push['bread_to_wall_distance'],
                marker='o',
                color='orange',
                label='Bread to Wall Distance'
            )
            plt.xlabel("Time")
            plt.ylabel("Distance (meters)")
            plt.title(f"Push Distances")
            plt.legend()  # 🔹 Adds legend using the labels above
            plt.grid(True)
            plt.savefig(os.path.join(distance_plot_episode_dir, f"pushDistances.png"))
            plt.close()

            
            rollout_video_path = os.path.join(episode_video_dir, f"Rollout {rollout + 1}.mp4")
            video_writer = imageio.get_writer(rollout_video_path, fps=20)
            for frame in path['frames']:
                video_writer.append_data(np.flipud(frame))
            video_writer.close()
            # Velocities grafikleri
            if len(path["Velocity"]) > 0:
                velocities = np.array(path["Velocity"])  # Numpy array'e çevir
                plt.figure()
                for joint_idx in range(velocities.shape[1]):  # Her bir joint için
                    plt.plot(range(len(velocities[:, joint_idx])), velocities[:, joint_idx], marker='o', label=f"Joint {joint_idx + 1}")
                plt.xlabel("Time")
                plt.ylabel("Velocity")
                plt.title(f"Episode {episode} - push_Velocities")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(velocity_plot_episode_dir, f"Rollout {rollout + 1} Joint Velocities.png"))
                plt.close()

            # Torques grafikleri
            if len(path["Torques"]) > 0:
                torques = np.array(path["Torques"])  # Numpy array'e çevir
                plt.figure()
                for joint_idx in range(torques.shape[1]):  # Her bir joint için
                    plt.plot(range(len(torques[:, joint_idx])), torques[:, joint_idx], marker='x', label=f"Joint {joint_idx + 1}")
                plt.xlabel("Time")
                plt.ylabel("Torque")
                plt.title(f"Episode {episode} - push_Torques")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(torque_plot_episode_dir, f"Rollout {rollout + 1} Joint Torques.png"))
                plt.close()
            """
            plt.plot(range(1, n_rollouts + 1), rollout_push_rewards, label="Push Reward", marker='o')
            plt.xlabel("Rollout Number")
            plt.ylabel("Reward")
            plt.title(f"Episode {episode} - Rollout Rewards")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(reward_plot_dir, f"Episode {episode} Rewards.png"))
            plt.close()
            """
        
        
        
    noise_scale_push = max(noise_scale_push * noise_decay, final_noise) 

    success_count = sum([p['is_successful_push'] for p in paths_push])
    success_rate = success_count / n_rollouts
    
    total_success_count += success_count
    total_rollouts_count += n_rollouts
    
    with open(info_path, 'a') as f:
        f.write("\nFinal Elite Information:\n")
        f.write("\nPush Phase:\n")
        elite_theta_1_push, elite_theta_2_push, elite_indices_push = cem.select_elite_thetas_push( paths_push)

        
        f.write(f"Standard Deviation of Rewards: {np.std(rollout_push_rewards):.2f}\n")
        f.write(f"Noise Scale: {noise_scale_push:.5f}\n")
        f.write(f"Success Rate: {success_rate:.2%}\n")
        f.write("---------------------------------------\n")
        f.write("Elite Rollouts for Push Phase:\n")
        for idx in elite_indices_push:
            elite_path = paths_push[idx]
            f.write(f"Rollout {idx + 1} - Total Reward: {elite_path['total_reward_push']}\n")
            f.write(f"Final Distance of Gripper to Target: {elite_path['final_distance']}\n")
            f.write(f"Theta_1:\n{elite_path['theta_push']['theta_1']}\n")
            f.write(f"Theta_2:\n{elite_path['theta_push']['theta_2']}\n\n")
            
    
    print(f"Episode {episode} bilgileri kaydedildi: {info_path}")

    base_theta_1_push, base_theta_2_push = elite_theta_1_push, elite_theta_2_push

    np.savez("model_push.npz", theta_1_push = base_theta_1_push, theta_2_push = base_theta_2_push)
    # Update last saved episode number
    # Save the last episode number and noise level
    with open(episode_log_path, "w") as f:
        f.write(f"{episode}\n{noise_scale_push}")

elapsed_time = time.time() - start_time
elapsed_hours = int(elapsed_time // 3600)
elapsed_minutes = int((elapsed_time % 3600) // 60)
elapsed_seconds = int(elapsed_time % 60)

print(f"\nTotal Training Time: {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s")
overall_success_rate = total_success_count / max(1, total_rollouts_count)

best_log_path = os.path.join(best_info_dir, "Best_Results.txt")

with open(best_log_path, 'w') as f:
    f.write("🔹 Best Results Across All Episodes 🔹\n\n")
    f.write(f" Overall Success Rate: {overall_success_rate:.2%}\n\n")  

    # ✅ Best Reward (En iyi ödül bazlı path)
    if best_reward_data:
        f.write(f"🏆 Best Total Reward Ever: {best_reward_ever}\n")
        f.write(f"   - Episode: {best_reward_episode}\n")
        f.write(f"   - Rollout: {best_reward_rollout}\n")
        f.write(f"   - Total Reward (By Reward): {best_reward_data['total_reward_push']}\n")
        f.write(f"   - Final Distance: {best_reward_data['final_distance']}\n")
        f.write(f"   - Theta_1:\n{best_reward_data['theta_push']['theta_1']}\n")
        f.write(f"   - Theta_2:\n{best_reward_data['theta_push']['theta_2']}\n\n")

    # ✅ Lowest Distance (En düşük mesafe bazlı path)
    if lowest_distance_data:
        f.write(f"📏 Lowest Final Distance Ever: {lowest_distance_ever}\n")
        f.write(f"   - Episode: {lowest_distance_episode}\n")
        f.write(f"   - Rollout: {lowest_distance_rollout}\n")
        f.write(f"   - Total Reward (By Distance): {lowest_distance_data['total_reward_push']}\n")
        f.write(f"   - Theta_1:\n{lowest_distance_data['theta_push']['theta_1']}\n")
        f.write(f"   - Theta_2:\n{lowest_distance_data['theta_push']['theta_2']}\n\n")

print(f"\n✅ Best results saved to: {best_log_path}")


    
print("\n🔹 FINAL SUMMARY OF TRAINING 🔹")
print("--------------------------------------------------")
print(f"🏆 Best Total Reward Ever: {best_reward_ever} (Episode {best_reward_episode}, Rollout {best_reward_rollout})")

print(f"📏 Lowest Final Distance Ever: {lowest_distance_ever} (Episode {lowest_distance_episode}, Rollout {lowest_distance_rollout})")
print("--------------------------------------------------")
print(f"✅ Overall Success Rate: {overall_success_rate:.2%}")
print(f"🕒 Total Training Time: {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s")
print(f"📁 Best results saved to: {best_log_path}")
print("--------------------------------------------------")
print("\nTraining completed.")

env.close()

