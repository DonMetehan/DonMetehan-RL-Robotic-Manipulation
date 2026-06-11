#my_cem.py
import numpy as np

class CEM:
    def __init__(self, elite_percentage):
        self.elite_percentage = elite_percentage

    def select_elite_thetas(self, paths, reward_key, theta_key):
        
        paths_sorted = sorted(enumerate(paths), key=lambda x: x[1][reward_key], reverse=True)

        elite_count = max(1, int(len(paths_sorted) * self.elite_percentage))
        elite_indices = [x[0] for x in paths_sorted[:elite_count]]

        elite_theta_1s = [paths[i][theta_key]['theta_1'] for i in elite_indices]
        elite_theta_2s = [paths[i][theta_key]['theta_2'] for i in elite_indices]

        elite_theta_1_mean = np.mean(elite_theta_1s, axis=0)
        elite_theta_2_mean = np.mean(elite_theta_2s, axis=0)

        print(f"Selected elite rollout indices: {[i + 1 for i in elite_indices]} based on {reward_key}")
        return elite_theta_1_mean, elite_theta_2_mean, elite_indices

    def select_elite_thetas_converge(self, paths):
        
        return self.select_elite_thetas(
            paths=paths,
            reward_key='total_reward_converge',
            theta_key='theta_converge'
        )
        
    def select_elite_thetas_push(self, paths):
        
        return self.select_elite_thetas(
            paths=paths,
            reward_key='total_reward_push',
            theta_key='theta_push'
        )
