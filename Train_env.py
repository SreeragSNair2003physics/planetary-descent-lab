from stable_baselines3 import PPO                                      # Imports the PPO reinforcement-learning algorithm
from stable_baselines3.common.monitor import Monitor                   # Imports the wrapper that tracks episode scores and lengths
from Planetary_environment import PlanetaryLanderEnv                   # Imports our custom planetary landing environment

SCREEN_WIDTH = 1920                                                    # Sets the simulation width in pixels
SCREEN_HEIGHT = 1080                                                   # Sets the simulation height in pixels

world_weights={"Earth": 1/3, "Mars": 1/3, "Moon": 1/3}                 # Gives each planet an equal selection probability at reset

env = Monitor(PlanetaryLanderEnv(randomize_resets=True,screen_width=SCREEN_WIDTH,screen_height=SCREEN_HEIGHT, world_weights=world_weights,))  
# Creates the monitored training environment with randomized resets


model = PPO.load("planetary_lander_recovery_impact_plus200k", env=env) # Loads the existing PPO weights and attaches the training environment
model.learn(total_timesteps=100_000, reset_num_timesteps=False)        # Adds 100000 training steps while preserving the cumulative step counter
model.save("planetary_lander_recovery_impact_plus300k")                # Saves the resulting PPO checkpoint, overwrites an existing file with this name if there is one
env.close()                                                            # Closes the training environment