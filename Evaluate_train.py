
from Planetary_environment import PlanetaryLanderEnv                                        # Imports our custom landing environment


env = PlanetaryLanderEnv(randomize_resets=True)                                             # Creates an environment with randomized planet and starting-state selection

pad_landings = 0                      # Starts the safe pad-landing count at zero
emergency_landings = 0                # Starts the safe terrain-landing count at zero
crashes = 0                           # Starts the crash count at zero
timeouts = 0                          # Starts the timeout count at zero
 
for episode in range(30):                                                                   # Runs thirty evaluation attempts
    observation, info = env.reset(seed=1000 + episode)                                      # Resets with a reproducible seed for this attempt
    terminated = False                                                                      # Starts with no ground-contact termination
    truncated = False                                                                       # Starts with no time-limit termination
    steps = 0                                                                               # Starts this attempt's frame counter at zero

    while not (terminated or truncated):                                                    # Continues flying until contact or the time limit ends the attempt
        action = env.backup_action()                                                        # Gets an action from guidance system, not the PPO 
        observation, reward, terminated, truncated, info = env.step(action)                 # Applies the action and receives the new state
        steps += 1                                                                          # Counts this frame

    pad_landings += int(info["pad_landing"])                                                # Adds one when this attempt ended with a safe pad landing
    emergency_landings += int(info["emergency_landing"])                                    # Adds one when this attempt ended with a safe terrain landing
    crashes += int(info["crashed"])                                                         # Adds one when this attempt ended in a crash
    timeouts += int(truncated)                                                              # Adds one when this attempt reached the time limit

    if info["pad_landing"]:                                                                 # Checks whether the outcome was a safe pad landing
        result = "PAD"
    elif info["emergency_landing"]:                                                         # Checks whether the outcome was a safe terrain landing
        result = "EMERGENCY TERRAIN"
    elif info["crashed"]:                                                                   # Checks whether the craft crashed
        result = "CRASH"
    else:                                                                                   # Handles attempts that ended without a landing or crash
        result = "TIME LIMIT"

    print(
        f"Episode {episode + 1}: world={env.current_world}, "
        f"result={result}, steps={steps}, "
        f"x={env.x:.2f}, altitude={observation[1]:.2f}, "
        f"throttle={observation[6]:.2f}, "                                                  # Prints the details of the completed attempt
        f"impact_vx={info['impact_vx']}, "
        f"impact_vy={info['impact_vy']}, "
        f"impact_angle={info['impact_angle']}"
    )

print(f"Pad landings: {pad_landings}/30")
print(f"Emergency terrain landings: {emergency_landings}/30")
print(f"Crashes: {crashes}/30")
print(f"Time limits: {timeouts}/30")
env.close()