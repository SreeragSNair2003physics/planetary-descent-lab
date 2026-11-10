# Planetary Descent Lab

A Python landing simulator for the Moon, Mars and Earth, with manual flight, a physics-guided autopilot and an experimental PPO controller.

The project began with PPO and reward-shaping experiments in Gymnasium's LunarLander. It developed into a custom environment with a fullscreen Pygame interface and a guidance controller that can take over during manual flight.

## Simulation previews

<img width="1899" height="1188" alt="Screenshot 2026-11-10 203127" src="https://github.com/user-attachments/assets/5ecd971f-e629-4a6a-af20-1134bfe833da" />
<img width="1907" height="1188" alt="Screenshot 2026-11-10 203239" src="https://github.com/user-attachments/assets/b1ab34c1-03d9-4cc8-9a2e-7116afdc9943" />
<img width="1897" height="1188" alt="Screenshot 2026-11-10 203416" src="https://github.com/user-attachments/assets/fae8c5e5-cd42-47a2-bfb8-0882a98a7b8d" />



## Features

- Three worlds with different gravity, atmospheric density and wind settings.
- Keyboard control of engine firing, throttle, side thrust and tilt.
- Guidance takeover while the spacecraft is airborne.
- Pad-first landing planning, with candidate terrain sites considered when a pad landing is not predicted to succeed.
- Telemetry for altitude, velocity, tilt, throttle and wind.
- Landing checks based on impact speeds, tilt and supported terrain, with crash animations.
- Shared environment dynamics for manual flight, guidance, PPO training and evaluation.

## Controllers

**Physics-guided autopilot:** uses the terrain map and simulation dynamics to forecast candidate landings and control the descent. The recorded demonstration uses this controller.

**PPO baseline:** a learned policy trained with Stable-Baselines3. It remains experimental; its performance should be evaluated separately from guidance.

In `Simulation_window.py`, `USE_GUIDANCE = True` selects guidance when autopilot is enabled. Setting it to `False` selects the saved PPO policy.

## Run the simulation

Developed on Windows with Python 3.13. Open Command Prompt in the repository folder and run:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python Simulation_window.py
```

Keep `planetary_lander_recovery_impact_plus200k.zip` in the repository folder: the simulation loads this checkpoint at startup.

## Controls

| Control | Action |
| --- | --- |
| `1` / `2` / `3` | Select Moon / Mars / Earth |
| `W` / `S` | Increase / decrease throttle |
| `Space` | Fire the main engine |
| Left / right arrows | Side thrust |
| `A` / `D` | Change tilt |
| Autopilot button | Switch between manual control and the selected controller |
| `R` | Reset the flight |
| `Esc` | Exit |

## Evaluate guidance

```bat
python Evaluate_train.py
```

The current evaluation runs 30 seeded episodes across randomly selected worlds. It counts pad landings, emergency terrain landings, crashes and time limits. World counts can differ because world selection is random.

Report success counts with the full number of evaluated episodes. Demonstration clips illustrate individual flights; aggregate reliability comes from evaluation runs.

## Main files

| File | Purpose |
| --- | --- |
| `Planetary_environment.py` | Simulation dynamics, landing checks, rewards and guidance |
| `Simulation_window.py` | Fullscreen interface and manual/autopilot controls |
| `Evaluate_train.py` | Guidance evaluation |
| `Train_env.py` | Continue training the saved PPO policy |
| `planetary_lander_recovery_impact_plus200k.zip` | Saved PPO checkpoint |
| `requirements.txt` | Python dependencies |

## Scope and next steps

This is an educational 2D simulation with simplified dynamics, constant spacecraft mass and direct tilt control. Guidance has access to the terrain map and simulation model.

Planned work includes balanced evaluation on each world, broader tests of mid-flight takeover, and using guidance demonstrations to teach a learned policy.

## Development acknowledgment

AI tools assisted with code generation, explanations, debugging and documentation.

