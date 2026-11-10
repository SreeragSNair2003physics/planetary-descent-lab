import numpy as np                                                     
from stable_baselines3 import PPO
from Planetary_environment import PlanetaryLanderEnv
import math
import random
import pygame as pg                           # Loads the graphics library

pg.init()                                     # Starts Pygame’s systems, including its display and fonts

model = PPO.load("planetary_lander_recovery_impact_plus200k")        # Loads the trained model

screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)      # Creates a fullscreen display at the current screen resolution
pg.display.set_caption("Planetary Descent Lab")          # Gives the window it's name
clock = pg.time.Clock()                                  # Creates a timer to control how quickly the screen updates

title_font = pg.font.Font(None, 64)                      # Larger font for title
hint_font = pg.font.Font(None, 32)                       # Smaller font for instruction

screen_width, screen_height = screen.get_size()          # Reads the full screen display's width and height

title_surface = title_font.render("SPACECRAFT LANDING ON DIFFERENT PLANETS", True, (230, 235, 255))                     
hint_surface = hint_font.render("1/2/3: choose planet | W/S: throttle | SPACE: fire | A/D: tilt | arrows: side thrust | R: reset | ESC: exit",True,(170, 185, 210),)   
title_rect = title_surface.get_rect(topleft=(40, 30))                                                         # Positions title around center of screen
hint_rect = hint_surface.get_rect(topleft=(40, 82))                                                           # Positions instruction around center of screen

stars = [
    (0.03, 0.08), (0.08, 0.12), (0.12, 0.28), (0.15, 0.05), (0.19, 0.24),
    (0.22, 0.15), (0.28, 0.35), (0.34, 0.10), (0.38, 0.25), (0.42, 0.07),
    (0.48, 0.20), (0.52, 0.32), (0.55, 0.12), (0.61, 0.28), (0.67, 0.13),
    (0.72, 0.06), (0.76, 0.31), (0.85, 0.40), (0.88, 0.08),                # Each star entry is (horizontal position, vertical position)
    (0.92, 0.16), (0.95, 0.33), (0.98, 0.22), (0.05, 0.45), (0.13, 0.50),  # Written as a fraction of the screen size
    (0.27, 0.48), (0.35, 0.55), (0.60, 0.45), (0.68, 0.52), (0.74, 0.47)]

earth_clouds = [
    [0.12, 0.16, 0.025],
    [0.45, 0.24, 0.018],                                                   
    [0.78, 0.13, 0.030],
]

terrain_shape = [
    (0.00, 0.79),
    (0.13, 0.74),
    (0.25, 0.79),                                    # Describes the Moon’s outline using screen fractions
    (0.37, 0.76),  
    (0.43, 0.78),#---------------------------------- # The flat section between 0.43 and 0.57 is the landing area
    (0.57, 0.78),#--------------------------------------^                                       
    (0.64, 0.82),
    (0.79, 0.80),
    (0.89, 0.84),
    (1.00, 0.77),
    (1.00, 1.00),
    (0.00, 1.00),]

terrain_points = [ (int(x * screen_width), int(y * screen_height)) for x, y in terrain_shape ]                        

''' Converts the terrain's screen-size fractions into pixel coordinates that Pygame can draw '''

pad_y = int(screen_height * 0.78)                    # Stores the landing area’s vertical position

lander_x = screen_width // 2                         # Spacecraft co-ordinates; Horizontal center of screen
lander_size = int(screen_height * 0.035)             # Scales its shape to the screen height

PIXELS_PER_METER = 5.0                               # Controls how tall the flight looks on screen; It converts the simulation’s meters into pixels

WORLD_GRAVITY = {"Moon": 1.62,"Mars": 3.71,"Earth": 9.81,}     # The dictionary stores each world’s gravity in metres per second squared
current_world = "Moon"

WORLD_VISUALS = {
    "Moon": {"sky": (5, 8, 20),"terrain": (110, 113, 122),"orb": (60, 130, 210),},
    "Mars": {"sky": (38, 18, 16),"terrain": (151, 78, 49),"orb": (205, 148, 102),},                    # Background definitions for different planets
    "Earth": {"sky": (29, 76, 132),"terrain": (62, 116, 75),"orb": (255, 220, 135),}, }

WORLD_ATMOSPHERE = {
    "Moon": {"density": 0.0, "gust_amplitude": 0.0},
    "Mars": {"density": 0.016, "gust_amplitude": 30.0},                                                # Atmosphere definitions for different planets
    "Earth": {"density": 1.225, "gust_amplitude": 5.0},
}

DRAG_COEFFICIENT = 1.2                               # A number describing how strongly the lander’s shape resists moving through air
FRONTAL_AREA = 20.0                                  # The lander’s area facing the wind, which affects how much wind force it catches
WIND_CYCLE_RATE = 0.8                                # How quickly the simulated wind changes over time


LANDER_MASS = 1000.0                                 # Craft’s mass in kilograms
MAIN_ENGINE_THRUST = 12000.0                         # Engine’s upward force in newtons
START_ALTITUDE = 50.0                                # Places the lander 20 metres above the pad
SIDE_THRUSTER_FORCE = 900.0                          # Sideways engine force in newtons
THROTTLE_CHANGE_RATE = 2.0                           # Sets how quickly the engine power changes: half the full range per second
PAD_HALF_WIDTH_METERS = (screen_width * 0.07) / PIXELS_PER_METER   # half the visible pad width, converted from pixels to metres

''' angle is the craft's tilt. Zero means upright; We convert them to radians '''

ROTATION_SPEED = math.radians(30)                    # 30 degrees per second; controls how quickly A/D tilt the craft
MAX_TILT = math.radians(60)                          # 60 degrees per second; limits how far it can tilt
MAX_LANDING_TILT = math.radians(10)                  # Safe landing requires within 10 degrees of upright

throttle = 0.15                                      # Engine setting from 0.0 (off) to 1.0 (full power). It starts at 15%

wind_time = 0.0                                      # This tracks time so the wind gust can change as the simulation runs

explosion_particles = []                             # The list stores the sparks
explosion_played = False                             # The flag ensures we create the explosion only once per crash

horizontal_position = 0.0                            # Lander’s distance from the pad center. Positive means right
horizontal_speed = 0.0                               # Lander's sideway speed
crashed = False                                      # Tracks an unsafe landing 

angle= 0.0                                           # Starts the craft upright, with angle measured in radians

altitude = START_ALTITUDE                            # Sets the initial displayed altitude
vertical_speed = 0.0                                 # Stores how fast it moves vertically. Positive speed means upward
landed = False                                       # Tracks whether the craft has reached the ground

autopilot = False                                                         # autopilot remembers whether AI control is on
autopilot_button = pg.Rect(screen_width - 330, 30, 290, 55)               # Defines the clickable rectangle near the top-right of the screen


env = PlanetaryLanderEnv(                                                 # Creates the environment used for the actual flight physics
    world=current_world,
    screen_width=screen_width,
    screen_height=screen_height,
    max_steps=1_000_000,)

env.reset()                                                               # Initializes the environment's physical state

USE_GUIDANCE = True                                                       # True selects guidance system; False selects the saved PPO policy
previous_autopilot = autopilot                                            # Remembers the previous autopilot setting to detect a handover
flight_finished = False                                                   # Starts with an active flight that has not ended
action = np.array([0, 1, 1, 1], dtype=np.int64)                           # Initializes an idle action: engine off and all other controls held




def rotate_point(point, center_x, center_y, angle):
    point_x, point_y = point                                  # rotate_point takes a point on the lander and calculates where it should appear after the craft tilts
    offset_x = point_x - center_x
    offset_y = point_y - center_y                        
    rotated_x = offset_x * math.cos(angle) - offset_y * math.sin(angle)
    rotated_y = offset_x * math.sin(angle) + offset_y * math.cos(angle)
    return int(center_x + rotated_x), int(center_y + rotated_y)


def draw_rotated_line(surface, color, start, end, center_x, center_y, angle, width):
    rotated_start = rotate_point(start, center_x, center_y, angle)
    rotated_end = rotate_point(end, center_x, center_y, angle)                      # draw_rotated_line does that for both ends of a line, then draws the rotated line
    pg.draw.line(surface, color, rotated_start, rotated_end, width)                 # We'll use it for the legs

def draw_cloud(surface, x, y, size):
    cloud_color = (238, 243, 250)                                                   # The ellipse is the base; Circles make the bumps; size controls.. well, size.. lol

    pg.draw.ellipse(surface,cloud_color,(x, y + size // 3, size * 3, size),)

    pg.draw.circle(surface,cloud_color,(x + size, y + size // 2),size // 2,)

    pg.draw.circle(surface,cloud_color,(x + size * 2, y + size // 3),size // 2,)   


def terrain_altitude_at(x_pixel):                               # x_pixel is the horizontal screen coordinate where you want to check the terrain
    for index in range(len(terrain_points) - 1):                # Loops over pairs of neighbouring terrain points. The - 1 ensures index + 1 will still be a valid point
        x1, y1 = terrain_points[index]
        x2, y2 = terrain_points[index + 1]

        if x2 > x1 and x1 <= x_pixel <= x2:
            fraction = (x_pixel - x1) / (x2 - x1)
            surface_y = y1 + fraction * (y2 - y1)
            return (pad_y - surface_y) / PIXELS_PER_METER

    return 0.0



running = True                                                                   # Keeps loop running

while running:

    clock.tick(60)
    dt = 1.0 / 60.0                                                 # 'dt' is the time elapsed since the previous screen update, measured in seconds

    for event in pg.event.get():                                                 # Checks for keystrokes and window events

        if event.type == pg.QUIT:                                                # Detects a request to close the window

            running = False

        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:             # Detects a left mouse-button click
             
                if autopilot_button.collidepoint(event.pos):                     # Checks whether the click landed on the autopilot button
 
                    autopilot = not autopilot                                    # Toggles between manual flight and autopilot

        elif event.type == pg.KEYDOWN and event.key in (pg.K_1, pg.K_KP1):       # Detects either version of the number-one key

            current_world = "Moon"

        elif event.type == pg.KEYDOWN and event.key in (pg.K_2, pg.K_KP2):       # Detects either version of the number-two key

            current_world = "Mars"

        elif event.type == pg.KEYDOWN and event.key in (pg.K_3, pg.K_KP3):       # Detects either version of the number-three key

            current_world = "Earth"

        elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:              # Detects the Escape key

            running = False

        elif event.type == pg.KEYDOWN and event.key == pg.K_r:                   # Detects the reset key

            env.set_world(current_world)                                         # Selects the planet to use for the reset
            env.reset()                                                          # Restores the environment's starting flight state

            explosion_particles.clear()                                          # Removes particles from the previous explosion
            explosion_played = False                                             # Allows a new explosion to be created after another crash
            landed = False                                                       # Clears the successful-landing flag
            crashed = False                                                      # Clears the crash flag
            flight_finished = False                                              # Allows flight updates to resume


        
    if env.current_world != current_world:                                       # Detects a change between the selected planet and the environment's planet
        env.set_world(current_world)                                             # Updates the environment's planet and gravity
        env.restart_guidance()                                                   # Clears the previous landing plan and restarts its time budget

        if not landed and not crashed:                                           # Allows resuming only while the craft has neither landed nor crashed
            flight_finished = False

    if autopilot != previous_autopilot:
        env.restart_guidance()
        env.max_steps = 1000 if autopilot else 1_000_000

        if not landed and not crashed:
            flight_finished = False

        previous_autopilot = autopilot

    if not flight_finished:
        if autopilot:
            if USE_GUIDANCE:
                action = env.backup_action()
            else:
                action, _ = model.predict(
                    env._get_observation(),
                    deterministic=True,
                )
        else:
            keys = pg.key.get_pressed()

            action = np.array([
                int(keys[pg.K_SPACE]),
                1 + int(keys[pg.K_w]) - int(keys[pg.K_s]),
                1 + int(keys[pg.K_RIGHT]) - int(keys[pg.K_LEFT]),
                1 + int(keys[pg.K_d]) - int(keys[pg.K_a]),
            ], dtype=np.int64)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            landed = info["landed_safely"]
            crashed = info["crashed"]
            flight_finished = True

            print(
                f"pad={info['pad_landing']}, "
                f"terrain={info['emergency_landing']}, "
                f"crash={crashed}, timeout={truncated}, "
                f"impact_vx={info['impact_vx']}, "
                f"impact_vy={info['impact_vy']}"
            )

    horizontal_position = env.x
    horizontal_speed = env.vx
    altitude = env.altitude
    vertical_speed = env.vy
    angle = env.angle
    throttle = env.throttle
    wind_time = env.wind_time
    wind_speed = env.wind_speed
    wind_acceleration = env.wind_acceleration()

    thrusting = (
        bool(action[0])
        and throttle > 0
        and not flight_finished
    )



    screen.fill(WORLD_VISUALS[current_world]["sky"])                                          # This looks up the selected world’s sky color

    if autopilot:
        button_text = "GUIDANCE: ON" if USE_GUIDANCE else "PPO: ON"
    else:
        button_text = "AUTOPILOT: OFF"
    button_color = (35, 150, 105) if autopilot else (75, 85, 105)

    pg.draw.rect(screen, button_color, autopilot_button, border_radius=12)

    button_surface = hint_font.render(button_text, True, (255, 255, 255))
    button_rect = button_surface.get_rect(center=autopilot_button.center)
    screen.blit(button_surface, button_rect)
    
    pg.draw.circle(screen,WORLD_VISUALS[current_world]["orb"],(int(screen_width * 0.82), int(screen_height * 0.20)),int(screen_height * 0.06),)  # Changes BG planet

    ''' 

    pygame.draw.circle draws a filled circle.
    - screen is where to draw it.
    - (int(screen_width * 0.82), int(screen_height * 0.20)) is its center: 82% of the way across the screen and 20% down.
    - int(screen_height * 0.06) is its radius: 6% of the screen height.
    - int(...) changes a decimal position into a whole number of pixels  
    
    '''

    if current_world != "Earth":

        for star_x, star_y in stars:                                                 # Goes through the list 'stars'; each pair has both co-ordinates

            pg.draw.circle(screen,(220, 230, 255),(int(star_x * screen_width), int(star_y * screen_height)), 2,) # Converts to pixel co-ordinates; radius is 2 pixels

    if current_world == "Earth":
        cloud_size = max(12, int(screen_height * 0.025))

        for cloud in earth_clouds:
            cloud[0] += cloud[2] * dt                                          # Each frame, cloud[2] * dt advances the cloud according to its speed and elapsed time

            if cloud[0] > 1.1:
                cloud[0] = -0.2

            cloud_x = int(cloud[0] * screen_width)
            cloud_y = int(cloud[1] * screen_height)

            draw_cloud(screen, cloud_x, cloud_y, cloud_size)        


    pg.draw.polygon(screen,WORLD_VISUALS[current_world]["terrain"],terrain_points,)                  # Draws the terrain

    pg.draw.line(screen,(240, 205, 95), (int(screen_width * 0.43), pad_y), (int(screen_width * 0.57), pad_y), 6,)  # Draws the landing pad; and thickness is 6 pixels

    left_flag_x = int(screen_width * 0.43)                # Left flag’s horizontal position to 43% of the screen width
    right_flag_x = int(screen_width * 0.57)               # Right flag’s horizontal position to 57% of the screen width

    pg.draw.line(screen,(210, 215, 225), (left_flag_x, pad_y), (left_flag_x, pad_y - 45), 3)         # Draws the left pole
    pg.draw.line(screen,(210, 215, 225), (right_flag_x, pad_y), (right_flag_x, pad_y - 45), 3)       # Draws the right pole

    pg.draw.polygon(screen, (245, 210, 65), [(left_flag_x, pad_y - 45),(left_flag_x + 24, pad_y - 38),(left_flag_x, pad_y - 31),])  # Draws the left flag

    pg.draw.polygon(screen, (245, 210, 65), [(right_flag_x, pad_y - 45),    (right_flag_x + 24, pad_y - 38),(right_flag_x, pad_y - 31),]) # Draws the right flag 


    lander_x = screen_width // 2 + int(horizontal_position * PIXELS_PER_METER)               # Converts sideways position into a screen x-coordinate


    foot_offset = int( lander_size * 1.5 * math.cos(angle) + lander_size * abs(math.sin(angle)))  # Estimates vertical distance from lander’s center to its lowest foot
    lander_y = pad_y - foot_offset - int(altitude * PIXELS_PER_METER)                             # Sets the lander’s vertical screen position


    if crashed and not explosion_played:

        for _ in range(32):

            direction = random.uniform(0.0, 2.0 * math.pi)
            speed = random.uniform(90.0, 240.0)
            lifetime = random.uniform(0.35, 0.8)

            explosion_particles.append([lander_x ,lander_y + foot_offset ,math.cos(direction) * speed,math.sin(direction) * speed,lifetime,])

        explosion_played = True



    world_surface = hint_font.render(f"World: {current_world} == Gravity: {WORLD_GRAVITY[current_world]:.2f} m/s²",True,(220, 225, 240),)
    world_rect = world_surface.get_rect(topleft=(40, 245))

    wind_surface = hint_font.render(f"Wind: {wind_speed:+.1f} m/s | Push: {wind_acceleration:+.2f} m/s^2",True,(220, 225, 240),)
    wind_rect = wind_surface.get_rect(topleft=(40, 285))

    status_surface = hint_font.render(f"Altitude: {altitude:.1f} m | Vertical speed: {vertical_speed:.1f} m/s | Throttle: {throttle * 100:.0f}%",True,(220, 225, 240),)

    status_rect = status_surface.get_rect(topleft=(40, 125))

    horizontal_surface = hint_font.render(f"Horizontal speed: {horizontal_speed:.1f} m/s | Offset: {horizontal_position:.1f} m",True,(220, 225, 240),)

    horizontal_rect = horizontal_surface.get_rect(topleft=(40, 165))

    angle_surface = hint_font.render(f"Tilt: {math.degrees(angle):.1f} deg",True,(220, 225, 240),)
    angle_rect = angle_surface.get_rect(topleft=(40, 205))

    ''' hint_font.render(...) turns the current numbers into drawable text.
        The f before the text lets Python insert the current variable values.
        (:.1f) displays each value rounded to one decimal place.
        True smooths the text edges; the color tuple is its RGB color.
        get_rect(topleft=...) positions the readout below the instructions '''


    if thrusting:

        flame_points = [
            (lander_x - lander_size // 4, lander_y + lander_size // 2),
            (lander_x + lander_size // 4, lander_y + lander_size // 2),               # Adds the flame for upward acceleration 
            (lander_x, lander_y + lander_size * 3 // 2),]

        flame_points = [rotate_point(point, lander_x, lander_y, angle) for point in flame_points]            # This turns the flame in the same direction as the craft
        
        pg.draw.polygon(screen, (255, 130, 35), flame_points)



    
    lander_body = [
    (lander_x, lander_y - lander_size),
    (lander_x + lander_size, lander_y - lander_size // 3),
    (lander_x + lander_size * 3 // 4, lander_y + lander_size // 2),
    (lander_x - lander_size * 3 // 4, lander_y + lander_size // 2),                   # Draws the lander body
    (lander_x - lander_size, lander_y - lander_size // 3),]

    lander_body = [rotate_point(point, lander_x, lander_y, angle) for point in lander_body]


    lander_window = [
        (lander_x, lander_y - lander_size * 2 // 3),
        (lander_x + lander_size // 3, lander_y - lander_size // 4),
        (lander_x, lander_y),                                                         # Draws the lander window
        (lander_x - lander_size // 3, lander_y - lander_size // 4),]

    lander_window = [rotate_point(point, lander_x, lander_y, angle) for point in lander_window]


    leg_lines = [
    ((lander_x - lander_size // 2, lander_y + lander_size // 3),
     (lander_x - lander_size, lander_y + lander_size * 3 // 2)),
    ((lander_x - lander_size, lander_y + lander_size * 3 // 2),
     (lander_x - lander_size // 2, lander_y + lander_size * 3 // 2)),          # Each pair in leg_lines is the start and end of one leg segment 
    ((lander_x + lander_size // 2, lander_y + lander_size // 3),               # The loop rotates and draws all four segments
     (lander_x + lander_size, lander_y + lander_size * 3 // 2)),
    ((lander_x + lander_size, lander_y + lander_size * 3 // 2),
     (lander_x + lander_size // 2, lander_y + lander_size * 3 // 2)),]

    for start, end in leg_lines:

        draw_rotated_line(screen,(205, 210, 220),start,end,lander_x,lander_y,angle,4,)

    pg.draw.polygon(screen, (145, 115, 235), lander_body)         # Draws the filled purple body by connecting the points stored in lander_body
    pg.draw.polygon(screen, (95, 220, 235), lander_window)        # Draws the cyan window from the points in lander_window


    for particle in explosion_particles[:]:
        particle[0] += particle[2] * dt
        particle[1] += particle[3] * dt
        particle[3] += 300.0 * dt
        particle[4] -= dt

        if particle[4] <= 0:
            explosion_particles.remove(particle)
            continue

        radius = max(1, int(8 * particle[4]))
        spark_color = (255, max(35, int(220 * particle[4])), 20)

        pg.draw.circle(screen,spark_color,(int(particle[0]), int(particle[1])),radius,)



    screen.blit(title_surface, title_rect)                        # Copies the prepared title image onto the screen at the position in title_rect
    screen.blit(hint_surface, hint_rect)                          # Copies the prepared instruction image onto screen at position in hint_rect
    screen.blit(world_surface, world_rect)                        # Displays the current world
    screen.blit(status_surface, status_rect)                      # Displays the live readout
    screen.blit(horizontal_surface, horizontal_rect)
    screen.blit(angle_surface, angle_rect)
    screen.blit(wind_surface, wind_rect)              

    
    pg.display.flip()                                                            # Shows latest frame in game window

pg.quit()    