# External libraries
import copy
import matplotlib.pyplot as plt
import math
import numpy as np
import random

# Local libraries
import parameters
import data_handling

# Helper function to make sure all angles are between -pi and pi
def angle_wrap(angle):
    while angle > math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle

# Helper class to store and manipulate your states.
class State:

    # Constructor
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    # Get the euclidean distance between 2 states
    def distance_to(self, other_state):
        return math.sqrt(math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2))
        
    # Get the distance squared between two states
    def distance_to_squared(self, other_state):
        return math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2)

    # return a deep copy of the state.
    def deepcopy(self):
        return copy.deepcopy(self)
        
    # Print the state
    def print(self):
        print("State: ",self.x, self.y, self.theta)


# Class to store walls as objects (specifically when represented as line segments in a 2D map.)
class Wall:

    # Constructor
    def __init__(self, wall_corners):
        self.corner1 = State(wall_corners[0], wall_corners[1], 0)
        self.corner2 = State(wall_corners[2], wall_corners[3], 0)
        self.corner1_mm = State(wall_corners[0] * 1000, wall_corners[1] * 1000, 0)
        self.corner2_mm = State(wall_corners[2] * 1000, wall_corners[3] * 1000, 0)
        
        self.m = (wall_corners[3] - wall_corners[1])/(0.0001 + wall_corners[2] -  wall_corners[0])
        self.b = wall_corners[3] - self.m * wall_corners[2]
        self.b_mm =  wall_corners[3] * 1000 - self.m * wall_corners[2] * 1000
        self.length = self.corner1.distance_to(self.corner2)
        self.length_mm_squared = self.corner1_mm.distance_to_squared(self.corner2_mm)
        
        if self.m > 1000:
            self.vertical = True
        else:
            self.vertical = False
        if abs(self.m) < 0.1:
            self.horizontal = True
        else:
            self.horizontal = False


# A class to store 2D maps
class Map:
    def __init__(self, wall_corner_list):
        self.wall_list = []
        for wall_corners in wall_corner_list:
            self.wall_list.append(Wall(wall_corners))
        min_x = 999999
        max_x = -99999
        min_y = 999999
        max_y = -99999
        for wall in self.wall_list:
            min_x = min(min_x, min(wall.corner1.x, wall.corner2.x))
            max_x = max(max_x, max(wall.corner1.x, wall.corner2.x))
            min_y = min(min_y, min(wall.corner1.y, wall.corner2.y))
            max_y = max(max_y, max(wall.corner1.y, wall.corner2.y))
        border = 0.5
        self.plot_range = [min_x - border, max_x + border, min_y - border, max_y + border]
        
        self.particle_range = [min_x , max_x , min_y, max_y]

    # Function to calculate the distance between any state and its closest wall, accounting for directon of the state.
    def closest_distance_to_walls(self, state):
        closest_distance = 999999999999
        for wall in self.wall_list:
            closest_distance = self.get_distance_to_wall(state, wall, closest_distance)
        
        return closest_distance
        
    # Function to get distance to a wall from a state, in the direction of the state's theta angle.
    # Or return the distance currently believed to be the closest if its closer.
    def get_distance_to_wall(self, state, wall, closest_distance):
        # Ray-Segment Intersection
        # Ray: R(t) = [x + t*cos(theta), y + t*sin(theta)], t > 0
        # Wall: W(u) = [x1 + u*(x2-x1), y1 + u*(y2-y1)], 0 <= u <= 1
        
        x, y, theta = state.x, state.y, state.theta
        x1, y1 = wall.corner1.x, wall.corner1.y
        x2, y2 = wall.corner2.x, wall.corner2.y
        
        dx = x2 - x1
        dy = y2 - y1
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        
        # Denominator for intersection
        div = (sin_th * dx - cos_th * dy)
        if abs(div) < 1e-6:
            # Ray is parallel to the wall, or robot is exactly on it
            return closest_distance
            
        # Parameter t along the ray (must be > 0, i.e., in front of the robot)
        t = (dx * (y1 - y) - dy * (x1 - x)) / div
        
        # Parameter u along the wall segment (must be between 0 and 1, i.e., between the corners)
        u = (cos_th * (y1 - y) - sin_th * (x1 - x)) / div
        
        if t > 0.01 and 0 <= u <= 1:
            if t < closest_distance:
                return t
        
        return closest_distance


# Class to hold a particle
class Particle:
    
    def __init__(self):
        self.state = State(0, 0, 0)
        self.weight = 1.0
        
    # Function to create a new random particle state within a range
    def randomize_uniformly(self, xy_range):
        # We need to ensure we don't start particles *inside* or outside the walls.
        # xy_range is [min_x, max_x, min_y, max_y]
        self.state.x = random.uniform(xy_range[0], xy_range[1])
        self.state.y = random.uniform(xy_range[2], xy_range[3])
            
        self.state.theta = random.uniform(-math.pi, math.pi)
        self.weight = 1.0 / parameters.num_particles

    # Function to create a new random particle state with a normal distribution
    def randomize_around_initial_state(self, initial_state, state_stdev):
        self.state.x = random.gauss(initial_state.x, state_stdev.x)
        self.state.y = random.gauss(initial_state.y, state_stdev.y)
        self.state.theta = angle_wrap(random.gauss(initial_state.theta, state_stdev.theta))
        self.weight = 1.0 / parameters.num_particles
        
    # Function to take a particle and "randomly" propagate it forward according to a motion model.
    def propagate_state(self, last_state, delta_encoder_counts, steering, delta_t):
        # Full Fancy Slip-Bias Motion Model
        ds_mean = parameters.K_SE * delta_encoder_counts
        ds_std = math.sqrt(parameters.K_SS * abs(delta_encoder_counts))
        ds = random.gauss(ds_mean, ds_std)
        
        dth_mean = parameters.C_R * delta_encoder_counts * steering
        dth_std = math.sqrt(parameters.SIGMA_W2_CONST * delta_t)
        # Random walk for bias integrated over dt
        bias_noise = random.gauss(0, parameters.SIGMA_BIAS * math.sqrt(delta_t))
        dth = dth_mean + random.gauss(0, dth_std) + bias_noise * delta_t
        
        # Slip angle noise
        beta = random.gauss(0, parameters.SIGMA_BETA)
        
        # Midpoint integration
        th_mid = last_state.theta + 0.5 * dth
        self.state.x = last_state.x + ds * math.cos(th_mid + beta)
        self.state.y = last_state.y + ds * math.sin(th_mid + beta)
        self.state.theta = angle_wrap(last_state.theta + dth)
        
    # Function to determine a particles log-weight based how well the lidar measurement matches up with the map.
    def calculate_log_weight(self, lidar_signal, map_obj):
        # We use a log-likelihood sum to completely avoid underflow.
        # We also subsample the rays (e.g. use ~30 rays) to speed up math and prevent overconfidence.
        self.log_weight = 0.0
        # The sensor static noise is parameters.distance_variance (e.g. 5.62e-6 m^2)
        # We add 0.05 m^2 for environmental model/map uncertainty so the uniform cloud can mathematically converge.
        variance = parameters.distance_variance + 0.05 
        valid_rays = 0
        
        step = max(1, len(lidar_signal.angles) // 30) # Subsample to ~30 evenly spaced rays
        for i in range(0, len(lidar_signal.angles), step):
            actual_dist = lidar_signal.convert_hardware_distance(lidar_signal.distances[i])
            
            # Skip invalid or out-of-range sensor readings
            if actual_dist < 0.05 or actual_dist > 8.0:
                continue
                
            # convert_hardware_angle handles the LIDAR-to-robot transform
            angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i])
            
            # Particle pose in map
            ray_state = State(self.state.x, self.state.y, angle_wrap(self.state.theta + angle))
            expected_dist = map_obj.closest_distance_to_walls(ray_state)
            
            # Cap expected distance to sensor max range, just in case of ray escapes the map
            if expected_dist > 8.0:
                expected_dist = 8.0
            
            # Gaussian Log-Likelihood component: -0.5 * (err^2 / process_variance)
            self.log_weight += -0.5 * ((actual_dist - expected_dist) ** 2) / variance
            valid_rays += 1
            
        if valid_rays > 0:
            # Average the log weight so that adding more rays doesn't disproportionately sharpen the distribution
            # This allows symmetric multi-modal clusters to form and survive accurately without instant collapse
            self.log_weight /= valid_rays
        else:
            self.log_weight = -1e9 # Penalize completely invalid particles
        
    # Return the normal distribution function output (kept for backward compatibility if needed).
    def gaussian(self, expected_distance, distance):
        variance = parameters.distance_variance + 0.05
        return (1.0/math.sqrt(2 * math.pi * variance)) * math.exp(-0.5 * (distance - expected_distance)**2 / variance)
        
    # Deep copy the particle
    def deepcopy(self):
        new_particle = Particle()
        new_particle.state.x = self.state.x
        new_particle.state.y = self.state.y
        new_particle.state.theta = self.state.theta
        new_particle.weight = self.weight
        if hasattr(self, 'log_weight'):
            new_particle.log_weight = self.log_weight
        return new_particle
        
    # Print the particle
    def print(self):
        print("Particle: ", self.state.x, self.state.y, self.state.theta, " w: ", self.weight)


# This class holds the collection of particles.
class ParticleSet:
    
    # Constructor, which calls the known start or unknown start initialization.
    def __init__(self, num_particles, xy_range, initial_state, state_stdev, known_start_state):
        self.num_particles = num_particles
        self.particle_list = []
        if known_start_state:
            self.generate_initial_state_particles(initial_state, state_stdev)
        else:
            self.generate_uniform_random_particles(xy_range)
        self.mean_state = State(0, 0, 0)
        self.update_mean_state()
        
    # Function to reset particles and random locations in the workspace.
    def generate_uniform_random_particles(self, xy_range):
        for i in range(self.num_particles):
            random_particle = Particle()
            random_particle.randomize_uniformly(xy_range)
            self.particle_list.append(random_particle)

    # Function to reset particles, normally distributed around the initial state. 
    def generate_initial_state_particles(self, initial_state, state_stdev):
        for i in range(self.num_particles):
            random_particle = Particle()
            random_particle.randomize_around_initial_state(initial_state, state_stdev)
            self.particle_list.append(random_particle)

    # Function to resample the particles set, i.e. make a new one with more copies of particles with higher weights.  
    def resample(self, max_weight):
        new_particles = []
        index = random.randint(0, self.num_particles - 1)
        beta = 0.0
        for i in range(self.num_particles):
            beta += random.uniform(0, 2 * max_weight)
            while beta > self.particle_list[index].weight:
                beta -= self.particle_list[index].weight
                index = (index + 1) % self.num_particles
            new_particles.append(self.particle_list[index].deepcopy())
            
        self.particle_list = new_particles
        # Normalize weights back to uniform and add roughening noise to prevent collapse
        for p in self.particle_list:
            p.weight = 1.0 / self.num_particles
            p.state.x += random.gauss(0, 0.01) # 1cm jitter
            p.state.y += random.gauss(0, 0.01) # 1cm jitter
            p.state.theta = angle_wrap(p.state.theta + random.gauss(0, 0.02)) # ~1deg jitter
            
    # Calculate the mean state. 
    def update_mean_state(self):
        sum_x = 0
        sum_y = 0
        sum_sin = 0
        sum_cos = 0
        
        for p in self.particle_list:
            sum_x += p.state.x * p.weight
            sum_y += p.state.y * p.weight
            sum_sin += math.sin(p.state.theta) * p.weight
            sum_cos += math.cos(p.state.theta) * p.weight
            
        self.mean_state.x = sum_x
        self.mean_state.y = sum_y
        self.mean_state.theta = math.atan2(sum_sin, sum_cos)
        
    # Print the particle set. Useful for debugging.
    def print_particles(self):
        for particle in self.particle_list:
            particle.print()
        print()

# Class to hold the particle filter and its functions.
class ParticleFilter:
    
    # Constructor
    def __init__(self, num_particles, map, initial_state, state_stdev, known_start_state, encoder_counts_0):
        self.map = map
        self.particle_set = ParticleSet(num_particles, map.particle_range, initial_state, state_stdev, known_start_state)
        self.state_estimate = self.particle_set.mean_state
        self.state_estimate_list = []
        self.last_time = 0
        self.last_encoder_counts = encoder_counts_0

    # Update the states given new measurements
    def update(self, odometery_signal, measurement_signal, delta_t):
        self.prediction(odometery_signal, delta_t)
        if len(measurement_signal.angles)>0:
            self.correction(measurement_signal)
        self.particle_set.update_mean_state()
        self.state_estimate_list.append(self.state_estimate.deepcopy())

    # Predict the current state from the last state.
    def prediction(self, odometry_signal, delta_t):
        # odometry_signal: [encoder_counts, steering_angle]
        delta_encoder = odometry_signal[0] - self.last_encoder_counts
        steering = odometry_signal[1]
        
        for p in self.particle_set.particle_list:
            p.propagate_state(p.state.deepcopy(), delta_encoder, steering, delta_t)
            
        self.last_encoder_counts = odometry_signal[0]
        
    # Corrrect the predicted states.
    def correction(self, measurement_signal):
        # Calculate log weights for all particles
        for p in self.particle_set.particle_list:
            p.calculate_log_weight(measurement_signal, self.map)
            
        # Log-Sum-Exp Trick to avoid underflow
        # Find the max log_weight
        max_log_weight = max([p.log_weight for p in self.particle_set.particle_list])
        
        # Convert log weights to regular weights strictly relative to the maximum
        max_weight = 0.0
        for p in self.particle_set.particle_list:
            # Shift by max_log_weight keeps the largest exponent at e^0 = 1.0
            p.weight = math.exp(p.log_weight - max_log_weight)
            if p.weight > max_weight:
                max_weight = p.weight
                
        if max_weight > 0.0:
            self.particle_set.resample(max_weight)
        else:
            pass
        
    # Output to terminal the mean state.
    def print_state_estimate(self):
        print("Mean state: ", self.particle_set.mean_state.x, self.particle_set.mean_state.y, self.particle_set.mean_state.theta)
    

# Class to help with plotting PF data.
class ParticleFilterPlot:

    # Constructor
    def __init__(self, map):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig
        self.map = map

    # Clear and update the plot with new PF data
    def update(self, state_mean, particle_set, lidar_signal, hold_show_plot):
        plt.clf()
        
        # Plot walls
        for wall in self.map.wall_list:
            plt.plot([wall.corner1.x, wall.corner2.x],[wall.corner1.y, wall.corner2.y],'k')

        # Plot lidar
        for i in range(len(lidar_signal.angles)):
            distance = lidar_signal.convert_hardware_distance(lidar_signal.distances[i])
            angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i]) + state_mean.theta
            x_ray = [state_mean.x, state_mean.x + distance * math.cos(angle)]
            y_ray = [state_mean.y, state_mean.y + distance * math.sin(angle)]
            plt.plot(x_ray, y_ray, 'r')


        # Plot state estimate
        plt.plot(state_mean.x, state_mean.y,'ro')
        plt.plot([state_mean.x, state_mean.x+ self.dir_length*math.cos(state_mean.theta) ], [state_mean.y, state_mean.y+ self.dir_length*math.sin(state_mean.theta) ],'r')
        x_particles, y_particles = self.to_plot_data(particle_set)
        plt.plot(x_particles, y_particles, 'g.')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis(self.map.plot_range)
        plt.grid()
        if hold_show_plot:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.1)

    # Helper function to make the particles easy to plot.
    def to_plot_data(self, particle_set):
        x_list = []
        y_list = []
        for p in particle_set.particle_list:
            x_list.append(p.state.x)
            y_list.append(p.state.y)
        return x_list, y_list
        
# NEW: Function for multi-subplot visualization
def plot_pf_snapshots(snapshots, map_obj):
    # Determine grid size based on number of snapshots
    num_snaps = len(snapshots)
    cols = min(4, num_snaps)
    rows = math.ceil(num_snaps / cols)
    
    # White background by default (remove dark_background)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows), squeeze=False)
    fig.suptitle("Particle Filter Localization Progress (SI Units: Meters)", fontsize=18, color='black')
    
    for i, (time_pt, mean, p_list, z_t, dr_path_x, dr_path_y, dr_th) in enumerate(snapshots):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        
        # Plot walls
        for wall in map_obj.wall_list:
            ax.plot([wall.corner1.x, wall.corner2.x], [wall.corner1.y, wall.corner2.y], 'k-', linewidth=2)
            
        # Plot particles (brighter, larger, fully opaque)
        px = [p.state.x for p in p_list]
        py = [p.state.y for p in p_list]
        ax.plot(px, py, 'g.', markersize=3, alpha=0.6, label='Particles' if i==0 else "")
        
        # Plot Dead Reckoning path
        if len(dr_path_x) > 0:
            ax.plot(dr_path_x, dr_path_y, 'b-', linewidth=2, label='Predicted (DR)' if i==0 else "")
            dx_dr = 0.2 * math.cos(dr_th)
            dy_dr = 0.2 * math.sin(dr_th)
            ax.arrow(dr_path_x[-1], dr_path_y[-1], dx_dr, dy_dr, head_width=0.08, color='blue')

        # Plot mean estimate
        ax.plot(mean.x, mean.y, 'ro', markersize=8, label='Mean Estimate' if i==0 else "")
        dx = 0.2 * math.cos(mean.theta)
        dy = 0.2 * math.sin(mean.theta)
        ax.arrow(mean.x, mean.y, dx, dy, head_width=0.08, color='red')
        
        ax.set_title(f"T = {time_pt:.1f} s", color='black')
        ax.set_xlabel("X (m)", color='black')
        ax.set_ylabel("Y (m)", color='black')
        ax.axis('equal')
        ax.grid(True, alpha=0.2, color='lightgray')
        ax.set_xlim(map_obj.plot_range[0], map_obj.plot_range[1])
        ax.set_ylim(map_obj.plot_range[2], map_obj.plot_range[3])
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

        # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])
        
    # Format the global legend for the white background
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), facecolor='white', edgecolor='black', fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.9)
    plt.savefig('pf_offline_snapshots.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved offline particle filter snapshots to 'pf_offline_snapshots.png'")
    # Reset style so we don't accidentally affect other plots in the session
    plt.style.use('default')
    plt.show()

# Function used to test your PF offline with logged data.
def offline_pf():
    map_obj = Map(parameters.wall_corner_list)

    # Use a specific data file from your data directory
    # Note: Replace with your actual file name
    import glob
    files = glob.glob('./data/*.pkl')
    if not files:
        print("Error: No data files found in ./data/")
        return
    # filename = files[-1]
    filename = './data/robot_data_100_-15_07_03_26_17_57_53.pkl'  # Use the most recent file
    print(f"Loading data from: {filename}")
    
    pf_data = data_handling.get_file_data_for_pf(filename)

    # Instantiate PF (Adjust initial_state as needed)
    start_row = pf_data[0]
    initial_x, initial_y, initial_th = 0.2, 0.09, 0.0 # Example starting pose
    
    particle_filter = ParticleFilter(
        parameters.num_particles, 
        map_obj, 
        initial_state=State(initial_x, initial_y, initial_th), 
        state_stdev=State(0.2, 0.2, 0.1), 
        known_start_state=False, # Changed to False for Unknown Start
        encoder_counts_0=start_row[2].encoder_counts
    )

    # Dead reckoning tracking
    dr_x, dr_y, dr_th = initial_x, initial_y, initial_th
    dr_path_x, dr_path_y = [dr_x], [dr_y]

    snapshots = []
    
    # Capture T=0 Snapshot BEFORE the movement loop
    snapshots.append((
        0.0,
        particle_filter.particle_set.mean_state.deepcopy(),
        [p.deepcopy() for p in particle_filter.particle_set.particle_list],
        start_row[2],
        list(dr_path_x), list(dr_path_y), dr_th
    ))

    # Time interval between snapshots in seconds
    snapshot_interval = 0.72 
    next_snapshot_time = snapshot_interval
    
    print("Running Particle Filter Offline...")
    for t in range(1, len(pf_data)):
        row = pf_data[t]
        delta_t = pf_data[t][0] - pf_data[t-1][0]
        current_time = row[0] - pf_data[0][0]
        u_t = np.array([row[2].encoder_counts, row[2].steering])
        z_t = row[2]
        
        # Dead Reckoning Update
        delta_e = row[2].encoder_counts - pf_data[t-1][2].encoder_counts
        dt_val = max(delta_t, 1e-9)
        ds = parameters.K_SE * delta_e
        w = (parameters.C_R * delta_e * row[2].steering) / dt_val
        dth = w * dt_val
        th_mid = dr_th + 0.5 * dth
        dr_x += ds * math.cos(th_mid)
        dr_y += ds * math.sin(th_mid)
        dr_th = angle_wrap(dr_th + dth)
        dr_path_x.append(dr_x)
        dr_path_y.append(dr_y)

        particle_filter.update(u_t, z_t, delta_t)
        
        if current_time >= next_snapshot_time or t == len(pf_data)-1:
            snapshots.append((
                current_time, 
                particle_filter.particle_set.mean_state.deepcopy(),
                [p.deepcopy() for p in particle_filter.particle_set.particle_list],
                z_t,
                list(dr_path_x),
                list(dr_path_y),
                dr_th
            ))
            next_snapshot_time += snapshot_interval

    # Single plot with multiple subplots
    plot_pf_snapshots(snapshots, map_obj)

        


####### MAIN #######
if __name__ == '__main__':
    offline_pf()
