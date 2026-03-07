# External Libraries
import math
import random

# Internal Libraries
import parameters

# -----------------------------
# Helpers
# -----------------------------
def wrap_to_pi(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

# -----------------------------
# Distance & Yaw-rate wrappers
# -----------------------------
def distance_travelled_s(delta_encoder_counts):
    return parameters.K_SE * float(delta_encoder_counts)

def rotational_velocity_w(delta_encoder_counts, steering_angle_command, delta_t):
    if float(delta_t) < 1e-9:
        return 0.0
    return (parameters.C_R * float(delta_encoder_counts) * float(steering_angle_command)) / float(delta_t)

# -----------------------------
# Motion Model class
# -----------------------------
class MyMotionModel:
    """
    State x = [x_G, y_G, theta_G] (m, m, rad)
    Uses the Fancy Slip-Bias parameters from parameters.py
    """

    def __init__(self, initial_state, last_encoder_count, v_cmd_default=None, add_noise=True, alpha_is_degrees=True):
        self.state = [float(initial_state[0]), float(initial_state[1]), float(initial_state[2])]
        self.last_encoder_count = float(last_encoder_count)
        self.v_cmd_default = v_cmd_default
        self.add_noise = add_noise
        self.alpha_is_degrees = alpha_is_degrees

    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        enc = float(encoder_counts)
        delta_e = enc - self.last_encoder_count
        self.last_encoder_count = enc

        dt = float(delta_t)
        if dt < 1e-9:
            dt = 1e-9

        # Nominal Distance & Yaw Rate
        ds = distance_travelled_s(delta_e)
        w_cmd = rotational_velocity_w(delta_e, steering_angle_command, dt)
        dth = w_cmd * dt

        # Add Noise (Optional)
        if self.add_noise:
            # ds noise
            var_s = parameters.K_SS * abs(delta_e)
            if var_s > 0:
                ds += random.gauss(0.0, math.sqrt(var_s))
            
            # dth noise
            var_w = parameters.SIGMA_W2_CONST * dt
            if var_w > 0:
                dth += random.gauss(0.0, math.sqrt(var_w))
                
            # Random Walk Bias
            bias_noise = random.gauss(0.0, parameters.SIGMA_BIAS * math.sqrt(dt))
            dth += bias_noise * dt

        # Midpoint integration (like our particle filter)
        th_mid = self.state[2] + 0.5 * dth
        
        # Slip angle noise
        if self.add_noise:
            beta = random.gauss(0.0, parameters.SIGMA_BETA)
            th_mid += beta

        x_new = self.state[0] + ds * math.cos(th_mid)
        y_new = self.state[1] + ds * math.sin(th_mid)
        th_new = wrap_to_pi(self.state[2] + dth)

        self.state = [x_new, y_new, th_new]
        return self.state

    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]

        self.last_encoder_count = float(encoder_count_list[0])

        for i in range(1, len(encoder_count_list)):
            delta_t = float(time_list[i]) - float(time_list[i - 1])
            new_state = self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])

        return x_list, y_list, theta_list

    def generate_simulated_traj(self, duration):
        delta_t = 0.1
        t_list, x_list, y_list, theta_list = [], [], [], []
        t = 0.0
        encoder_counts = self.last_encoder_count
        steer = 0.0

        while t < duration:
            t_list.append(t)
            x_list.append(self.state[0])
            y_list.append(self.state[1])
            theta_list.append(self.state[2])

            encoder_counts += -50.0 
            self.step_update(encoder_counts, steer, delta_t)
            t += delta_t

        return t_list, x_list, y_list, theta_list
