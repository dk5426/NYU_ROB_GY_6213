# External Libraries
import math
import random

# -----------------------------
# Motion Model calibration constants (from your notebook)
# -----------------------------

# Distance mapping: s = k_se * e_fwd, where e_fwd = -(delta encoder counts)
K_SE = 0.00028851708972299  # meters per count

# Distance variance mapping: sigma_s^2 = k_ss * e_fwd
K_SS = 7.342396346880917e-08  # (m^2) per count

# Yaw-rate mapping coefficients (no symmetry constraint)
# w = b0 + b1*v + b2*a + b3*v*a + b4*a^2 + b5*v^2 + b6*v*a^2
# where a = steering angle in radians; v = speed command (80/100, etc.)
B0 = -3.97536968e-05
B1 = -1.76673990e-03
B2 = -5.16057305e-01
B3 =  2.47746185e-02
B4 = -4.29707073e-05
B5 =  1.63932739e-05
B6 = -4.22979202e-03

# Yaw-rate variance (constant)
SIGMA_W2_CONST = 0.0008853605431128998  # (rad/s)^2

# Numerics
DTH_EPS = 1e-6


# -----------------------------
# Helpers
# -----------------------------
def wrap_to_pi(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


# -----------------------------
# Section 3: encoder -> distance + variance
# -----------------------------
def variance_distance_travelled_s(e_fwd_counts):
    """
    sigma_s^2 = k_ss * e_fwd
    Always nonnegative for e_fwd >= 0.
    """
    e = float(e_fwd_counts)
    if e < 0.0:
        # If someone accidentally passes raw counts, protect ourselves
        e = abs(e)
    return K_SS * e


def distance_travelled_s(delta_encoder_counts):
    """
    s = k_se * e_fwd, e_fwd = -delta_e (because forward raw counts are negative)
    """
    delta_e = float(delta_encoder_counts)
    e_fwd = -delta_e
    return K_SE * e_fwd


# -----------------------------
# Section 5: steering+speed -> yaw rate + variance
# -----------------------------
def variance_rotational_velocity_w(v_cmd, steering_angle_command):
    """
    Baseline: constant yaw-rate variance.
    """
    return SIGMA_W2_CONST


def rotational_velocity_w(v_cmd, steering_angle_command, alpha_is_degrees=True):
    """
    w = f_w(v, alpha) with fitted coefficients (no symmetry constraint).
    v_cmd: speed command (e.g., 80, 100)
    steering_angle_command: alpha (deg by default)
    Returns w in rad/s.
    """
    v = float(v_cmd)
    a = float(steering_angle_command)
    if alpha_is_degrees:
        a = math.radians(a)  # radians

    w = (
        B0
        + B1 * v
        + B2 * a
        + B3 * (v * a)
        + B4 * (a ** 2)
        + B5 * (v ** 2)
        + B6 * (v * (a ** 2))
    )
    return w


# -----------------------------
# Motion Model class
# -----------------------------
class MyMotionModel:
    """
    State x = [x_G, y_G, theta_G] (m, m, rad)

    step_update() integrates:
      - s from encoder delta using f_se
      - w from f_w using (v_cmd_default, steering_angle_command)

    IMPORTANT:
      Your fitted w-model uses v as the SPEED COMMAND (80/100),
      so you should pass v_cmd_default per trial (or per dataset).
    """

    def __init__(self, initial_state, last_encoder_count, v_cmd_default=None, add_noise=True, alpha_is_degrees=True):
        self.state = [float(initial_state[0]), float(initial_state[1]), float(initial_state[2])]
        self.last_encoder_count = float(last_encoder_count)
        self.v_cmd_default = v_cmd_default  # set this! e.g. 80 or 100 for that trial
        self.add_noise = add_noise
        self.alpha_is_degrees = alpha_is_degrees

    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        """
        Implements x_t = f(x_{t-1}, u_t), where u_t is derived from:
          s = distance_travelled_s(delta_encoder)
          w = rotational_velocity_w(v_cmd_default, steering_angle_command)
        """
        # --- 1) distance increment from encoders ---
        enc = float(encoder_counts)
        delta_e = enc - self.last_encoder_count
        self.last_encoder_count = enc

        s = distance_travelled_s(delta_e)  # meters

        # --- 2) yaw rate from steering & speed command ---
        if self.v_cmd_default is None:
            raise ValueError(
                "MyMotionModel requires v_cmd_default (e.g., 80 or 100), "
                "because your fitted yaw-rate model uses speed COMMAND units."
            )

        w = rotational_velocity_w(self.v_cmd_default, steering_angle_command, alpha_is_degrees=self.alpha_is_degrees)

        # --- 3) optionally inject noise using calibrated variances ---
        if self.add_noise:
            e_fwd = -delta_e  # forward counts
            var_s = variance_distance_travelled_s(e_fwd)
            s_noisy = s + random.gauss(0.0, math.sqrt(var_s)) if var_s > 0.0 else s

            var_w = variance_rotational_velocity_w(self.v_cmd_default, steering_angle_command)
            w_noisy = w + random.gauss(0.0, math.sqrt(var_w)) if var_w > 0.0 else w
        else:
            s_noisy = s
            w_noisy = w

        # --- 4) exact constant-curvature integration ---
        x, y, th = self.state
        dt = float(delta_t)
        dth = w_noisy * dt

        if abs(dth) < DTH_EPS:
            # straight approximation
            x_new = x + s_noisy * math.cos(th)
            y_new = y + s_noisy * math.sin(th)
            th_new = th + dth
        else:
            R = s_noisy / dth
            th2 = th + dth
            x_new = x + R * (math.sin(th2) - math.sin(th))
            y_new = y - R * (math.cos(th2) - math.cos(th))
            th_new = th2

        self.state = [x_new, y_new, wrap_to_pi(th_new)]
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
        # Optional demo, not required for the lab
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

            # Fake encoder increments for illustration only
            encoder_counts += -50.0  # negative forward counts
            self.step_update(encoder_counts, steer, delta_t)
            t += delta_t

        return t_list, x_list, y_list, theta_list
