# Trajectory sequence: (speed_cmd, steering_angle, duration_sec)
# speed_cmd: Speed command to the robot (e.g., 80, 100).
# steering_angle: Steering angle in degrees (positive left, negative right).
# duration_sec: Duration to hold the command in seconds.
import numpy as np

TRAJECTORY_SEQUENCE = [
    (100, 0, 1.0),   # Straight for 2s
    (100, 10, 2.0),  # Left turn for 2s
    (80, -10, 2.0),  # Right turn for 2s
    (100, 0, 1.0)    # Straight for 2s
]

# Mapping from speed command to encoder counts per second
# These values should match your calibration data (or generate_traj.py)
# COUNTS_RATE_MAP = {
#     80: 1500, # Approx counts/sec for speed 80
#     100: 2000 # Approx counts/sec for speed 100
# }
# Cubic through-origin fit: r(v) = a1*v + a2*v^2 + a3*v^3  (counts/sec)
A1 = -2.38974830e+01
A2 =  8.13690476e-01
A3 = -4.65850340e-03

def counts_per_sec_from_speed(v_cmd, v_min=40.0, v_max=100.0):
    v = float(np.clip(v_cmd, v_min, v_max))
    r = A1*v + A2*(v**2) + A3*(v**3)
    return max(0.0, r)
