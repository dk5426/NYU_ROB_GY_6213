import numpy as np

def generate_random_trial(
    duration_s=12.0,
    dt=0.05,
    v_cmd=100,               # use 80 or 100 (matches your yaw-rate fit)
    steer_max=20.0,          # setpoint limits
    steer_rw_sigma=1.0,      # random-walk step (setpoint units) per timestep
    steer_slew_per_s=120.0,  # max steering change rate (setpoint units / second)
    counts_rate_map={80: 1500, 100: 2000},  # counts/second (tunable)
    counts_noise_frac=0.08,  # relative noise on counts increment
    seed=None
):
    """
    Returns:
      time_list (N,)
      encoder_counts_abs (N,)  absolute encoder count
      steering_setpoint (N,)   in [-steer_max, steer_max]
      v_cmd (scalar)
    """
    rng = np.random.default_rng(seed)

    N = int(np.floor(duration_s / dt)) + 1
    time_list = np.linspace(0.0, duration_s, N)

    # --- steering: smooth random walk with slew limit + clamp ---
    steer = np.zeros(N, dtype=float)
    slew_per_step = steer_slew_per_s * dt

    for i in range(1, N):
        # random walk proposal
        propose = steer[i-1] + rng.normal(0.0, steer_rw_sigma)

        # slew limit
        delta = np.clip(propose - steer[i-1], -slew_per_step, slew_per_step)
        steer[i] = steer[i-1] + delta

        # clamp
        steer[i] = np.clip(steer[i], -steer_max, steer_max)

    # --- encoder counts: negative increments for forward motion ---
    rate = counts_rate_map.get(int(v_cmd), 1800)  # counts/sec
    encoder_counts_abs = np.zeros(N, dtype=float)
    enc = 0.0
    encoder_counts_abs[0] = enc

    for i in range(1, N):
        mean_counts = rate * dt

        # noisy positive magnitude then negate (forward is negative delta_e)
        mag = mean_counts * (1.0 + rng.normal(0.0, counts_noise_frac))
        mag = max(0.0, mag)

        delta_e = -mag
        enc = enc + delta_e
        encoder_counts_abs[i] = enc

    return time_list.tolist(), encoder_counts_abs.tolist(), steer.tolist(), v_cmd
