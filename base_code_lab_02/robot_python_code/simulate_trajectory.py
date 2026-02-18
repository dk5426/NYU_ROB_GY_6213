import math
import numpy as np
import matplotlib.pyplot as plt
from motion_models import MyMotionModel
from my_trajectory import TRAJECTORY_SEQUENCE, counts_per_sec_from_speed


def simulate_trajectory(num_trials=50):
    all_x = []
    all_y = []
    
    plt.figure(figsize=(10, 8))
    
    print(f"Running {num_trials} Monte Carlo simulations...")
    
    for i in range(num_trials):
        # Initialize motion model
        # State: [x, y, theta]
        # Start at origin (0,0,0) with 0 last encoder count
        # Default v_cmd is overridden in loop, but we need an initial one
        model = MyMotionModel([0, 0, 0], 0, v_cmd_default=100) 
        
        x_list = [0]
        y_list = [0]
        
        # Simulation loop
        current_encoder_counts = 0.0
        
        for speed, steering, duration in TRAJECTORY_SEQUENCE:
            # Set the speed command for the model (affects calibration)
            model.v_cmd_default = speed
            
            # Estimate counts/sec
            rate = counts_per_sec_from_speed(speed)

            
            # Discretize time
            dt = 0.1
            steps = int(duration / dt)
            
            for _ in range(steps):
                # Calculate encoder increment (negative for forward)
                # Add some noise to the counts if desired, but the model adds noise too
                # Here we just feed the mean counts rate to the model, and let the model add process noise
                d_counts = rate * dt
                
                # Update total encoder counts (simulated)
                # In the real robot, forward is negative counts
                current_encoder_counts -= d_counts 
                
                # Update model
                model.step_update(current_encoder_counts, -steering, dt)
                
                state = model.state
                x_list.append(state[0])
                y_list.append(state[1])
        
        # Store end point
        all_x.append(x_list[-1])
        all_y.append(y_list[-1])
        
        # Plot path (faintly)
        plt.plot(x_list, y_list, 'b-', alpha=0.1)

    # Calculate statistics
    mean_x = np.mean(all_x)
    mean_y = np.mean(all_y)
    cov = np.cov(all_x, all_y)
    
    print(f"Mean End Position: x={mean_x:.4f}, y={mean_y:.4f}")
    print(f"Covariance Matrix:\n{cov}")
    
    # Plot end points
    plt.scatter(all_x, all_y, c='r', marker='x', label='End Points')
    plt.plot(mean_x, mean_y, 'ko', label='Mean Position')
    
    plt.title("Simulated Robot Trajectories (Monte Carlo)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate_trajectory()
