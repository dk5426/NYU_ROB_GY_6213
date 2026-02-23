import time
import numpy as np
import parameters
import robot_python_code
from robot import Robot

def send_trajectory():
    # Initialize Robot and UDP Communication
    robot = Robot()
    
    print(f"Connecting to robot at {parameters.arduinoIP}:{parameters.arduinoPort}...")
    udp, udp_success = robot_python_code.create_udp_communication(
        parameters.arduinoIP, 
        parameters.localIP, 
        parameters.arduinoPort, 
        parameters.localPort, 
        parameters.bufferSize
    )
    
    if not udp_success:
        print("Failed to setup UDP communication. Exiting.")
        return

    robot.setup_udp_connection(udp)
    robot.connected_to_hardware = True

    # Define Trajectory (matching ekf_simulation.ipynb)
    # Format: (speed_cmd, steering_cmd, duration_sec)
    TRAJECTORY_SEQUENCE = [
        (100,  0,  1.0),   # Straight
        (100, 10,  2.0),   # Left turn
        ( 80,-10,  2.0),   # Right turn
        (100,  0,  1.0),   # Straight
    ]

    print("Starting trajectory execution with Online EKF monitoring...")
    print("Format: [Time] X: {:.3f}, Y: {:.3f}, Theta: {:.3f} | Cam: {}")
    
    # Capture initial pose for relative display
    mu_0 = np.copy(robot.extended_kalman_filter.state_mean)
    start_time_total = time.perf_counter()
    
    try:
        for speed, steering, duration in TRAJECTORY_SEQUENCE:
            print(f"\n--- Executing: Speed={speed}, Steering={steering}, Duration={duration}s ---")
            
            step_start_time = time.perf_counter()
            while time.perf_counter() - step_start_time < duration:
                loop_start = time.perf_counter()
                
                # 1. Update control signals
                robot.msg_sender.update_signals(speed, steering)
                robot.msg_sender.send_control_signal()
                
                # 2. Receive sensor data from robot
                robot.msg_receiver.receive_robot_sensor_signal()
                robot.robot_sensor_signal = robot.msg_receiver.robot_sensor_signal
                
                # 3. Get camera signal (if any)
                robot.camera_sensor_signal = robot.camera_sensor.get_signal(robot.camera_sensor_signal)
                
                # 4. Update EKF state estimate
                robot.update_state_estimate()
                
                # 5. Print current RELATIVE pose
                mu = robot.extended_kalman_filter.state_mean
                mu_rel = mu - mu_0
                # Wrap relative theta to [-pi, pi]
                mu_rel[2] = (mu_rel[2] + np.pi) % (2 * np.pi) - np.pi
                
                cam_seen = "YES" if (len(robot.camera_sensor_signal) >= 7 and any(robot.camera_sensor_signal[:3])) else "NO"
                elapsed = time.perf_counter() - start_time_total
                print(f"[{elapsed:6.2f}s] X: {mu_rel[0]:6.3f}, Y: {mu_rel[1]:6.3f}, Th: {mu_rel[2]:6.3f} | Cam: {cam_seen}", end='\r')
                
                # 6. Maintain loop rate (DT = 0.05s)
                loop_duration = time.perf_counter() - loop_start
                sleep_time = parameters.DT - loop_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Stop the robot
        print("\n\nTrajectory complete. Stopping robot.")
        for _ in range(5):
            robot.msg_sender.update_signals(0, 0)
            robot.msg_sender.send_control_signal()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted! Stopping robot.")
        for _ in range(5):
            robot.msg_sender.update_signals(0, 0)
            robot.msg_sender.send_control_signal()
            time.sleep(0.1)
    finally:
        robot.camera_sensor.close()

if __name__ == "__main__":
    send_trajectory()
