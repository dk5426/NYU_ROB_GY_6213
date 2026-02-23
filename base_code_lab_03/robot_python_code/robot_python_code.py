# External libraries
import serial
import time
import pickle
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import socket
from time import strftime

# Local libraries
import parameters


# Function to try to connect to the robot via udp over wifi
def create_udp_communication(arduinoIP, localIP, arduinoPort, localPort, bufferSize):
    try:
        udp = UDPCommunication(arduinoIP, localIP, arduinoPort, localPort, bufferSize)
        print("Success in creating udp communication")
        return udp, True
    except:
        print("Failed to create udp communication!")
        return _, False
        
        
# Class to hold the UPD over wifi connection setup
class UDPCommunication:
    def __init__(self, arduinoIP, localIP, arduinoPort, localPort, bufferSize):
        self.arduinoIP = arduinoIP
        self.arduinoPort = arduinoPort
        self.localIP = localIP
        self.localPort = localPort
        self.bufferSize = bufferSize
        self.UDPServerSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
        self.UDPServerSocket.bind((localIP, localPort))
        
    # Receive a message from the robot
    def receive_msg(self):
        bytesAddressPair = self.UDPServerSocket.recvfrom(self.bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]
        clientMsg = "{}".format(message.decode())
        clientIP = "{}".format(address)
        
        return clientMsg
       
    # Send a message to the robot
    def send_msg(self, msg):
        bytesToSend = str.encode(msg)
        self.UDPServerSocket.sendto(bytesToSend, (self.arduinoIP, self.arduinoPort))


# Class to hold the data logger that records data when needed
class DataLogger:

    # Constructor
    def __init__(self, filename_start, data_name_list):
        self.filename_start = filename_start
        self.filename = filename_start
        self.line_count = 0
        self.dictionary = {}
        self.data_name_list = data_name_list
        for name in data_name_list:
            self.dictionary[name] = []
        self.currently_logging = False

    # Open the log file
    def reset_logfile(self, control_signal):
        self.filename = self.filename_start + "_"+str(control_signal[0])+"_"+str(control_signal[1]) + strftime("_%d_%m_%y_%H_%M_%S.pkl")
        self.dictionary = {}
        for name in self.data_name_list:
            self.dictionary[name] = []

        
    # Log one time step of data
    def log(self, logging_switch_on, time, control_signal, robot_sensor_signal, camera_sensor_signal, state_mean, state_covariance):
        if not logging_switch_on:
            if self.currently_logging:
                self.currently_logging = False
        else:
            if not self.currently_logging:
                self.currently_logging = True
                self.reset_logfile(control_signal)

        if self.currently_logging:
            self.dictionary['time'].append(time)
            self.dictionary['control_signal'].append(control_signal)
            self.dictionary['robot_sensor_signal'].append(robot_sensor_signal)
            self.dictionary['camera_sensor_signal'].append(camera_sensor_signal)
            self.dictionary['state_mean'].append(state_mean)
            self.dictionary['state_covariance'].append(state_covariance)

            self.line_count += 1
            if self.line_count > parameters.max_num_lines_before_write:
                self.line_count = 0
                with open(self.filename, 'wb') as file_handle:
                    pickle.dump(self.dictionary, file_handle)

# Utility for loading saved data
class DataLoader:

    # Constructor
    def __init__(self, filename):
        self.filename = filename
        
    # Load a dictionary from file.
    def load(self):
        with open(self.filename, 'rb') as file_handle:
            loaded_dict = pickle.load(file_handle)
        return loaded_dict

# Class to hold a message sender
class MsgSender:

    # Time step size between message to robot sends, in seconds
    delta_send_time = 0.1

    # Constructor
    def __init__(self, last_send_time, msg_size, udp_communication):
        self.last_send_time = last_send_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
        
    # Pack and send a control signal to the robot.
    def send_control_signal(self, control_signal):
        packed_send_msg = self.pack_msg(control_signal)
        self.send(packed_send_msg)
    
    # If its time, send the control signal to the robot.
    def send(self, msg):
        new_send_time = time.perf_counter()
        if new_send_time - self.last_send_time > self.delta_send_time:
            message = ""
            for data in msg:
                message = message + str(data)
            self.udp_communication.send_msg(message)
            self.last_send_time = new_send_time
      
    # Pack a message so it is in the correct format for the robot to receive it.
    def pack_msg(self, msg):
        packed_msg = ""
        for data in msg:
            if packed_msg == "":
                packed_msg = packed_msg + str(data)
            else:
                packed_msg = packed_msg + ", "+ str(data)
        packed_msg = packed_msg + "\n"
        return packed_msg
        
# The robot's message receiver
class MsgReceiver:

    # Determines how often to look for incoming data from the robot.
    delta_receive_time = 0.05

    # Constructor
    def __init__(self, last_receive_time, msg_size, udp_communication):
        self.last_receive_time = last_receive_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
      
    # Check if its time to look for a new message from the robot.
    def receive(self):
        new_receive_time = time.perf_counter()
        if new_receive_time - self.last_receive_time > self.delta_receive_time:
            received_msg = self.udp_communication.receive_msg()
            self.last_receive_time = new_receive_time
            return True, received_msg
            
        return False, ""
    
    # Given a new message, put it in a digestable format
    def unpack_msg(self, packed_msg):
        unpacked_msg = []
        msg_list = packed_msg.split(',')
        if len(msg_list) >= self.msg_size:
            for data in msg_list:
                unpacked_msg.append(float(data))
            return True, unpacked_msg

        return False, unpacked_msg
        
    # Check for new message and unpack it if there is one.
    def receive_robot_sensor_signal(self, last_robot_sensor_signal):
        robot_sensor_signal = last_robot_sensor_signal
        receive_ret, packed_receive_msg = self.receive()
        if receive_ret:
            unpack_ret, unpacked_receive_msg = self.unpack_msg(packed_receive_msg)
            if unpack_ret:
                robot_sensor_signal = RobotSensorSignal(unpacked_receive_msg)
            
        return robot_sensor_signal

# Class to hold a camera sensor data. Not needed for lab 1.
class CameraSensor:

    # Constructor
    def __init__(self, camera_id):
        self.camera_id = camera_id
        # Use AVFOUNDATION for Mac
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        
        # Set resolution to 1280x720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Hardware Camera Controls (from cali_check.py)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 64.0)
        self.cap.set(cv2.CAP_PROP_SHARPNESS, 7.0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Manual
        self.cap.set(cv2.CAP_PROP_EXPOSURE, parameters.EXPOSURE_VAL)
        
        # Configure Aruco detector
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
        self.detector_params = aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # Define 3D object points for the marker (centered at origin, in meters)
        half_length = parameters.marker_length / 2.0
        self.obj_points = np.array([
            [-half_length,  half_length, 0],
            [ half_length,  half_length, 0],
            [ half_length, -half_length, 0],
            [-half_length, -half_length, 0]
        ], dtype=np.float32)
        
        self.prev_z = None # for temporal smoothing

    # Get a new pose estimate from a camera image
    def get_signal(self, last_camera_signal):
        ret_estimate, camera_sensor_signal_new = self.get_pose_estimate()
        if ret_estimate:
            return camera_sensor_signal_new
        else:
            return [0.0] * 7
        
    # If there is a new image, calculate a pose estimate from the fiducial tag on the robot.
    def get_pose_estimate(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, []
        
        # --- Software Filter (from aruco_test.py) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Software contrast & brightness
        alpha = max(0.1, parameters.CONTRAST_VAL / 50.0)
        gray_f = gray.astype(np.float32)
        gray_f = gray_f * alpha + parameters.BETA_VAL
        gray_f = np.clip(gray_f, 0, 255)
        gray = gray_f.astype(np.uint8)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=parameters.CLAHE_VAL, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is not None:
            for i in range(len(ids)):
                if ids[i][0] == parameters.TARGET_MARKER_ID:
                    # Estimate pose using SQPNP (Stable Depth)
                    ret_pnp, rvec, tvec = cv2.solvePnP(
                        self.obj_points, corners[i][0], 
                        parameters.camera_matrix, parameters.dist_coeffs, 
                        flags=cv2.SOLVEPNP_SQPNP
                    )
                    
                    if ret_pnp:
                        # Extract translation (in meters for EKF)
                        tx = tvec[0][0]
                        ty = tvec[1][0]
                        tz = tvec[2][0]
                        
                        # Z temporal smoothing
                        if self.prev_z is not None:
                            tz = 0.75 * self.prev_z + 0.25 * tz
                        self.prev_z = tz
                        
                        # Extract the rotation matrix to find the yaw (theta)
                        R, _ = cv2.Rodrigues(rvec)
                        # raw theta is rotation of marker's X around camera's Z
                        theta_raw = np.arctan2(R[1, 0], R[0, 0])
                        
                        # Apply 180-degree offset (0 points to -x in original)
                        # theta_ekf = wrap(theta_cam + pi)
                        theta_aligned = (theta_raw + np.pi + np.pi) % (2 * np.pi) - np.pi
                        
                        # Return in format expected by EKF: [tx, ty, tz, rx, ry, rz, theta]
                        return True, [tx, ty, tz, float(rvec[0][0]), float(rvec[1][0]), float(rvec[2][0]), theta_aligned]
        
        return False, []
    
    # Close the camera stream
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


# A storage vessel for an instance of a robot signal
class RobotSensorSignal:

    # Constructor
    def __init__(self, unpacked_msg):
        self.encoder_counts = int(unpacked_msg[0])
        self.steering = int(unpacked_msg[1])
        self.num_lidar_rays = int(unpacked_msg[2])
        self.angles = []
        self.distances = []
        for i in range(self.num_lidar_rays):
            index = 3 + i*2
            self.angles.append(unpacked_msg[index])
            self.distances.append(unpacked_msg[index+1])
    
    # Print the robot sensor signal contents.
    def print(self):
        print("Robot Sensor Signal")
        print(" encoder: ", self.encoder_counts)
        print(" steering:" , self.steering)
        print(" num_lidar_rays: ", self.num_lidar_rays)
        print(" angles: ",self.angles)
        print(" distances: ", self.distances)
    
    # Convert the sensor signal to a list of ints and floats.
    def to_list(self):
        sensor_data_list = []
        sensor_data_list.append(self.encoder_counts)
        sensor_data_list.append(self.steering)
        sensor_data_list.append(self.num_lidar_rays)
        for i in range(self.num_lidar_rays):
            sensor_data_list.append(self.angles[i])
            sensor_data_list.append(self.distances[i])
        
        return sensor_data_list
