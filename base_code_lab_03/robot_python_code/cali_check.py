import cv2
import numpy as np
import parameters
import glob
import os

# Physical properties of the checkerboard (Ignore the Aruco markers inside)
# Note: For OpenCV findChessboardCorners, we count INNER corners. 
# A 9x12 square board has 8x11 inner corners! But if your board has 10x13 squares, it will have 9x12 inner corners.
# Start with what you specified and we will adjust if it doesn't find it.
CHECKERBOARD = (6, 8)  
SQUARE_SIZE = 0.026     # 30mm squares

# Termination criteria for subpixel corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def main():
    print("==========================================================")
    print("            INTERACTIVE CAMERA CALIBRATION                ")
    print("==========================================================\n")
    print("1. Hold the checkerboard clearly in view of the camera.")
    print(f"2. When OpenCV sees all {CHECKERBOARD[0]}x{CHECKERBOARD[1]} internal corners, it will ")
    print("   draw a colorful grid on the board.")
    print("3. Press [SPACE] to capture that frame!")
    print("4. Move the board to a different angle/edge, and repeat.")
    print("5. Once you have captured 20+ frames, press [c] to Calibrate.")
    print("6. Press [q] to Quit.\n")

    # Prepare standard 3D object points based on the square size
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    # Arrays to store object points and image points from all the captured frames
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    
    captured_count = 0

    # Explicitly load DSHOW and 720p to bypass the 1080p hardware sensor crop
    cap = cv2.VideoCapture(parameters.camera_id, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Optimize hardware camera properties for black/white corner detection
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)    # Lock focal length physically 
    cap.set(cv2.CAP_PROP_CONTRAST, 64.0)  # Bump contrast up (default was 32) so white paper is distinct
    cap.set(cv2.CAP_PROP_SHARPNESS, 7.0)  # Bump sharpness up (default was 4) for sub-pixel math

    # Force lower brightness / exposure to reduce glare
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual exposure mode (mac compatible)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)          # try range -5 to -9 if needed
    cap.set(cv2.CAP_PROP_BRIGHTNESS, -20)       # reduce brightness

    cv2.namedWindow("Interactive Calibrator")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab physical camera frame.")
                break

            # --- Fixed Filter Configuration ---
            exposure_value = 0
            brightness_value = -50
            contrast_value = 22
            clahe_value = 1.6
            beta_value = -16
            blur_kernel = 0

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Software contrast & brightness
            alpha = max(0.1, contrast_value / 50.0)
            gray = gray.astype(np.float32)
            gray = gray * alpha + beta_value
            gray = np.clip(gray, 0, 255)
            gray = gray.astype(np.uint8)

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=clahe_value, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Optional blur
            if blur_kernel > 0:
                k = blur_kernel * 2 + 1
                gray = cv2.GaussianBlur(gray, (k, k), 0)

            # Use the more stable SB (Symmetric Board) detector
            ret_corners, corners = cv2.findChessboardCornersSB(
                gray,
                CHECKERBOARD,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            # Show grayscale feed instead of color
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if ret_corners:
                # Apply light temporal smoothing to reduce jitter
                corners2 = corners.copy()

                if hasattr(main, "prev_corners"):
                    corners2 = 0.7 * main.prev_corners + 0.3 * corners2

                main.prev_corners = corners2
                
                # Draw the colorful grid
                cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners2, ret_corners)
                cv2.putText(display_frame, "BOARD DETECTED - PRESS [SPACE] TO CAPTURE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, f"Searching for full {CHECKERBOARD[0]}x{CHECKERBOARD[1]} internal corners...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Overlay capture count summary
            cv2.putText(display_frame, f"Captured: {captured_count}/20+", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            cv2.putText(display_frame, "Press 'c' when ready to calculate calibration.", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Shrink the view just for the laptop display
            preview = cv2.resize(display_frame, (int(1920/2), int(1080/2)))
            cv2.imshow("Interactive Calibrator", preview)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                # Only save if OpenCV successfully mapped the corners
                if ret_corners:
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    captured_count += 1
                    print(f"Captured Grid {captured_count} successfully! Move board to a new angle...")
                    
                    # Flash the screen green to indicate a capture
                    flash = np.zeros_like(preview)
                    flash[:] = (0, 255, 0)
                    cv2.imshow("Interactive Calibrator", flash)
                    cv2.waitKey(200)
                else:
                    print("Cannot capture! Make sure the entire checkerboard runs clear to the edges.")
                    
            elif key == ord('c'):
                if captured_count < 10:
                    print(f"You only have {captured_count} images. OpenCV needs at least 10 (ideally 20+) to calibrate accurately. Keep capturing!")
                else:
                    print(f"\n--- Starting Calibration on {captured_count} captured frames ---")
                    print("Running math, please wait...\n")
                    
                    # Seed the math with a mathematically sane guess for a standard 1280x720 webcam.
                    # This prevents the solver from spiralling into insane focal lengths (like 16000)
                    # if the user's hand movements lacked severe pitch/yaw angles.
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    guess_mtx = np.array([
                        [800.0, 0, width / 2.0],
                        [0, 800.0, height / 2.0],
                        [0, 0, 1.0]
                    ], dtype=np.float32)
                    
                    calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO

                    # Math time - Compute the camera matrix and dist_coeffs!
                    ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                        objpoints, imgpoints, gray.shape[::-1], guess_mtx, None, flags=calib_flags
                    )
                    
                    print("\n=======================================================")
                    print(f"                   CALIBRATION SUCCESSFUL!")
                    print(f"                   Reprojection Error: {ret_calib:.4f}")
                    print("=======================================================\n")
                    
                    print("Copy paste these NEW values directly into parameters.py:\n")
                    
                    # Format camera matrix natively
                    print("camera_matrix = np.array([")
                    for row in mtx:
                        print(f"    [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}],")
                    print("], dtype=np.float32)\n")
                    
                    # Format dist coeffs natively
                    print("dist_coeffs = np.array([")
                    d = dist[0]
                    print(f"    [{d[0]:.8f}, {d[1]:.8f}, {d[2]:.8f}, {d[3]:.8f}, {d[4]:.8f}]")
                    print("], dtype=np.float32)")
                    
                    print("\n=======================================================\n")
                    
            elif key == ord('s'):
                print("\n--- CURRENT SLIDER SETTINGS ---")
                print(f"Exposure: {exposure_value}")
                print(f"Brightness: {brightness_value}")
                print(f"CLAHE ClipLimit: {clahe_value}")
                print(f"Digital Beta: {beta_value}")
                print("--------------------------------\n")

            elif key == ord('q'):
                print("Quitting Interactive Calibrator.")
                break

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()