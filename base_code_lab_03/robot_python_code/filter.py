import cv2
import numpy as np

def nothing(x):
    pass

def main():

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Camera not opening.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create window
    cv2.namedWindow("Camera Filter Tuner")

    # ------------------------
    # Hardware Sliders
    # ------------------------
    cv2.createTrackbar("Exposure", "Camera Filter Tuner", 7, 10, nothing)
    cv2.createTrackbar("Brightness", "Camera Filter Tuner", 50, 100, nothing)
    cv2.createTrackbar("Contrast", "Camera Filter Tuner", 64, 128, nothing)
    cv2.createTrackbar("Sharpness", "Camera Filter Tuner", 7, 15, nothing)

    # ------------------------
    # Software Sliders
    # ------------------------
    cv2.createTrackbar("CLAHE Clip x10", "Camera Filter Tuner", 20, 50, nothing)
    cv2.createTrackbar("Digital Beta", "Camera Filter Tuner", 50, 100, nothing)
    cv2.createTrackbar("Blur Kernel", "Camera Filter Tuner", 1, 10, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not captured.")
            break

        # Read sliders
        exposure = -cv2.getTrackbarPos("Exposure", "Camera Filter Tuner")
        brightness = cv2.getTrackbarPos("Brightness", "Camera Filter Tuner") - 50
        contrast = cv2.getTrackbarPos("Contrast", "Camera Filter Tuner")
        sharpness = cv2.getTrackbarPos("Sharpness", "Camera Filter Tuner")

        clahe_clip = max(1.0, cv2.getTrackbarPos("CLAHE Clip x10", "Camera Filter Tuner") / 10.0)
        beta = cv2.getTrackbarPos("Digital Beta", "Camera Filter Tuner") - 50
        blur_val = cv2.getTrackbarPos("Blur Kernel", "Camera Filter Tuner")

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Specular Highlight Suppression + Contrast Control ---

        gray_f = gray.astype(np.float32)

        # Detect saturated / reflective regions
        highlight_mask = gray_f > 220

        # Smooth only highlight areas to reduce glare bloom
        blurred = cv2.GaussianBlur(gray_f, (15, 15), 0)
        gray_f[highlight_mask] = blurred[highlight_mask]

        # Now apply software contrast and brightness
        alpha = max(0.1, contrast / 50.0)
        beta_soft = brightness + beta

        gray_f = gray_f * alpha + beta_soft
        gray_f = np.clip(gray_f, 0, 255)

        gray = gray_f.astype(np.uint8)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Gaussian blur (odd kernel only)
        if blur_val > 0:
            k = blur_val * 2 + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        # Convert back for display
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Show current values overlay
        cv2.putText(display, f"Exposure: {exposure}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(display, f"CLAHE: {clahe_clip:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Camera Filter Tuner", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("\n--- CURRENT SETTINGS ---")
            print(f"Exposure = {exposure}")
            print(f"Brightness = {brightness}")
            print(f"Contrast = {contrast}")
            print(f"Sharpness = {sharpness}")
            print(f"CLAHE ClipLimit = {clahe_clip}")
            print(f"Digital Beta = {beta}")
            print(f"Blur Kernel = {blur_val}")
            print("------------------------\n")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()