import cv2
import os
import sys
import ctypes

# --- Configuration ---
OUTPUT_FOLDER = 'image/hph_hw3'
IMAGE_FORMAT = 'jpg'
VIDEO_PATH = 'clip/hph_03.mp4'
FRAME_INTERVAL = 5  # Extract every Nth frame (1 = every frame, 3 = every 3rd frame, etc.)

# Global variables for mouse callback
ref_point = []
cropping = False
drawing_frame = None
original_frame = None
current_blocks = []

def create_resizable_window(window_name, width, height):
    """Creates a resizable OpenCV window that fits within the screen resolution."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Default fallback resolution
    screen_w, screen_h = 1920, 1080
    
    try:
        if sys.platform == 'win32':
            user32 = ctypes.windll.user32
            # Set process DPI aware to get actual resolution
            user32.SetProcessDPIAware()
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
    except Exception:
        pass

    # Scale to fit (e.g., 85% of screen)
    scale = min(0.85 * screen_w / width, 0.85 * screen_h / height, 1.0)
    
    new_w = int(width * scale)
    new_h = int(height * scale)
    
    cv2.resizeWindow(window_name, new_w, new_h)

def mouse_callback(event, x, y, flags, param):
    global ref_point, drawing_frame

    # Left click: Add a point
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ref_point) >= 2:
            # Reset if user clicks a 3rd time
            ref_point = []
            drawing_frame = original_frame.copy()

        ref_point.append((x, y))
        
        # Visual feedback
        # 1. Draw a green dot where user clicked
        cv2.circle(drawing_frame, (x, y), 5, (0, 255, 0), -1)

        # 2. If we have 2 points, draw the red box
        if len(ref_point) == 2:
            cv2.rectangle(drawing_frame, ref_point[0], ref_point[1], (0, 0, 255), 2)
            print(f"   Selected region: {ref_point[0]} to {ref_point[1]}")
            print("   Press SPACE to confirm, or click again to reset.")

        cv2.imshow("Select Crop Area", drawing_frame)

def get_crop_coordinates(frame):
    global drawing_frame, original_frame, ref_point
    
    original_frame = frame.copy()
    drawing_frame = frame.copy()
    ref_point = []
    
    h, w = frame.shape[:2]
    create_resizable_window("Select Crop Area", w, h)
    cv2.setMouseCallback("Select Crop Area", mouse_callback)

    print("\n--- INSTRUCTIONS ---")
    print("1. Click the TOP-LEFT corner.")
    print("2. Click the BOTTOM-RIGHT corner.")
    print("   (You will see green dots and a red box)")
    print("3. Press SPACE or ENTER to confirm.")
    print("4. Press 'p' to pass (skip crop & block, use full image).")
    print("5. Press 'q' or 'c' to quit without saving.")
    print("--------------------")

    while True:
        cv2.imshow("Select Crop Area", drawing_frame)
        key = cv2.waitKey(1) & 0xFF

        # Press 'Space' or 'Enter' to confirm
        if key == 32 or key == 13:
            if len(ref_point) == 2:
                break
            else:
                print("⚠ Please click TWO points to define the box first.")

        # Press 'p' to pass (full image, skip blocking)
        elif key == ord("p"):
            cv2.destroyAllWindows()
            print("Skipping crop and block selection. Using full frame...")
            return (0, 0, w, h, True) # Return full dimensions and skip_block=True

        # Press 'q' or 'c' to quit
        elif key == ord("q") or key == ord("c"):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    # Calculate final (x, y, w, h) handling user clicking in any direction
    p1, p2 = ref_point
    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    x2 = max(p1[0], p2[0])
    y2 = max(p1[1], p2[1])
    
    return (x, y, x2-x, y2-y, False) # Return coords and skip_block=False

def block_mouse_callback(event, x, y, flags, param):
    global ref_point, drawing_frame, current_blocks

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point.append((x, y))
        cv2.circle(drawing_frame, (x, y), 5, (0, 255, 0), -1)

        if len(ref_point) == 2:
            # Draw filled brown box (42, 42, 165)
            cv2.rectangle(drawing_frame, ref_point[0], ref_point[1], (255, 255, 255), -1)
            current_blocks.append((ref_point[0], ref_point[1]))
            ref_point = [] # Reset for next block
            print(f"   Block added: {current_blocks[-1]}")
            
        cv2.imshow("Select Blocking Areas", drawing_frame)

def get_block_coordinates(frame):
    global drawing_frame, original_frame, ref_point, current_blocks
    
    original_frame = frame.copy()
    drawing_frame = frame.copy()
    ref_point = []
    current_blocks = [] 
    
    h, w = frame.shape[:2]
    create_resizable_window("Select Blocking Areas", w, h)
    cv2.setMouseCallback("Select Blocking Areas", block_mouse_callback)

    print("\n--- BLOCKING MODE ---")
    print("1. Click two points to define a block (Brown box).")
    print("2. Repeat for multiple blocks.")
    print("3. Press SPACE to confirm and finish.")
    print("---------------------")

    while True:
        cv2.imshow("Select Blocking Areas", drawing_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32 or key == 13: # Space or Enter
            break
        elif key == ord("q"):
             current_blocks = []
             break

    cv2.destroyAllWindows()
    return current_blocks

def main():
    video_path = VIDEO_PATH

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video.")
        return

    # --- Step 1: Manual Selection ---
    roi_result = get_crop_coordinates(frame)
    
    if roi_result is None:
        print("Operation cancelled.")
        return

    # Unpack the returned variables (including our new skip_block flag)
    x, y, w, h, skip_block = roi_result
    
    # --- SAFETY CHECK: Prevent the crash ---
    if w <= 0 or h <= 0:
        print("❌ Error: Invalid selection (Width or Height is 0).")
        print("Please make sure you click two different points.")
        return

    print(f"\nProcessing area: x={x}, y={y}, w={w}, h={h}")

    blocks = []
    
    # Only prompt for blocking if the user didn't press 'p'
    if not skip_block:
        # --- Step 1.5: Blocking Selection (Optional) ---
        print("Press 'b' to add blocking regions, or any other key to start processing...")
        
        h_orig, w_orig = frame.shape[:2]
        create_resizable_window("Confirm Selection", w_orig, h_orig)
        # Draw the crop rect for visualization on a copy
        preview = frame.copy()
        cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Confirm Selection", preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('b'):
            blocks = get_block_coordinates(frame)

    # --- Step 2: Process Video ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind
    video_frame_idx = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames based on interval
        if video_frame_idx % FRAME_INTERVAL != 0:
            video_frame_idx += 1
            continue

        # Apply blocks
        for b in blocks:
            # b is ((x1, y1), (x2, y2))
            cv2.rectangle(frame, b[0], b[1], (0, 0, 0), -1)

        # Crop
        cropped_frame = frame[y:y+h, x:x+w]

        # Double check crop is valid before saving
        if cropped_frame.size == 0:
            video_frame_idx += 1
            continue

        out_name = f"{OUTPUT_FOLDER.split('/')[-1]}_frame_{video_frame_idx:05d}.{IMAGE_FORMAT}"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)

        cv2.imwrite(out_path, cropped_frame)
        saved_count += 1
        video_frame_idx += 1
        
        if saved_count % 50 == 0:
            print(f"Saved {saved_count} frames (current frame: {video_frame_idx}/{total_frames})...", end='\r')

    print(f"\n✅ Done! Saved {saved_count} frames to '{OUTPUT_FOLDER}/'")
    cap.release()

if __name__ == "__main__":
    main()