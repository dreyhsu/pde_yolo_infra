import cv2
import os

# --- Configuration ---
IMAGE_PATH = os.path.join("images", "cable_1_frame_00076.jpg")
OUTPUT_TXT = "saved_rois.txt"

# --- Global Variables ---
ref_point = []       # Stores the two clicks for the current ROI
saved_rois = []      # Stores tuples of (x, y, w, h)
original_image = None
display_image = None

def redraw_image():
    """Redraws the image with all saved ROIs and the current active selection."""
    global display_image
    display_image = original_image.copy()
    
    # Draw all previously saved ROIs in Green
    for (x, y, w, h) in saved_rois:
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Draw the currently active selection in Red
    if len(ref_point) >= 1:
        cv2.circle(display_image, ref_point[0], 4, (0, 0, 255), -1)
    if len(ref_point) == 2:
        cv2.circle(display_image, ref_point[1], 4, (0, 0, 255), -1)
        cv2.rectangle(display_image, ref_point[0], ref_point[1], (0, 0, 255), 2)
        
    cv2.imshow("ROI Selector", display_image)

def mouse_callback(event, x, y, flags, param):
    global ref_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # If we already have 2 points, start a new selection
        if len(ref_point) >= 2:
            ref_point = []

        ref_point.append((x, y))
        redraw_image()

def main():
    global original_image, display_image, ref_point

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image not found at '{IMAGE_PATH}'")
        return

    original_image = cv2.imread(IMAGE_PATH)
    display_image = original_image.copy()

    cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Selector", mouse_callback)

    print("\n--- INSTRUCTIONS ---")
    print("1. Left Click TWO points to draw a red bounding box.")
    print("2. Press 's' to SAVE the current box (it will turn green).")
    print("3. Press 'r' to RESET/clear the current red box.")
    print("4. Press 'q' or 'ESC' to QUIT and write coordinates to file.")
    print("--------------------\n")

    redraw_image()

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Press 's' to save the current ROI
        if key == ord("s"):
            if len(ref_point) == 2:
                x1, y1 = ref_point[0]
                x2, y2 = ref_point[1]
                
                # Calculate x, y, w, h
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                if w > 0 and h > 0:
                    saved_rois.append((x, y, w, h))
                    print(f"✅ Saved ROI: x={x}, y={y}, w={w}, h={h}")
                    ref_point = [] # Reset for the next box
                    redraw_image()
                else:
                    print("⚠ Invalid ROI (width or height is 0).")
            else:
                print("⚠ Please select exactly two points before saving.")

        # Press 'r' to reset the current selection
        elif key == ord("r"):
            if len(ref_point) > 0:
                print("🔄 Current selection reset.")
                ref_point = []
                redraw_image()

        # Press 'q' or 'ESC' to quit
        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()

    # Save to text file
    if saved_rois:
        with open(OUTPUT_TXT, "w") as f:
            f.write("x,y,w,h\n")
            for (x, y, w, h) in saved_rois:
                f.write(f"{x},{y},{w},{h}\n")
        print(f"\n📁 Successfully saved {len(saved_rois)} ROIs to '{OUTPUT_TXT}'.")
    else:
        print("\nℹ No ROIs were saved.")

if __name__ == "__main__":
    main()