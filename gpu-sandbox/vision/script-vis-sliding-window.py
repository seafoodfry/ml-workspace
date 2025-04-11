"""
    uv run python script-vis-sliding-window.py
"""
import cv2

halp = """
Press:
- q to see the next image
- n to skip to the next row
- p to increase the cv2.waitKey delay
- m to reset the cv2.waitKey delay
"""

def vis_sliding_window(img, delay=5):
    # Create a named window for the zoomed patch nd
    # resize the window (adjust size as needed).
    cv2.namedWindow("Current Window", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("Current Window", 620, 620)

    original_img = img.copy()
    height, width = original_img.shape[:2]
    window_size = 32
    stride = 8
    original_delay = delay
    for y in range(0, height - window_size, stride):                
            for x in range(0, width - window_size, stride):
                # Start with a fresh copy for this iteration.
                display_img = original_img.copy()

                # Extract the window.
                window = img[y:y+window_size, x:x+window_size]   

                # Draw rectangle.
                top_left = (x, y)
                bottom_right = (x+window_size, y+window_size)
                cv2.rectangle(display_img, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle.

                # Show the image with current rectangle.
                cv2.imshow("Sliding Window", display_img)    
                cv2.imshow("Current Window", window)               
                key = cv2.waitKey(delay)
                if key == ord('q') or key == ord('n'):
                    break

                if key == ord('p'):
                    delay += 50
                if key == ord('m'):
                    delay = original_delay

            # If the key was pressed, then exit out of here too.
            if key == ord('q'):
                    break
    
    cv2.destroyAllWindows()

def build_pyramid(image, levels=3):
    """
    Build an image pyramid for hierarchical LK
    Returns a list of images, each half the size of the previous

    See https://github.com/seafoodfry/ml-workspace/blob/main/gpu-sandbox/vision/009-lk-gaussian-pyramids.ipynb
    """
    img = image.copy()
    pyramid = [img]
    for _ in range(levels-1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

if __name__ == "__main__":
    print(halp)

    img_path = './cnns/img/buildings-finetune/img-01.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    downsample_pyr = build_pyramid(img, levels=3)

    delays = [1, 2, 3]
    for pimg, delay in zip(downsample_pyr, delays):
        vis_sliding_window(pimg, delay=delay)