"""
    uv run python cnns/script-classify-on-window.py
"""
import cv2
import torch

from svhn_script_even_deepercnn import DeeperCNN
from svhn_script_digit_detector import DigitDetector


if torch.cuda.is_available():
    _device = 'cuda'
elif torch.backends.mps.is_available():
    _device = 'mps'
else:
    _device = 'cpu'
device = torch.device(_device)
print(f'{device=}')

halp = """
Press:
- q to see the next image
- n to skip to the next row
- p to increase the cv2.waitKey delay
- m to reset the cv2.waitKey delay
"""

def is_really_a_digit(probabilities):
    # Get top two probabilities
    top_probs, _ = torch.topk(probabilities, 2)
    
    # Calculate the ratio between top probability and runner-up
    ratio = top_probs[0] / (top_probs[1] + 1e-6)
    
    # Real digits typically have a much higher ratio
    # (Tune this threshold on your validation set)
    return ratio #> 3.0  # Example threshold

def clf_sliding_window(model, clf_model, img, delay=5, confidence_threshold=0.5):
    # Create a named window for the zoomed patch nd
    # resize the window (adjust size as needed).
    cv2.namedWindow("Current Window", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("Current Window", 620, 620)

    mean = torch.tensor([0.4377, 0.4438, 0.4728]).view(3, 1, 1).to(device)
    std = torch.tensor([0.1980, 0.2010, 0.1970]).view(3, 1, 1).to(device)

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
                
                # Normalize and convert to tensor.
                # Chaneg the dimensions:
                window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                window_tensor = torch.from_numpy(window_rgb.transpose(2, 0, 1)).float() / 255.0
                window_tensor = window_tensor.unsqueeze(0).to(device)
                window_tensor = (window_tensor - mean) / std

                # Run the classifier.
                # The output tensor has shape [batch_size, num_classes]
                # Dimension 0 is the batch dimension.
                # Dimension 1 is the class dimension.
                with torch.no_grad():
                    outputs = model(window_tensor)
                    print(f'-> {outputs=}')
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probabilities, 1)
                    confidence = confidence.item()
                    prediction = prediction.item()

                # 1 is for digits.
                is_digit = False
                if confidence > confidence_threshold and prediction == 1:
                    #ratio = is_really_a_digit(probabilities[0])
                    #print(f'{prediction=} {confidence=:.3f} -> {ratio=:.3f}')
                    print(f'{prediction=} {confidence=:.3f}')
                    is_digit = True

                    with torch.no_grad():
                        outputs = clf_model(window_tensor)
                        print(f'---> {outputs=}')
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, prediction = torch.max(probabilities, 1)
                        confidence = confidence.item()
                        prediction = prediction.item()
                        print(f'DIGIT PRED: {prediction=} {confidence=:.3f}')
                #print(f'> {prediction=} {confidence=:.3f}')


                # Draw rectangle.
                top_left = (x, y)
                bottom_right = (x+window_size, y+window_size)
                rect_color = (0, 255, 0) if is_digit else (0, 0, 255)
                cv2.rectangle(display_img, top_left, bottom_right, rect_color, 2)  # Green rectangle.

                # Show the image with current rectangle.
                cv2.imshow("Sliding Window", display_img)    
                cv2.imshow("Current Window", window)
                if is_digit:
                    delay = 1_000
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

    # Load the model.
    # Update the path.
    model_weights = './cnns/trained/deepcnn_model.pth'
    model = DeeperCNN()
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model = model.to(device) 
    model.eval()

    digit_model_weights = './cnns/trained/digit_model.pth'
    digit_model = DigitDetector()
    digit_model.load_state_dict(torch.load(digit_model_weights, map_location=device))
    digit_model = digit_model.to(device)
    digit_model.eval()

    # Load images to classify.
    # Update the path.
    img_path = './img/015-images/six.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    downsample_pyr = build_pyramid(img, levels=3)
    delays = [1, 20, 50]
    for pimg, delay in zip(downsample_pyr, delays):
        clf_sliding_window(digit_model, model, pimg, delay=delay)