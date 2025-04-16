"""
    uv run python cnns/script-classify-on-window.py
"""
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

import torchvision.models as models

from finetune_model import CNN


device = torch.device('cpu')


class VGG16Transfer(torch.nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()

        # Load pretrained VGG16.
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Freeze all feature extraction layers.
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
            
        # Replace the final layer.
        # VGG16's last layer has 4096 inputs and 1000 outputs (ImageNet classes).
        # We modify it to output our 11 classes (0-9 digits + class 10 for non-digits).
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = torch.nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.vgg16(x)


def dedup_boxes(filtered_bboxes, distance_threshold=3):
    # MSER will return a TON of duplicates.
    # So let's do a simple dedup first.
    unique_boxes = []
    seen = set()
    for box in filtered_bboxes:
        # np.ndarrays are not hashable, so we gotta make tuples out of them to dedup them.
        box_tuple = tuple(box)
        if box_tuple not in seen:
            seen.add(box_tuple)
            unique_boxes.append(box)

    print(f'unique bboxes: {len(unique_boxes)}')

    # Sort the dedup'd bboxes by area, highest to lowest.
    # We are doing this because we will give preferential treatment to larger bounding boxes
    # in the next step.
    sorted_boxes = sorted(unique_boxes, key=lambda box: box[2] * box[3], reverse=True)

    # Now we will dedup based on how close the bounding boxes are.
    deduped = {}

    # Can't use two values for a dict key so let's make something up.
    # The thing we will do is that we will favour the biggest bbox and we will use it
    # as "representative" of any other bboxes that were kinda close.
    # But we store the first bbox into the dict right away 'cas we need a non-empty
    # dict in order to have the next for loop work (can't add to the dict by iterating
    # through its items if there are no items to begin).
    dict_key = f'{sorted_boxes[0][0]},{sorted_boxes[0][1]}'
    deduped[dict_key] = sorted_boxes[0]

    for box in sorted_boxes:
        x, y, w, h = box
        dict_key = f'{x},{y}'
        
        # Create a list of existing keys to avoid modifying during iteration.
        # Can't (shouldn't) modify a dict while iterating through it!
        is_duplicate = False
        existing_keys = list(deduped.keys())
        for existing_key in existing_keys:
            other_box = deduped[existing_key]
            x2, y2, w2, h2 = other_box
            
            # Calculate distance between centers.
            dist = np.sqrt(np.square(x-x2) + np.square(y-y2))
            if dist < distance_threshold:
                # If the bbox is "too close" then we consider it a dup.
                # so check out the next bbox.
                is_duplicate = True
                break
        
        # If we made it here, then the distance of this bbox is far enough apart
        # from all others.
        if not is_duplicate:
            deduped[dict_key] = box

    print(f'unique bboxes that are far enough from oneanother: {len(unique_boxes)}')

    # One more deduping step!
    # This time sort based on x coords.
    deduped_boxes = sorted(list(deduped.values()), key=lambda box: box[1], reverse=True)

    # Find groups of bboxes.
    # ASSUMPTION: We want to find the group w/ the most members.
    # The most member are prob digits.
    groups = []
    current_group = [deduped_boxes[0]]
    max_distance = 150
    for i in range(1, len(deduped_boxes)):
        curr_x, curr_y = deduped_boxes[i][0], deduped_boxes[i][1]
        prev_x, prev_y = deduped_boxes[i-1][0], deduped_boxes[i-1][1]
        
        # Euclidean distance between this digit and the previous one
        distance = np.sqrt(np.square(curr_x - prev_x) + np.square(curr_y - prev_y))
        if distance <= max_distance:
            # Close enough to be part of the same number.
            current_group.append(deduped_boxes[i])
        else:
            # Too far, start a new group.
            groups.append(current_group)
            current_group = [deduped_boxes[i]]
    
    # Don't forget the last group.
    if current_group:
        groups.append(current_group)

    # Return the group with the most bounding boxes (likely to be the digits).
    largest_group = max(groups, key=len)

    print(f'unique bboxes deduped by which members are more clustered: {len(largest_group)}')

    # Let's add one more deduping step based on area: remove any boxes that are too different in size.
    largest_group = sorted(largest_group, key=lambda box: box[2] * box[3], reverse=True)
    comparison_box = largest_group[0]
    comparison_box_area = comparison_box[-2] * comparison_box[-1]
    deduped_by_area = []
    for box in largest_group:
        box_area = box[-2] * box[-1]
        diff = np.abs(box_area - comparison_box_area)
        tolerance = 36**2
        print(f'{box_area} - {comparison_box_area} = {diff} < {tolerance}')
        if np.abs(box_area - comparison_box_area) < tolerance:
            deduped_by_area.append( box )

    print(f'unique bboxes deduped by area: {len(deduped_by_area)}')

    # One more dedup based on building numbers not being that long.
    return deduped_by_area[:4]


def resize_image_by_factor(image, scale_factor):
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)

    # Choose appropriate interpolation method based on scaling direction.
    if scale_factor < 1:  # Downsampling
        interp = cv2.INTER_AREA  # Better for reducing image size
    else:  # Upsampling
        interp = cv2.INTER_LINEAR  # Better for increasing image size
    
    #image = cv2.bilateralFilter(image, 9, 15, 15)
    #image = cv2.filter2D(image, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
    # Calculate appropriate sigma based on downsampling factor.
    # Rule of thumb: sigma = factor/2
    sigma = scale_factor/2
    
    # Kernel size should be odd and roughly 6*sigma.
    # 3 sigma deviations on each side capture 99.7% of the "energy".
    kernel_size = int(6*sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # Apply Gaussian blur.
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    return cv2.resize(image, (new_width, new_height), interpolation=interp)


def detect_digits(clf_model, img, idx, scale_factor=1, save_dir = './tmp-mser', vgg=False):
    mser = cv2.MSER_create(
        delta=10,
        min_area=20*20,
        max_area=32*32,
    )
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 5, 35, 35)
    regions, bboxes = mser.detectRegions(img_gray)
    print(f'MSER found {len(bboxes)} initial regions')

    # Filter regions based on properties that are likely to be digits.
    img_copy = img.copy()
    filtered_bboxes = []
    for i, region in enumerate(regions):
        x, y, w, h = bboxes[i]
        aspect_ratio = w / h
        if 0.35 < aspect_ratio < 1.5:
            filtered_bboxes.append(bboxes[i])
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)

    print(f'Number of bounding boxes after filtering based on aspect ratio: {len(filtered_bboxes)}')
    unique_bboxes = dedup_boxes(filtered_bboxes, distance_threshold=20*scale_factor)
    print(f'Final number of bounding boxes: {len(unique_bboxes)}')


    mean = torch.tensor([0.4377, 0.4438, 0.4728]).view(3, 1, 1).to(device)
    std = torch.tensor([0.1980, 0.2010, 0.1970]).view(3, 1, 1).to(device)
    last_match = -1 # canary value.
    for i, box in enumerate(unique_bboxes):
        vis = img_copy.copy()

        x, y, w, h = box
        aspect_ratio = w / h
        area = h * w
        to_window_ratio = area/(32**2)
        print(f'{aspect_ratio=}, {box=}, {to_window_ratio=}')
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Extract the window.
        offset = 0 if max(h, w) > 100 else 5
        y_min = y - offset
        y_max = y + h + offset
        x_min = x - offset
        x_max = x + w + offset
        window_orig = img[y_min:y_max, x_min:x_max]
        #filtered_window = cv2.bilateralFilter(window_orig, d=5, sigmaColor=75, sigmaSpace=75)
        if vgg:
            resize_window = (224, 224)
        else:
            resize_window = (32, 32)
        window = cv2.resize(window_orig, resize_window, interpolation=cv2.INTER_AREA if max(w, h) > 32 else cv2.INTER_LINEAR)

        
        # Normalize and convert to tensor.
        window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
        window_tensor = torch.from_numpy(window_rgb.transpose(2, 0, 1)).float() / 255.0
        window_tensor = window_tensor.unsqueeze(0).to(device)
        window_tensor = (window_tensor - mean) / std

        # Run the digit detector.
        # The output tensor has shape [batch_size, num_classes]
        # Dimension 0 is the batch dimension.
        # Dimension 1 is the class dimension.
        with torch.no_grad():
            outputs = clf_model(window_tensor)
            #print(f'-> {outputs=}')
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            confidence = confidence.item()
            prediction = prediction.item()

        # Update the canary value
        last_match = prediction

        plt.figure(figsize=(12, 8))
        plt.subplot(223)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Digit: {prediction}')

        plt.subplot(221)
        plt.imshow(cv2.cvtColor(window_orig, cv2.COLOR_BGR2RGB))
        plt.title(f'Window view')

        plt.subplot(222)
        plt.imshow(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
        plt.title(f'Resized window view')

        # Plot the probability distribution.
        plt.subplot(224)
        probs_np = probabilities[0].cpu().numpy()
        plt.bar(range(11), probs_np)
        plt.xticks(range(11))
        plt.xlabel('Digit Class')
        plt.ylabel('Probability')
        plt.title(f'Probability Distribution')

        # Add text annotation with location and confidence
        plt.figtext(0.5, 0.01, f'Position: ({x}, {y}), confidence: {confidence:.6f}', ha='center', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{idx}_digit{i}_pred{prediction}_x{x}_y{y}.png')
        plt.savefig(save_path)
        plt.close()

    return last_match == 10 and len(unique_bboxes) == 1



if __name__ == "__main__":
    """
    uv run python cnns/finetune-script-classify-mser.py
    """
    # Load the model.
    use_vgg = False
    if use_vgg:
        model_weights = './cnns/trained/vgg_model.pth'
        model = VGG16Transfer(num_classes=11)
    else:
        model_weights = './cnns/trained/deepcnn_model.pth'
        model = CNN()
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model = model.to(device) 
    model.eval()

    save_dir = './tmp-mser'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))


    # Load images to classify.
    # Update the path.
    images = [
        './cnns/img/buildings-samples/h0.jpg',
        './cnns/img/buildings-samples/h0-rotated.jpg',
        './cnns/img/buildings-samples/h1.jpg',
        './cnns/img/buildings-samples/h2.jpg', # scale_factor = 1/2
        './cnns/img/buildings-samples/h1-noisy.jpg',
    ]
    #img_path = './cnns/img/buildings-samples/h1.jpg'
    for idx, img_path in enumerate(images):
        print(f'>>> Processing {img_path}')
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        retry = detect_digits(model, img, idx, scale_factor=1, save_dir=save_dir, vgg=use_vgg)
        if retry:
            print(f'Retrying run on {img_path} with a smaller scale factor')
            idx += 0.5
            scale_factor = 1/2
            img = resize_image_by_factor(img, scale_factor)
            detect_digits(model, img, idx, scale_factor=scale_factor, save_dir=save_dir, vgg=use_vgg)