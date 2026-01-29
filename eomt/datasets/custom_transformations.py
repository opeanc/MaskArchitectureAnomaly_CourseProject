import cv2
import numpy as np
import pandas as pd
import random


def load_sample(df_annotations):


    # random sample an anomaly object
    row = df_annotations.sample(1).iloc[0]
    label_type = row['label'] # "air", "ground", o "obj"

    return (row['image_path'], row['mask_path'], label_type)



def extract_object(image_path, mask_path):
    """
    Crops the object from the original image using the binary mask.
    Returns the cropped image and its cropped mask.

    Args:
        image_path: Path to the anomaly image file
        mask_path: Path to the anomaly mask file
    """
    # Load image and mask
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Binarization    
    # makes sure mask is perfectly black (0) or white (255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Bounding Box
    # it takes all the contours from the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No object found in the mask!")
        return None, None

    # it takes the largest contour (in case there are other small artifacts)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Crop
    # We crop both image and mask using the coordinates
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = binary_mask[y:y+h, x:x+w]

    # Background Masking
    # we apply the mask to the cropped image to make the 
    # immediate background (grass that might have remained inside the rectangle) black
    cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)

    return cropped_img, cropped_mask


def get_random_scale(img, mask, min_scale=0.3, max_scale=1.0):
    """
    Scales the object randomly within the given range.

    Args:
        img: anomaly image (after extraction)
        mask: anomaly mask (after extraction)
        min_scale: minimum scaling factor
        max_scale: maximum scaling factor
    """
    h, w = img.shape[:2]
    scale = random.uniform(min_scale, max_scale)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Avoid zero dimensions
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    new_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    new_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return (new_image, new_mask)


def check_ego_vehicle_overlap(bg_mask, x, y, h_obj, w_obj, void_id=255, threshold=0.3):
    """
    Checks if the proposed position overlaps with the ego-vehicle.
    In Cityscapes, the ego-vehicle is always labeled as VOID (255).
    Additionally, the ego-vehicle is always in the lower part of the image.

    Args:
        bg_mask: Cityscapes mask
        x: top-left x coordinate of the object
        y: top-left y coordinate of the object
        h_obj: height of the object
        w_obj: width of the object
        void_id: pixel value representing the void class (default is 255)
        threshold: maximum allowed overlap ratio with the void class
    """
    H, W = bg_mask.shape

    # extract the ROI from the background mask
    roi = bg_mask[y:y+h_obj, x:x+w_obj]

    # Calculate how many pixels in the ROI are void (255)
    void_pixels = np.sum(roi == void_id)
    total_pixels = h_obj * w_obj
    void_ratio = void_pixels / total_pixels

    # If the object is too overlapped with the void (often black borders or ego-vehicle), discard.
    # Add a positional check: the ego vehicle is at the bottom center.
    # If y is very low (e.g., > 80% of the image) and there is void, it is almost certainly the car.
    is_bottom = y > (H * 0.75)

    if is_bottom and void_ratio > 0.1: # Tight tolerance at the bottom
        return True # overlap detected (Reject)

    if void_ratio > threshold: # Wider tolerance elsewhere (e.g., black borders)
        return True

    return False # Valid position

def get_smart_position(bg_shape, obj_shape, label_type, bg_mask, max_attempts=50, margin=70):
    """
    It finds a coordinate (x, y) having a look at the label type and avoiding
    overlaps with the ego-vehicle.

    Args:
        bg_shape: shape (C, H, W) of the background image
        obj_shape: shape (C, H, W) of the object image
        label_type: ("air", "ground", "obj")
        bg_mask: mask of the background image (Cityscapes)
        max_attempts: maximum number of attempts to find a valid position
        margin: minimum distance in pixels from the image borders.
    """
    H_bg, W_bg = bg_shape[:2]
    h_obj, w_obj = obj_shape[:2]

    # defining a safe area to place the object
    x_min = margin
    x_max = W_bg - w_obj - margin
    y_min = margin
    y_max = H_bg - h_obj - margin

    # if the object is too large to respect the margins, remove margins
    if x_max <= x_min or y_max <= y_min:
        # try to insert it without margins (or reject if it doesn't fit)
        x_min, y_min = 0, 0
        x_max = W_bg - w_obj
        y_max = H_bg - h_obj
        if x_max < 0 or y_max < 0:
            return None # obj too large for the background

    for _ in range(max_attempts):
        # determine y (Height)

        if label_type == 'air':
            # Upper zone
            limit = max(y_min, y_max // 2)
            y = random.randint(y_min, limit)

        elif label_type == 'ground':
            # Lower zone
            # Avoid starting too high, but respect y_min
            start_y = max(y_min, y_max // 3)
            y = random.randint(start_y, y_max)

        else: # label_type == 'obj'
            y = random.randint(y_min, y_max)

        # determine x (Width)
        x = random.randint(x_min, x_max)

        # Ego-Vehicle Check
        if not check_ego_vehicle_overlap(bg_mask, x, y, h_obj, w_obj):
            return (x, y)

    return None



def augment_object(img, mask):
    """
    Applies geometric and photometric Data Augmentation to the object.
    
    Args:
        img: Cropped object image (BGR)
        mask: Object mask (Gray)
    """
    h, w = img.shape[:2]

    # --- 1. Random Flip ---
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    # --- 2. Random Rotation (-15degree to +15degree) ---
    if random.random() < 0.7: # 70% probability
        angle = random.uniform(-15, 15)
        center = (w // 2, h // 2)

        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # --- 3. Color Jitter (brightness and contrast) ---
    # slightly modify the colors to better blend with the background
    if random.random() < 0.8:
        # contrast: multiplier ( 0.8 to 1.2)
        alpha = random.uniform(0.8, 1.2)
        # brightness: addend (-30 to +30)
        beta = random.uniform(-30, 30)

        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # --- 4. sSaturation Jitter ---
    # change the saturation to have more/less vivid objects
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")

        sat_factor = random.uniform(0.7, 1.3)
        hsv[:, :, 1] *= sat_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    return img, mask


def perform_color_transfer(source, target, strength=0.4):
    """
    Applies color transfer but mixes it with the original to avoid a "ghost/camouflage" effect.

    Args:
        source: source image (object)
        target: target image (background)
        strength (float): 0.0 = original color, 1.0 = total camouflage (between 0.4 and 0.6 recommended)
    """
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    (l_mean_src, l_std_src) = cv2.meanStdDev(source_lab)
    (l_mean_tar, l_std_tar) = cv2.meanStdDev(target_lab)

    # Apply the full mathematical transfer
    source_lab_t = source_lab.copy()

    # Subtract source mean
    source_lab_t -= np.array([l_mean_src.flatten()]).astype("float32")

    # Scale standard deviation
    # Limit scaling: if the background is too flat (std close to 0),
    # the object loses all details. Add a clamp.
    scale = l_std_tar.flatten() / (l_std_src.flatten() + 1e-5)
    scale = np.clip(scale, 0.5, 1.5) # Avoid too explosive or flat contrasts

    source_lab_t *= scale

    # Add target mean
    source_lab_t += np.array([l_mean_tar.flatten()]).astype("float32")
    source_lab_t = np.clip(source_lab_t, 0, 255)

    # Convert the transferred result to BGR
    transfer = cv2.cvtColor(source_lab_t.astype("uint8"), cv2.COLOR_LAB2BGR)

    # low strength -> more similar to the original
    # high strength -> more similar to the background
    final_result = cv2.addWeighted(transfer, strength, source, 1.0 - strength, 0)

    return final_result

def apply_depth_blur(img, y_pos, total_height):
    """
    Simulates "depth blur based on the position".
    Assumption used: the higher the object, the farther it is.

    Args:
        img: image to apply the depth blur (object)
        y_pos: vertical position of the object (top-left corner)
        total_height: total height of the image
    """
    # normalize y position between 0 and 1
    rel_y = y_pos / total_height

    # Higher = more blur (distance blur)

    k_size = 0
    if rel_y < 0.5: # Far (slight blur)
        k_size = 3
    elif rel_y > 0.85: # Very close (Rapid motion blur of the road)
        k_size = 5

    if k_size > 0:
        return cv2.GaussianBlur(img, (k_size, k_size), 0)

    return img

def add_noise(img, strength=0.05):
    """
    Adds "color noise" to simulate sensor grain.

    Args:
        img: image to add noise (object)
        strength: noise strength (default 0.05)
    """
    noise = np.random.randn(*img.shape).astype(np.float32)
    # scale the noise and add it
    noisy_img = img.astype(np.float32) + (noise * strength * 255)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)




def inject_anomaly(city_img, city_mask, obj_img_path, obj_mask_path, label, anomaly_id=254):
    """
    Main Function:
    1. Extracts the object + mask
    2. Performs trasformations of the object
    3. Updates the segmentation mask

    Args:
        city_img: cityscapes image
        city_mask: cityscapes mask
        obj_img_path: object image PATH
        obj_mask_path: object mask PATH
        label: object label
        anomaly_id: The numerical ID to assign to the anomaly pixels in the mask (254)
    """

    # ON-THE-FLY EXTRACTION
    obj_img, obj_mask = extract_object(obj_img_path, obj_mask_path)

    if obj_img is None:
        # if something is wrong during extraction, return the original
        print("Warning: Object not found.")
        return city_img, city_mask

    # DATA AUGMENTATION
    obj_img, obj_mask = augment_object(obj_img, obj_mask)

    # suggested by the Fishyscapes paper
    obj_img_rescale, obj_mask_rescale = get_random_scale(obj_img, obj_mask, min_scale=0.3, max_scale=1.0)

    # Dimensions of the cropped object
    h_obj, w_obj = obj_img_rescale.shape[:2]

    # SMART POSITIONING
    position = get_smart_position(city_img.shape, obj_img_rescale.shape, label, city_mask)
    print(position)

    if position is None:
        return city_img, city_mask

    x_pos, y_pos = position

    # safety checks on the edges
    if x_pos + w_obj > city_img.shape[1] or y_pos + h_obj > city_img.shape[0]:
        print("Warning: The object goes out of the Cityscapes image boundaries.")
        return city_img, city_mask

    # --- DOMAIN ADAPTATION ---
    roi = city_img[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj]

    # COLOR TRANSFER
    obj_img_adjusted = perform_color_transfer(obj_img_rescale, roi)

    # DEPTH BLUR
    obj_img_adjusted = apply_depth_blur(obj_img_adjusted, y_pos, city_img.shape[0])

    # NOISE
    obj_img_adjusted = add_noise(obj_img_adjusted)

    # EDGE SMOOTHING
    float_mask = obj_mask_rescale.astype(float) / 255.0
    float_mask_blurred = cv2.GaussianBlur(float_mask, (3, 3), 0) # Kernel ridotto per dettagli fini
    alpha = np.dstack([float_mask_blurred] * 3)

    # COMPOSITING
    blended_roi = (alpha * obj_img_adjusted) + ((1 - alpha) * roi)

    final_img = city_img.copy()
    final_img[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = blended_roi.astype(np.uint8)

    # --- MASK UPDATE ---
    final_mask = city_mask.copy()
    roi_mask = final_mask[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj]
    roi_mask[obj_mask_rescale > 127] = anomaly_id
    final_mask[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = roi_mask

    return final_img, final_mask
