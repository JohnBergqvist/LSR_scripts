import os
import cv2
import numpy as np

def detect_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)

    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=150, 
        param1=50, 
        param2=40, 
        minRadius=470, 
        maxRadius=510  
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        filtered_circles = []
        for circle in circles:
            x, y, r = circle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, thickness=30)
            mean_val = cv2.mean(gray, mask=mask)[0]
            if 470 <= r <= 510 and mean_val < 110:
                filtered_circles.append(circle)
        return filtered_circles
    return None

def distance(circle, stats):
    x, y, r = circle
    return np.sqrt((x - stats["x"])**2 + (y - stats["y"])**2 + (r - stats["radius"])**2)

def draw_detected_circles(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame of {input_video_path}.")
        return

    circles = detect_circles(first_frame)
    if circles is None:
        circles = []

    # Define average and standard deviation values for each quadrant
    quadrant_stats = {
        "upper_left": {"x": 1122, "y": 749.75, "radius": 505},
        "upper_right": {"x": 2609.25, "y": 746.625, "radius": 505},
        "lower_left": {"x": 1050, "y": 2240, "radius": 510},
        "lower_right": {"x": 2603.375, "y": 2254, "radius": 505}
    }

    height, width, _ = first_frame.shape
    quadrants = {
        "upper_left": (0, 0, width // 2, height // 2),
        "upper_right": (width // 2, 0, width, height // 2),
        "lower_left": (0, height // 2, width // 2, height),
        "lower_right": (width // 2, height // 2, width, height),
    }

    # Filter circles to keep only the closest one to the quadrant stats
    filtered_circles = []
    for quadrant, (qx1, qy1, qx2, qy2) in quadrants.items():
        quadrant_circles = [circle for circle in circles if qx1 <= circle[0] < qx2 and qy1 <= circle[1] < qy2]
        if quadrant_circles:
            closest_circle = min(quadrant_circles, key=lambda circle: distance(circle, quadrant_stats[quadrant]))
            filtered_circles.append(closest_circle)
        else:
            avg_values = quadrant_stats[quadrant]
            filtered_circles.append((int(avg_values["x"]), int(avg_values["y"]), int(avg_values["radius"])))

    for circle in filtered_circles:
        x, y, r = circle
        cv2.circle(first_frame, (x, y), r, (0, 255, 0), 10)
        cv2.rectangle(first_frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_image_path = os.path.join(os.path.dirname(input_video_path), f"{base_name}_detected_circles.jpg")

    cv2.imwrite(output_image_path, first_frame)
    print(f"Image with detected circles saved to {output_image_path}")

    quadrant_images = []
    for i, (qx1, qy1, qx2, qy2) in enumerate(quadrants.values()):
        quadrant_image = first_frame[qy1:qy2, qx1:qx2]
        quadrant_images.append(quadrant_image)
        quadrant_output_path = os.path.join(os.path.dirname(input_video_path), f"{base_name}_quadrant_{i+1}.jpg")
        cv2.imwrite(quadrant_output_path, quadrant_image)
        print(f"Quadrant {i+1} image saved to {quadrant_output_path}")

    # Generate a txt file with the properties of all the rings found
    properties_output_path = os.path.join(os.path.dirname(input_video_path), f"{base_name}_circle_properties.txt")
    with open(properties_output_path, 'w') as f:
        f.write(f"Image: {output_image_path}\n")
        f.write("x, y, radius\n")
        for circle in filtered_circles:
            x, y, r = circle
            f.write(f"{x}, {y}, {r}\n")
    print(f"Circle properties saved to {properties_output_path}")

    cap.release()

def process_videos_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.avi')):
                video_path = os.path.join(root, file)
                draw_detected_circles(video_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python detect_circles.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    process_videos_in_directory(directory_path)