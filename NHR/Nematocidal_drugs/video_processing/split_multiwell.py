import os
import cv2
import numpy as np

def detect_circles(frame, input_video_path):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)  # Contrast enhancement

    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=150, 
        param1=50,  # Lower threshold for Canny edge detector
        param2=40,  # Lower accumulator threshold for circle detection
        minRadius=470,  # Adjusted to match detected wells
        maxRadius=510  
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Filter out unwanted circles
        filtered_circles = []
        for circle in circles:
            x, y, r = circle
            # Create a mask for the border of the circle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, thickness=130)  # Border thickness of 10 pixels
            # Calculate the average intensity along the border of the circle
            mean_val = cv2.mean(gray, mask=mask)[0]
            # Add your filtering criteria here
            if 470 <= r <= 510 and mean_val < 110:  # Example: filter based on radius and border intensity
                filtered_circles.append(circle)
        circles = filtered_circles
    else:
        circles = []

    # Define average and standard deviation values for each quadrant
    quadrant_stats = {
        "upper_left": {"x": 1200, "y": 749.75, "radius": 505},
        "upper_right": {"x": 2609.25, "y": 746.625, "radius": 505},
        "lower_left": {"x": 1150, "y": 2240, "radius": 510},
        "lower_right": {"x": 2603.375, "y": 2254, "radius": 505}
    }
    # Define a tolerance range for validation
    tolerance_x = 200
    tolerance_y = 200

    def is_within_range(detected_circle, quadrant_stat):
        x, y, r = detected_circle
        return (abs(x - quadrant_stat["x"]) <= tolerance_x and
                abs(y - quadrant_stat["y"]) <= tolerance_y)

    def get_circle_for_quadrant(detected_circle, quadrant_name):
        if detected_circle is not None and len(detected_circle) > 0 and is_within_range(detected_circle, quadrant_stats[quadrant_name]):
            return detected_circle
        else:
            print(f"Warning: Circle outside of tolerance parameters was found. Using defined positions for {input_video_path}")
            return (quadrant_stats[quadrant_name]["x"],
                    quadrant_stats[quadrant_name]["y"],
                    quadrant_stats[quadrant_name]["radius"])

    height, width, _ = frame.shape
    quadrants = {
        "upper_left": (0, 0, width // 2, height // 2),
        "upper_right": (width // 2, 0, width, height // 2),
        "lower_left": (0, height // 2, width // 2, height),
        "lower_right": (width // 2, height // 2, width, height),
    }

    def distance(circle, stats):
        x, y, r = circle
        return np.sqrt((x - stats["x"])**2 + (y - stats["y"])**2 + (r - stats["radius"])**2)

    filtered_circles = []
    for quadrant, (qx1, qy1, qx2, qy2) in quadrants.items():
        quadrant_circles = [circle for circle in circles if qx1 <= circle[0] < qx2 and qy1 <= circle[1] < qy2]
        if quadrant_circles:
            closest_circle = min(quadrant_circles, key=lambda circle: distance(circle, quadrant_stats[quadrant]))
            filtered_circles.append(closest_circle)
        else:
            avg_values = quadrant_stats[quadrant]
            filtered_circles.append((int(avg_values["x"]), int(avg_values["y"]), int(avg_values["radius"])))

    if len(filtered_circles) != 4:
        if len(filtered_circles) > 4:
            print(f"Warning: {len(filtered_circles)} wells found, expected 4. Closest circles to defined positions were chosen for video {input_video_path}")
        else:
            print(f"Warning: {len(filtered_circles)} wells found, expected 4. Missing wells added for video {input_video_path}")

    final_circles = []
    for quadrant_name in quadrant_stats.keys():
        detected_circle = next((circle for circle in filtered_circles if is_within_range(circle, quadrant_stats[quadrant_name])), None)
        final_circle = get_circle_for_quadrant(detected_circle, quadrant_name)
        final_circles.append(final_circle)

    return final_circles

def get_well_names(folder_name):
    # Assuming folder_name is in the format 'A5_B6'
    parts = folder_name.split('_')
    if len(parts) == 2:
        part1, part2 = parts
        num1 = int(part1[1:])
        num2 = int(part2[1:])
        if num1 == 9 and num2 == 10:
            return [f"{part1[0]}9", f"{part1[0]}10", f"{part2[0]}9", f"{part2[0]}10"]
        elif num1 == 11 and num2 == 12:
            return [f"{part1[0]}11", f"{part1[0]}12", f"{part2[0]}11", f"{part2[0]}12"]
        else:
            return [part1, part1[0] + str(num1 + 1), part2[0] + str(num2 - 1), part2]
    return []

def crop_and_save_videos(input_video_path, output_dir, folder_name):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame of {input_video_path}.")
        return

    circles = detect_circles(first_frame, input_video_path)
    if circles is None or len(circles) != 4:
        print(f"Error: Expected 4 circles, but found {len(circles) if circles is not None else 0}.")
        return

    well_names = get_well_names(folder_name)
    height, width, _ = first_frame.shape

    # Extract components from the input video path
    path_parts = input_video_path.split(os.sep)
    component1 = path_parts[-4]  # '250213_40h'
    component2 = path_parts[-3]  # 'not_viscous_1_test'
    component3 = path_parts[-1].split('.')[0]  # 'A5_B6_20250213_142541'

    # Prepare video writers for each well
    video_writers = []
    for i, circle in enumerate(circles):
        x, y, r = circle
        # Determine the quadrant
        if x < width // 2 and y < height // 2:
            well_name = well_names[0]  # Upper left
        elif x >= width // 2 and y < height // 2:
            well_name = well_names[1]  # Upper right
        elif x < width // 2 and y >= height // 2:
            well_name = well_names[2]  # Lower left
        else:
            well_name = well_names[3]  # Lower right

        # Define the cropping size increase (250 pixels on each side)
        crop_increase = 200

        # Adjust the cropping coordinates
        x1, y1 = int(max(0, x - r - crop_increase)), int(max(0, y - r - crop_increase))
        x2, y2 = int(min(width, x + r + crop_increase)), int(min(height, y + r + crop_increase))

        # Construct the new filename
        new_filename = f"{component1}_{component2}_{component3}_{well_name}.avi"
        output_video_path = os.path.join(output_dir, new_filename)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (x2 - x1, y2 - y1))
        video_writers.append((out, (x1, y1, x2, y2)))

    # Read and process frames
    while ret:
        for out, (x1, y1, x2, y2) in video_writers:
            cropped_frame = first_frame[y1:y2, x1:x2]
            out.write(cropped_frame)
        ret, first_frame = cap.read()

    # Release all video writers
    for out, _ in video_writers:
        out.release()

    cap.release()
    print(f"Cropped videos saved to {output_dir}")

def process_directory(main_directory):
    for subdir, _, files in os.walk(main_directory):
        for file in files:
            if file.endswith('.avi'):
                input_video_path = os.path.join(subdir, file)
                output_dir = subdir
                folder_name = os.path.basename(subdir)
                crop_and_save_videos(input_video_path, output_dir, folder_name)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python split_multiwell.py <main_directory_path>")
        sys.exit(1)

    main_directory = sys.argv[1]
    process_directory(main_directory)
