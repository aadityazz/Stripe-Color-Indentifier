from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

app = Flask(__name__)

def process_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to the LAB color space for better color segmentation
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Define color boundaries for each color on the urine strip (in LAB color space)
    color_boundaries = {
        'URO': ([90, 10, 20], [230, 130, 170]),
        'BIL': ([90, 50, 100], [220, 160, 200]),
        'KET': ([100, 60, 100], [220, 180, 210]),
        'BLD': ([80, 50, 100], [210, 170, 200]),
        'PRO': ([90, 50, 80], [210, 150, 200]),
        'NIT': ([90, 30, 90], [220, 160, 200]),
        'LEU': ([80, 30, 100], [210, 170, 200]),
        'GLU': ([70, 40, 100], [190, 180, 200]),
        'SG':  ([60, 60, 80], [180, 170, 200]),
        'PH':  ([70, 60, 100], [200, 180, 200]),
    }

    # Dictionary to store the RGB values of each color
    result = {}

    # Dictionary to store the integer RGB values of each color
    result = {}

    # Iterate through each color and extract its RGB value
    for color, (lower_bound, upper_bound) in color_boundaries.items():
        # Create a binary mask for the current color
        mask = cv2.inRange(lab_image, np.array(lower_bound), np.array(upper_bound))

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if there are valid contours for the current color
        if len(contours) > 0:
            # Get the average RGB value of the color from the contour area
            rgb_values = []
            for contour in contours:
                # Compute the mean RGB value of the color in the contour area
                mask_area = cv2.contourArea(contour)
                mask = np.zeros_like(mask)  # Create an empty mask of the same shape
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                mean_color = cv2.mean(image, mask=mask)[:3]
                rgb_values.append(mean_color)

            # Take the average of all RGB values for the current color
            avg_rgb = np.mean(rgb_values, axis=0)

            # Convert the float RGB values to integers
            avg_rgb_int = [int(round(value)) for value in avg_rgb]

            # Store the RGB value in the result dictionary
            result[color] = avg_rgb_int
        else:
            # Set a default RGB value (e.g., white [255, 255, 255]) for the color when no valid contour is found
            result[color] = [255, 255, 255]


    return result

@app.route('/api/identify_colors', methods=['POST'])
def identify_colors():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = request.files['image']
    image_path = "temp_image.png"
    image.save(image_path)

    result = process_image(image_path)

    return jsonify(result)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
