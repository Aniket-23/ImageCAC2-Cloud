from flask import Flask, render_template, request, redirect
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
from io import BytesIO
import boto3
from botocore.exceptions import NoCredentialsError
from flask_s3 import FlaskS3

app = Flask(__name__)

# AWS credentials
AWS_ACCESS_KEY_ID = 'ASIARUO2LU22FAEUNMFJ'
AWS_SECRET_ACCESS_KEY = '/gTr/AWg1WQcwCD7Zeic32JTxIaM2cNdXdvTGFPR'
AWS_REGION = 'us-east-1'
S3_BUCKET = 'imagecac2'
S3_ENDPOINT_URL = 'https://s3.amazonaws.com'  # For example, 'https://s3.amazonaws.com'

# Set AWS credentials directly in the Flask app
app.config['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
app.config['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
app.config['AWS_REGION'] = AWS_REGION
app.config['S3_BUCKET'] = S3_BUCKET
app.config['S3_ENDPOINT_URL'] = S3_ENDPOINT_URL

# Initialize FlaskS3 with the app
s3 = FlaskS3(app)

UPLOAD_FOLDER = 'uploads'  # Local temporary upload folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_s3(local_path, s3_path):
    try:
        s3.upload_file(local_path, app.config['S3_BUCKET'], s3_path)
        return True
    except NoCredentialsError:
        print('Credentials not available')
        return False

def apply_sobel_operations(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Sobel operator to compute gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel operator for horizontal gradient
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel operator for vertical gradient

    # Compute magnitude and angle of the gradient
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Save Sobel magnitude and angle images
    magnitude_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sobel_magnitude_' + secure_filename(image_path))

    cv2.imwrite(magnitude_image_path, magnitude)

    return magnitude_image_path

def apply_canny_edge_detection(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 0, 50)

    # Save the processed image in the static/uploads folder
    edge_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'edge_' + secure_filename(image_path))
    cv2.imwrite(edge_image_path, edges)

    return edge_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploadImage', methods=['GET', 'POST'])
def upload_image():
    original_image = None
    edge_image = None

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'imageFile' not in request.files:
            return redirect(request.url)

        file = request.files['imageFile']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            # Save the uploaded file locally
            local_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(local_filename)

            # Upload to S3
            s3_path = f'uploads/{filename}'  # S3 path for the uploaded image
            if upload_to_s3(local_filename, s3_path):
                original_image = s3.url(s3_path)
                edge_image = None  # Modify this if you want to process the image and upload the result

                # Render the template with S3 image paths
                return render_template('uploadImage.html', original_image=original_image, edge_image=edge_image)
            else:
                return "Upload to S3 failed"

    # Render the template with image paths
    return render_template('uploadImage.html', original_image=original_image, edge_image=edge_image)


@app.route('/uploadSobelImage', methods=['GET', 'POST'])
def upload_sobel_image():
    original_image = None
    sobel_magnitude = None
    sobel_angle = None

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'imageFile' not in request.files:
            return redirect(request.url)

        file = request.files['imageFile']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            # Save the uploaded file locally
            local_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(local_filename)

            # Upload to S3
            s3_path = f'uploads/{filename}'  # S3 path for the uploaded image
            if upload_to_s3(local_filename, s3_path):
                original_image = s3.url(s3_path)

                # Apply Sobel operations
                magnitude_image_path = apply_sobel_operations(local_filename)

                # Upload Sobel magnitude to S3
                sobel_magnitude_path = f'uploads/sobel_magnitude_{filename}'  # S3 path for Sobel magnitude image
                if upload_to_s3(magnitude_image_path, sobel_magnitude_path):
                    sobel_magnitude = s3.url(sobel_magnitude_path)

                    # Render the template with S3 image paths
                    return render_template('uploadSobel.html', original_image=original_image, sobel_magnitude=sobel_magnitude)
                else:
                    return "Upload Sobel magnitude to S3 failed"
            else:
                return "Upload to S3 failed"

    # Render the template with image paths
    return render_template('uploadSobel.html', original_image=original_image, sobel_magnitude=sobel_magnitude)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
