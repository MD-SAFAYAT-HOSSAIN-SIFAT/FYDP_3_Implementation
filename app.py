from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from roboflow import Roboflow
from pdf2image import convert_from_path

app = Flask(__name__)

# Initialize Roboflow model
rf = Roboflow(api_key="BmOQVSckGdhRKhCcE4hR")
project = rf.workspace("uiu-lxaq9").project("fydp-vmxwc")
model = project.version("7").model

model.confidence = 50
model.overlap = 25

# Define color map for segmentation
color_map = {
    'paragraph': (0, 255, 0),
    'image': (255, 0, 0),
    'title': (0, 0, 255),
    'caption': (255, 255, 0),
    'text': (0, 255, 255),
    'page number': (255, 0, 255),
    'advertisement': (255, 165, 0),
    'table': (75, 0, 130)
}

def draw_segmentation(image, prediction):
    overlay = image.copy()
    for obj in prediction['predictions']:
        points = obj['points']
        points = [(int(p['x']), int(p['y'])) for p in points]
        points = np.array(points, dtype=np.int32)
        color = color_map.get(obj['class'], (0, 255, 0))
        cv2.fillPoly(overlay, [points], color + (127,))
        label = obj['class']
        confidence = obj['confidence']
        text = f"{label} ({confidence:.2f})"
        centroid = np.mean(points, axis=0).astype(int)

        # Draw the text with a black background for better visibility
        cv2.putText(image, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(image, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(image, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

def convert_to_jpg(file_path, upload_dir):
    images = []
    if file_path.endswith('.pdf'):
        pages = convert_from_path(file_path, 300)
        for i, page in enumerate(pages):
            image_path = os.path.join(upload_dir, f"page_{i + 1}.jpg")
            page.save(image_path, 'JPEG')
            images.append(image_path)
    else:
        image = cv2.imread(file_path)
        image_path = os.path.join(upload_dir, secure_filename(os.path.splitext(file_path)[0] + ".jpg"))
        cv2.imwrite(image_path, image)
        images.append(image_path)
    return images

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/uploader', methods=['POST'])
def uploader_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        upload_dir = "static/uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, secure_filename(file.filename))
        file.save(file_path)
        
        # Convert to JPG if necessary
        image_paths = convert_to_jpg(file_path, upload_dir)
        image_paths = [os.path.relpath(p, 'static') for p in image_paths]  
        return redirect(url_for('show_images', images=image_paths))
    return "File upload failed"

@app.route('/show_images')
def show_images():
    images = request.args.getlist('images')
    return render_template('show_images.html', images=images)

@app.route('/infer_image/<path:image_path>')
def infer_image(image_path):
    # Perform inference
    image_full_path = os.path.join('static', image_path)
    print(f"Inferencing image at path: {image_full_path}")  # Debug print
    prediction = model.predict(image_full_path).json()

    # Load and process the image
    image = cv2.imread(image_full_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    draw_segmentation(image, prediction)

    # Save the output image
    output_path = os.path.join('static', 'segmented_image.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
