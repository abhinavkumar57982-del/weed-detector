from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from ultralytics import YOLO
import cv2
import datetime
import requests
from urllib.parse import urlparse
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
MODEL_PATH = 'runs/detect/weed_detector/weights/best.pt'
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print("✅ Trained model loaded!")
    print(f"📋 Model classes: {model.names}")
else:
    print("⚠️  No trained model found. Downloading pre-trained...")
    model = YOLO('yolov8n.pt')
    print("✅ Pre-trained model loaded!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_weed_detections(img, detections):
    """Draw ONLY WEED detections (Red boxes)"""
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Only weed - Red color
        color = (0, 0, 255)      # Red for weed
        label = f"🌿 WEED {confidence}"
        
        # Draw main rectangle with thicker border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Draw background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-30), (x1+text_w, y1-5), color, -1)  # -1 = filled
        
        # Draw text in white
        cv2.putText(img, label, (x1, y1-10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw small corner markers for better visibility
        marker_len = 20
        cv2.line(img, (x1, y1), (x1+marker_len, y1), color, 3)
        cv2.line(img, (x1, y1), (x1, y1+marker_len), color, 3)
        cv2.line(img, (x2, y2), (x2-marker_len, y2), color, 3)
        cv2.line(img, (x2, y2), (x2, y2-marker_len), color, 3)
    
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or GIF'}), 400
        
        # Save file with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Get original image dimensions
        h, w = img.shape[:2]
        
        # Resize if image is too large
        max_size = 1280
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # Try multiple confidence thresholds
        conf_levels = [0.02, 0.01, 0.005, 0.001]
        weed_detections = []  # Only weed detections
        
        for conf in conf_levels:
            results = model(img, conf=conf)
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        
                        if cls in model.names:
                            class_name = model.names[cls]
                        else:
                            class_name = f"class_{cls}"
                        
                        # 👇 ONLY WEED - SKIP CROP
                        if class_name.lower() == 'weed':
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            weed_detections.append({
                                'class': 'weed',
                                'confidence': f"{conf_score:.1%}",
                                'bbox': [x1, y1, x2, y2]
                            })
                        # CROP IS IGNORED - NOT ADDED TO DETECTIONS
            
            if weed_detections:
                break
        
        # Count only weed (crop_count always 0)
        weed_count = len(weed_detections)
        crop_count = 0  # 👈 Always 0
        
        # Draw ONLY weed detections
        if weed_detections:
            img = draw_weed_detections(img, weed_detections)
        
        # Save annotated image
        output_filename = f"annotated_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, img)
        
        # Prepare result message - ONLY WEED
        result_text = []
        if weed_count > 0:
            result_text.append(f"🚨 Found {weed_count} weed(s)!")
        else:
            result_text.append("🌿 No weeds detected")
        
        return jsonify({
            'success': True,
            'result': result_text,
            'weed_count': weed_count,
            'crop_count': 0,  # 👈 Always 0
            'total_detections': len(weed_detections),
            'annotated_image': f'/uploads/{output_filename}',
            'detections': weed_detections,
            'image_size': f"{w}x{h}"
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-url', methods=['POST'])
def upload_from_url():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Download image from URL
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download image'}), 400
        
        # Get filename from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or '.' not in filename:
            filename = f"url_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return jsonify({'success': True, 'filename': filename})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/paste', methods=['POST'])
def paste_image():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Remove data:image/png;base64, part
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Save image
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pasted_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        return jsonify({'success': True, 'filename': filename})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌿 WEED DETECTOR WEB APP (Weed Only)")
    print("="*60)
    print("📍 Open: http://localhost:5000")
    print("📁 Upload folder:", os.path.abspath(UPLOAD_FOLDER))
    print("🤖 Model classes:", model.names if 'model' in locals() else "Not loaded")
    print("🎯 Output: Only WEED detection (Crops ignored)")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)