"""
Flask API Server for Papercut Portrait Vectorization v2.0
With proper edge detection and mode-optimized processing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

# Preset mode configurations with optimal defaults
MODES = {
    'posterized': {
        'name': 'Posterized Portrait',
        'layers': 6,
        'description': 'High-contrast facial silhouettes (5-7 layers)',
        'default_contrast': 70,  # High contrast for bold shapes
        'blur_strength': 5,
        'edge_preserve': True,
        'min_area': 200,
        'use_edge_detection': False
    },
    'lineart': {
        'name': 'Line-Art Outline',
        'layers': 2,
        'description': 'Minimal contour for vinyl/cut decals (1-2 layers)',
        'default_contrast': 80,  # Very high for clean edges
        'blur_strength': 2,
        'edge_preserve': True,
        'min_area': 300,
        'use_edge_detection': True,  # Use Canny edge detection
        'canny_low': 50,
        'canny_high': 150
    },
    'popart': {
        'name': 'Pop-Art Duo-Tone',
        'layers': 4,
        'description': 'Bold limited-palette versions (3-4 layers)',
        'default_contrast': 85,  # Very high for bold pop-art effect
        'blur_strength': 7,
        'edge_preserve': False,
        'min_area': 150,
        'use_edge_detection': False
    },
    'layered': {
        'name': 'Layered Color Blend',
        'layers': 10,
        'description': 'Decorative gradient builds (8-12 layers)',
        'default_contrast': 40,  # Lower for smooth gradients
        'blur_strength': 4,
        'edge_preserve': True,
        'min_area': 100,
        'use_edge_detection': False
    }
}

def process_image_to_layers(image_data, mode='posterized', contrast=None):
    """Process an image into layered masks based on selected mode"""
    try:
        # Get mode configuration
        config = MODES.get(mode, MODES['posterized'])
        num_layers = config['layers']
        
        # Use mode's default contrast if not specified
        if contrast is None:
            contrast = config['default_contrast']
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError('Failed to decode image')
        
        # Resize to optimal size
        max_width = 800
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_width = max_width
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        height, width = img.shape[:2]
        
        # Special processing for line-art mode
        if config.get('use_edge_detection'):
            return process_lineart_mode(img, width, height, config, contrast)
        
        # Standard processing for other modes
        return process_standard_mode(img, width, height, config, contrast, num_layers)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def process_lineart_mode(img, width, height, config, contrast):
    """Special processing for line-art mode using edge detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply contrast
    alpha = 1.0 + (contrast / 100.0)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)
    
    # Use Canny edge detection to find actual borders
    edges = cv2.Canny(gray, config['canny_low'], config['canny_high'])
    
    # Dilate edges slightly to make them more visible
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and simplify contours
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= config['min_area']:
            # Simplify contour for cleaner lines
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 3:
                filtered_contours.append(approx)
    
    # Create outline layer (edges)
    outline_path = contours_to_svg_path(filtered_contours, width, height)
    
    # Create fill layer (inverted threshold for solid areas)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    fill_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_fill = []
    for contour in fill_contours:
        area = cv2.contourArea(contour)
        if area >= config['min_area'] * 2:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 3:
                filtered_fill.append(approx)
    
    fill_path = contours_to_svg_path(filtered_fill, width, height)
    
    layers = [
        {
            'index': 0,
            'pathData': fill_path,
            'brightness': 50,
            'tonalRange': 'fill',
            'name': 'Fill Layer'
        },
        {
            'index': 1,
            'pathData': outline_path,
            'brightness': 200,
            'tonalRange': 'outline',
            'name': 'Outline Layer'
        }
    ]
    
    return {
        'success': True,
        'layers': layers,
        'width': width,
        'height': height,
        'mode': 'lineart',
        'modeName': config['name']
    }

def process_standard_mode(img, width, height, config, contrast, num_layers):
    """Standard processing for posterized, pop-art, and layered modes"""
    # Preprocessing based on mode
    if config['edge_preserve']:
        # Use bilateral filter for edge-preserving smoothing
        img_smooth = cv2.bilateralFilter(img, 9, 75, 75)
    else:
        # Use Gaussian blur for smoother results
        kernel_size = config['blur_strength'] * 2 + 1
        img_smooth = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Convert to LAB color space for better perceptual uniformity
    lab = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Apply CLAHE for better detail
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)
    
    # Apply contrast adjustment
    alpha = 1.0 + (contrast / 100.0)
    l_channel = cv2.convertScaleAbs(l_channel, alpha=alpha, beta=0)
    
    # Generate layers with equal tonal distribution
    layers = []
    step = 256 / num_layers
    
    for i in range(num_layers):
        lower = int(i * step)
        upper = int((i + 1) * step) if i < num_layers - 1 else 256
        
        # Create mask for this tonal range
        mask = np.zeros_like(l_channel, dtype=np.uint8)
        mask[(l_channel >= lower) & (l_channel < upper)] = 255
        
        # Morphological operations to clean up
        kernel_size = 3 if num_layers > 6 else 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and simplify contours
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= config['min_area']:
                # Adaptive simplification based on layer count
                epsilon = 0.003 * cv2.arcLength(contour, True) if num_layers > 6 else 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) >= 3:
                    filtered_contours.append(approx)
        
        # Convert to SVG path
        svg_path = contours_to_svg_path(filtered_contours, width, height)
        
        layers.append({
            'index': i,
            'pathData': svg_path,
            'brightness': int((lower + upper) / 2),
            'tonalRange': f'{lower}-{upper}'
        })
    
    return {
        'success': True,
        'layers': layers,
        'width': width,
        'height': height,
        'mode': config.get('mode', 'standard'),
        'modeName': config['name']
    }

def contours_to_svg_path(contours, width, height):
    """Convert OpenCV contours to SVG path data"""
    if not contours or len(contours) == 0:
        return ""
    
    path_data = []
    
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Create path
        first_point = contour[0][0]
        path = f"M {first_point[0]} {first_point[1]}"
        
        for point in contour[1:]:
            path += f" L {point[0][0]} {point[0][1]}"
        
        path += " Z"
        path_data.append(path)
    
    return " ".join(path_data)

@app.route('/api/vectorize', methods=['POST'])
def vectorize():
    """API endpoint to vectorize an image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        result = process_image_to_layers(
            data['image'],
            data.get('mode', 'posterized'),
            data.get('contrast')  # Will use mode default if None
        )
        
        return jsonify(result), 200 if result['success'] else 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/modes', methods=['GET'])
def get_modes():
    """Get available modes with their default settings"""
    return jsonify({
        'success': True,
        'modes': {
            key: {
                'name': val['name'],
                'description': val['description'],
                'layers': val['layers'],
                'defaultContrast': val['default_contrast']
            } 
            for key, val in MODES.items()
        }
    }), 200

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'service': 'papercut-vectorizer-v2.1'}), 200

if __name__ == '__main__':
    print("Papercut Portrait API v2.1 running on port 5001")
    print("Available modes:", list(MODES.keys()))
    print("Mode defaults:", {k: f"{v['default_contrast']}% contrast" for k, v in MODES.items()})
    app.run(host='0.0.0.0', port=5001, debug=False)
