from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from rec3 import predict_compatibility_image_only, get_recommendations_for_item, ALL_CATEGORIES, ALL_GENDERS
import os
import shutil
import tempfile
import atexit

app = Flask(__name__)

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------------------------------------------------------
# TEMP DIR CLEANUP
# --------------------------------------------------------------------------
def cleanup_temp_dir():
    """Deletes the temporary upload directory on app exit."""
    try:
        shutil.rmtree(UPLOAD_FOLDER)
        print(f"🧹 Cleaned up temporary upload directory: {UPLOAD_FOLDER}")
    except Exception as e:
        print(f"⚠️ Error cleaning up temp directory: {e}")

atexit.register(cleanup_temp_dir)

def allowed_file(filename):
    """Checks if the uploaded file is valid (jpg/jpeg/png)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------------------------
@app.route('/')
def home():
    """Render homepage with upload form."""
    return render_template('index1.html',
                           categories=ALL_CATEGORIES,
                           genders=ALL_GENDERS)

# --------------------------------------------------------------------------
# 1️⃣ COMPATIBILITY CHECK (TWO IMAGES)
# --------------------------------------------------------------------------
@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles two uploaded images and predicts compatibility score."""
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')

    if not image1 or not image2:
        return render_template('result1.html', error='Please upload two images.'), 400

    if not (allowed_file(image1.filename) and allowed_file(image2.filename)):
        return render_template('result1.html', error='Invalid file type. Only JPG, JPEG, PNG allowed.'), 400

    temp_path_1, temp_path_2 = None, None

    try:
        filename1 = secure_filename(image1.filename)
        filename2 = secure_filename(image2.filename)

        temp_path_1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        temp_path_2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        image1.save(temp_path_1)
        image2.save(temp_path_2)

        score = predict_compatibility_image_only(temp_path_1, temp_path_2)
        score_percent = round(score * 100, 2)

        return render_template('result1.html',
                               score=score_percent,
                               mode='Compatibility')

    except (ValueError, RuntimeError) as e:
        return render_template('result1.html', error=str(e), mode='Compatibility')
    except Exception as e:
        print(f"❌ Unexpected error in compatibility: {e}")
        return render_template('result1.html',
                               error="An unexpected server error occurred.",
                               mode='Compatibility')
    finally:
        for path in [temp_path_1, temp_path_2]:
            if path and os.path.exists(path):
                os.remove(path)

# --------------------------------------------------------------------------
# 2️⃣ SINGLE IMAGE → RECOMMENDATIONS
# --------------------------------------------------------------------------
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Handles a single uploaded image and metadata to find compatible outfit pairings."""
    if 'image_item' not in request.files:
        return render_template('result1.html', error='Please upload a single image for recommendations.'), 400
    
    image_item = request.files['image_item']
    gender = request.form.get('gender_item')
    category = request.form.get('category_item')

    if not all([image_item, gender, category]):
        return render_template('result1.html',
                               error='Please upload image and select both Gender & Category.'), 400

    if not allowed_file(image_item.filename):
        return render_template('result1.html',
                               error='Invalid file type. Only JPG, JPEG, PNG allowed.'), 400

    temp_path = None
    
    try:
        # Save uploaded image temporarily
        filename = secure_filename(image_item.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_item.save(temp_path)

        # Call the recommendation function
        recommendations = get_recommendations_for_item(
            temp_path, gender, category, top_n=5
        )

        # --- NEW: Copy recommended images to static/images/ ---
        STATIC_IMG_FOLDER = os.path.join(app.root_path, 'static', 'images')
        os.makedirs(STATIC_IMG_FOLDER, exist_ok=True)

        for rec in recommendations:
            original_path = rec['image_path']  # original path from CSV
            if os.path.exists(original_path):
                file_basename = os.path.basename(original_path)
                dst_path = os.path.join(STATIC_IMG_FOLDER, file_basename)
                
                # Copy if not already in static
                if not os.path.exists(dst_path):
                    shutil.copy(original_path, dst_path)

                # Update path for HTML
                rec['image_path'] = f'images/{file_basename}'
            else:
                # If the image file doesn't exist, show a placeholder or skip
                rec['image_path'] = 'images/placeholder.png'

        # Render the results page
        return render_template('result1.html', 
                               item_category=category,
                               item_gender=gender,
                               recommendations=recommendations, 
                               mode='Recommendation')
        
    except (ValueError, RuntimeError) as e:
        return render_template('result1.html', error=str(e), mode='Recommendation')
    except Exception as e:
        print(f"Unexpected error in recommendation route: {e}")
        return render_template('result1.html', error="Unexpected server error occurred.", mode='Recommendation')
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
# --------------------------------------------------------------------------
# RUN APP
# --------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
