import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import logging
from datetime import datetime
from flask import  session
from itsdangerous import URLSafeTimedSerializer as Serializer
from sqlalchemy.exc import IntegrityError
from unittest.mock import patch, MagicMock
import re
import uuid  

from datetime import datetime, timedelta 
logging.basicConfig(level=logging.INFO) 

app = Flask(__name__) 
app.config['SECRET_KEY'] = 'mysecretkey1234567890' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///SkinDisease.db' 
UPLOAD_FOLDER = 'static/uploads' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

class Member(db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    password = db.Column(db.String(50), nullable=False)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    username = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)  

    

disease_names = [
    'Acne and Rosacea', 
    'Actinic Keratosis Basal Cell Carcinoma',
    'Bullous',
    'Eczema',
    'Hair Loss',
    'Light Disease',
    'Melanoma',
    'Nail Fungus', 
    'Psoriasis',
    'Seborrheic Keratoses', 
    'Tinea Ringworm Candidiasis', 
    'Warts Molluscum'
]

def allowed_file(filename): 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    if isinstance(image, str): 
        image = cv2.imread(image) 

    if image is None:
        print("Error: Unable to load the image.")
        return None

    if len(image.shape) < 3 or image.shape[2] < 3:
        print("Error: Input image should have at least 3 channels.")
        return None

    # Convert BGR to YUV color space
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    image = cv2.resize(image, (224, 224))

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = image.astype(np.float32) / 255.0
    return image

model_path = 'models/MobileNetV1.h5' 
model = load_model(model_path)
logging.info("Custom-trained MobileNet-based model loaded successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST']) 
def detect():
    try:
        if 'file' not in request.files:  
            return jsonify({'error': 'No file part'}), 400  

        file = request.files['file']  
        if file.filename == '':  
            return jsonify({'error': 'No selected file'}), 400  

        if file and allowed_file(file.filename):  
            filename = str(uuid.uuid4()) + '.png'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image_path = os.path.join('uploads', filename).replace('\\', '/')
            image = preprocess_image(file_path)

            predictions = model.predict(np.expand_dims(image, axis=0))
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class = disease_names[int(predicted_class_index)]
            accuracy = np.max(predictions) * 100

            if 'user_id' in session:
                new_image = Image(disease=predicted_class, accuracy=accuracy, username=session['user_id'], image_path=image_path, upload_date=datetime.utcnow() )
                db.session.add(new_image)
                db.session.commit()

            return jsonify({
                'result': predicted_class,
                'accuracy': f"{accuracy:.2f}%"
            }), 200
    except Exception as e: 
        logging.error(f"Error predicting image: {e}")  
        return jsonify({'error': 'Error predicting the image.'}), 500  



from flask import flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash 

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST': 
        username = request.form.get('username') 
        password = request.form.get('password') 

        # Check if any field is empty
        if any (field is None or field == '' for field in [username, password]): 
            error = 'All fields are required' 
            return render_template('login.html', error=error)

        user = Member.query.filter_by(username=username).first()

        if user:
            if check_password_hash(user.password, password):
               # Login successful
                session.clear()
                session['user_id'] = user.id 
                session['username'] = user.username
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                # Login failed
                error = 'Invalid username or password'
                return render_template('login.html', error=error)
        else:
            # User not found
            error = 'User does not exist. Please check your username.'
            return render_template('login.html', error=error)
    return render_template('login.html')

from datetime import datetime
from sqlalchemy.exc import IntegrityError

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        dob_str = request.form.get('dob')
        sex = request.form.get('sex')
        password = request.form.get('password')
        confirm_password = request.form.get('confirmPassword')

        # Check if any field is empty
        if any (field is None or field == '' for field in [username, email, dob_str, sex, password, confirm_password]):
            error = 'All fields are required'
            return render_template('signup.html', error=error)
    
        # Check if passwords match
        if password != confirm_password:
            error = 'Passwords do not match'
            return render_template('signup.html', error=error)
        if len(password) < 8:
            error = 'Password must be at least 8 characters long.'
            return render_template('signup.html', error=error)
        
        if not re.search("[a-z]", password):
            error = 'Password must containg at least one lowercase letter.'
            return render_template('signup.html', error=error)
        
        if not re.search("[A-Z]", password):
            error = 'Password must contain at least one uppercase letter.'
            return render_template('signup.html', error=error)
        
        if not re.search("[0-9]", password):
            error = 'Password must contain at least one digit.'
            return render_template('signup.html', error=error)
        
        if not re.search("[!@#$%^&*(),.?\":{}|<>]", password):
            error = 'Password must contain at least one special character.'
            return render_template('signup.html', error=error)

        try:
            # Convert dob_str to date object
            dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256') #  the user's password is hashed
            new_user = Member(username=username, email=email, date_of_birth=dob, sex=sex, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))

        except IntegrityError:
            error = 'Username or Email already exists, Please use a different one.'
            return render_template('signup.html', error=error)

        except Exception as e:
            logging.error(f"Error creating user: {e}")
            error = f'An error occurred while creating your account: {e}'
            return render_template('signup.html', error=error)
        
    return render_template('signup.html')

@app.route('/logout')
def logout():
    # Clear the user session
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/view_history')
def view_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    detections = Image.query.filter_by(username=user_id).all()

    return render_template('viewHistroy.html', detections=detections)

@app.route('/check_db')
def check_db():
    try:
        user_count = Member.query.count()
        return f"Database connected! Total users: {user_count}", 200
    except Exception as e:
        return f"Database connection error: {e}", 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
