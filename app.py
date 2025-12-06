from flask import Flask, request, jsonify
from database import Database
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# Initialize Database
db = Database()

print("=" * 50)
print("Loading ML Models...")
print("=" * 50)

# Load trained models
with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
print("✓ SVM model loaded")

with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
print("✓ Random Forest model loaded")

with open('models/nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)
print("✓ Naive Bayes model loaded")

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("✓ Label encoder loaded")

with open('models/symptoms.pkl', 'rb') as f:
    all_symptoms = pickle.load(f)
print(f"✓ Loaded {len(all_symptoms)} symptoms")

# Load description and precaution data
desc_data = pd.read_csv('datasets/symptom_Description.csv')
precaution_data = pd.read_csv('datasets/symptom_precaution.csv')
print("✓ Description and precaution data loaded")

# Load medication, diet, workout data
try:
    medication_data = pd.read_csv('datasets/Medication.csv')
    print("✓ Medication data loaded")
except Exception as e:
    medication_data = pd.DataFrame(columns=['Disease', 'Medication'])
    print(f"⚠ Medication data not found: {e}")

try:
    diet_data = pd.read_csv('datasets/Diet.csv')
    diet_data.columns = diet_data.columns.str.strip()
    print(f"✓ Diet data loaded ({len(diet_data)} diseases)")
except Exception as e:
    diet_data = pd.DataFrame(columns=['Disease', 'Diet'])
    print(f"⚠ Diet data not found: {e}")

try:
    workout_data = pd.read_csv('datasets/workout.csv')
    workout_data.columns = workout_data.columns.str.strip()
    print(f"✓ Workout data loaded ({len(workout_data)} diseases)")
except Exception as e:
    workout_data = pd.DataFrame(columns=['Disease', 'Workout'])
    print(f"⚠ Workout data not found: {e}")

print("=" * 50)
print("Flask API Ready!")
print("=" * 50)


# Helper function: Convert symptoms to input vector
def symptoms_to_vector(symptoms_list):
    """Convert user symptoms to binary vector"""
    vector = np.zeros(len(all_symptoms))
    
    for symptom in symptoms_list:
        symptom = symptom.strip().lower().replace(' ', '_')
        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            vector[idx] = 1
    
    return vector.reshape(1, -1)


# Helper function: Get disease info
def get_disease_info(disease_name):
    """Get complete information for a disease"""
    info = {
        'disease': disease_name,
        'description': 'Description not available',
        'precautions': [],
        'medications': [],
        'diet': [],
        'workout': []
    }
    
    # Get description
    desc_row = desc_data[desc_data['Disease'].str.strip() == disease_name]
    if not desc_row.empty:
        info['description'] = desc_row.iloc[0]['Description']
    
    # Get precautions
    prec_row = precaution_data[precaution_data['Disease'].str.strip() == disease_name]
    if not prec_row.empty:
        precautions = []
        for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
            if col in prec_row.columns:
                prec = prec_row.iloc[0][col]
                if pd.notna(prec) and str(prec).strip():
                    precautions.append(str(prec).strip())
        info['precautions'] = precautions

    # Get medications
    if not medication_data.empty:
        med_row = medication_data[medication_data['Disease'].str.strip() == disease_name]
        if not med_row.empty:
            meds = str(med_row.iloc[0]['Medication']).split(';')
            info['medications'] = [m.strip() for m in meds if m.strip()]
    
    # Get diet recommendations
    if not diet_data.empty and 'Diet' in diet_data.columns:
        diet_row = diet_data[diet_data['Disease'].str.strip() == disease_name]
        if not diet_row.empty:
            diets = str(diet_row.iloc[0]['Diet']).split(';')
            info['diet'] = [d.strip() for d in diets if d.strip()]
    
    # Get workout suggestions
    if not workout_data.empty and 'Workout' in workout_data.columns:
        work_row = workout_data[workout_data['Disease'].str.strip() == disease_name]
        if not work_row.empty:
            workouts = str(work_row.iloc[0]['Workout']).split(';')
            info['workout'] = [w.strip() for w in workouts if w.strip()]
    
    return info


# ============================================
# AUTHENTICATION ENDPOINTS
# ============================================

@app.route('/api/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()
        
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not name or not email or not password:
            return jsonify({
                'success': False,
                'message': 'All fields are required'
            }), 400
        
        if len(password) < 5:
            return jsonify({
                'success': False,
                'message': 'Password must be at least 5 characters'
            }), 400
        
        result = db.register_user(name, email, password)
        
        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({
                'success': False,
                'message': 'Email and password are required'
            }), 400
        
        result = db.login_user(email, password)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 401
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# ============================================
# PREDICTION ENDPOINTS
# ============================================

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """Return list of all available symptoms"""
    try:
        symptoms_formatted = [s.replace('_', ' ').title() for s in all_symptoms]
        return jsonify({
            'status': 'success',
            'symptoms': symptoms_formatted,
            'total': len(symptoms_formatted)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """Predict disease from symptoms"""
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Please provide symptoms'
            }), 400
        
        symptoms = data['symptoms']
        model_choice = data.get('model', 'random_forest')
        
        if not symptoms or len(symptoms) == 0:
            return jsonify({
                'status': 'error',
                'message': 'Symptoms list cannot be empty'
            }), 400
        
        # Convert symptoms to vector
        input_vector = symptoms_to_vector(symptoms)
        
        # Select model
        if model_choice == 'svm':
            model = svm_model
        elif model_choice == 'naive_bayes':
            model = nb_model
        else:
            model = rf_model
        
        # Predict
        prediction = model.predict(input_vector)
        disease_encoded = prediction[0]
        disease_name = label_encoder.inverse_transform([disease_encoded])[0]
        
        # Get confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_vector)[0]
            confidence = float(probabilities[disease_encoded]) * 100
        else:
            confidence = 85.0
        
        # Get disease information
        disease_info = get_disease_info(disease_name)
        
        return jsonify({
            'status': 'success',
            'disease': disease_name,
            'confidence': round(confidence, 2),
            'model_used': model_choice,
            'description': disease_info['description'],
            'precautions': disease_info['precautions'],
            'medications': disease_info['medications'],
            'diet': disease_info['diet'],
            'workout': disease_info['workout'],
            'input_symptoms': symptoms
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================
# HISTORY ENDPOINTS
# ============================================

@app.route('/api/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    """Get prediction history for user"""
    try:
        history = db.get_user_history(user_id)
        return jsonify({
            'success': True,
            'history': history
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/save_prediction', methods=['POST'])
def save_prediction():
    """Save prediction to history"""
    try:
        data = request.json
        
        result = db.save_prediction(
            user_id=data.get('user_id'),
            disease=data.get('disease'),
            confidence=data.get('confidence'),
            model_used=data.get('model_used'),
            symptoms=data.get('symptoms', []),
            description=data.get('description'),
            medications=data.get('medications', []),
            precautions=data.get('precautions', []),
            diet=data.get('diet', []),
            workout=data.get('workout', [])
        )
        
        if result['success']:
            return jsonify({"status": "success", "message": "Prediction saved"}), 201
        else:
            return jsonify({"status": "error", "message": result['message']}), 500
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/history/<int:prediction_id>', methods=['DELETE'])
def delete_history(prediction_id):
    """Delete prediction from history"""
    try:
        result = db.delete_prediction(prediction_id)
        
        if result['success']:
            return jsonify({"status": "success", "message": "Deleted"}), 200
        else:
            return jsonify({"status": "error", "message": result['message']}), 500
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================
# PROFILE ENDPOINTS
# ============================================

@app.route('/api/user/update_profile', methods=['POST'])
def update_profile():
    """Update user profile (name, email, photo)"""
    try:
        data = request.json
        user_id = data.get('user_id')
        new_name = data.get('name', '').strip()
        new_email = data.get('email', '').strip().lower()
        profile_photo = data.get('profile_photo')
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'User ID required'
            }), 400
        
        result = db.update_user_profile(
            user_id=user_id,
            new_name=new_name if new_name else None,
            new_email=new_email if new_email else None,
            profile_photo=profile_photo
        )
        
        if result['success']:
            return jsonify({'status': 'success', 'message': result['message']}), 200
        else:
            return jsonify({'status': 'error', 'message': result['message']}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/user/change_password', methods=['POST'])
def change_password():
    """Change user password"""
    try:
        data = request.json
        user_id = data.get('user_id')
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not all([user_id, current_password, new_password]):
            return jsonify({
                'status': 'error',
                'message': 'All fields required'
            }), 400
        
        if len(new_password) < 5:
            return jsonify({
                'status': 'error',
                'message': 'Min 5 characters'
            }), 400
        
        result = db.change_user_password(user_id, current_password, new_password)
        
        if result['success']:
            return jsonify({'status': 'success', 'message': result['message']}), 200
        else:
            return jsonify({'status': 'error', 'message': result['message']}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'API Running',
        'total_symptoms': len(all_symptoms),
        'total_diseases': len(label_encoder.classes_)
    }), 200


# if __name__ == '__main__':
#     print("\n" + "=" * 50)
#     print("Flask API Starting...")
#     print("=" * 50)
#     app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Flask API Starting...")
    print("=" * 50)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)