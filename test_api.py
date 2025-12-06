import requests
import json

API_URL = "http://localhost:5000"

print("=" * 60)
print("Testing Flask API")
print("=" * 60)

# Test 1: Check if API is running
print("\n[Test 1] Checking API status...")
try:
    response = requests.get(f"{API_URL}/api/test")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ API Running")
        print(f"  Total Diseases: {data['total_diseases']}")
        print(f"  Total Symptoms: {data['total_symptoms']}")
    else:
        print("✗ API not responding")
        exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    print("Make sure Flask server is running: python app.py")
    exit(1)

# Test 2: Get all symptoms
print("\n[Test 2] Fetching symptoms list...")
try:
    response = requests.get(f"{API_URL}/api/symptoms")
    data = response.json()
    symptoms = data['symptoms']
    print(f"✓ Retrieved {len(symptoms)} symptoms")
    print(f"  First 5 symptoms: {symptoms[:5]}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Predict disease - Common Cold
print("\n[Test 3] Testing prediction - Common Cold symptoms")
test_symptoms_1 = [
    "continuous sneezing",
    "chills",
    "fatigue",
    "cough",
    "high fever",
    "headache"
]
try:
    payload = {
        "symptoms": test_symptoms_1,
        "model": "random_forest"
    }
    response = requests.post(f"{API_URL}/api/predict", json=payload)
    data = response.json()
    
    if data['status'] == 'success':
        print(f"✓ Prediction successful")
        print(f"  Disease: {data['disease']}")
        print(f"  Confidence: {data['confidence']}%")
        print(f"  Model: {data['model_used']}")
        print(f"\n  Description: {data['description'][:100]}...")
        print(f"\n  Precautions:")
        for i, prec in enumerate(data['precautions'], 1):
            print(f"    {i}. {prec}")
    else:
        print(f"✗ Prediction failed: {data.get('message')}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Predict disease - Diabetes
print("\n[Test 4] Testing prediction - Diabetes symptoms")
test_symptoms_2 = [
    "fatigue",
    "weight loss",
    "restlessness",
    "lethargy",
    "irregular sugar level",
    "blurred and distorted vision",
    "increased appetite",
    "polyuria"
]
try:
    payload = {
        "symptoms": test_symptoms_2,
        "model": "svm"
    }
    response = requests.post(f"{API_URL}/api/predict", json=payload)
    data = response.json()
    
    if data['status'] == 'success':
        print(f"✓ Prediction successful")
        print(f"  Disease: {data['disease']}")
        print(f"  Confidence: {data['confidence']}%")
        print(f"  Model: {data['model_used']}")
    else:
        print(f"✗ Prediction failed: {data.get('message')}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Predict disease - Malaria
print("\n[Test 5] Testing prediction - Malaria symptoms")
test_symptoms_3 = [
    "chills",
    "vomiting",
    "high fever",
    "sweating",
    "headache",
    "nausea",
    "muscle pain"
]
try:
    payload = {
        "symptoms": test_symptoms_3,
        "model": "naive_bayes"
    }
    response = requests.post(f"{API_URL}/api/predict", json=payload)
    data = response.json()
    
    if data['status'] == 'success':
        print(f"✓ Prediction successful")
        print(f"  Disease: {data['disease']}")
        print(f"  Confidence: {data['confidence']}%")
        print(f"  Model: {data['model_used']}")
    else:
        print(f"✗ Prediction failed: {data.get('message')}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 6: Test with invalid symptoms
print("\n[Test 6] Testing with invalid symptoms")
test_symptoms_4 = [
    "random symptom",
    "xyz",
    "invalid"
]
try:
    payload = {
        "symptoms": test_symptoms_4
    }
    response = requests.post(f"{API_URL}/api/predict", json=payload)
    data = response.json()
    
    if data['status'] == 'success':
        print(f"✓ API responded")
        print(f"  Disease: {data['disease']}")
        print(f"  Confidence: {data['confidence']}%")
    else:
        print(f"✗ Error (expected): {data.get('message')}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("Testing Complete!")
print("=" * 60)
print("\nNext Steps:")
print("1. Backend is ready ✓")
print("2. Start Android development")
print("3. Connect Android app to this API")
print("=" * 60)