from flask import Flask, request, jsonify
import numpy as np
import joblib
import json
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load the mappings
with open('enhanced_api_mappings.json', 'r') as f:
    mappings = json.load(f)

def get_area_rate_from_address(address):
    """Get area rate from address by matching with mappings"""
    # Convert address to lowercase for case-insensitive matching
    address_lower = address.lower()
    
    # Search through area mappings
    for area, details in mappings['enhanced_area_mapping'].items():
        if area.lower() in address_lower:
            return details['rate']
    
    # Return None if no match found
    return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Extract features
        area = float(data['area'])
        beds = int(data['beds'])
        bathrooms = int(data['bathrooms'])
        furnishing = int(data['furnishing'])  # Now expecting furnishing as number (0, 1, or 2)
        bhk = int(data['bhk'])
        
        # Get area_rate from address if provided
        if 'address' in data:
            area_rate = get_area_rate_from_address(data['address'])
            if area_rate is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not find matching area rate for provided address'
                }), 400
        else:
            # If no address provided, expect area_rate directly
            area_rate = float(data['area_rate'])
        
        # Create feature array
        features = np.array([[
            area,
            beds,
            bathrooms,
            furnishing,
            area_rate,
            bhk
        ]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return prediction
        return jsonify({
            'status': 'success',
            'predicted_rent': float(prediction[0]),
            'message': 'Rent prediction successful',
            'used_area_rate': area_rate
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/get_mappings', methods=['GET'])
def get_mappings():
    """Return the city and area mappings"""
    return jsonify(mappings)

if __name__ == '__main__':
    app.run(debug=True, port=5000)