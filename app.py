import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Initialize with smaller, memory-efficient model
try:
    # Use a lighter sentiment model to reduce memory usage
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_length=512,
        truncation=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to even smaller model
    sentiment_analyzer = pipeline("sentiment-analysis")

@app.route('/', methods=['GET'])
def home():
    return "Sentiment Analysis API with optimized BERT is running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Empty input text'}), 400
        
        # Limit text length to reduce memory usage
        if len(text) > 512:
            text = text[:512]
        
        results = sentiment_analyzer(text)
        label = results[0]['label'].lower()
        confidence = results[0]['score']
        
        return jsonify({
            'sentiment': label,
            'confidence': round(confidence, 4)
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# CRITICAL: Proper port binding for Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
