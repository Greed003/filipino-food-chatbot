from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model = tf.keras.models.load_model('text_generation_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_sequence_length = model.input_shape[1]

def generate_text(seed_text, next_words=15):
    """Generates text based on a seed question without repeating words."""
    used_words = set(seed_text.split())
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')
        
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        output_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_index and word not in used_words:
                output_word = word
                used_words.add(word)
                break
        
        if output_word is None:
            break
            
        seed_text += " " + output_word
    
    return seed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Please enter a question'}), 400
    
    try:
        response = generate_text(question, next_words=15)
        return jsonify({
            'question': question,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)