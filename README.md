# 🍲 Filipino Food AI Chatbot

A chatbot web application that provides information about Filipino cuisine using TensorFlow/Keras

## ✨ Features

- 🍛 Generates authentic Filipino food descriptions
- 💬 Interactive chat interface
- 🚀 REST API endpoint

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
# Clone repository
git clone https://github.com/yourusername/filipino-food-chatbot.git
cd filipino-food-chatbot

# Install dependencies
pip install flask flask-cors transformers torch tensorflow numpy pyttsx3

## 📁 Project Structure
filipino-food-chatbot/
├── server.py                 # Main Flask application
├── templates/
│   └── index.html            # Chat interface
├── text_generation_model.keras  # TensorFlow model
├── tokenizer.pkl             # Keras tokenizer
└── README.md
└── train_model.py            # Script to train the model
└── food_dataset.txt          # Dataset

## 🚀 Running the Application
# Run in Terminal
python server.py
# Open in Browser
http://127.0.0.1:5000