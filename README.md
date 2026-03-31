🎤 Singer Detection Using Audio Files

A Machine Learning-based web application that identifies the singer of an uploaded audio file using audio feature extraction and classification techniques.

🌐 Live Demo

👉 https://singerdetection-byaudiofiles-siri.streamlit.app/

📌 Project Overview

This project focuses on audio classification, where the goal is to predict the singer from a given audio clip. It processes the audio, extracts meaningful features, and uses a trained machine learning model to make predictions.

The application is deployed using Streamlit, providing a simple and interactive interface for users.

🚀 Features

🎧 Upload audio files (WAV format recommended)
🔍 Extracts advanced audio features
🤖 Predicts the singer using ML model
⚡ Fast and user-friendly interface
🌐 Fully deployed web application

🧠 How It Works

1. Audio Preprocessing
Load audio file
Normalize and standardize sampling rate
2. Feature Extraction

The following features are extracted:

MFCC (Mel-Frequency Cepstral Coefficients)
Chroma Features
Spectral Contrast
Zero Crossing Rate

3. Model Training
Dataset labeled with singer names
Features used to train a classification model

5. Prediction
Extract features from uploaded audio
Feed into trained model
Display predicted singer


🛠️ Tech Stack

Programming Language: Python

Libraries:

Librosa (audio processing)
NumPy, Pandas (data handling)
Scikit-learn (machine learning)
Framework: Streamlit (web app)

📂 Project Structure

Singer_Detection_Using_AudioFiles/
│
├── app.py                 # Streamlit app
├── model.pkl             # Trained model
├── features.py           # Feature extraction code
├── requirements.txt      # Dependencies
├── dataset/              # Audio dataset (optional)
└── README.md             # Project documentation

▶️ Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/Sirisha-Nukala/Singer_Detection_Using_AudioFiles.git
cd Singer_Detection_Using_AudioFiles

2️⃣ Create virtual environment (recommended)

python -m venv venv
venv\Scripts\activate   # For Windows

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run the application

streamlit run app.py

📸 Usage

Open the web app

Upload an audio file

Click on Predict

View the predicted singer 🎶

⚠️ Limitations

Works best with clean and noise-free audio
Limited to trained singers only
Accuracy depends on dataset size and quality

🔮 Future Enhancements
🎯 Improve accuracy using Deep Learning (CNN/RNN)
🎵 Support more audio formats (MP3, FLAC)
📊 Display confidence scores
📚 Expand dataset with more singers
🎛️ Add audio visualization

🙌 Acknowledgements

Librosa Documentation
Scikit-learn Community
Streamlit for easy deployment

📬 Contact

Feel free to reach out for feedback or collaboration!

⭐ If you like this project

Give it a ⭐ on GitHub and share it!
