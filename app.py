import streamlit as st
import librosa
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Singer Recognition", page_icon="🎤", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🎵 Predict Singer", "📊 Upload Your Own Dataset"])

# ==========================================
# PAGE 1: PREDICT SINGER (PRE-TRAINED MODEL)
# ==========================================
if page == "🎵 Predict Singer":
    st.title("🎤 Audio Singer Recognition")
    st.write("Upload an audio file (.wav, .mp3) to identify the singer based on the pre-trained model.")

    @st.cache_resource
    def load_models():
        try:
            model = joblib.load('svc_singer_model.pkl')
            encoder = joblib.load('label_encoder.pkl')
            return model, encoder
        except FileNotFoundError:
            return None, None

    model, encoder = load_models()

    if model is None or encoder is None:
        st.warning("⚠️ Pre-trained model files not found. Please ensure 'svc_singer_model.pkl' and 'label_encoder.pkl' are in the directory.")

    def extract_features(file_path):
        y, sr = librosa.load(file_path, duration=30) 
        mfccs_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
        chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        features = list(mfccs_mean) + [centroid_mean, zcr_mean, chroma_mean]
        return np.array(features).reshape(1, -1)

    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format='audio/wav')
        
        if st.button("Predict Singer"):
            if model is not None and encoder is not None:
                with st.spinner("Analyzing audio and extracting features..."):
                    temp_file_path = f"temp_{uploaded_audio.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_audio.getbuffer())
                    
                    try:
                        features = extract_features(temp_file_path)
                        prediction_num = model.predict(features)
                        singer_name = encoder.inverse_transform(prediction_num)[0]
                        st.success(f"### Predicted Singer: **{singer_name}**")
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

# ==========================================
# PAGE 2: UPLOAD YOUR OWN DATASET
# ==========================================
elif page == "📊 Upload Your Own Dataset":
    st.title("Upload Your Own Dataset")
    st.write("Train a custom Support Vector Classifier (SVC) by uploading your extracted feature datasets in **CSV format**.")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("1. Upload Training Dataset (CSV)", type=["csv"])
    with col2:
        test_file = st.file_uploader("2. Upload Testing Dataset (CSV)", type=["csv"])

    if train_file is not None and test_file is not None:
        # Load datasets
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        st.write("### Training Data Preview")
        st.dataframe(train_df.head())

        # Let the user select the target/label column
        target_col = st.selectbox("Select the Target/Label Column (e.g., 'Singer' or 'Label')", train_df.columns)

        if st.button("Train Model & Get Outputs"):
            with st.spinner("Training Custom Model..."):
                try:
                    # Separate features (X) and labels (y)
                    X_train = train_df.drop(columns=[target_col])
                    y_train = train_df[target_col]
                    
                    X_test = test_df.drop(columns=[target_col])
                    y_test = test_df[target_col]

                    # Create and train the pipeline (Scaler + SVC)
                    pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
                    pipeline.fit(X_train, y_train)

                    # Make predictions on the test set
                    predictions = pipeline.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)

                    st.success(f"### Custom Model Accuracy: {accuracy * 100:.2f}%")

                    # Display the results dataframe
                    results_df = pd.DataFrame({
                        'Actual Label': y_test.values,
                        'Predicted Label': predictions
                    })
                    
                    st.write("### Testing Dataset Output Comparison")
                    st.dataframe(results_df)

                except Exception as e:
                    st.error(f"An error occurred during training: {e}. Please ensure your CSV contains only numeric features and one label column.")