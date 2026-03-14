# VocalSet Singer Classification (SingerSense)

## Overview
This project applies machine learning to identify individual singers based on their vocal audio characteristics. By processing `.wav` audio files from the VocalSet dataset, the pipeline extracts key acoustic features and trains multiple classification models to recognize the singer's unique vocal footprint.


## Dataset
The project uses **VocalSet11**, a dataset of professional singers. The data processed consists of 3,613 individual audio samples across 20 distinct singers (9 female, 11 male). 

## Project Structure
* `feature_extraction.ipynb`: Crawls the `VocalSet11/FULL` directory, processes `.wav` files using `librosa`, and generates a labeled tabular dataset (`singer_features.csv`).
* `model_training.ipynb`: Loads the extracted features, scales the data, and evaluates multiple machine learning classifiers using `scikit-learn`.

## Features Extracted
For every 30-second audio window, the following features are extracted using `librosa`:
* **MFCCs (1-13):** Mel-frequency cepstral coefficients representing the short-term power spectrum of the sound.
* **Spectral Centroid:** Indicates where the "center of mass" of the spectrum is located (brightness of the sound).
* **Zero-Crossing Rate (ZCR):** The rate at which the signal changes from positive to zero to negative.
* **Chroma Mean:** Relates to the 12 different pitch classes.

## Models Evaluated
The extracted features are scaled using `StandardScaler` and passed into several models:
* **Logistic Regression:** Tested on both raw and scaled data, achieving ~73% accuracy on the scaled dataset.
* **K-Nearest Neighbors (KNN):** Evaluated across various `k` values (from 2 to 24) using scaled data to find the optimal neighbor count.
* **Decision Tree Classifier:** Evaluated on the dataset, achieving ~54-56% accuracy.
* **Linear Regression:** Used as a baseline/experimental model, evaluated using R-squared.

Evaluation metrics used across classification models include Accuracy, Precision, Recall, and F1-score (macro average).
