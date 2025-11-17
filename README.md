# Prompt Injection Detector

This project implements a hybrid machine learning system using **TF-IDF + Logistic Regression** and **BERT (bert-base-uncased)** to detect malicious prompt injections.
This AI is still being fine-tuned and in-progress. Expect frequent updates.

## Features

* Data cleaning and preprocessing
* TF-IDF vectorization + Logistic Regression with GridSearchCV
* BERT fine-tuning using HuggingFace Trainer
* Confidence-based model switching
* Interactive user input for live predictions

## Dataset

This program uses a csv file that is 2500 lines. Future updates will increase the size for more reliable consistency. 
You can currently replace the prompt.csv file with proper format csv or a parquet file with read_parquet.

with columns:
text,label


Labels must be:
injection, safe


## Running the Program

The script will:

1. Clean and vectorize text for TF-IDF
2. Fine-tune a BERT model
3. Evaluate both models
4. Enter an interactive prompt mode where you can type sample prompts

## Output

The program displays:

* Logistic Regression accuracy
* Confusion matrix heatmap
* BERT evaluation metrics
* Real-time prompt injection predictions

# Future Improvements:

1. This model can currently be manipulated do to in insufficient dataset size. Working on updating this model to analyze a 200000 prompt dataset.
2. This model only uses BERT when TF-IDF is confidence is low. This allows longer prompts to manipulate the output. In the future both model answers will always be taken into account.
