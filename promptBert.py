import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import seaborn as sns
import re
import numpy as np

#reads data and data analysis

data = pd.read_csv("prompt.csv")
data.info
print(data)

#data cleaning and feature Engineering

data["label"]=data["label"].str.lower().str.strip()
data["label"]=data["label"].map({'injection':1, 'safe':0})

def skPrepdata(df): 
    #removes noise of unneccesary charactors/spaces from text
    df["text"] = df["text"].apply(lambda x: re.sub(r'[^a-z\s]', '', re.sub(r'\s+', ' ', x.lower())).strip())
    df.drop_duplicates(subset="text", inplace=True)
    #convert label to readable format
    df.dropna(subset="label", inplace=True)
    return df

skData=skPrepdata(data)

#Prepping data to be used by bert
def bertPrepdata(df):
    #remove whitespace and invalid charactors
    df["text"]= df["text"].apply(lambda x: re.sub(r'[^\x00-\x7F]+','',re.sub(r'\s+',' ', x.lower())).strip())
    return df

bertData=bertPrepdata(data)

#Dataset for data
class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


#Create Features / Target Variables (Make Flashcards)

X=skData.drop(columns=["label"])
y=skData["label"]

#TF-IDF 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
#BERT
XBertTrain, XBertTest, yBertTrain, yBertTest = train_test_split(bertData["text"], bertData["label"], test_size=0.25, random_state=42, stratify=bertData["label"])

#ML preprocessing

#TF-IDF Vectorization
vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train["text"])
X_test=vectorizer.transform(X_test["text"])

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

trainToken = tokenizer(
    XBertTrain.tolist(),
    padding=True,
    truncation=True,
    max_length=384,
    return_tensors="pt"
)

testToken = tokenizer(
    XBertTest.tolist(),
    padding=True,
    truncation=True,
    max_length=384,
    return_tensors="pt"
)

#Create Datasets
trainDataset=BertDataset(trainToken, yBertTrain.tolist())
testDataset=BertDataset(testToken, yBertTest.tolist())


#Hyperparemeter Tuning - Logistic Regression and Train Bert

def log_regress(X_train,y_train):
    param_grid= {
        #Implements regularization using the option with the highest cross-validation accuracy
        "C": [0.01,0.1,1,10],
        #l2 promotes small but non-zero values, higher stability
        "penalty": ["l2"],
        #lbfgs works with l2 and is fast on given dataset size
        "solver": ["lbfgs"]
    }
    #This function uses a Logistic Regression model
    model=LogisticRegression(max_iter=200)
    grid_search=GridSearchCV(model,param_grid,cv=5,n_jobs=-1)
    grid_search.fit(X_train,y_train)
    return grid_search.best_estimator_

best_model=log_regress(X_train,y_train)

trainingModel=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
#set bert arguments for trainer
arguments = TrainingArguments(
    output_dir='./trainedDetector',
    metric_for_best_model='eval_loss',
    overwrite_output_dir=True,
    do_train=True,
    load_best_model_at_end=True,
    save_strategy='epoch',
    eval_strategy='epoch',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01
)

def metrics(eval):
    logits,labels=eval
    predictions=np.argmax(logits,axis=-1)
    precision,recall,f1,_=precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy=accuracy_score(labels,predictions)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#Initialize Bert trainer
trainer = Trainer(
    model=trainingModel,
    args=arguments,
    train_dataset=trainDataset,
    eval_dataset=testDataset,
    tokenizer=tokenizer,
    compute_metrics=metrics
)

trainer.train()
stats = trainer.evaluate()
print(stats)

#Predictions and evaluation

def evaluate_model(model,X_test,y_test):
    #Gives dataset+model statistics
    predictions=model.predict(X_test)
    accuracy=accuracy_score(y_test,predictions)
    matrix=confusion_matrix(y_test,predictions)
    return accuracy,matrix

accuracy, matrix=evaluate_model(best_model, X_test,y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)


#Plot on heatmap

def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Safe","Injection"], yticklabels=["Safe", "Injection"])
    plt.title(f"Accuracy: {accuracy*100:.2f}%")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Values")
    plt.show()

plot_model(matrix)

#Allow for user interaction

#Burt Processing for input
def inputTokenizer(user_input): 
    #Tokenize user input
    clean_input = re.sub(r'[^\x00-\x7F]+','', re.sub(r'\s+', ' ', user_input.lower())).strip()
    trainToken = tokenizer(
        clean_input,
        padding=True,
        truncation=True,
        max_length=384,
        return_tensors="pt"
        )
    #Runs input through trained model 
    with torch.no_grad():
        bertOutput=trainingModel(**trainToken)

    #Preps and outputs probability(prob) and prediction(pred)
    logits=bertOutput.logits
    prob=torch.softmax(logits, dim=-1).detach().numpy()[0]
    pred=np.argmax(prob)
    return prob, pred
    
    #Recieve user input
while True:
    user_input = input("Enter your prompt or EXIT: ")
    if user_input == "EXIT":
        exit()

    else:
        #EMBEDDINGS on Prompt
        prob, pred = inputTokenizer(user_input)
        #TF-IDF on Prompt
        clean_input = re.sub(r'[^a-z\s]', '', re.sub(r'\s+', ' ', user_input.lower())).strip()
        input_vector = vectorizer.transform([clean_input])
        prediction=best_model.predict(input_vector)[0]
        probability = best_model.predict_proba(input_vector)[0]
    
        #When TF-IDF isn't confident use bert instead
        #TF-IDF has lower confidence in its output when manipulation methods are used
        if 0.3 < probability[1] < 0.7:
            outputResult = pred
        else:
            outputResult = prediction
        
        if outputResult == 1:
            print("\033[31mThis prompt may be malicious.\033[0m\n")
        elif outputResult == 0:
            print("\033[32mThis prompt is safe.\033[0m\n")
        else:
            print("\033[31mThere was an error.\033[0m\n")


"""
ADD PREDICTION OUTPUT:
print(f'The probability this prompt is malicious is: {probability[1]*100:.2f}%.',"\033[0m\n")
"""
