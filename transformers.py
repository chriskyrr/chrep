import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
texts = df['v2'].values
labels = df['v1'].values

# Split the data into training and test sets
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.9, random_state=42)

# BERT
# Tokenize the texts
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_texts_bert = tokenizer_bert(texts_train.tolist(), padding=True, truncation=True, max_length=32, return_tensors='pt')

# Prepare the labels
label_dict_bert = {'ham': 0, 'spam': 1}
label_dict_bert_reverse = {v: k for k, v in label_dict_bert.items()}  
labels_bert = torch.tensor([label_dict_bert[label] for label in labels_train])

# Load the BERT model
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Train the BERT model
optimizer = torch.optim.Adam(model_bert.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 3
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(tokenized_texts_bert['input_ids']), batch_size):
        inputs = {key: val[i:i+batch_size] for key, val in tokenized_texts_bert.items()}
        labels = labels_bert[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model_bert(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Make predictions using BERT
tokenized_texts_test_bert = tokenizer_bert(texts_test.tolist(), padding=True, truncation=True, max_length=32, return_tensors='pt')
inputs_test_bert = {key: val for key, val in tokenized_texts_test_bert.items()}
labels_test_bert = torch.tensor([label_dict_bert[label] for label in labels_test])
outputs_test_bert = model_bert(**inputs_test_bert)
predicted_labels_bert = torch.argmax(outputs_test_bert.logits, dim=1).numpy()
predicted_labels_bert = [label_dict_bert_reverse[label] for label in predicted_labels_bert]

# Calculate accuracy for BERT
accuracy_bert = accuracy_score(labels_test, predicted_labels_bert)
print(f"BERT Accuracy: {accuracy_bert}")

# Convert predicted_labels_bert to a Pandas DataFrame
df_predicted_bert = pd.DataFrame({'Label': predicted_labels_bert})

# Plot the bar chart for BERT
plt.figure(figsize=(8, 6))
sns.countplot(data=df_predicted_bert, x='Label')
plt.title('BERT Predicted Labels Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Plot the confusion matrix for BERT
confusion_mat_bert = confusion_matrix(labels_test, predicted_labels_bert)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_bert, annot=True, fmt='d', cmap='Blues')
plt.title('BERT Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# RoBERTa
# Tokenize the texts
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')
tokenized_texts_roberta = tokenizer_roberta(texts_train.tolist(), padding=True, truncation=True, max_length=32, return_tensors='pt')

# Prepare the labels
label_dict_roberta = {'ham': 0, 'spam': 1}
label_dict_roberta_reverse = {v: k for k, v in label_dict_roberta.items()}  # Define the reverse label mapping

labels_roberta = torch.tensor([label_dict_roberta[label] for label in labels_train])

# Load the RoBERTa model
model_roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Train the RoBERTa model
optimizer = torch.optim.AdamW(model_roberta.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 3
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(tokenized_texts_roberta['input_ids']), batch_size):
        inputs = {key: val[i:i+batch_size] for key, val in tokenized_texts_roberta.items()}
        labels = labels_roberta[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model_roberta(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Make predictions using RoBERTa
tokenized_texts_test_roberta = tokenizer_roberta(texts_test.tolist(), padding=True, truncation=True, max_length=32, return_tensors='pt')
inputs_test_roberta = {key: val for key, val in tokenized_texts_test_roberta.items()}
labels_test_roberta = np.array([label_dict_roberta[label] for label in labels_test])
outputs_test_roberta = model_roberta(**inputs_test_roberta)
predicted_labels_roberta = torch.argmax(outputs_test_roberta.logits, dim=1).numpy()
predicted_labels_roberta = [label_dict_roberta_reverse[label] for label in predicted_labels_roberta]

# Calculate accuracy for RoBERTa
accuracy_roberta = accuracy_score(labels_test, predicted_labels_roberta)
print(f"RoBERTa Accuracy: {accuracy_roberta}")

# Convert predicted_labels_roberta to a Pandas DataFrame
df_predicted_roberta = pd.DataFrame({'Label': predicted_labels_roberta})

# Plot the bar chart for RoBERTa
plt.figure(figsize=(8, 6))
sns.countplot(data=df_predicted_roberta, x='Label')
plt.title('RoBERTa Predicted Labels Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
# Visualize the confusion matrix for RoBERTa
confusion_mat_roberta = confusion_matrix(labels_test, predicted_labels_roberta)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_roberta, annot=True, fmt='d', cmap='Blues')
plt.title('RoBERTa Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Naive Bayes
# Vectorize the texts
vectorizer = CountVectorizer()
vectorized_texts = vectorizer.fit_transform(texts_train)

# Prepare the labels
label_dict_nb = {'ham': 0, 'spam': 1}
label_dict_nb_reverse = {v: k for k, v in label_dict_nb.items()}  # Reverse label mapping
labels_nb = np.array([label_dict_nb[label] for label in labels_train])

# Train the Naive Bayes model
model_nb = MultinomialNB()
model_nb.fit(vectorized_texts, labels_nb)

# Make predictions using Naive Bayes
vectorized_texts_test = vectorizer.transform(texts_test)
predicted_labels_nb = model_nb.predict(vectorized_texts_test)
predicted_labels_nb = [label_dict_nb_reverse[label] for label in predicted_labels_nb]  # Convert back to original labels

# Calculate accuracy for Naive Bayes
accuracy_nb = accuracy_score(labels_test, predicted_labels_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")

# Convert predicted_labels_roberta_series to a Pandas DataFrame
df_predicted_roberta = pd.DataFrame({'Label': predicted_labels_roberta_series})

# Plot the bar chart for Naive Bayes
plt.figure(figsize=(8, 6))
sns.countplot(data=df_predicted_roberta, x='Label')
plt.title('Naive Bayes Predicted Labels Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Visualize the confusion matrix for Naive Bayes
confusion_mat_nb = confusion_matrix(labels_test, predicted_labels_nb)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_nb, annot=True, fmt='d', cmap='Blues')
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()