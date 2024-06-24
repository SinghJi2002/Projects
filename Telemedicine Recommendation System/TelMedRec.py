#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import torch


# In[40]:


#Loading Dataset.
dataframe=pd.read_excel("TeleMedicine.xlsx")
#Converting integer type columns to string for searching
dataframe['Age']=dataframe["Age"].astype(str)
dataframe['PatientID']=dataframe["PatientID"].astype(str)
dataframe=dataframe.set_index("PatientID")
#Converting each element to uppercase for convienence in searching
dataframe["Medical Condition"] = dataframe["Medical Condition"].apply(lambda x: x.upper())
dataframe["Other Demographics"] = dataframe["Other Demographics"].apply(lambda x: x.upper())
dataframe["Insurance Type"] = dataframe["Insurance Type"].apply(lambda x: x.upper())
dataframe["Tech Device"] = dataframe["Tech Device"].apply(lambda x: x.upper())
dataframe["Tech Access"] = dataframe["Tech Access"].apply(lambda x: x.upper())
dataframe["Symptoms"] = dataframe["Symptoms"].apply(lambda x: x.upper())


# In[73]:


#Taking user input and processing it
print('Enter the difficulties and potential symptoms being expierenced by you')
prompt=input()
prompt = prompt.upper()


# In[74]:


#Processed Dataset
dataframe.head()


# In[75]:


'''from transformers import BertTokenizer, BertForSequenceClassification

# Load BioBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1')

# Prepare the sentence
encoded_sentence = tokenizer("I have a headache and feel nauseous.", return_tensors="pt")

# Pass through the model and interpret predictions
with torch.no_grad():
    outputs = model(**encoded_sentence)
    labels = tokenizer.convert_ids_to_tokens(outputs.logits.argmax(-1).squeeze().tolist())

# Filter predictions based on labels and confidence
potential_symptoms = [label for label in labels if label.startswith("##") and outputs.logits[0][labels.index(label)] > 0.5]

print("Potential symptoms:", potential_symptoms)  # Output: ['headache', 'nausea']'''


# In[76]:


#List of symptoms of various diseases.
symptom_list=['Joint pain', 'stiffness', 'swelling', 'redness', 'warmth', 'decreased range of motion',
              'fatigue','Wheezing', 'chest tightness', 'shortness of breath', 'cough',
              'difficulty breathing during exercise','Itching', 'redness', 'scaling', 'dryness',''
              'cracking','blistering', 'pain','Depressed mood',
              'loss of interest or pleasure in activities','changes in appetite or sleep', 'fatigue',
              'feelings of worthlessness or guilt','difficulty concentrating', 'suicidal thoughts',
              'Throbbing headache','nausea', 'vomiting','sensitivity to light and sound','Inattention'
              ,'hyperactivity', 'impulsivity','Pain in the lower back',
              'radiating pain down the leg (sciatica)', 'stiffness','difficulty moving',
              'Excessive worry','fear', 'restlessness', 'fatigue','difficulty concentrating',
              'irritability', 'muscle tension', 'sleep problems','Frequent urination',
              'increased thirst', 'fatigue', 'blurred vision','slow healing of wounds','infections']
for i in range(0,len(symptom_list),1):
  symptom_list[i]=symptom_list[i].upper()


# In[77]:


#Identifying symptoms from user prompt
con_symptom=[]
for start in range(0,len(prompt),1):
  for end in range(start,len(prompt)+1,1):
    if(prompt[start:end] in symptom_list):
      con_symptom.append(prompt[start:end])


# In[78]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[79]:


# Vectorize symptoms in the dataframe and user input
vectorizer = TfidfVectorizer()
symptoms_matrix = vectorizer.fit_transform(dataframe['Symptoms'].fillna(''))
user_input_vector = vectorizer.transform([" ".join(con_symptom)])


# In[80]:


# Calculate cosine similarity
cosine_similarities = linear_kernel(user_input_vector, symptoms_matrix).flatten()
# Get the index of the highest similarity
highest_similarity_index = cosine_similarities.argmax()
# Print the matched row
matched_row = dataframe.iloc[highest_similarity_index]


# In[81]:


print(f"User input symptoms: {con_symptom}")
print(f"Matched Medical Condition: {matched_row['Medical Condition']}")

