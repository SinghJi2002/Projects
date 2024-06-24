import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import chardet

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(input_text):
    cleaned_text = re.sub(r'[^\w\s]', '', input_text)
    cleaned_text = cleaned_text.lower()
    tokenized_words = word_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))
    final_words = [word for word in tokenized_words if word not in stop_words]

    return final_words

# Load dataset and drop NaN values
with open('dataset.csv', 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']
data = pd.read_csv('dataset.csv', encoding=encoding)
data.dropna(inplace=True)  # Drop NaN values
my_dataframe_sympt = pd.DataFrame(data)

def find_matching_diseases(symptoms, dataframe):
    disease_matches = {}

    for _, row in dataframe.iterrows():
        disease = row['Disease']

        for column in dataframe.columns[1:]:
            cell_value = str(row[column]).lower()
            for symptom in symptoms:
                if symptom in cell_value:
                    if disease not in disease_matches:
                        disease_matches[disease] = set()
                    disease_matches[disease].add(symptom)

    return disease_matches

# User input for symptoms
user_input = input("Enter your symptoms (comma-separated): ")
user_symptoms = preprocess_text(user_input)
print("Processed Symptoms:", user_symptoms)

# Find matching diseases
matching_diseases = find_matching_diseases(user_symptoms, my_dataframe_sympt)

# Display results
if matching_diseases:
    print("Related Diseases:")
    for disease, symptoms_set in matching_diseases.items():
        symptoms_list = ', '.join(symptoms_set)
        print(f"{disease}: Matching symptoms - {symptoms_list}")
else:
    print("No diseases found for the given symptoms.")
