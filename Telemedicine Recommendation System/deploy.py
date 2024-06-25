import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from transformers import BioGptTokenizer, BioGptForCausalLM
import pandas as pd
import chardet
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

# Load datasets
df = pd.read_csv('dataset.csv')
df2 = df.fillna("null")
with open('commonsympt.csv', 'rb') as f:
    result = chardet.detect(f.read())
df_sympt = pd.read_csv('commonsympt.csv', encoding=result['encoding'])
df_sympt.rename(columns={'Headache': 'Symptoms'}, inplace=True)

# Preprocess text
def preprocess_text(input_text):
    cleaned_text = re.sub(r'[^\w\s]', '', input_text)  # Remove special characters
    cleaned_text = cleaned_text.lower()  # Convert to lower case
    tokenized_words = word_tokenize(cleaned_text)  # Tokenize
    stop_words = set(stopwords.words('english'))  # Retrieve stopwords
    final_words = [word for word in tokenized_words if word not in stop_words]
    return final_words

# Find matching tokens
def find_matching_tokens(tokens, dataframe):
    matched_tokens = []
    for token in tokens:
        matching_rows = dataframe[dataframe['Symptoms'].str.contains(token, case=False)]
        if not matching_rows.empty:
            matched_tokens.append(token)
    return matched_tokens

# Prediction function
def prediction(user_input):
    processed_tokens = preprocess_text(user_input)
    report_disease = pd.DataFrame(columns=['disease_name'])
    add_dis = pd.DataFrame(columns=['Symptoms'])
    list_from_df = []

    for i in processed_tokens:
        for index1, row in df2.iterrows():
            list_from_df = row.tolist()
            for index in range(len(list_from_df)):
                if i.lower() == list_from_df[index].lower():
                    flag = list_from_df[0]
                    new_row = {'disease_name': flag}
                    for sym in range(1, len(row)):
                        if row[sym] != 'null':
                            new_row2 = {'Symptoms': row[sym]}
                            add_dis = add_dis.append(new_row2, ignore_index=True)
                    report_disease = report_disease.append(new_row, ignore_index=True)
                    break

    add_dis = add_dis.drop_duplicates(subset='Symptoms', keep='first')
    possibleDiseases = report_disease.drop_duplicates(subset='disease_name', inplace=False)

    listOfDiseases = list(possibleDiseases['disease_name'])
    model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")  # Load pre-trained BioGPT model
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    lifestyleChanges = []
    medicines = []

    def prompt(Input1, Input2):
        lifestyleChanges.append(generator(Input1, max_length=50, num_return_sequences=1, do_sample=True))
        medicines.append(generator(Input2, max_length=50, num_return_sequences=1, do_sample=True))

    for i in listOfDiseases:
        Input1 = "Lifestyle changes to be adopted for " + i + " are as follows"
        Input2 = "Medicines to be used for " + i + " are as follows"
        prompt(Input1, Input2)

    st.text("Here are the list of lifestyle changes to be adopted for each disease:")
    for change in lifestyleChanges:
        st.success(change[0]['generated_text'])

    st.text("Here are the list of medication changes to be adopted for each disease:")
    for medicine in medicines:
        st.success(medicine[0]['generated_text'])

# Main function
def main():
    st.title('Enter your description:')
    user_input = st.text_area("Enter Your Symptoms")
    if(st.button('Check')):
        prediction(user_input)

if __name__ == '__main__':
    main()
