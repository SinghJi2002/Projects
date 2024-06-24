
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

with open('commonsympt.csv', 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']
data = pd.read_csv('commonsympt.csv', encoding=encoding)
my_dataframe_sympt = pd.DataFrame(data)

def find_matching_tokens(tokens, dataframe):
    matched_tokens = []
    for token in tokens:
        matching_rows = dataframe[dataframe['Symptoms'].str.contains(token, case=False)]
        if not matching_rows.empty:
            matched_tokens.append(token)
    return matched_tokens

# Example usage
user_input=input("Enter your description: ")
#user_input = "A 45-year-old male with persistent cough , fatigue and a hearing sensitivity"
processed_tokens = preprocess_text(user_input)
print(processed_tokens)
matching_words = find_matching_tokens(processed_tokens, my_dataframe_sympt)
print("Matched words:", matching_words)
