import re
import nltk
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

@st.cache(allow_output_mutation=True)
def load_model():
    # load the model from disk
    model = pickle.load(open('logistic_regression.model', 'rb'))
    vectorizer = pickle.load(open('tfidf_vector.model', 'rb'))
    return model, vectorizer 
with st.spinner('Model is being loaded..'):
    model, vectorizer = load_model()


def only_alpha(text):
    return re.sub('[^A-Za-z0-9 ]+', '', text)

def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase


#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    text = text.split(' ')
    output= [i for i in text if i not in stopwords]
    return ' '.join(output)

#defining the object for stemming
porter_stemmer = PorterStemmer()
#defining a function for stemming
def stemming(text):
    text = text.split(' ')
    stem_text = [porter_stemmer.stem(word) for word in text]
    return ' '.join(stem_text)


#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
#defining the function for lemmatization
def lemmatizer(text):
    text = text.split(' ')
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return ' '.join(lemm_text)

def all_preprocessing(text_df):    
    text_df = text_df.apply(lambda x : only_alpha(x))
    text_df = text_df.apply(lambda x : x.lower())
    text_df = text_df.apply(lambda x : decontractions(x))
    text_df = text_df.apply(lambda x:remove_stopwords(x))
    text_df = text_df.apply(lambda x: stemming(x))
    text_df = text_df.apply(lambda x:lemmatizer(x))
    return text_df


st.write("""# BBC News Article Sorting""")

message_text = st.text_input("Enter the news for classifying")

def classify_message(model, message):
    new_news = all_preprocessing(pd.Series(message))
    vec_news = vectorizer.transform(new_news)
    label = model.predict(vec_news)
    return label


classes = ['business', 'entertainment', 'politics', 'sport', 'tech']
if message_text != '':
    result = classify_message(model, message_text)
    out = classes[int(result)]
    st.write('This news belongs from {0} category!!!'.format(out.upper()))

