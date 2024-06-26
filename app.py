from io import BytesIO
from flask import Flask, jsonify, request, render_template, url_for, redirect
import requests
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import PyPDF2
from PyPDF2 import PdfReader
from translate import Translator
import spacy
from langdetect import detect, detect_langs
import en_core_web_lg
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

warnings.filterwarnings(action='ignore')

nlp = spacy.load("en_core_web_lg")
pdf_target_url = "https://api.akabot-staging.zetta-demo.space/fileuploads/Artificial-Intelligence-in-Finance-6a364d95-f26c-41e6-a3a1-54f9b9f975d2.pdf"


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'


data = []

def process_the_pdf():
    ### File Traitement ###
    # Opening it
    response = requests.get(pdf_target_url)
    if response.status_code == 200:
        myfile = BytesIO(response.content)
    pdf_reader = PdfReader(myfile)
    pages=[]
    # Storing it
    for i in range(pdf_reader._get_num_pages()):
        pages.append(pdf_reader.pages[i])
    #closing the reeader
    myfile.close
    ### Spacy traitment ###
    # Storing the language model
    example=pages[0].extract_text()
    

    # Tokenization
    data = []
    doc1 = example.replace('\n',' ').split('.')

    for token in doc1:
        data.append(token)

    
    # data = example.lower().split('.')
    # corpus.remove('')
    # corpus.remove(' ')
    print(data)
    return data[1:]

def vectorization(corpus):
    # Vectorization
    
    vectoriser = TfidfVectorizer()

    #learn the vocab
    X = vectoriser.fit_transform(corpus)
    print(vectoriser.get_feature_names_out)
    return X
    

def detect_translate(text):

# Language detection & translation
    corpus = text
    translated_corpus = []

    for sentence in corpus:
        

        if detect(sentence) == 'en':
            translate_in_french = Translator(from_lang=(detect(sentence)), to_lang='fr')
            print('This sentence is in English.')
            translated_corpus.append(translate_in_french.translate(sentence))
        elif detect(sentence) == 'fr':
            print('This sentence is in French.')
            translated_corpus.append(sentence)
        elif (detect(sentence) != 'fr' or detect(sentence) != 'en') :
            print("This sentence is neither an English nor a French sentence.")
            translated_corpus.append(sentence)
        else :
            print('No feature')
            translated_corpus.append(sentence)

    print("- Final Result -")

    for sentence in translated_corpus:
        print(sentence)

    return translated_corpus
    
def calculate_similarity(text1, text2):
    # logger.info("Calculating Similarity between two texts")
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)


@app.route('/process_pdf', methods=['POST', 'GET'])

def process_pdf():
    final_tab = []
    the_data = process_the_pdf()
    
    the_translation = detect_translate(the_data[0:5])
    print("translation", the_translation)
    for i in range(5):
        the_similarity = calculate_similarity(the_data[i], the_translation[i])

        print(f"step {i}")
        result_analyze = {
            'sentence' : the_data[i],
            'translation' : the_translation[i],
            'similarity' : the_similarity
        }
        print("result : ", result_analyze)
        final_tab.append(result_analyze)
        print("final_tab : ", final_tab)
    return final_tab


@app.route('/view_pdf', methods=['GET'])
def view_pdf():
    # Call your process_pdf function to get the data
    final_tab = process_pdf()
    return render_template('index.html', data=final_tab)




if __name__ == "__main__":
    app.run(debug=True)