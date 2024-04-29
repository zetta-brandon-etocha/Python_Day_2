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

### File Traitement ###
# Opening it
myfile = open('NLP.pdf'.replace('\n', ' '), mode= 'rb')
pdf_reader = PdfReader(myfile)
pages=[]
# Storing it
for i in range(pdf_reader._get_num_pages()):
    pages.append(pdf_reader.pages[i])
all_pages = pdf_reader.pages[0:]
#closing the reeader
myfile.close

### Spacy traitment ###
# Storing the language model
nlp = spacy.load("en_core_web_lg")
example=pages[6].extract_text()

# Tokenization
data = []
doc1 = nlp(example)

for token in doc1:
    data.append(token.text)

# Vectorization
corpus = example.replace('\n','.').replace('•','').lower().split('.')
corpus.remove('')
corpus.remove(' ')
print(corpus)

vectoriser = TfidfVectorizer()

#learn the vocab
X = vectoriser.fit_transform(corpus)
print(vectoriser.get_feature_names_out)

# Language detection & translation

translate_in_french = Translator(provider='libre', from_lang='fr', to_lang='en')
translated_corpus = []

for sentence in corpus:
    if detect(sentence) == 'en':
        print('This sentence is in English.')
        translated_corpus.append(translate_in_french.translate(sentence))
    elif detect(sentence) == 'fr':
        print('This sentence is in French.')
        translated_corpus.append(sentence)
    else :
        print("This neither an English nor a French sentence.")
        translated_corpus.append(sentence)

print("- Final Result -")

for sentence in translated_corpus:
    print(sentence)
    