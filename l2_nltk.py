from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.chunk import RegexpParser

# Tokenizing using NLTK
s = '''Good muffins cost $3.88 \n in New York.  Please buy me ... two of them.\n \n Thanks.'''

word_tokenize(s) 
print(word_tokenize(s) )
# print(sent_tokenize(s))

# Filtering using stop word
nltk.download('stopwords')
print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(s)
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

# Stemming using nltk 
ps = PorterStemmer()

# choose some words to be stemmed
words = ["program", "programs", "programmer", "programming", "programmers"]
 
for w in words:
    print(w, " : ", ps.stem(w))


# Parts of speech tagging 
s = '''Good muffins cost $3.88 \n in New York.  Please buy me ... two of them.\n \n Thanks.'''

word_tokens = word_tokenize(s) 

tagged = nltk.pos_tag(word_tokens)
print(tagged)

# chunking example 
chunk_patterns = r"""
    NP: {<DT>?<JJ>*<NN>}  # Chunk noun phrases
    VP: {<VB.*><NP|PP>}  # Chunk verb phrases
"""
nltk.download('averaged_perceptron_tagger_eng')

# Named Entity Recognition
namedEnt = nltk.ne_chunk(tagged, binary=False)
namedEnt.draw()