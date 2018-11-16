from model import *
import matplotlib.pyplot as plt

PATH_TO_DATA = 'lyrics'

RAP_POS_LABEL = 'popular_rap'
RAP_NEG_LABEL = 'unpopular_rap'
COUNTRY_POS_LABEL = 'popular_country'
COUNTRY_NEG_LABEL = 'unpopular_country'

RAP_DIR = os.path.join(PATH_TO_DATA, "rap")
COUNTRY_DIR = os.path.join(PATH_TO_DATA, "country")

RAP_TRAIN_DIR = os.path.join(RAP_DIR, "train")
RAP_TEST_DIR = os.path.join(RAP_DIR, "test")
COUNTRY_TRAIN_DIR = os.path.join(COUNTRY_DIR, "train")
COUNTRY_TEST_DIR = os.path.join(COUNTRY_DIR, "test")

for label in [RAP_POS_LABEL, RAP_NEG_LABEL]:
	print(label)
	if label == RAP_POS_LABEL:
		print(os.listdir(RAP_TRAIN_DIR + "/" + label))
		if len(os.listdir(RAP_TRAIN_DIR + "/" + label)) == 12:
			print("Great! You have 12 {} reviews in {}".format(label, RAP_TRAIN_DIR + "/" + label))
		else:
			print("Oh no! Something is wrong. Check your code which loads the reviews")
	else:
		print(os.listdir(RAP_TRAIN_DIR + "/" + label))
		if len(os.listdir(RAP_TRAIN_DIR + "/" + label)) == 7:
			print("Great! You have 7 {} reviews in {}".format(label, RAP_TRAIN_DIR + "/" + label))
		else:
			print("Oh no! Something is wrong. Check your code which loads the reviews")

import glob
import codecs
from collections import Counter

word_counts = Counter() # Counters are often useful for NLP in python

corpus = ''
for label in [RAP_POS_LABEL, RAP_NEG_LABEL]:
    for directory in [RAP_TRAIN_DIR, RAP_TEST_DIR]:
        for fn in glob.glob(directory + "/" + label + "/*txt"):
            doc = codecs.open(fn, 'r', 'utf8') # Open the file with UTF-8 encoding
            corpus += doc.read() + ' '
    
word_counts = Counter(tokenize_doc(corpus))
print(word_counts)
print ("there are {} word types in the corpus".format(n_word_types(word_counts)))
print ("there are {} word tokens in the corpus".format(n_word_tokens(word_counts)))

import operator
sorted_dict = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)

for i in range(30):
    print(sorted_dict[i])

import math
import operator
x = []
y = []
X_LABEL = "log(rank)"
Y_LABEL = "log(frequency)"

for i in range(len(sorted_dict)):
    x += [math.log(i+1)]
    y += [math.log(sorted_dict[i][1])]

plt.scatter(x, y)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)