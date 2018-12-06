from __future__ import division

# import matplotlib.pyplot as plt
import math
import os
import time
import operator
import numpy
from   collections import defaultdict, Counter
from   sklearn import datasets
from   sklearn.linear_model import LogisticRegression

# Global class labels.
RAP_POS_LABEL = 'popular_rap'
RAP_NEG_LABEL = 'unpopular_rap'
COUNTRY_POS_LABEL = 'popular_country'
COUNTRY_NEG_LABEL = 'unpopular_country'

# Global popularity labels.
RAP_EXTREMELY_POS_LABEL = 'extremely_popular_rap'
RAP_VERY_POS_LABEL = 'very_popular_rap'
RAP_STANDARD_POS_LABEL = 'standard_popular_rap'
RAP_STANDARD_NEG_LABEL = 'extremely_unpopular_rap'
RAP_VERY_NEG_LABEL = 'very_unpopular_rap'
RAP_EXTREMELY_NEG_LABEL = 'standard_unpopular_rap'
COUNTRY_EXTREMELY_POS_LABEL = 'extremely_popular_country'
COUNTRY_VERY_POS_LABEL = 'very_popular_country'
COUNTRY_STANDARD_POS_LABEL = 'standard_popular_country'
COUNTRY_STANDARD_NEG_LABEL = 'extremely_unpopular_country'
COUNTRY_VERY_NEG_LABEL = 'very_unpopular_country'
COUNTRY_EXTREMELY_NEG_LABEL = 'standard_unpopular_country'

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)

def popularity_labeling(genre, plays):
        if genre == 'rap':
            if int(plays) < 100000:
                return RAP_EXTREMELY_NEG_LABEL
            elif int(plays) >= 100000 and int(plays) < 500000:
                return RAP_VERY_NEG_LABEL
            elif int(plays) >= 500000 and int(plays) < 1000000:
                return RAP_STANDARD_NEG_LABEL
            elif int(plays) >= 1000000 and int(plays) < 10000000:
                return RAP_STANDARD_POS_LABEL
            elif int(plays) >= 10000000 and int(plays) < 100000000:
                return RAP_VERY_POS_LABEL
            elif int(plays) >= 100000000:
                return RAP_EXTREMELY_POS_LABEL
        elif genre == 'country':
            if int(plays) < 100000:
                return COUNTRY_EXTREMELY_NEG_LABEL
            elif int(plays) >= 100000 and int(plays) < 500000:
                return COUNTRY_VERY_NEG_LABEL
            elif int(plays) >= 500000 and int(plays) < 1000000:
                return COUNTRY_STANDARD_NEG_LABEL
            elif int(plays) >= 1000000 and int(plays) < 10000000:
                return COUNTRY_STANDARD_POS_LABEL
            elif int(plays) >= 10000000 and int(plays) < 100000000:
                return COUNTRY_VERY_POS_LABEL
            elif int(plays) >= 100000000:
                return COUNTRY_EXTREMELY_POS_LABEL


class NaiveBayesTextClassification:
    def __init__(self, path_to_data, tokenizer, popularity=popularity_labeling):
        # Vocabulary is a set that stores every word seen in the training data
        self.rap_vocab = set()
        self.country_vocab = set()
        self.path_to_data = path_to_data
        self.tokenize_doc = tokenizer
        self.popularity_labeling = popularity
        self.rap_dir = os.path.join(path_to_data, "rap")
        self.country_dir = os.path.join(path_to_data, "country")
        self.rap_train_dir = os.path.join(self.rap_dir, "train")
        self.rap_test_dir = os.path.join(self.rap_dir, "test")
        self.country_train_dir = os.path.join(self.country_dir, "train")
        self.country_test_dir = os.path.join(self.country_dir, "test")

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { RAP_EXTREMELY_POS_LABEL: 0.0,
                                        RAP_VERY_POS_LABEL: 0.0,
                                        RAP_STANDARD_POS_LABEL: 0.0,
                                        RAP_STANDARD_NEG_LABEL: 0.0,
                                        RAP_VERY_NEG_LABEL: 0.0,
                                        RAP_EXTREMELY_NEG_LABEL: 0.0,
                                        COUNTRY_EXTREMELY_POS_LABEL: 0.0,
                                        COUNTRY_VERY_POS_LABEL: 0.0,
                                        COUNTRY_STANDARD_POS_LABEL: 0.0,
                                        COUNTRY_STANDARD_NEG_LABEL: 0.0,
                                        COUNTRY_VERY_NEG_LABEL: 0.0,
                                        COUNTRY_EXTREMELY_NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { RAP_EXTREMELY_POS_LABEL: 0.0,
                                         RAP_VERY_POS_LABEL: 0.0,
                                         RAP_STANDARD_POS_LABEL: 0.0,
                                         RAP_STANDARD_NEG_LABEL: 0.0,
                                         RAP_VERY_NEG_LABEL: 0.0,
                                         RAP_EXTREMELY_NEG_LABEL: 0.0,
                                         COUNTRY_EXTREMELY_POS_LABEL: 0.0,
                                         COUNTRY_VERY_POS_LABEL: 0.0,
                                         COUNTRY_STANDARD_POS_LABEL: 0.0,
                                         COUNTRY_STANDARD_NEG_LABEL: 0.0,
                                         COUNTRY_VERY_NEG_LABEL: 0.0,
                                         COUNTRY_EXTREMELY_NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { RAP_EXTREMELY_POS_LABEL: defaultdict(float),
                                   RAP_VERY_POS_LABEL: defaultdict(float),
                                   RAP_STANDARD_POS_LABEL: defaultdict(float),
                                   RAP_STANDARD_NEG_LABEL: defaultdict(float),
                                   RAP_VERY_NEG_LABEL: defaultdict(float),
                                   RAP_EXTREMELY_NEG_LABEL: defaultdict(float),
                                   COUNTRY_EXTREMELY_POS_LABEL: defaultdict(float),
                                   COUNTRY_VERY_POS_LABEL: defaultdict(float),
                                   COUNTRY_STANDARD_POS_LABEL: defaultdict(float),
                                   COUNTRY_STANDARD_NEG_LABEL: defaultdict(float),
                                   COUNTRY_VERY_NEG_LABEL: defaultdict(float),
                                   COUNTRY_EXTREMELY_NEG_LABEL: defaultdict(float) }

    def train_model(self):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.
        """
        pos_rap_path = os.path.join(self.rap_train_dir, RAP_POS_LABEL)
        neg_rap_path = os.path.join(self.rap_train_dir, RAP_NEG_LABEL)
        pos_country_path = os.path.join(self.country_train_dir, COUNTRY_POS_LABEL)
        neg_country_path = os.path.join(self.country_train_dir, COUNTRY_NEG_LABEL)

        for (p, label) in [ (pos_rap_path, RAP_POS_LABEL), (neg_rap_path, RAP_NEG_LABEL) ]:
            genre = 'rap'
            for f in os.listdir(p):
                with open(os.path.join(p,f), encoding="ISO-8859-1") as doc:
                    play_count = f.split('~')[1].split('.')[0]
                    content = doc.read()
                    self.tokenize_and_update_model(content, genre, play_count)

        for (p, label) in [ (pos_country_path, COUNTRY_POS_LABEL), (neg_country_path, COUNTRY_NEG_LABEL) ]:
            genre = 'country'
            for f in os.listdir(p):
                with open(os.path.join(p,f), encoding="ISO-8859-1") as doc:
                    play_count = f.split('~')[1].split('.')[0]
                    content = doc.read()
                    self.tokenize_and_update_model(content, genre, play_count)

        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print ("REPORTING CORPUS STATISTICS")

        print ("NUMBER OF RAP SONGS IN EXTREMELY POPULAR CLASS:", self.class_total_doc_counts[RAP_EXTREMELY_POS_LABEL])
        print ("NUMBER OF RAP SONGS IN VERY POPULAR CLASS:", self.class_total_doc_counts[RAP_VERY_POS_LABEL])
        print ("NUMBER OF RAP SONGS IN STANDARD POPULAR CLASS:", self.class_total_doc_counts[RAP_STANDARD_POS_LABEL])
        print ("NUMBER OF RAP SONGS IN EXTREMELY UNPOPULAR CLASS:", self.class_total_doc_counts[RAP_EXTREMELY_NEG_LABEL])
        print ("NUMBER OF RAP SONGS IN VERY UNPOPULAR CLASS:", self.class_total_doc_counts[RAP_VERY_NEG_LABEL])
        
        print ("NUMBER OF RAP SONGS IN STANDARD UNPOPULAR CLASS:", self.class_total_doc_counts[RAP_STANDARD_NEG_LABEL])
        print ("NUMBER OF COUNTRY SONGS IN EXTREMELY POPULAR CLASS:", self.class_total_doc_counts[COUNTRY_EXTREMELY_POS_LABEL])
        print ("NUMBER OF COUNTRY SONGS IN VERY POPULAR CLASS:", self.class_total_doc_counts[COUNTRY_VERY_POS_LABEL])
        print ("NUMBER OF COUNTRY SONGS IN STANDARD POPULAR CLASS:", self.class_total_doc_counts[COUNTRY_STANDARD_POS_LABEL])
        print ("NUMBER OF COUNTRY SONGS IN EXTREMELY UNPOPULAR CLASS:", self.class_total_doc_counts[COUNTRY_EXTREMELY_NEG_LABEL])
        print ("NUMBER OF COUNTRY SONGS IN VERY UNPOPULAR CLASS:", self.class_total_doc_counts[COUNTRY_VERY_NEG_LABEL])
        print ("NUMBER OF COUNTRY SONGS IN STANDARD UNPOPULAR CLASS:", self.class_total_doc_counts[COUNTRY_STANDARD_NEG_LABEL])

        print ("NUMBER OF TOKENS IN RAP EXTREMELY POPULAR CLASS:", self.class_total_word_counts[RAP_EXTREMELY_POS_LABEL])
        print ("NUMBER OF TOKENS IN RAP VERY POPULAR CLASS:", self.class_total_word_counts[RAP_VERY_POS_LABEL])
        print ("NUMBER OF TOKENS IN RAP STANDARD POPULAR CLASS:", self.class_total_word_counts[RAP_STANDARD_POS_LABEL])
        print ("NUMBER OF TOKENS IN RAP EXTREMELY UNPOPULAR CLASS:", self.class_total_word_counts[RAP_EXTREMELY_NEG_LABEL])
        print ("NUMBER OF TOKENS IN RAP VERY UNPOPULAR CLASS:", self.class_total_word_counts[RAP_VERY_NEG_LABEL])
        print ("NUMBER OF TOKENS IN RAP STANDARD UNPOPULAR CLASS:", self.class_total_word_counts[RAP_STANDARD_NEG_LABEL])
        
        print ("NUMBER OF TOKENS IN COUNTRY EXTREMELY POPULAR CLASS:", self.class_total_word_counts[COUNTRY_EXTREMELY_POS_LABEL])
        print ("NUMBER OF TOKENS IN COUNTRY VERY POPULAR CLASS:", self.class_total_word_counts[COUNTRY_VERY_POS_LABEL])
        print ("NUMBER OF TOKENS IN COUNTRY STANDARD POPULAR CLASS:", self.class_total_word_counts[COUNTRY_STANDARD_POS_LABEL])
        print ("NUMBER OF TOKENS IN COUNTRY EXTREMELY UNPOPULAR CLASS:", self.class_total_word_counts[COUNTRY_EXTREMELY_NEG_LABEL])
        print ("NUMBER OF TOKENS IN COUNTRY VERY UNPOPULAR CLASS:", self.class_total_word_counts[COUNTRY_VERY_NEG_LABEL])
        print ("NUMBER OF TOKENS IN COUNTRY STANDARD UNPOPULAR CLASS:", self.class_total_word_counts[COUNTRY_STANDARD_NEG_LABEL])

        print ("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS FOR RAP:", len(self.rap_vocab))
        print ("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS FOR COUNTRY:", len(self.country_vocab))

    def tokenize_and_update_model(self, doc, genre, plays):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """
        label = self.popularity_labeling(genre, plays)
        word_counts = Counter(tokenize_doc(doc))
        self.update_model(word_counts, label, genre)

    def update_model(self, bow, label, genre):
        """
        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each label (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each label (self.class_total_doc_counts)
        """        
        for item in bow.items():
            if genre == 'rap':
                if item[0] not in self.rap_vocab: self.rap_vocab.add(item[0])
            elif genre == 'country':
                if item[0] not in self.country_vocab: self.country_vocab.add(item[0])
            if item[0] in self.class_word_counts[label]:
                self.class_word_counts[label][item[0]] += item[1]
            else: 
                self.class_word_counts[label][item[0]] = item[1]
            self.class_total_word_counts[label] += item[1]
        self.class_total_doc_counts[label] += 1

    def p_word_given_label_and_alpha(self, word, label, genre, alpha):
        """
        Returns the probability of word given label wrt psuedo counts.
        alpha - smoothing parameter
        """
        if genre == 'rap':
            return (self.class_word_counts[label][word] + alpha) / (self.class_total_word_counts[label] + len(self.rap_vocab) * alpha)
        elif genre == 'country':
            return (self.class_word_counts[label][word] + alpha) / (self.class_total_word_counts[label] + len(self.country_vocab) * alpha)

    def log_likelihood(self, bow, label, genre, alpha):
        """
        Computes the log likelihood of a set of words given a label and smoothing.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; smoothing parameter
        """
        log_sum = 0.0
        for item in bow.items():
            if label == RAP_EXTREMELY_POS_LABEL or label == RAP_EXTREMELY_NEG_LABEL or label == COUNTRY_EXTREMELY_POS_LABEL or label ==COUNTRY_EXTREMELY_NEG_LABEL:
                multiplier = 3
            elif label == RAP_VERY_POS_LABEL or label == RAP_VERY_NEG_LABEL or label == COUNTRY_VERY_POS_LABEL or label == COUNTRY_VERY_NEG_LABEL:
                multiplier = 2
            elif label == RAP_STANDARD_POS_LABEL or label == RAP_STANDARD_NEG_LABEL or label == COUNTRY_STANDARD_POS_LABEL or label == COUNTRY_STANDARD_NEG_LABEL:
                multiplier = 1
            log_sum += item[1]*math.log(multiplier) + item[1]*math.log(self.p_word_given_label_and_alpha(item[0], label, genre, alpha))
        return log_sum

    def log_prior(self, label, genre):
        """
        Returns the log prior of a document having the class 'label'.
        """
        if genre == 'rap':
            return math.log(self.class_total_doc_counts[label] / (self.class_total_doc_counts[RAP_EXTREMELY_POS_LABEL] + self.class_total_doc_counts[RAP_VERY_POS_LABEL] + self.class_total_doc_counts[RAP_STANDARD_POS_LABEL] + self.class_total_doc_counts[RAP_EXTREMELY_NEG_LABEL] + self.class_total_doc_counts[RAP_VERY_NEG_LABEL] + self.class_total_doc_counts[RAP_STANDARD_NEG_LABEL]))
        elif genre == 'country':
            return math.log(self.class_total_doc_counts[label] / (self.class_total_doc_counts[COUNTRY_EXTREMELY_POS_LABEL] + self.class_total_doc_counts[COUNTRY_VERY_POS_LABEL] + self.class_total_doc_counts[COUNTRY_STANDARD_POS_LABEL] + self.class_total_doc_counts[COUNTRY_EXTREMELY_NEG_LABEL] + self.class_total_doc_counts[COUNTRY_VERY_NEG_LABEL] + self.class_total_doc_counts[COUNTRY_STANDARD_NEG_LABEL]))

    def unnormalized_log_posterior(self, bow, label, genre, alpha):
        """
        Computes the unnormalized log posterior (of doc being of class 'label').
        bow - a bag of words (i.e., a tokenized document)
        """
        unnormalized_likelihood = self.log_likelihood(bow, label, genre, alpha)
        unnormalized_prior = self.log_prior(label, genre)
        return unnormalized_likelihood + unnormalized_prior

    def classify(self, bow, genre, alpha):
        """
        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior)
        bow - a bag of words (i.e., a tokenized document)
        """
        if genre == 'rap':
            max_label = None
            labels = [RAP_EXTREMELY_POS_LABEL, RAP_VERY_POS_LABEL, RAP_STANDARD_POS_LABEL, RAP_STANDARD_NEG_LABEL, RAP_VERY_NEG_LABEL, RAP_EXTREMELY_NEG_LABEL]
            for l in labels:
                if max_label:
                    max_label = max(max_label, (l, self.unnormalized_log_posterior(bow, l, genre, alpha)), key=lambda x: x[1])
                else:
                    max_label = (l, self.unnormalized_log_posterior(bow, l, genre, alpha))
            return max_label[0]
        elif genre == 'country':
            max_label = None
            labels = [COUNTRY_EXTREMELY_POS_LABEL, COUNTRY_VERY_POS_LABEL, COUNTRY_STANDARD_POS_LABEL, COUNTRY_STANDARD_NEG_LABEL, COUNTRY_VERY_NEG_LABEL, COUNTRY_EXTREMELY_NEG_LABEL]
            for l in labels:
                if max_label:
                    max_label = max(max_label, (l, self.unnormalized_log_posterior(bow, l, genre, alpha)), key=lambda x: x[1])
                else:
                    max_label = (l, self.unnormalized_log_posterior(bow, l, genre, alpha))
            return max_label[0]

    def evaluate_classifier_accuracy(self, genre, alpha):
        """
        alpha - pseudocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        generally_correct = 0.0
        partially_correct = 0.0
        exact_correct = 0.0
        total = 0.0

        if genre == 'rap':
            pos_path = os.path.join(self.rap_test_dir, RAP_POS_LABEL)
            neg_path = os.path.join(self.rap_test_dir, RAP_NEG_LABEL)
            for (p, label) in [ (pos_path, RAP_POS_LABEL), (neg_path, RAP_NEG_LABEL) ]:
                for f in os.listdir(p):
                    with open(os.path.join(p,f), encoding="ISO-8859-1") as doc:
                        content = doc.read()
                        bow = self.tokenize_doc(content)
                        play_count = f.split('~')[1].split('.')[0]
                        correct_label = self.popularity_labeling(genre, play_count)
                        calculated_label = self.classify(bow, genre, alpha)
                        # print(correct_label, calculated_label)
                        if calculated_label == correct_label:
                            generally_correct += 1.0
                            partially_correct += 1.0
                            exact_correct += 1.0
                        elif calculated_label.split('_')[1] == correct_label.split('_')[1]:
                            generally_correct += 1.0
                            partially_correct += 0.5
                        total += 1.0
            return (100 * generally_correct / total, 100 * partially_correct / total, 100 * exact_correct / total)
        elif genre == 'country':
            pos_path = os.path.join(self.country_test_dir, COUNTRY_POS_LABEL)
            neg_path = os.path.join(self.country_test_dir, COUNTRY_NEG_LABEL)
            for (p, label) in [ (pos_path, COUNTRY_POS_LABEL), (neg_path, COUNTRY_NEG_LABEL) ]:
                for f in os.listdir(p):
                    with open(os.path.join(p,f), encoding="ISO-8859-1") as doc:
                        content = doc.read()
                        bow = self.tokenize_doc(content)
                        play_count = f.split('~')[1].split('.')[0]
                        correct_label = self.popularity_labeling(genre, play_count)
                        calculated_label = self.classify(bow, genre, alpha)
                        # print(correct_label, calculated_label)
                        if calculated_label == correct_label:
                            generally_correct += 1.0
                            partially_correct += 1.0
                            exact_correct += 1.0
                        elif calculated_label.split('_')[1] == correct_label.split('_')[1]:
                            generally_correct += 1.0
                            partially_correct += 0.5
                        total += 1.0
            return (100 * generally_correct / total, 100 * partially_correct / total, 100 * exact_correct / total)

class LogisticRegressionTextClassification:
    def __init__(self, path_to_data, tokenizer, popularity):
        self.rap_features = []
        self.country_features = []
        self.rap_labels = []
        self.country_labels = []
        self.path_to_data = path_to_data
        self.tokenize_doc = tokenizer
        self.popularity_labeling = popularity
        self.rap_dir = os.path.join(path_to_data, "rap")
        self.country_dir = os.path.join(path_to_data, "country")
        self.rap_train_dir = os.path.join(self.rap_dir, "train")
        self.rap_test_dir = os.path.join(self.rap_dir, "test")
        self.country_train_dir = os.path.join(self.country_dir, "train")
        self.country_test_dir = os.path.join(self.country_dir, "test")

    def feature_generation(self):
        """
        Feature generation for the document after tokenizing.
        """
        pos_rap_train_path = os.path.join(self.rap_train_dir, RAP_POS_LABEL)
        neg_rap_train_path = os.path.join(self.rap_train_dir, RAP_NEG_LABEL)
        pos_rap_test_path = os.path.join(self.rap_test_dir, RAP_POS_LABEL)
        neg_rap_test_path = os.path.join(self.rap_test_dir, RAP_NEG_LABEL)
        pos_country_train_path = os.path.join(self.country_train_dir, COUNTRY_POS_LABEL)
        neg_country_train_path = os.path.join(self.country_train_dir, COUNTRY_NEG_LABEL)
        pos_country_test_path = os.path.join(self.country_test_dir, COUNTRY_POS_LABEL)
        neg_country_test_path = os.path.join(self.country_test_dir, COUNTRY_NEG_LABEL)

        for (p, label) in [ (pos_rap_train_path, RAP_POS_LABEL), (neg_rap_train_path, RAP_NEG_LABEL), (pos_rap_test_path, RAP_POS_LABEL), (neg_rap_test_path, RAP_NEG_LABEL) ]:
            genre = 'rap'
            for f in os.listdir(p):
                with open(os.path.join(p,f), encoding="ISO-8859-1") as doc:
                    content = doc.read()
                    word_counts = Counter(tokenize_doc(content))
                    title_words = f.split('-')[1].split('~')[0].split('+')
                    play_count = f.split('~')[1].split('.')[0]
                    popularity_label = popularity_labeling(genre, play_count)

                    # Features
                    length = self.song_length(word_counts)
                    unique = self.unique_words(word_counts, length)
                    repeated = self.total_repeated_words(word_counts, length)
                    repeated_unique = self.repeated_unique_words(word_counts)
                    most_frequent = self.most_frequent_word(word_counts)
                    average = self.average_word_length(word_counts, length)
                    title = self.frequency_title_words(word_counts, title_words, length)
                    self.rap_features.append([unique, repeated, repeated_unique, most_frequent, length, average, title])

                    if popularity_label == RAP_EXTREMELY_NEG_LABEL:
                        self.rap_labels.append(0)
                    elif popularity_label == RAP_VERY_NEG_LABEL:
                        self.rap_labels.append(1)
                    elif popularity_label == RAP_STANDARD_NEG_LABEL:
                        self.rap_labels.append(2)
                    elif popularity_label == RAP_STANDARD_POS_LABEL:
                        self.rap_labels.append(3)
                    elif popularity_label == RAP_VERY_POS_LABEL:
                        self.rap_labels.append(4)
                    elif popularity_label == RAP_EXTREMELY_POS_LABEL:
                        self.rap_labels.append(5)

        for (p, label) in [ (pos_country_train_path, COUNTRY_POS_LABEL), (neg_country_train_path, COUNTRY_NEG_LABEL), (pos_country_test_path, COUNTRY_POS_LABEL), (neg_country_test_path, COUNTRY_NEG_LABEL) ]:
            genre = 'country'
            for f in os.listdir(p):
                with open(os.path.join(p,f), encoding="ISO-8859-1") as doc:
                    content = doc.read()
                    word_counts = Counter(tokenize_doc(content))
                    title_words = f.split('-')[1].split('~')[0].split('+')
                    play_count = f.split('~')[1].split('.')[0]
                    popularity_label = self.popularity_labeling(genre, play_count)

                    # Features
                    length = self.song_length(word_counts)
                    unique = self.unique_words(word_counts, length)
                    repeated = self.total_repeated_words(word_counts, length)
                    repeated_unique = self.repeated_unique_words(word_counts)
                    most_frequent = self.most_frequent_word(word_counts)
                    average = self.average_word_length(word_counts, length)
                    title = self.frequency_title_words(word_counts, title_words, length)
                    self.country_features.append([unique, repeated, repeated_unique, most_frequent, length, average, title])

                    if popularity_label == COUNTRY_EXTREMELY_NEG_LABEL:
                        self.country_labels.append(0)
                    elif popularity_label == COUNTRY_VERY_NEG_LABEL:
                        self.country_labels.append(1)
                    elif popularity_label == COUNTRY_STANDARD_NEG_LABEL:
                        self.country_labels.append(2)
                    elif popularity_label == COUNTRY_STANDARD_POS_LABEL:
                        self.country_labels.append(3)
                    elif popularity_label == COUNTRY_VERY_POS_LABEL:
                        self.country_labels.append(4)
                    elif popularity_label == COUNTRY_EXTREMELY_POS_LABEL:
                        self.country_labels.append(5)

    # Feature Generation Methods
    def unique_words(self, bow, length):
        """
        Return the ratio of the number of unique words 
        against the total number of words.
        """
        return len(bow) / length

    def total_repeated_words(self, bow, length):
        """
        Return the ratio of the total number of repeated words
        against the total number of words.
        """
        total_repetitions = 0
        for word, count in bow.items():
            if count > 1:
                total_repetitions += count
        return total_repetitions / length

    def repeated_unique_words(self, bow):
        """
        Return the ratio of the number of unique words repeated 
        against the total number of unique words.
        """
        unique_repetitions = 0
        for word, count in bow.items():
            if count > 1:
                unique_repetitions += count
        return unique_repetitions / len(bow)

    def most_frequent_word(self, bow):
        """
        Return most frequent count of the word.
        """
        answer = None
        highest = 0
        for word, count in bow.items():
            if count > highest:
                highest = count
                answer = word
        return highest

    def song_length(self, bow):
        """
        Return total number of words in a song.
        """
        total = 0
        for word, count in bow.items():
            total = total + count
        return int(total)

    def average_word_length(self, bow, length):
        """
        Return the average word length of a song.
        """
        max = 0.0
        for word, count in bow.items():
            max = max + (len(word)*count)
        else:
            return max / length


    def frequency_title_words(self, bow, title_words, length):
        """
        Return the ratio of the frequency of title words appearing in a song
        against the length of the song.
        """
        total = 0
        for word in title_words:
            if word in bow:
                total += bow[word]
        return int(total) / length

    def create_model(self, genre):
        LogReg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

        if genre == 'rap':
            feature_array = numpy.array([numpy.array(row) for row in self.rap_features])
            label_array = numpy.array(self.rap_labels)

            X = feature_array[:, :]
            Y = label_array

            LogReg.fit(X, Y)
            return 100 * LogReg.score(X, Y)
        elif genre == 'country':
            feature_array = numpy.array([numpy.array(row) for row in self.country_features])
            label_array = numpy.array(self.country_labels)

            X = feature_array[:, :]
            Y = label_array

            LogReg.fit(X, Y)
            return 100 * LogReg.score(X, Y)


def main():
    nb = NaiveBayesTextClassification('lyrics', tokenizer=tokenize_doc, popularity=popularity_labeling)
    print('\n#### NAIVE BAYES MODEL TRAINING ####\n')
    nb.train_model()

    print('\n#### RAP ACCURACY TEST ####\n')
    rap_results = nb.evaluate_classifier_accuracy('rap', 0.2)
    print('GENERALLY_CORRECT RESULT: ' + str(rap_results[0]))
    print('PARTIALLY_CORRECT RESULT: ' + str(rap_results[1]))
    print('EXACT_CORRECT RESULT: ' + str(rap_results[2]))

    print('\n#### COUNTRY ACCURACY TEST ####\n')
    country_results = nb.evaluate_classifier_accuracy('country', 0.2)
    print('GENERALLY_CORRECT RESULT: ' + str(country_results[0]))
    print('PARTIALLY_CORRECT RESULT: ' + str(country_results[1]))
    print('EXACT_CORRECT RESULT: ' + str(country_results[2]))

    print('\n#### GENERATING FEATURES ####\n')
    lr = LogisticRegressionTextClassification('lyrics', tokenizer=tokenize_doc, popularity=popularity_labeling)
    lr.feature_generation()

    print('\n#### RAP MODEL RESULTS ####\n')
    print(lr.create_model('rap'))

    print('\n#### COUNTRY MODEL RESULTS ####\n')
    print(lr.create_model('country'))

main()