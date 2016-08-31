import sys
import collections
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from collections import Counter
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
	"""
	Returns the feature vectors for all text in the train and test datasets.
	"""
	# English stopwords from nltk
	stopwords = set(nltk.corpus.stopwords.words('english'))

	# Determine a list of words that will be used as features. 
	# This list should have the following properties:
	#   (1) Contains no stop words
	#   (2) Is in at least 1% of the positive texts or 1% of the negative texts
	#   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
	features = []

	pos_length = len(train_pos)	
	neg_length = len(train_neg)
	
	pos_tweets = [set(tweet) for tweet in train_pos]
	pos_features = [word for tweet in pos_tweets for word in tweet]

	neg_tweets = [set(tweet) for tweet in train_neg]
	neg_features = [word for tweet in neg_tweets for word in tweet]
	
	pos_features_no_stopwords = [word for word in pos_features if word not in list(stopwords)]
	neg_features_no_stopwords = [word for word in neg_features if word not in list(stopwords)]
	
	pos_features_count = Counter(pos_features_no_stopwords)
	neg_features_count = Counter(neg_features_no_stopwords)
	
	pos_features_1_percent = [word for word in pos_features_no_stopwords if pos_features_count[word] >= (0.01*pos_length)]
	neg_features_1_percent = [word for word in neg_features_no_stopwords if neg_features_count[word] >= (0.01*neg_length)]

	pos_features_1_count = Counter(pos_features_1_percent)
	neg_features_1_count = Counter(neg_features_1_percent)

	pos_features_final = [word for word in pos_features_1_percent if pos_features_1_count[word] >= (2*neg_features_1_count.get(word, 0))]
	neg_features_final = [word for word in neg_features_1_percent if neg_features_1_count[word] >= (2*pos_features_1_count.get(word, 0))]

	features = set(pos_features_final + neg_features_final)
	
	# Using the above words as features, construct binary vectors for each text in the training and test set.
	# These should be python lists containing 0 and 1 integers.
	train_pos_vec = [] 
	train_neg_vec = []
	test_pos_vec = []
	test_neg_vec = []

	for tweet in train_pos:
		wordlist = []
		for word in features:
			if word in tweet:
				wordlist.append(1)
			else:
				wordlist.append(0)
		train_pos_vec.append(wordlist)
	
	for tweet in train_neg:
		wordlist = []
		for word in features:
			if word in tweet:
				wordlist.append(1)
			else:
				wordlist.append(0)
		train_neg_vec.append(wordlist)

	for tweet in test_pos:
		wordlist = []
		for word in features:
			if word in tweet:
				wordlist.append(1)
			else:
				wordlist.append(0)
		test_pos_vec.append(wordlist)

	for tweet in test_neg:
		wordlist = []
		for word in features:
			if word in tweet:
				wordlist.append(1)
			else:
				wordlist.append(0)
		test_neg_vec.append(wordlist)
		
	# Return the four feature vectors
	return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
	"""
	Returns the feature vectors for all text in the train and test datasets.
	"""
	# Doc2Vec requires LabeledSentence objects as input.
	# Turn the datasets from lists of words to lists of LabeledSentence objects.
	labeled_train_pos = [] 
	labeled_train_neg = []
	labeled_test_pos = []
	labeled_test_neg = []

	for i,tweet in enumerate(train_pos):
		label = 'TRAIN_POS_%s'%i
		labeled_train_pos.append(LabeledSentence(tweet, [label]))

	for i,tweet in enumerate(train_neg):
		label = 'TRAIN_NEG_%s'%i
		labeled_train_neg.append(LabeledSentence(tweet, [label]))

	for i,tweet in enumerate(test_pos):
		label = 'TEST_POS_%s'%i
		labeled_test_pos.append(LabeledSentence(tweet, [label]))

	for i,tweet in enumerate(test_neg):
		label = 'TEST_NEG_%s'%i
		labeled_test_neg.append(LabeledSentence(tweet, [label]))

	# Initialize model
	model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
	sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
	model.build_vocab(sentences)

	# Train the model
	# This may take a bit to run 
	for i in range(5):
		  print "Training iteration %d" % (i)
		  random.shuffle(sentences)
		  model.train(sentences)

	# Use the docvecs function to extract the feature vectors for the training and test data
	train_pos_vec = [] 
	train_neg_vec = []
	test_pos_vec = []
	test_neg_vec = []
	for i in range(len(train_pos)):
		label = 'TRAIN_POS_%s'%i
		train_pos_vec.append(model.docvecs[label])

	for i in range(len(train_neg)):
		label = 'TRAIN_NEG_%s'%i
		train_neg_vec.append(model.docvecs[label])

	for i in range(len(test_pos)):
		label = 'TEST_POS_%s'%i
		test_pos_vec.append(model.docvecs[label])

	for i in range(len(test_neg)):
		label = 'TEST_NEG_%s'%i
		test_neg_vec.append(model.docvecs[label])

	# Return the four feature vectors
	return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def build_models_NLP(train_pos_vec, train_neg_vec):
	"""
	Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
	"""
	Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

	# Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
	# For BernoulliNB, use alpha=1.0 and binarize=None
	# For LogisticRegression, pass no parameters
	
	X = train_pos_vec + train_neg_vec
	
	nb_model = BernoulliNB(alpha=1.0, binarize=None)
	nb_model.fit(X, Y)

	lr_model = LogisticRegression()
	lr_model.fit(X, Y)

	return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
	"""
	Returns a GaussianNB and LosticRegression Model that are fit to the training data.
	"""
	Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

	# Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
	# For LogisticRegression, pass no parameters
	X = train_pos_vec + train_neg_vec

	nb_model = GaussianNB()
	nb_model.fit(X, Y)

	lr_model = LogisticRegression()
	lr_model.fit(X, Y)	
	
	return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
	"""
	Prints the confusion matrix and accuracy of the model.
	"""
	# Use the predict function and calculate the true/false positives and true/false negative.
	predicted_pos = model.predict(test_pos_vec)
	predicted_neg = model.predict(test_neg_vec)
	tp = 0
	fn = 0
	tn = 0
	fp = 0
	for sentiment in predicted_pos:
		if sentiment == 'pos':
			tp = tp + 1
		else:
			fn = fn + 1
	for sentiment in predicted_neg:
		if sentiment == 'neg':
			tn = tn + 1
		else:
			fp = fp + 1
	accuracy = (float)(tp + tn)/(tp + fn + tn + fp)
	if print_confusion:
		  print "predicted:\tpos\tneg"
		  print "actual:"
		  print "pos\t\t%d\t%d" % (tp, fn)
		  print "neg\t\t%d\t%d" % (fp, tn)
	print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
