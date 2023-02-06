from featureExtraction import train_gsd, test_gsd
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.feature_extraction import DictVectorizer

#https://www.cs.bgu.ac.il/~elhadad/nlp17/hw1.html

def add_pseudo_word(ls_toks):
    # add two pseudo words to the start and the end of the sentence to simplify the treatment
    tok = ls_toks
    tok[:0] = ["#", "#"]
    tok.extend(["#", "#"])
    return tok

def has_number(str):
    return any(char.isdigit() for char in str)

def to_feature_representation(corpus):
    #obtain feature representation of every word in the corpus
    feature_vec = []
    pos_encoder = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PRON", "PROPN", "PUNCT", "SCONJ","SYM", "VERB", "X"]
    y = []
    for sent in corpus:
        for tok, label in zip(make_feature_vec(sent[0]), sent[1]):
            feature_vec.append(tok)
            y.append(pos_encoder.index(label))
    return feature_vec, y

def make_feature_vec(sent):
    ls_prefix = ["prev_prev_word", "prev_word", "curr_word", "next_word", "next_next_word"]

    # a temporary list where the unfinished feature vectors of the sentence is stored
    feature_vec = []
    for i in range(2, len(sent) - 2):
        # initiate the feature vector for the current word
        todo = {prefix: w for prefix, w in zip(ls_prefix, sent[(i - 2):(i + 3)]) if w != "#"}
        # verify other features specific to the current word tok[i]
        todo["biais"] = 1
        if sent[i][0].isupper():
            todo["start_with_upper"] = 1
        if has_number(sent[i]):
            todo["has_number"] = 1
        # update the feature vector
        feature_vec.append(todo)
        #print(todo)
    return feature_vec

feat_train, y_train = to_feature_representation(train_gsd)
feat_test, y_test = to_feature_representation(test_gsd)
print(len(feat_train))
print(len(feat_test))
print(len(y_train))
print(len(y_test))
vec = DictVectorizer()
X_train = vec.fit_transform(feat_train)
X_test = vec.transform(feat_test)

print(X_train.shape)
print(X_test.shape)

#print(y_train)
#print(X_test)

clf_perceptron = Perceptron()
clf_perceptron.fit(X = X_train, y= y_train)
clf_perceptron.score(X_test, y_test)
print(f"Sklearn perceptrion: The score on the train corpus is: " "{:.00%}".format(clf_perceptron.score(X_train, y_train)))
print(f"Sklearn perceptionThe score on the test corpus is: " " {:.00%}".format(clf_perceptron.score(X_test, y_test)))

clf_logistic_r = LogisticRegression(solver='lbfgs', max_iter=100)
clf_logistic_r.fit(X_train, y_train)
print(f"Sklearn LR: The score on the train corpus is {clf_perceptron.score(X_train, y_train)}")
print(f"Sklearn LR: the score on the test corpus is: {clf_perceptron.score(X_test, y_test)}")