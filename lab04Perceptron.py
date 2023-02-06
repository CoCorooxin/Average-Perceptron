import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import random
import json
from featureExtraction import train_gsd, test_gsd, to_feature_representation
import pandas as pd
import seaborn as sns
import logging


logger = logging.getLogger('__name__')
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)

class Average_perceptron:
    """
    Initialize the parameter vector for weight update;
    the average vector for average perceptron;
    the last update where stores the time stamp of the last prediction error is made for a certain label.
    """
    parameters_vector = dict()
    average_vector = dict()
    last_update = defaultdict(int)
    ls_labels=[]
    # store the precision score for every iteration on the corpus for further analysis
    score_by_epochs = {"train":[], "test": []}

    def __init__(self, labels: List[str]):
        self.ls_labels = labels
        self.parameters_vector = {key: defaultdict(float) for key in labels}
        self.average_vector = {key: defaultdict(float) for key in labels}

    def predict(self, observation: List[str])->str:
        # observation : a list of active features
        #y is a dict that map the label with its scores
        y = defaultdict(float) #Dict[str, float]
        for label in self.average_vector.keys():
            #print(f"label is {label} and parameters is {parameters} and obersvation is {observation}")
            y[label] = self._dot(observation, label)
            #print(f"the dot product for label {label} is {y[label]}")

        return max(y, key=y.get)

    def _dot(self, observation: List[str], label: str)->float:

        return sum(self.average_vector[label][feat] for feat in observation)

    def score(self, corpus: List[Tuple[List[str], str]])->float:
        res = [self.predict(feat_vec) == gold_label for feat_vec, gold_label in corpus]
        return sum(res)/len(res)

    def _update_single_example(self, observation: List[str], gold_label: str, n_done: int)->None:
        prediction = self.predict(observation)
        if prediction != gold_label:
            for feat in observation:
                self.average_vector[gold_label][feat] += (n_done-self.last_update[(gold_label, feat)]) * self.parameters_vector[gold_label][feat]
                self.last_update[gold_label, feat] = n_done

                self.average_vector[prediction][feat] += (n_done-self.last_update[prediction, feat]) * self.parameters_vector[prediction][feat]
                self.last_update[prediction, feat] = n_done

                self.parameters_vector[gold_label][feat] += 0.1
                self.parameters_vector[prediction][feat] -= 0.1

    def fit(self, train_corpus: List[Tuple[List[str], str]], test_corpus:List[Tuple[List[str], str]], max_iter: int)-> None:
        n_iter, n_done = 0,0
        # the number of prediction that has been accomplished so far
        n_vocab = len(train_corpus)
        while n_iter < max_iter:
            n_iter += 1
            # 296695 is the # of toks in the corpus
            random.shuffle(train_corpus)
            for obser, gold_label in train_corpus:
                self._update_single_example(obser, gold_label, n_done)
                n_done += 1

            self._last_update_for_average(n_vocab, n_done)

            score_train = self.score(train_corpus)
            score_test = self.score(test_corpus)
            self.score_by_epochs["train"].append(score_train)
            self.score_by_epochs["test"].append(score_test)
            print(f"Iter {n_iter}: the score on the train corpus {score_train}" 
                  f"\n the score on the test corpus {score_test}")

    def _last_update_for_average(self, n_vocab:int, n_done: int):
        if n_done == n_vocab:
            for key, last_error in self.last_update.items():
                label, feat = key[0], key[1]
                self.average_vector[label][feat] += (n_vocab - last_error) * self.parameters_vector[label][feat]


def plot_risks(err_simple, err_average):
    """
    :param dict_risk: the mapping between a str indicating the train/test corpus and the list of errors on different epoches
    """

    df = pd.DataFrame({"simple perceptron train error": err_simple["train"],
                       "simple perceptron test error": err_simple["test"],
                       "average perceptron train error": err_average["train"],
                       "average perception test error": err_average["test"]})
    ax = sns.lineplot(data=df)
    ax.set(xlabel="epochs", ylabel="error(risk)")

def main():

    train_corpus = to_feature_representation(train_gsd)
    test_corpus = to_feature_representation(test_gsd)

    # print(type(list(test_corpus)))
    # print(test_corpus)

    ls_labels = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PRON", "PROPN", "PUNCT", "SCONJ","SYM", "VERB", "X"]
    #perceptron_clf = Average_perceptron(ls_labels)
    #perceptron_clf.fit(train_corpus, test_corpus, max_iter= 20)
    #print(perceptron_clf.score_by_epochs)



if __name__ == "__main__":
    main()
