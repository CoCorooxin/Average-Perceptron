import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter
import numpy as np

def read_conllu(file_path, sep="\t"):
    """
    Read corpus in CoNILL format and returns tokens and gold label lists
    :param file_path: file path to the conllu file
    :param sep(str, optional): Column separator. Defaults to "\t"
    :return: a list of tuples(pairs)--->pair the list of toks representing the sentence, and the list of corresponding labels
    """
    ls_examples = []
    with open(file_path) as f:
        data = f.readlines()
    todo_toks, todo_lbls = [], []
    #print(data)
    for line in data:
        line = line.strip().split(sep)
        #if a new sentence is detected
        if line[0] == "1":
             # add the pair of complete observation for one sentence
            ls_examples.append((todo_toks, todo_lbls))
            #recreate the todo lists for sentences
            todo_toks, todo_lbls = [line[1]], [line[3]]
        # if it is a new tok for the same sentence, add the tok to the old sub list
        elif line[0].isdigit():
            todo_toks.append(line[1])
            todo_lbls.append(line[3])
    # print(len(ls_examples)-1)
    #strip the first empty element
    return ls_examples[1:]

def plot_labels_Distribution(ls_train, ls_test):

    dist_train = Counter(chain.from_iterable(pair[1] for pair in ls_train)).most_common()
    dist_test = Counter(chain.from_iterable(pair[1] for pair in ls_test)).most_common()

    plt.subplot(1,2,1)
    plt.bar(*zip(*(dist_train)))
    plt.title("The distribution of POS in the train set")

    plt.subplot(1,2,2)
    plt.bar(*zip(*(dist_test)))
    plt.title("The distribution of POS in the test set")
    plt.show()

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
    #by list of pairs(feacture_vect, labels)
    res = []
    for sent in corpus:
        for tok, label in zip(make_feature_vec(sent[0]), sent[1]):
            res.append((tok, label))
    #print(len(res))
    return res

def make_feature_vec(sent):
    ls_prefix = ["prev_prev_word_", "prev_word_", "curr_word_", "next_word_", "next_next_word_"]
    # a temporary list where the unfinished feature vectors of the sentence is stored
    feature_vec = []
    for i in range(2, len(sent) - 2):
        # initiate the feature vector for the current word
        todo = ["".join(feature).lower() for feature in zip(ls_prefix, sent[(i - 2):(i + 3)]) if feature[1] != "#"]
        # verify other features specific to the current word tok[i]
        todo.append("biais")
        if sent[i][0].isupper():
            todo.append("start_with_upper")
        if has_number(sent[i]):
            todo.append("has_number")
        # update the feature vector
        feature_vec.append(todo)
        #print(todo)
    return feature_vec


def plot_risks():
    """
    :param dict_risk: the mapping between a str indicating the train/test corpus and the list of errors on different epoches
    """
    score_by_epochs = {
        'train': [0.1659212322418645, 0.1420718246010213, 0.10954347056741776, 0.08945550144087366, 0.0915721532213215,0.06881140565226918, 0.05749001499856754, 0.05231298134447837, 0.04190498660240316,0.04080284467213802,0.03473263789413372, 0.03040833178853708, 0.026704191172753133, 0.023522472572844122,0.022046209069920275, 0.020205935388193264, 0.01730733581624222, 0.016201823421358674, 0.015888370211833647, 0.014209878831796918],
        'test': [0.1791796593907412, 0.17006476373230994, 0.14655792756056607, 0.1324058527224754, 0.14140081554329575,0.11561525545694407, 0.11981290477332696, 0.11525545694411132, 0.10805948668745502,0.10841928520028787,0.10398177020868316, 0.10326217318301756, 0.1014631806188534, 0.09954425521707844, 0.09810506116574713,0.09402734468697527, 0.09690573278963777, 0.09966418805468935, 0.09690573278963777, 0.09630606860158308]}
    plt.plot(np.array([i for i in range(1,21)], dtype=int), score_by_epochs["train"], label="train error")
    plt.plot(np.array([i for i in range(1,21)], dtype=int), score_by_epochs["test"], label="test error")
    plt.title("The error rates on the two set by epoch(20 epochs all together)")
    plt.show()

path_train = "../lab03/fr_gsd-ud-train.conllu"
path_test = "../lab03/fr_gsd-ud-test.conllu"
train_gsd = read_conllu(path_train)
test_gsd = read_conllu(path_test)

def main():
    path_train = "../lab03/fr_gsd-ud-train.conllu"
    path_test = "../lab03/fr_gsd-ud-test.conllu"
    train_gsd = read_conllu(path_train)
    test_gsd = read_conllu(path_test)
    #plot_risks()
    #plot_labels_Distribution(train_gsd, test_gsd)

    #train_corpus = to_feature_representation(train_gsd)
    #test_corpus = to_feature_representation(test_gsd)

    #print(test_corpus)

if __name__ == "__main__":
    main()


