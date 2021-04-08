#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from random import uniform

from nltk.corpus import reuters, stopwords
from nltk import word_tokenize
from tqdm import tqdm
import re

# ==== Text processing constants
BLACKLIST_STOPWORDS = ["over", "only", "very", "not", "no"]
ENGLISH_STOPWORDS = set(stopwords.words("english")) - set(BLACKLIST_STOPWORDS)
NEG_CONTRACTIONS = [
    (r"aren\'t", "are not"),
    (r"can\'t", "can not"),
    (r"couldn\'t", "could not"),
    (r"daren\'t", "dare not"),
    (r"didn\'t", "did not"),
    (r"doesn\'t", "does not"),
    (r"don\'t", "do not"),
    (r"isn\'t", "is not"),
    (r"hasn\'t", "has not"),
    (r"haven\'t", "have not"),
    (r"hadn\'t", "had not"),
    (r"mayn\'t", "may not"),
    (r"mightn\'t", "might not"),
    (r"mustn\'t", "must not"),
    (r"needn\'t", "need not"),
    (r"oughtn\'t", "ought not"),
    (r"shan\'t", "shall not"),
    (r"shouldn\'t", "should not"),
    (r"wasn\'t", "was not"),
    (r"weren\'t", "were not"),
    (r"won\'t", "will not"),
    (r"wouldn\'t", "would not"),
    (r"ain\'t", "am not"),  # not only but stopword anyway
]
OTHER_CONTRACTIONS = {
    "'m": "am",
    "'ll": "will",
    "'s": "has",  # or 'is' but both are stopwords
    "'d": "had",  # or 'would' but both are stopwords
}


fileids = reuters.fileids()

all_text = ""

for file in tqdm(fileids[:1000]):
    raw_doc = reuters.raw(file)
    raw_doc = raw_doc.lower()
    for t in NEG_CONTRACTIONS:
        doc = re.sub(t[0], t[1], raw_doc)
    # tokenize
    tokens = word_tokenize(doc)
    # transform other contractions (e.g 'll --> will)
    tokens = [
        OTHER_CONTRACTIONS[token] if OTHER_CONTRACTIONS.get(token) else token
        for token in tokens
    ]

    # tokens = ["" if token in ENGLISH_STOPWORDS else token for token in tokens]
    # remove punctuation
    r = r"^[a-zA-Z\s]*$"
    tokens = [word for word in tokens if re.search(r, word)]
    tokens_string = " ".join(word for word in tokens)
    all_text += tokens_string
data = all_text

# Convert every word to one hot
unique_words_list = list(set(data.split(" ")))
lb = LabelBinarizer()
lb.fit(unique_words_list)
onehot_words = lb.transform(unique_words_list)
print(len(unique_words_list))
# Create word-onehot dict. Also reverse for sampling
word_onehot_dict = {
    word: k for k, word in zip(map(tuple, onehot_words), unique_words_list)
}
onehot_word_dict = {v: k for k, v in word_onehot_dict.items()}

# Initialize matrices & vectors (w and bias)

vocab_size = len(set(data.split(" ")))  # Number of unique words + space
hidden_size = 100

w_xh = np.random.rand(hidden_size, vocab_size) / 100
w_hh = np.random.rand(hidden_size, hidden_size) / 100
w_hy = np.random.rand(vocab_size, hidden_size) / 100

b_h = np.random.rand(hidden_size, 1)
b_y = np.random.rand(vocab_size, 1)


def sample(h, starting_word, sentence_length):
    """
    sample a sequence of words
    h is memory state
    """
    x = np.array(word_onehot_dict[starting_word])[:, np.newaxis]
    generated_words = []
    for t in range(sentence_length):
        h = np.tanh(np.dot(w_xh, x) + np.dot(w_hh, h) + b_h)
        y = np.dot(w_hy, h) + b_y
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1

        generated_words.append(onehot_word_dict[tuple(x.flatten())])
    return generated_words


def gradCheck(inputs, hprev):
    """
    Check if the gradients are computed correctly (evaluate function and take two-sided gradient manually)
    """
    global w_xh, w_hh, w_hy, b_h, b_y
    num_checks, delta = 10, 1e-5
    _, dLdw_xh, dLdw_hh, dLdw_hy, dLb_h, dLb_y, _ = loss_function(inputs, hprev)
    for param, dparam, name in zip(
        [w_xh, w_hh, w_hy, b_h, b_y],
        [dLdw_xh, dLdw_hh, dLdw_hy, dLb_h, dLb_y],
        ["w_xh", "w_hh", "w_hy", "b_h", "b_y"],
    ):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, "Error dims dont match: %s and %s." % (repr(s0), repr(s1))
        for i in range(num_checks):
            ri = int(uniform(0, param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _, _, _, _, _ = loss_function(inputs, hprev)
            param.flat[ri] = old_val - delta
            cg1, _, _, _, _, _, _ = loss_function(inputs, hprev)
            param.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)
            rel_error = abs(grad_analytic - grad_numerical) / (
                abs(grad_numerical + grad_analytic) + np.spacing(1)
            )
            print("%f, %f => %e " % (grad_numerical, grad_analytic, rel_error))
            # rel_error should be on order of 1e-7 or less


def loss_function(inputs, hprev):

    """
    For a given input string of 5 words, computes the overall loss and gradients
    """

    ## Forward propagation
    h_dict = {}
    h_dict[-1] = np.copy(hprev)
    y_normalized_dict = {}
    target_dict = {}
    x_dict = {}
    loss = 0
    for i, word in enumerate(inputs[:-1]):

        x = np.array(word_onehot_dict[word])[
            :, np.newaxis
        ]  # input vector (one hot from vocabulary)
        h = np.tanh(np.dot(w_hh, h_dict[i - 1]) + np.dot(w_xh, x) + b_h)  # hidden state
        y = np.dot(w_hy, h) + b_y  # output
        y_normalized = np.exp(y) / np.sum(np.exp(y))  # softmax on output
        target = np.array(word_onehot_dict[inputs[i + 1]])[
            :, np.newaxis
        ]  # target (next word in sequence)
        loss += -np.log(
            np.multiply(target, y_normalized).sum()
        )  # Its just taking the y_normalized at index of correct wor

        # Store in dicts for backprop later on (and next iteration of h)
        h_dict[i] = h
        y_normalized_dict[i] = y_normalized
        target_dict[i] = target
        x_dict[i] = x

    ## Backpropagation

    # Initialize (they are accumulated over the sentence)
    dLdw_xh, dLdw_hh, dLdw_hy = (
        np.zeros_like(w_xh),
        np.zeros_like(w_hh),
        np.zeros_like(w_hy),
    )
    dLdb_h, dLdb_y = np.zeros_like(b_h), np.zeros_like(b_y)
    dhnext = np.zeros_like(h_dict[0])

    for j in range(len(inputs) - 1)[::-1]:  # Now we move right-to-left
        dLdy = y_normalized_dict[j] - target_dict[j]
        dLdw_hy += np.dot(
            dLdy, h_dict[j].T
        )  # Transpose for the shape maintenance of w_hy
        dLdb_y += dLdy
        dL_dh = np.dot(w_hy.T, dLdy) + dhnext
        helper = (
            1 - h_dict[j] * h_dict[j]
        ) * dL_dh  # derivative of tanh, used in chain rule for other derivatives
        dLdb_h += helper
        dLdw_xh += np.dot(helper, x_dict[j].T)
        dLdw_hh += np.dot(helper, h_dict[j - 1].T)
        dhnext = np.dot(w_hh.T, helper)

    #    for dparam in [dLdw_xh, dLdw_hh, dLdw_hy, dLdb_h, dLdb_y]:
    #        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dLdw_xh, dLdw_hh, dLdw_hy, dLdb_h, dLdb_y, h_dict[len(inputs) - 2]


# In[ ]:

print("Initialized everything!")
seq_length = 5
i = 0
learning_rate = 0.01
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

while i + seq_length < len(data):

    if i == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory

    inputs = data.split(" ")[seq_length * i : seq_length * i + seq_length]

    loss, dLdw_xh, dLdw_hh, dLdw_hy, dLdb_h, dLdb_y, hprev = loss_function(
        inputs, hprev
    )
    # gradCheck(inputs, hprev)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    mw_xh, mw_hh, mw_hy = np.zeros_like(w_xh), np.zeros_like(w_hh), np.zeros_like(w_hy)
    mb_h, mb_y = np.zeros_like(b_h), np.zeros_like(b_y)  # memory variables for Adagrad
    # perform parameter update with Adagrad
    for param, dparam, mem in zip(
        [w_xh, w_hh, w_hy, b_h, b_y],
        [dLdw_xh, dLdw_hh, dLdw_hy, dLdb_h, dLdb_y],
        [mw_xh, mw_hh, mw_hy, mb_h, mb_y],
    ):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    if i % 100 == 0:
        print(i)
        print(smooth_loss)
        generated_words = sample(hprev, starting_word="weather", sentence_length=200)
        txt = " ".join(generated_word for generated_word in generated_words)
        print("----\n %s \n----" % (txt,))

    i += 1
