#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from random import uniform
from nltk.corpus import reuters
from tqdm import tqdm
fileids = reuters.fileids()

all_text= ""

for file in tqdm(fileids):
    all_text+=reuters.raw(file)
data = all_text
# data = open("shakespeare.txt", "r").read()  # should be simple plain text file
data = data.lower()  # We dont want upper/lowercase to impact the word prediction
data = data.replace("\n", " ")
# data = data.replace("'", " ")
# data = data.replace("!", " ")
# data = data.replace("?", " ")
# data = data.replace(":", " ")
# #data = "I am once again doing a test sentence here, the purpose is not to train a network or anything fancy, we are just debugging why stuff isnt working. You may wonder why this sentence is so long, that is because it will take too long otherwise and so random words like pineapple and stuff will work"

# Convert every word to one hot
unique_words_list = list(set(data))

lb = LabelBinarizer()
lb.fit(unique_words_list)
onehot_words = lb.transform(unique_words_list)

# Create word-onehot dict. Also reverse for sampling. These are actually letters not words
word_onehot_dict = {
    word: k for k, word in zip(map(tuple, onehot_words), unique_words_list)
}
onehot_word_dict = {v: k for k, v in word_onehot_dict.items()}

print("Number of things: {}".format(len(unique_words_list)))
print(unique_words_list)
# Initialize matrices & vectors (w and bias)

vocab_size = len(set(data))  # Number of unique words + space
hidden_size = 400

w_xh = np.random.rand(hidden_size, vocab_size) / 100
w_hh = np.random.rand(hidden_size, hidden_size) / 100
w_hy = np.random.rand(vocab_size, hidden_size) / 100

b_h = np.random.rand(hidden_size, 1)
b_y = np.random.rand(vocab_size, 1)

def sample(h, starting_word, sentence_length):
    """
    sample a sequence of letters
    h is memory state
    sentence length is number of letters to sample
    """
    x = np.array(word_onehot_dict[starting_word])[:, np.newaxis]
    generated_letters = []
    for t in range(sentence_length):
        h = np.tanh(np.dot(w_xh, x) + np.dot(w_hh, h) + b_h)
        y = np.dot(w_hy, h) + b_y
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1

        generated_letters.append(onehot_word_dict[tuple(x.flatten())])
    return generated_letters


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
        print(name)
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

    for dparam in [dLdw_xh, dLdw_hh, dLdw_hy, dLdb_h, dLdb_y]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dLdw_xh, dLdw_hh, dLdw_hy, dLdb_h, dLdb_y, h_dict[len(inputs) - 2]


# In[ ]:

print("Initialized everything!")
seq_length = 150
i = 0
learning_rate = 0.001
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
print(len(data))
while i + seq_length < len(data):
    if i == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory

    inputs = data[seq_length * i : seq_length * i + seq_length]
    loss, dLdw_xh, dLdw_hh, dLdw_hy, dLdb_h, dLdb_y, hprev = loss_function(
        inputs, hprev
    )
    #gradCheck(inputs, hprev)

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
        generated_letters = sample(hprev, starting_word="a", sentence_length=200)
        txt = "".join(generated_letter for generated_letter in generated_letters)
        print("----\n %s \n----" % (txt,))

    i += 1
ii