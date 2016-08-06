# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import chainer.serializers
import os
import sys
import pickle
import numpy

ROOTDIR = os.path.dirname(__file__)
sys.path.append(ROOTDIR)
import narow_generator

VOCAB_SIZE = 3000
MID_SIZE = 100

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("model_path")
parser.add_argument("converter_path")
args = parser.parse_args()


model = narow_generator.model.FeatureWordModel(vocab_size=VOCAB_SIZE+2, midsize=MID_SIZE)
chainer.serializers.load_hdf5(args.model_path, model)

id_generator = None # type: narow_generator.io.IdConverter
with open(args.converter_path, "rb") as f:
    id_generator = pickle.load(f)

model.reset_state()
last_word = -1
for i in range(1000):
    word_probabilities = model.predict_word_probability(chainer.Variable(numpy.array([[last_word]], numpy.int32))).data[0]
    # last_word = word_probabilities.data.argmax()
    # print(word_probabilities/numpy.sum(word_probabilities))
    last_word = numpy.random.choice(list(range(word_probabilities.shape[0])), p=word_probabilities/numpy.sum(word_probabilities))
    if last_word == VOCAB_SIZE:
        sys.stdout.write("*")
    else:
        sys.stdout.write(id_generator.inverse(last_word))
