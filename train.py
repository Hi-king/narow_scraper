# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import logging
import os
import pickle
import random
import string
import sys

import chainer.serializers
import numpy

ROOTDIR = os.path.dirname(__file__)
DATADIR = os.path.join(ROOTDIR, "dataset")
MODELDIR = os.path.join(ROOTDIR, "models")
VOCAB_SIZE = 3000

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--num_lstm_layer", type=int, default=2)
parser.add_argument("--mid_size", type=int, default=100)
args = parser.parse_args()

sys.path.append(ROOTDIR)
import narow_generator

def data_loader():
    import json
    while True:
        for major_code in ["c", "d"]:
            minor_codes = list(string.ascii_lowercase)
            random.shuffle(minor_codes)
            for minor_code in minor_codes:
                numbers = list(range(10000))
                random.shuffle(numbers)
                for i in numbers:
                    target_id = "n{:04}{}{}".format(i, major_code, minor_code)
                    filepath = os.path.join(DATADIR, "{}.json".format(target_id))
                    if os.path.exists(filepath):
                        with open(filepath) as f:
                            yield json.load(f)

model = narow_generator.model.FeatureWordModel(vocab_size=VOCAB_SIZE + 3, midsize=args.mid_size)
logging.basicConfig(filename=os.path.join(MODELDIR, "{}_log.txt".format(id(model))), level=logging.DEBUG)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)
logging.info(args)
if args.gpu >= 0:
    import cupy
    chainer.cuda.get_device(args.gpu).use()
    xp = cupy
    model.to_gpu()
else:
    xp = numpy
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

id_converter = narow_generator.io.IdConverter(limit=VOCAB_SIZE)
for i, data in enumerate(data_loader()):
    target = [VOCAB_SIZE+1] + id_converter.convert_sentense(data["summary"]) + [VOCAB_SIZE+2]
    target_array = xp.array(target, dtype=xp.int32)
    model.reset_state()
    total_loss = chainer.Variable(xp.zeros((), dtype=xp.float32))
    for character_index in range(len(target) - 1):
        loss, predicted = model.loss_predict_word(
            chainer.Variable(target_array[character_index:character_index+1]),
            chainer.Variable(target_array[character_index + 1:character_index+2]))
        total_loss += loss
        total_loss /= len(target)
    logging.info("total_loss: {}".format(total_loss.data))
    optimizer.zero_grads()
    total_loss.backward()
    optimizer.update()

    if i % 1000 == 1:
        chainer.serializers.save_hdf5(os.path.join(MODELDIR, "{}_{}.model.npz".format(id(model), i)), model)
        chainer.serializers.save_hdf5(os.path.join(MODELDIR, "{}_{}.optiizer.npz".format(id(model), i)), optimizer)
        with open(os.path.join(MODELDIR, "{}_{}.converter.dump".format(id(model), i)), "wb+") as f:
            pickle.dump(id_converter, f)
