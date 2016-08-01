# -*- coding: utf-8 -*-
from __future__ import print_function
from typing import Iterable
import os
import sys
import string
import chainer.optimizers
import chainer.serializers
import numpy
import argparse
import random
import logging

ROOTDIR = os.path.dirname(__file__)
DATADIR = os.path.join(ROOTDIR, "dataset")
MODELDIR = os.path.join(ROOTDIR, "models")
VOCAB_SIZE = 3000
MID_SIZE = 100

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=-1)
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


class IdConverter(object):
    def __init__(self, limit=1000):
        self.id_dict = {}  # type: dict[str, int]
        self.character_dict = []  # type: list[str]
        self.limit = limit

    def convert(self, character: str) -> int:
        if character in self.id_dict:
            return self.id_dict[character]
        else:
            if len(self.character_dict) >= self.limit:
                return self.limit
            else:
                self.id_dict[character] = len(self.character_dict)
                self.character_dict.append(character)
                return self.id_dict[character]

    def convert_sentense(self, sentense: str) -> Iterable[int]:
        return [self.convert(character) for character in sentense]


model = narow_generator.model.FeatureWordModel(vocab_size=VOCAB_SIZE + 2, midsize=MID_SIZE)
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

id_converter = IdConverter(limit=VOCAB_SIZE)
for i, data in enumerate(data_loader()):
    target = [-1] + id_converter.convert_sentense(data["summary"]) + [-2]
    target_array = xp.array(target, dtype=xp.int32)
    model.reset_state()
    total_loss = chainer.Variable(xp.zeros((), dtype=xp.float32))
    for character_index in range(len(target) - 1):
        loss, predicted = model.loss_predict_word(
            chainer.Variable(target_array[character_index:character_index+1]),
            chainer.Variable(target_array[character_index + 1:character_index+2]))
        total_loss += loss
    logging.info("total_loss: {}".format(total_loss.data))
    optimizer.zero_grads()
    total_loss.backward()
    optimizer.update()

    if i % 1000 == 1:
        chainer.serializers.save_hdf5(os.path.join(MODELDIR, "{}_{}.model.npz".format(id(model), i)), model)
        chainer.serializers.save_hdf5(os.path.join(MODELDIR, "{}_{}.optiizer.npz".format(id(model), i)), optimizer)
