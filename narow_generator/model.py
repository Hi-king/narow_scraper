# -*- coding: utf-8 -*-
from __future__ import print_function
import chainer


class FeatureWordModel(chainer.Chain):
    def __init__(self, vocab_size, midsize, num_lstm_layer=3):
        if num_lstm_layer == 2:
            super(FeatureWordModel, self).__init__(
                word_embed=chainer.functions.EmbedID(vocab_size, midsize),
                lstm0=chainer.links.connection.lstm.LSTM(midsize, midsize),
                lstm1=chainer.links.connection.lstm.LSTM(midsize, midsize),
                word_out_layer=chainer.functions.Linear(midsize, vocab_size),
            )
        elif num_lstm_layer == 3:
            super(FeatureWordModel, self).__init__(
                word_embed=chainer.functions.EmbedID(vocab_size, midsize),
                lstm0=chainer.links.connection.lstm.LSTM(midsize, midsize),
                lstm1=chainer.links.connection.lstm.LSTM(midsize, midsize),
                word_out_layer=chainer.functions.Linear(midsize, vocab_size),
            )
        else:
            raise Exception("invalid num_lstm_layer")

    def predict_word_probability(self, x):
        word_predicted = chainer.functions.softmax(self._forward(x))
        return word_predicted

    def loss_predict_word(self, x, t):
        word_predicted = self._forward(x)
        loss = chainer.functions.softmax_cross_entropy(word_predicted, t)
        return loss, word_predicted

    def _forward(self, x):
        h = self.word_embed(x)
        if hasattr(self, "lstm0"):
            h = self.lstm0(h)
        if hasattr(self, "lstm1"):
            h = self.lstm1(h)
        if hasattr(self, "lstm2"):
            h = self.lstm2(h)
        word = self.word_out_layer(h)
        return word

    def reset_state(self):
        if hasattr(self, "lstm0"):
            self.lstm0.reset_state()
        if hasattr(self, "lstm1"):
            self.lstm1.reset_state()
        if hasattr(self, "lstm2"):
            self.lstm1.reset_state()