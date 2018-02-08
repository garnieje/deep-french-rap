
import torch
import re
import random
import functools

from torch.autograd import Variable


class DataReader:

    def __init__(self, path_lyrics):

        self.lyrics = open(path_lyrics, "r+", encoding="utf-8").read().split("\n")
        self.clean_text()
        self.get_char()

    def clean_text(self):

        new_lyrics = []

        for lyric in self.lyrics:
            lyric = re.sub("\\[[^\\]]+\\]", "", lyric)
            lyric = re.sub("(\s?\*BREAK\*\s?)+", "\n", lyric)
            new_lyrics.append(lyric)

        self.lyrics = new_lyrics

    def get_char(self):

        all_words = functools.reduce(lambda a, b: a + b, self.lyrics)
        self.chars = list(set(all_words))
        self.ix_to_char = {ix: char for ix, char in enumerate(self.chars)}
        self.char_to_ix = {char: ix for ix, char in enumerate(self.chars)}

    def char_tensor(self, string):

        tensor = torch.zeros(len(string)).long()
        for i in range(len(string)):
            try:
                tensor[i] = self.char_to_ix[string[i]]
            except Exception as e:
                continue
        return tensor

    def get_batch(self, batch_size, seq_len):

        inputs = torch.LongTensor(batch_size, seq_len)
        targets = torch.LongTensor(batch_size, seq_len)

        for batch_id in range(batch_size):
            inp, target = self.get_sequence(seq_len)
            inputs[batch_id] = inp
            targets[batch_id] = target
        inputs = Variable(inputs)
        targets = Variable(targets)

        return inputs, targets

    def get_sequence(self, seq_len):

        for i in range(1000):
            song_id = random.choice(range(len(self.lyrics)))
            if len(self.lyrics[song_id]) > seq_len:
                break
        else:
            raise ValueError("No song long enough")

        song = self.lyrics[song_id]
        start_id = random.randint(0, len(song) - seq_len - 1)
        inp = song[start_id:(start_id + seq_len)]
        target = song[(start_id + 1):(start_id + seq_len + 1)]
        return self.char_tensor(inp), self.char_tensor(target)

    def get_ix_to_char(self):
        return self.ix_to_char

    def get_char_to_ix(self):
        return self.char_to_ix
