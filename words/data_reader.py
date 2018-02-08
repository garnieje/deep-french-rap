import numpy as np

import re
import random
import functools


class DataReader:

    TOKENS_LOOKUP = {
        '.': ' ||period|| ',
        ',': ' ||comma|| ',
        '"': ' ||quotes|| ',
        ';': ' ||semicolon|| ',
        '!': ' ||exclamation-mark|| ',
        '?': ' ||question-mark|| ',
        '(': ' ||left-parentheses|| ',
        ')': ' ||right-parentheses|| ',
        '--': ' ||emm-dash|| '
    }

    def __init__(self, path_lyrics):

        self.lyrics = open(path_lyrics, "rb").read().split("\n")
        self.clean_text()
        self.get_vocab()

    def clean_text(self):

        new_lyrics = []

        for lyric in self.lyrics:
            lyric = lyric.decode("utf-8")
            lyric = lyric.lower()
            # remove text indication in between bracket, we might want to
            # create token for them
            lyric = re.sub("\\[[^\\]]+\\]", "", lyric)
            for punct, token in self.TOKENS_LOOKUP.items():
                lyric = re.sub(re.escape(punct), token, lyric)
            lyric = lyric.split()
            if lyric:
                lyric[0] = "*START*"
                lyric += ["*END*"]
                new_lyrics.append(lyric)

        self.lyrics = new_lyrics

    def get_vocab(self):

        all_words = functools.reduce(lambda a, b: a + b, self.lyrics)
        self.vocab = sorted(list(set(all_words)))
        print("we have a vocab of length {}".format(len(self.vocab)))

        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
        self.id_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        lyrics_id = []

        for lyric in self.lyrics:
            lyric_id = [self.word_to_id[word] for word in lyric]
            lyrics_id.append(lyric_id)
        self.lyrics_id = lyrics_id

    def get_batch(self, batch_size, seq_len):

        inputs = np.empty([batch_size, seq_len], dtype=int)
        targets = np.empty([batch_size, seq_len], dtype=int)

        for batch_id in range(batch_size):
            inp, target = self.get_sequence(seq_len)
            inputs[batch_id] = inp
            targets[batch_id] = target

        return inputs, targets

    def get_sequence(self, seq_len):

        for i in range(1000):
            song_id = random.choice(range(len(self.lyrics_id)))
            if len(self.lyrics_id[song_id]) > seq_len:
                break
        else:
            raise ValueError(
                "No song is long enough for a sequence of lenght {}".format(seq_len))

        song = self.lyrics_id[song_id]
        start_id = random.randint(0, len(song) - seq_len - 1)
        inp = song[start_id:(start_id + seq_len)]
        target = song[(start_id + 1): (start_id + seq_len + 1)]
        return inp, target

    def get_batches(self, batch_size, seq_len):

        words_per_batch = batch_size * seq_len
        for i in range(1000):
            song_id = random.choice(range(len(self.lyrics_id)))
            if len(self.lyrics_id[song_id]) > words_per_batch:
                break
        else:
            raise ValueError("No song is long enough")
        int_text = self.lyrics_id[song_id]
        num_batches = len(int_text)//words_per_batch
        int_text = int_text[:num_batches*words_per_batch]
        y = np.array(int_text[1:] + [int_text[0]])
        x = np.array(int_text)
        
        x_batches = np.split(x.reshape(batch_size, -1), num_batches, axis=1)
        y_batches = np.split(y.reshape(batch_size, -1), num_batches, axis=1)
        
        batch_data = list(zip(x_batches, y_batches))
        
        return np.array(batch_data)
