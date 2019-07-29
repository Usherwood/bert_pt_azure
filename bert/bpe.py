#!/usr/bin/env python

"""Functions for Byte Pair ENcoding BPE language representation."""

__author__ = "Peter J Usherwood and Steve Smit"
__python_version__ = '3.6'

import re, collections
import numpy as np
import requests
from bs4 import BeautifulSoup
import pickle
import os
import regex


def get_emoji_constituent_codes():
    # Note at time of writing 'https://unicode.org/emoji/charts/full-emoji-list.html' and 'https://unicode.org/emoji/charts/full-emoji-modifiers.html'
    # are allowed to be scraped as given by unicode.org/robots.txt (20-02-2019)

    codes = []
    for page in ['https://unicode.org/emoji/charts/full-emoji-list.html',
                 'https://unicode.org/emoji/charts/full-emoji-modifiers.html']:
        r = requests.get(page)
        print(r.status_code)

        soup = BeautifulSoup(r.text, 'html.parser')
        tbl = soup.find_all('table')[0]
        rows = tbl.find_all('tr')
        for row in rows:
            if len(row.find_all('th')) == 0:
                codes.append(row.find('td', {"class": "code"}).text)

    print("Raw size of emoji", len(codes))
    codes = [l for code in codes for l in code.split()]
    print("Raw size of emoji constituents", len(codes))
    codes = list(set(codes))
    print("Final size of emoji constituents", len(codes))
    codes = [code[2:] for code in codes]

    # codes from hex to int
    codes = [int(code, 16) for code in codes]
    return codes


def create_token_vocabulary():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    cs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs_emoji = get_emoji_constituent_codes()
    cs += cs_emoji
    cs = [chr(n) for n in set(cs)]
    return dict(zip(np.arange(len(cs)), cs))


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(r''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def load_obj(name, directory):
    with open(os.path.join(directory, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, byte_decoder, unk_token, unk_id, spc_token, spc_id, errors='replace'):
        self.encoder = encoder
        self.unk_token = unk_token
        self.unk_id = unk_id
        self.spc_token = spc_token
        self.spc_id = spc_id
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.decoder[self.unk_id] = self.unk_token
        self.errors = errors # how to handle errors in decoding
        self.byte_decoder = byte_decoder
        self.byte_encoder = {v:k for k, v in self.byte_decoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in regex.findall(self.pat, text):
            token_ids = [self.encoder.get(l, self.unk_id) if l is not " " else self.spc_id for l in token]
            token = "".join([self.decoder[l] for l in token_ids])
            bpe_tokens.extend(self.encoder.get(bpe_token, self.unk_id) for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def tokenize(self, text):
        tokens = []
        for token in regex.findall(self.pat, text):
            token_ids = [self.encoder.get(l, self.unk_id) if l is not " " else self.spc_id for l in token]
            token = "".join([self.decoder[l] for l in token_ids])
            tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = ''.join([t if t != self.spc_token else " " for t in text])
        return text


def get_encoder(language_map_dir):
    encoder = load_obj("id_to_vocab", language_map_dir)
    bpe_merges = load_obj("bpe_merges", language_map_dir)
    byte_decoder = load_obj("byte_encodings", language_map_dir)
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
        byte_decoder=byte_decoder,
        unk_token=chr(int("E000", 16)),
        unk_id=len(byte_decoder)-2,
        spc_token=chr(int("E001", 16)),
        spc_id=len(byte_decoder)-1
    )
