import re
import unicodedata
import numpy as np

#normalize the string, covert to lower case, trim and remove non-letter characters.
def normalizeString(names):
    names = [unicodeToAscii(str(name).lower().strip()) for name in names]
    names = [name for name in names if re.match("^[A-Z a-z]*$", name)]
    return names

#function to covert to ascii values
def unicodeToAscii(name):
    return ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    )

#function to create a lookup table for each charcter and each charcter is mapped to an index
def create_lookup_tables(vocab):
    #text = ' '.join(name for name in names)
    chars = tuple(set(vocab))
    n_characters = (len(chars))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    return int2char, char2int, n_characters

#function to build n-grams based on the batch size and sequence length
def build_ngrams(vocab, batch_size, seq_length):
    total_count = len(vocab)
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = total_count // batch_size_total
    input_words = vocab[:(n_batches*batch_size_total)]
    word_grams = [(input_words[n:(n+seq_length)], input_words[(n+1):(n+seq_length+1)]) for n in range(0, len(input_words), seq_length)
                                                                                      if (n+seq_length+1) < len(input_words)]
    return word_grams

#function to build a one hot encoder for a given input
def one_hot_encode(input, n_characters):
    # Initialize the the encoded array
    one_hot = np.zeros((input.size, n_characters), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), input.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*input.shape, n_characters))

    return one_hot
