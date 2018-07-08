import numpy as np

from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[ii:ii + window_size] for ii in range(len(series) - window_size)]
    y = [series[ii + window_size] for ii in range(len(series) - window_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    ## "only ascii" as written above is not enough for the project assistant
    # ascii_chars = {ii for ii in set(text) if (ord(ii) >= 128)}
    # for ascii_char in ascii_chars:
        # text = text.replace(ascii_char, ' ')
    chars_to_exclude = {ii for ii in set(text) if (not ii.isalpha()) or (ord(ii) >= 128)}
    chars_to_exclude -= set(punctuation) # keep punctuation
    for char_to_exclude in chars_to_exclude:
        text = text.replace(char_to_exclude, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[ii:ii + window_size] for ii in range(0, len(text) - window_size, step_size)]
    outputs = [text[ii + window_size] for ii in range(0, len(text) - window_size, step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    
    return model
