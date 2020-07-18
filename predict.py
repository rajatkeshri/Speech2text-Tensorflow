import json
import tensorflow as tf
import numpy as np
import librosa

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050


mappings = [
        "down",
        "off",
        "on",
        "no",
        "yes",
        "stop",
        "up",
        "right",
        "left",
        "go"
    ]


def predict(file_path, num_mfcc=13, hop_length=512, n_fft=2048):

    #load model
    model = tf.keras.models.load_model(SAVED_MODEL_PATH)

    #load file
    signal, sample_rate = librosa.load(file_path)

    #extract mfcc
    if len(signal) >= SAMPLES_TO_CONSIDER:
        # ensure consistency of the length of the signal
        signal = signal[:SAMPLES_TO_CONSIDER]
        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        MFCCs = MFCCs.T
        print(MFCCs.shape)

        #change dimensions
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        #predict
        predictions = model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = mappings[predicted_index]

        print(predicted_keyword)
    else:

        print("less than 1 second, please try another audio")

        """
        length_of_sig = len(signal)
        num_of_zeros = SAMPLES_TO_CONSIDER - length_of_sig
        zero_array = [0]*num_of_zeros

        signal = list(signal) + list(zero_array)
        signal = np.array(signal)

        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        MFCCs = MFCCs.T
        print(MFCCs.shape)

        #change dimensions
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        #predict
        predictions = model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = mappings[predicted_index]

        print(predicted_keyword)
        """


if __name__ == "__main__":

    file_path = "test/stop.wav"
    predict(file_path)
