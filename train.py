import json
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

DATA_PATH = "data.json"
SAVE_PATH = "model.h5"

LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32

NUM_OF_OUTPUTS = 10

####################################################################################
def load_data(data_path):

    with open(data_path,"r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Data Loaded!")

    return X, y
####################################################################################

def prepare_dataset(data_path, test_size=0.1, test_validation=0.1):

    # Loading the data from json in np array
    X, y = load_data(data_path)

    # splitting to train, test, validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_validation)

    # adding extra axis for CNN
    X_train = X_train[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    X_val = X_val[...,np.newaxis]

    return X_train, y_train, X_val, y_val, X_test, y_test
####################################################################################
def build_model(input_shape, learning_rate=0.0001, loss="sparse_categorical_crossentropy"):

    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(NUM_OF_OUTPUTS, activation='softmax'))

    optimiser = tf.keras.optimizers.Adam(lr=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model
####################################################################################

def main():

    #load the data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset(DATA_PATH)

    #build the model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (segment, mfccs, 1)
    model = build_model(input_shape, LEARNING_RATE)

    #train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    #evaluate
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(test_error, test_accuracy)

    #save the model
    model.save(SAVE_PATH)
####################################################################################

if __name__ == "__main__":
    main()
