#!/usr/bin/env python3
"""Transfer learning Resnet50-Cifar10"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """Function to pre-processes the data for the model"""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(x_train, y_train)
    X_test, Y_test = preprocess_data(x_test, y_test)
    input_i = K.Input(shape=(32, 32, 3))
    base_model = K.applications.ResNet50(
                                         include_top=False,
                                         weights="imagenet",
                                         input_shape=(224, 224, 3))
    base_model.trainable = False
    model = K.models.Sequential()
    model.add(K.layers.Lambda(lambda i: K.backend.resize_images(i,
                                                                height_factor=7,
                                                                width_factor=7,
                                                                data_format='channels_last',
                                                                interpolation='bilinear')))
    model.add(base_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(512, activation='relu'))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(10, activation='softmax'))
    check_model = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                              monitor="val_accuracy",
                                              mode="max",
                                              save_best_only=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.SGD(lr=0.001,),
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=128, epochs=5, verbose=1,
              validation_data=(X_test, Y_test),
              callbacks=[check_model])
    model.summary
    model.save("cifar10.h5")
