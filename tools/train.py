import tensorflow as tf

import numpy as np
from tensorflow import keras

def build_network(output_dim, n_hidden, n_neurons, learning_rate):

    """

    output_dim: what do we want this to output
    Generator output n_columns of data
    Discriminator output 1, p(data_real|data_seen)

    n_hiden: number of layers of the neural net

    n_neurons: number of neuros in the network

    learning_rate: duhhh

    This outputs a keras neural net
    
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="selu"))
        # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(output_dim + 3, activation="selu"))  
    model.add(keras.layers.Dense(output_dim, activation="sigmoid"))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)
    return model

def train_gan(
    generator, discriminator, dataset, n_epochs=100, n_noise=20000
):
    """
    # TODO: UPDATE ARGS
    Inputs: 

    gan, this is a keras gan object made by combining two neural nets and
    restricting the trainability of one of them.

    dataset, this takes in regular tabular data. now this is training rowwise
    however i may change this to matrix wise like a picture.

    n_epochs, numper of times the gans go though training iterationations

    iterationations, number of times in gan iterationaton loop, 
    it would be a good idea to reduct this after the warmup period

    n_noise, this is the size of fake data generated

    
    Output:

    generators_saved, this is an iterationable list of keras objects that can be used
    
    discriminators_saved, same thing, these can be used to test

    for generator, discriminator in zip(gen, desc):
        noise = tf.random.normal(shape=dims)
        generated_data = generator(noise)
        judgement = discriminator(generated_data) # probs data is real
    """
    gan = keras.models.Sequential([generator, discriminator])
  
    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    generator, discriminator = gan.layers
    data_out = np.empty((0, dataset.shape[1]))


    for epoch in range(250):
        np.random.seed(epoch)
        random_index = tf.random.uniform(shape=(n_noise,), minval=0, maxval=len(dataset), dtype=tf.int32)
        X_batch = dataset[random_index, :]
        for iteration in range(5):

            noise = tf.random.normal(shape=X_batch.shape,
                                     mean=0,
                                     stddev=1) 

            generated_data = generator(noise)
            X_fake_and_real = tf.concat([generated_data, X_batch], axis=0)
            y1 = tf.concat([tf.zeros(n_noise), tf.ones(n_noise)], axis=0)
            
            # training discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # training the generator

            noise = tf.random.normal(shape=X_batch.shape,
                                     mean=0,
                                     stddev=1) 
            
            discriminator.trainable = False
            gan.train_on_batch(noise, tf.ones(n_noise))
         
        generated_data = generator(noise)
        rand = tf.random.uniform(shape=(1,), minval=0, maxval=X_batch.shape[0], dtype=tf.int32)

        data_out = np.concatenate([data_out, generated_data[ :1 , :]])
    
    return data_out
    
