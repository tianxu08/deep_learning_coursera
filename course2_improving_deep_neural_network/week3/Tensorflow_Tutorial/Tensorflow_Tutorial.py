import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

# np.random.seed(1)

def play():
    y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
    y = tf.constant(39, name='y')                    # Define y. Set to 39

    loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

    init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                    # the loss variable will be initialized and ready to be computed
    with tf.Session() as session:                    # Create a session and print the output
        session.run(init)                            # Initializes the variables
        print(session.run(loss))                     # Prints the loss

if __name__ == "__main__":
    play()
    
'''
What you should remember:

Tensorflow is a programming framework used in deep learning
The two main object classes in tensorflow are Tensors and Operators.
When you code in tensorflow you have to take the following steps:
Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)
Create a session
Initialize the session
Run the session to execute the graph
You can execute the graph multiple times as you've seen in model()
The backpropagation and optimization is automatically done when running the session on the "optimizer" object.
'''