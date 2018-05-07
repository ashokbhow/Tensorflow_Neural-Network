# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## Neural Net Classification using Tensorflow on USA_Housing data
##
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import scale

train = pd.read_csv("US_House_Training.csv", dtype=float)
test = pd.read_csv("US_Housing_Test.csv", dtype=float)

X_train = train.drop('Price', axis=1).values
Y_train = train[['Price']].values

X_test = test.drop('Price', axis=1).values
Y_test = test[['Price']].values

# Scaling the features
from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled_train = X_scaler.fit_transform(X_train)
Y_scaled_train = Y_scaler.fit_transform(Y_train)

X_scaled_test = X_scaler.fit_transform(X_test)
Y_scaled_test = Y_scaler.fit_transform(Y_test)

## Defining Tensorflow Model Parameters

learning_rate = 0.001
training_epochs = 100
display_step = 5

number_of_inputs = 5
number_of_outputs = 1

layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

## Defining Neural Net Layer Structure

# Input Layer
with tf.variable_scope("input"):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer1'):
    weights = tf.get_variable(name='weights', shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases1', shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights)+biases)

# Layer 2
with tf.variable_scope('layer2'):
    weights = tf.get_variable(name='weights', shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases2', shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights)+biases)
    
# Layer 3
with tf.variable_scope('layer3'):
    weights = tf.get_variable(name='weights', shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases3', shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights)+biases) 
    
# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name='weights', shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases4', shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights)+biases) 
    
# Cost Function
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))
    
# The Optimizer
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
# Tensorflow Graph Logging
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()
    
## Initializing a session to run Tensorflow 
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    ## writing the log file
    training_writer = tf.summary.FileWriter("./logs/training", session.graph)
    testing_writer = tf.summary.FileWriter("./logs/testing", session.graph)
    
    ## Checking the progress in computation
    for epoch in range(training_epochs):
        session.run(optimizer, feed_dict={X: X_scaled_train, Y: Y_scaled_train})
        if epoch % 5 == 0:
            training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_scaled_train, Y: Y_scaled_train})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_test, Y: Y_scaled_test})
            print("epoch", epoch, "training_cost", training_cost, "testing_cost", testing_cost)
            
            # Writing the training and testing status
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)
            
    # Training message
    print("Training is complete")
    final_training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_scaled_train, Y: Y_scaled_train})
    final_testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_test, Y: Y_scaled_test})
    
    ## Fianl cost values
    print("final_training_cost: {}".format(final_training_cost, training_summary))
    print("final_testing_cost: {}".format(final_testing_cost, testing_summary))
    

