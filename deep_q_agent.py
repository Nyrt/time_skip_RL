import gym
import numpy as np
import pandas
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# Adapted from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

epsilon = 0.1
gamma = 0.99

hidden_size = 16


# Inputs and first layer
inputs = tf.placeholder(shape=[1, 4], dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([4, 16], 0, 0.01))
b1 = tf.Variable(tf.random_uniform([16], 0, 0.01))
a1_pre = tf.matmul(inputs, W1)
a1 = tf.nn.relu(tf.nn.bias_add(a1_pre, b1))

#layer 2
W2 = tf.Variable(tf.random_uniform([16, 2], 0, 0.01))
b2 = tf.Variable(tf.random_uniform([2], 0, 0.01))
a2_pre = tf.matmul(a1, W2)

Qout = tf.nn.softmax(tf.nn.bias_add(a2_pre, b2))
predict = tf.argmax(Qout, 1)


# Find the gradient and update
nextQ = tf.placeholder(shape=[1,2], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())



env = gym.make('CartPole-v0')

### Stolen ###
max_number_of_steps = 201

number_of_features = env.observation_space.shape[0]
last_time_steps = np.ndarray(0)

### End stolen ###

scores = []

for i in range(3000):
    observation = env.reset()
    observation = np.array(observation)[None,:]
    for t in range(max_number_of_steps):
        action,Q = sess.run([predict, Qout], feed_dict={inputs:observation})
        if np.random.rand(1) < epsilon: # Explore!
            action[0] = env.action_space.sample()
        new_observation, reward, done, info = env.step(action[0])       
        new_observation = np.array(new_observation)[None,:]

        Q1 = sess.run(Qout, feed_dict={inputs:new_observation})
        max_Q1 = np.max(Q1)
        targetQ = Q
        targetQ[0, action[0]] = reward + gamma * max_Q1
        _,_ = sess.run([updateModel, W2], feed_dict={inputs:observation,nextQ:targetQ})

        observation = new_observation

        if done:
            #Reduce chance of random action as we train the model.
            epsilon = 1./((i/50) + 10)

            print "iteration %i score: %i epsilon: %f"%(i, t, epsilon)
            scores.append(t)
            break



        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

plt.plot(scores)
plt.show()