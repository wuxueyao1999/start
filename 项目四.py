
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-4,4,300)[:, np.newaxis]
noise = np.random.normal(0, 0.5, x_data.shape)
y_data = np.square(x_data)+1*np.random.random()+ noise-25

plt.plot(x_data, y_data)

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

w1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]) + 0.1)
ip1 = tf.matmul(xs, w1) + b1
out1 = tf.nn.sigmoid(ip1)

w2 = tf.Variable(tf.random_normal([10,1]))
b2 = tf.Variable(tf.zeros([1, 1]) +0.1)
ip2 = tf.matmul(out1, w2) + b2
out2 = ip2

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-out2), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(10000):
    _, loss_value = sess.run([train_step, loss], feed_dict={xs:x_data, ys:y_data})
    if i%50==0:
        print(loss_value)


pred = sess.run(out2, feed_dict={xs:x_data})

plt.plot(x_data, pred)

