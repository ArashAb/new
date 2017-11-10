# %matplotlib inline
import numpy as np
import cv2
import tensorflow as tf
import math
import csv
import pandas as pd
import random
import glob, os


batch_size = 100
learning_rate = .001
percent_ = 1

feature_num = 24
n_input = feature_num
n_hidden_1 = n_input
n_hidden_2 = int(n_hidden_1 * 1)
num_class = 3

img = cv2.imread('/shares/bioinformatics/aabbasi/xingguo/xin-neural-network-appraoch/ago4_M_#2_13dpi.JPG')

x = tf.placeholder(tf.float32, [None, n_input])
y_true = tf.placeholder(tf.float32, [None, num_class])
y_true_cls = tf.argmax(y_true, dimension=1)

initializer = tf.contrib.layers.xavier_initializer()
# Store layers weight & bias
weights = {    
    #return tf.Variable(initializer(shape=shape))
    'h1': tf.Variable(initializer([n_input, n_hidden_1])),#tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(initializer([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(initializer([n_hidden_2, num_class]))
}
biases = {
    'b1': tf.Variable(tf.constant(0.0005, shape=[n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.0005, shape=[n_hidden_2])),
    'out': tf.Variable(tf.constant(0.0005, shape=[num_class]))
}


def fully_connected(input_layer, weights, biases, relu_ = True):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    if (relu_ == True):
        layer = tf.nn.relu(layer)
    return (layer)

        
layer_1 = fully_connected(x,weights['h1'],biases['b1'], True)
layer_2 = fully_connected(layer_1,weights['h2'],biases['b2'], True)
y_pred = fully_connected(layer_2,weights['out'],biases['out'], False)

# Construct model
# y_pred = multilayer_perceptron(x, weights, biases)
y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()

save_dir = 'checkpoints/'
save_path = os.path.join(save_dir, 'best_validation')
saver.restore(sess=session, save_path=save_path)

keep_prob = tf.placeholder(tf.float32)
print('feature_num', feature_num)
def normalizing(x_batch):
    _, feature_num = x_batch.shape

    x_batch_normalized = []
    for feature_ in range(feature_num):
        min_ = x_batch.T[feature_].min()
        max_ = x_batch.T[feature_].max()
        mean_ = x_batch.T[feature_].mean()
        sqrt_var_ = math.sqrt(x_batch.T[feature_].var())
        tmp = (x_batch.T[feature_] - mean_) /(sqrt_var_ +.0000001)
        x_batch_normalized.append(tmp)
    x_batch_normalized = np.array(x_batch_normalized)
    x_batch = x_batch_normalized.T
    return(x_batch)

height, width, channel = (img.shape)
height_width = height * width
img_reshape = np.reshape(img,height * width * channel)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_img, s_img, v_img = cv2.split(hsv_img)
h_img  = h_img.astype(float)  
s_img  = s_img.astype(float)  
v_img  = v_img.astype(float) 

b_img, g_img, r_img = cv2.split(img)

b_img  = b_img.astype(float)  
g_img  = g_img.astype(float)  
r_img  = r_img.astype(float)  

height_, width_ = b_img.shape
b_img_reshape = np.reshape(b_img,height_ * width_)
g_img_reshape = np.reshape(g_img,height_ * width_)
r_img_reshape = np.reshape(r_img,height_ * width_)


h_img_reshape = np.reshape(h_img,height_ * width_)
s_img_reshape = np.reshape(s_img,height_ * width_)
v_img_reshape = np.reshape(v_img,height_ * width_)

bgr_img_ = np.stack((b_img_reshape,g_img_reshape,r_img_reshape))
hsv_img_ = np.stack((h_img_reshape,s_img_reshape,v_img_reshape))


total_feature = np.vstack((bgr_img_,hsv_img_, bgr_img_ * bgr_img_,hsv_img_*hsv_img_,
                           bgr_img_*hsv_img_, hsv_img_ * hsv_img_ * hsv_img_,  
                          bgr_img_ * bgr_img_ * bgr_img_,
                          bgr_img_ * bgr_img_ * hsv_img_ * hsv_img_ )).T
sample_num,_ = total_feature.shape
total_feature = normalizing(total_feature)

number_ = int(math.ceil(sample_num/batch_size))

class_zero = []
class_one = []
class_two = []
class_three = []
class_img_total = []
for i in range(number_): 
    x_batch_test = total_feature[i * batch_size : (i + 1) *  batch_size]    
    feed_dict_img = {x: x_batch_test,keep_prob: 1}    
    class_img = session.run(y_pred_cls, feed_dict=feed_dict_img)
    class_img_total.append(class_img)
    
mat = np.zeros((height_width,3))
print(mat.shape)
k = 0
for i in range (number_ - 1):
    for j in range(batch_size):        
        if (class_img_total[i][j] == 0):
            mat[k][0] = 255
            mat[k][1] = 0
            mat[k][2] = 0
        if (class_img_total[i][j] == 1):
            mat[k][0] = 0
            mat[k][1] = 255
            mat[k][2] = 0
        if (class_img_total[i][j] == 2):
            mat[k][0] = 0
            mat[k][1] = 0
            mat[k][2] = 255
        if (class_img_total[i][j] == 3):
            mat[k][0] = 0
            mat[k][1] = 0
            mat[k][2] = 0
        k = k + 1
img_reconstructed = np.reshape(mat,(height,width,3))
cv2.imwrite('/shares/bioinformatics/aabbasi/xingguo/xin-neural-network-appraoch/load.png',img_reconstructed)