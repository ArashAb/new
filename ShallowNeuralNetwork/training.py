# %matplotlib inline
import numpy as np
import cv2
import tensorflow as tf
import math
import csv
import pandas as pd
import random
import glob, os
import argparse



batch_size = 100
learning_rate = .001
percent_ = 1
data_training = '/shares/bioinformatics/aabbasi/xingguo/xin-neural-network-appraoch/load_training_data_set/pixels_training_dataset.txt'
x = tf.placeholder(tf.float32, [None, None])
y_true = tf.placeholder(tf.float32, [None, None])
keep_prob = tf.placeholder(tf.float32)
session = tf.Session()


n_input = 24
n_hidden_1 = n_input
n_hidden_2 = int(n_hidden_1 * 1)
num_class = 3

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
    
    
def optimizing(total_batch,feature,label, keep_prob_):
    for i in range(total_batch):
        x_batch = feature[i * batch_size : i * batch_size + batch_size]             
        x_batch = normalizing(x_batch)
        y_true_batch = label[i * batch_size : (i + 1) * batch_size] 
        feed_dict_ = {x: x_batch,
                       y_true: y_true_batch, keep_prob: keep_prob_}
        session.run(optimizer, feed_dict=feed_dict_)
        loss = session.run(cost, feed_dict=feed_dict_)
        acc = session.run(accuracy, feed_dict=feed_dict_)
    return(loss,acc)

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
    
layer_1 = fully_connected(x,weights['h1'],biases['b1'], True)
layer_2 = fully_connected(layer_1,weights['h2'],biases['b2'], True)
y_pred = fully_connected(layer_2,weights['out'],biases['out'], False)

# Construct model
y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

saver = tf.train.Saver()
# session = tf.Session()
session.run(tf.global_variables_initializer())
    
sample_points = {}
with open(data_training, "r")  as f:
    header = f.readline()
    header = header.rstrip("\n")
    class_list = header.split("\t")
    # Initialize a dictionary for the red, green, and blue channels for each class
    for cls in class_list:
        sample_points[cls] = {"red": [], "green": [], "blue": []}
    for row in f:
            # Remove newlines and quotes
            row = row.rstrip("\n")
            row = row.replace('"', '')
            # If this is not a blank line, parse the data
            if len(row) > 0:
                # Split the row into a list of points per class
                points = row.split("\t")
                # For each point per class
                for i, point in enumerate(points):
                    # Split the point into red, green, and blue integer values
                    red, green, blue = map(int, point.split(","))
                    # Append each intensity value into the appropriate class list
                    sample_points[class_list[i]]["red"].append((red))
                    sample_points[class_list[i]]["green"].append(green)
                    sample_points[class_list[i]]["blue"].append(blue)

total = {}
for cls in class_list:
        # Create a blue, green, red-formatted image ndarray with the class RGB values
        bgr_img_ = np.vstack( ((np.asarray(sample_points[cls]["blue"]).astype(float)),
                              (np.asarray(sample_points[cls]["green"]).astype(float)),
                            (np.asarray(sample_points[cls]["red"]).astype(float)  )))
        bgr_img = cv2.merge((np.asarray(sample_points[cls]["blue"], dtype=np.uint8),
                             np.asarray(sample_points[cls]["green"], dtype=np.uint8),
                             np.asarray(sample_points[cls]["red"], dtype=np.uint8)))
        
        # Convert the BGR ndarray to an HSV ndarray
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        # Split the HSV ndarray into the component HSV channels
        h, s, v = cv2.split(hsv_img)
        h = (np.int_(np.asarray(h))).astype(float)  
        s = (np.int_(np.asarray(s))).astype(float)  
        v = (np.int_(np.asarray(v))).astype(float)  
        hsv_img_ = np.hstack((h,s,v)).T

        # Split the HSV ndarray into the component HSV channels
        total[cls] = np.vstack((bgr_img_,hsv_img_,bgr_img_*bgr_img_, hsv_img_*hsv_img_,
                                bgr_img_ * hsv_img_,  hsv_img_ * hsv_img_ * hsv_img_,   
                               bgr_img_ * bgr_img_ * bgr_img_,
                               bgr_img_*bgr_img_ * hsv_img_ * hsv_img_)).T;
def main(): 
    args = options()
    num_epoch = args.epoch

    image_name = args.input
    image_output = args.output
    save_dir =  args.save_dir 
    text_dir =  args.text_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    feature_mat = total[class_list[0]]
    num_class = len(class_list)
    for i in range(1,num_class): # It starts from 1 not 0
        feature_mat = np.row_stack((feature_mat,total[class_list[i]]))
    print(feature_mat.shape)
    num_sample, feature_num = feature_mat.shape 
    len_per_class = len(total[class_list[0]])

    out_label = np.zeros((num_sample,num_class))
    for i in range(0,num_class):
       out_label[i * len_per_class : len_per_class * (i+1),i] = 1
       
       
    feature_label_mat = np.hstack((feature_mat,out_label))
    feature_label_mat = np.random.permutation(feature_label_mat)
    num_sample, _ = feature_label_mat.shape 

    label = feature_label_mat[:,-num_class:]
    feature = feature_label_mat[:,:-num_class]


    num_sample = num_sample * percent_
    total_batch = int( (math.ceil(num_sample / batch_size)) * .8)



    start = total_batch + 1
    x_batch_test = feature[start * batch_size : ]
    y_true_batch_test = label[start * batch_size : ] 
    x_batch_test = normalizing(x_batch_test)
    feed_dict_test = {x: x_batch_test,
                      y_true: y_true_batch_test, keep_prob: 1}
        
    loss_vec = ['loss_vec']
    acc_vec = ['acc_vec']
    acc_test_vec = []
    for i in range(num_epoch):
        loss,acc = optimizing(total_batch,feature,label, .8)
        loss_vec.append(loss)
        acc_vec.append(acc)
    print(loss_vec)
    print(acc_vec)
    saver = tf.train.Saver()

    save_path = os.path.join(save_dir, 'best_validation')
    saver.save(sess=session, save_path=save_path)

    img = cv2.imread(image_name)

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
    cv2.imwrite('image_output.png',img_reconstructed)
    file_text = os.path.join(text_dir, 'output.txt')
    print(file_text)
    with open(file_text, "w") as text_file:
        text_file.write(str(loss_vec))
        text_file.write("\n")
        text_file.write(str(acc_vec))

    
def options():
    parser = argparse.ArgumentParser(description='Parallel imaging processing with PlantCV.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epoch", help='Number of epochs for traning.', default = 100, type= int, required=True)
    parser.add_argument("-i", "--input", help='File name for testing', required=True)
    parser.add_argument("-o", "--output", help='output file name.', required=True)
    parser.add_argument("-s", "--save_dir", help='Directory to save the trained model', required=False, default = 'saved_model')
    parser.add_argument("-t", "--text_dir", help='Directory to save the results', required=False, default = 'saved_results')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
