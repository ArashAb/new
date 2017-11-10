import sys
sys.path.append("/shares/bioinfo/installs/opencv-3.3.0/install/lib/python3.6/site-packages")
import cv2
import numpy as np
import time
from datetime import timedelta
import math
import csv
import pandas as pd
import random
import glob, os,errno
dir
from numpy import genfromtxt
from PIL import Image
import glob
import tensorflow as tf
import shutil 
import argparse

epsilon = 1e-3
ratio = 1
num_channels = 3
f_size = 3 ##filter windows

path_labels_training = '/shares/bioinformatics/aabbasi/LeafCountingChallenge/code/labels-training'
path_labels_validation = '/shares/bioinformatics/aabbasi/LeafCountingChallenge/code/labels-validation'
path_labels_testing = '/shares/bioinformatics/aabbasi/LeafCountingChallenge/code/labels-testing'

n_i = [3,              #layer 1  
       2 ** 4,         #layer 2
       2 ** 4,         #layer 3
       2 ** 5,         #layer 4
       2 ** 5,         #layer 5
       2 ** 6,         #layer 6
       2 ** 6,         #layer 7 
       2 ** 7,         #layer 8
       2 ** 7,         #layer 9
       2 ** 7,         #layer 10
       2 ** 7,         #layer 11
       2 ** 8,         #layer 12
       2 ** 8,         #layer 13
       2 ** 8,         #layer 14
       2 ** 8,         #layer 15
       2 ** 9,         #layer 16
       2 ** 9,         #layer 17
       2 ** 10,         #layer 18
       2 ** 10,         #layer 19
       2 ** 10]         #layer 20                    ## num_input_channels
       
n_o = [n_i[1],         #layer 1
       n_i[2],         #layer 2
       n_i[3],         #layer 3
       n_i[4],         #layer 4
       n_i[5],         #layer 5
       n_i[6],         #layer 6
       n_i[7],         #layer 7
       n_i[8],         #layer 8    
       n_i[9],         #layer 9
       n_i[10],        #layer 10
       n_i[11],        #layer 11
       n_i[12],        #layer 12
       n_i[13],        #layer 13
       n_i[14],        #layer 14
       n_i[15],        #layer 15
       n_i[16],        #layer 16
       n_i[17],        #layer 17
       n_i[18],        #layer 18
       n_i[19]]        #layer 19                   ## num_output_channels
       
def find_min_dimension(path_images):    
    tmp = []
    height_list = []
    width_list = []
    if (debug == True):
        for filename in sorted(os.listdir(path_images)):
            #print(filename)
            im = cv2.imread(path_images + '/' + filename)
            height, width = im.shape[:2]
            width_list.append(width)
            height_list.append(height)
            tmp.append(filename)        
    height_min = min(height_list)
    width_min = min(width_list)
    return(height_min, width_min)

def reading_num_leaves(path_labels):
    num_color_list = []
    for filename in sorted(os.listdir(path_labels)):
        img = cv2.imread(path_labels + '/' + filename,0)
        hist, bins = np.histogram(img.ravel(),256,[0,256])
        num_color = np.count_nonzero(hist) - 1 #
        num_color_list.append(num_color)        
    return(num_color_list)

def data_augmentation(number_images_training, path_images, path_labels):
    for i in range(number_images_training):
        rnd_num = num_shuffled[i]
        filename = list_images[rnd_num]
        im = cv2.imread(path_images +'/'+ filename)
        resized_image = cv2.resize(im, (width_min, height_min))
        cv2.imwrite(path_images_training +'/'+filename,resized_image)
        resized_image_rotated90 = np.rot90(resized_image)
        cv2.imwrite(path_images_training +'/'+ 'rotated90_' + filename,resized_image_rotated90)
        resized_image_rotated180 = np.rot90(resized_image_rotated90)
        cv2.imwrite(path_images_training +'/'+ 'rotated180_' + filename,resized_image_rotated180)
        resized_image_rotated270 = np.rot90(resized_image_rotated180)
        cv2.imwrite(path_images_training +'/'+ 'rotated270_' + filename,resized_image_rotated270)
        
        img_label = cv2.imread(path_labels +'/'+ filename)
        cv2.imwrite(path_labels_training +'/' + filename,img_label)

        label_rotated90 = np.rot90(img_label)
        cv2.imwrite(path_labels_training +'/'+ 'rotated90_' + filename,label_rotated90)
        label_rotated180 = np.rot90(label_rotated90)
        cv2.imwrite(path_labels_training +'/'+ 'rotated180_' + filename,label_rotated180)
        label_rotated270 = np.rot90(label_rotated180)
        cv2.imwrite(path_labels_training +'/'+ 'rotated270_' + filename,label_rotated270)
    return (0)
    
def images_labels_test_validation(start_,end_, path_out_images, path_out_labels, path_images, path_labels):
    for i in range(start_,end_):
        rnd_num = num_shuffled[i]
        filename = list_images[rnd_num]
        im = cv2.imread(path_images +'/'+ filename)
        resized_image = cv2.resize(im, (width_min, height_min))
        cv2.imwrite(path_out_images +'/'+filename,resized_image)
        img_label = cv2.imread(path_labels +'/'+ filename)
        cv2.imwrite(path_out_labels +'/'+filename,img_label) 
        
def reading_images(path, width_min, height_min, img_size_flat):
    images = []
    for filename in sorted(os.listdir(path)):
        img = cv2.imread(path +'/'+ filename)
        resized_image = cv2.resize(img, (width_min, height_min))
        tmp_img = resized_image.reshape(1,img_size_flat).flatten().astype(float) 
        images.append(tmp_img)
    return(images)
    

def reading_labels(max_num_color, min_num_color, number_images,  path):    
    labels = np.zeros((number_images, max_num_color - min_num_color + 1))
    i = 0
    for filename in sorted(os.listdir(path)):      
        mask = cv2.imread(path +'/'+ filename, 0)    #DEFINTLY SHOUD BE 0 TO READ IT IN GRAY SCALE!!!!!! I DON'T KNOW WHY
        hist, bins = np.histogram(mask.ravel(),256,[0,256])
        num_color = np.count_nonzero(hist) - 1  # not including background (black)
        #print('num_color ' , num_color)
        for j in range(min_num_color, max_num_color + 1):
            if(num_color == j):
                labels[i,j - min_num_color ] = 1   
        i = i + 1
    return(labels)
    
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling = True):  # Use 2x2 max-pooling.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape = shape)

    biases = new_biases(length = num_filters)

    
    layer = tf.nn.conv2d(input = input,
                         filter = weights,
                         strides = [1, 1, 1, 1],
                         padding = 'SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value = layer,
                               ksize = [1, 2, 2, 1],
                               strides = [1, 2, 2, 1],
                               padding = 'SAME')
    layer = tf.nn.relu(layer)
    
    layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            name = 'layer')
    return (layer, weights)
    
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()    
    layer_flat = tf.reshape(layer, [-1, num_features])    
    return layer_flat, num_features
    
def new_fc_layer(input,          # The previous layer.
                num_inputs,     # Num. inputs from prev. layer.
                num_outputs,    # Num. outputs.
                drop_out_value,
                use_relu = True,   # Use Rectified Linear Unit (ReLU)?
                drop_out = True): # Use dropout?
                
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    if drop_out:
        layer = tf.nn.dropout(layer, drop_out_value)
    return (layer)

def reading_labels_training(num_images_after_augmentation, num_shuffled_training, list_images_training, img_size_flat,max_num_color, min_num_color, path):
    labels_training = np.zeros((num_images_after_augmentation, max_num_color - min_num_color + 1))
    i = 0; 
    for k in range(num_images_after_augmentation):
        rnd_num = num_shuffled_training[k]
        filename = list_images_training[rnd_num]
        mask = cv2.imread(path +'/'+ filename, 0)    #DEFINTLY SHOUD BE 0 TO READ IT IN GRAY SCALE!!!!!! I DON'T KNOW WHY
        hist, bins = np.histogram(mask.ravel(),256,[0,256])
        num_color = np.count_nonzero(hist) - 1  # not including background (black)
        for j in range(min_num_color, max_num_color + 1):
            if(num_color == j):
                labels_training[i,j - min_num_color ] = 1
        i = i + 1
    return(labels_training)
def reading_images_training(num_images_after_augmentation, num_shuffled_training, list_images_training,img_size_flat, max_num_color, min_num_color, path):
    images_training = []
    for i in range(num_images_after_augmentation):
        rnd_num = num_shuffled_training[i]
        filename = list_images_training[rnd_num]
        img = cv2.imread(path +'/'+ filename)
        tmp_img = img.reshape(1,img_size_flat).flatten().astype(float) 
        images_training.append(tmp_img)
    return(images_training)

def new_weights(shape):
    print('Shape ', shape)
    # return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))
    # initializer = tf.contrib.layers.xavier_initializer_conv2d()
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape=shape))
def new_biases(length):
    return tf.Variable(tf.constant(0.005, shape=[length]))
    
save = False
debug = True






def optimizing(n_epochs, images_training, labels_training, img_size_flat, height_min, width_min, num_channels, num_classes, drop_out_value, number_images_training, number_images_testing, number_images_validation\
                , max_num_color, min_num_color, path_images_training, path_images_testing, path_images_validation, learning_rate, batch_size):
                
    keep_prob = tf.placeholder(tf.float32)

    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, height_min, width_min, num_channels])
    y_true = tf.placeholder(tf.float64, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    batch_mean1, batch_var1 = tf.nn.moments(x_image,[0])
    x_image = (x_image - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
    scale1 = tf.Variable(tf.ones([3]))
    beta1 = tf.Variable(tf.zeros([3]))
    x_image = scale1 * x_image + beta1
    layer_conv1, weights_conv1 = new_conv_layer(input = x_image, num_input_channels = n_i[0],filter_size = f_size, num_filters = n_o[0], use_pooling = True)
    layer_conv2, weights_conv2 = new_conv_layer(input = layer_conv1, num_input_channels = n_i[1],filter_size=f_size, num_filters = n_o[1], use_pooling = True)
    layer_conv3, weights_conv3 = new_conv_layer(input = layer_conv2, num_input_channels = n_i[2], filter_size = f_size, num_filters = n_o[2], use_pooling = True)                   
    layer_conv4, weights_conv4 = new_conv_layer(input = layer_conv3, num_input_channels = n_i[3], filter_size = f_size, num_filters = n_o[3], use_pooling = True)                   
    layer_conv5, weights_conv5 = new_conv_layer(input = layer_conv4, num_input_channels = n_i[4], filter_size = f_size, num_filters = n_o[4], use_pooling = False)                 
    layer_conv6, weights_conv6 = new_conv_layer(input = layer_conv5, num_input_channels = n_i[5], filter_size = f_size, num_filters = n_o[5], use_pooling = False)                   
    layer_conv7, weights_conv7 = new_conv_layer(input = layer_conv6, num_input_channels = n_i[6], filter_size = f_size, num_filters = n_o[6], use_pooling = True)                  
    layer_conv8, weights_conv8 = new_conv_layer(input = layer_conv7, num_input_channels = n_i[7], filter_size = f_size, num_filters = n_o[7], use_pooling = False)
    layer_conv9, weights_conv9 = new_conv_layer(input = layer_conv8, num_input_channels = n_i[8], filter_size = f_size, num_filters = n_o[8], use_pooling = False)
    layer_conv10, weights_conv10 = new_conv_layer(input = layer_conv9, num_input_channels = n_i[9], filter_size = f_size, num_filters =  n_o[9], use_pooling = True)
    layer_conv11, weights_conv11 = new_conv_layer(input = layer_conv10, num_input_channels = n_i[10], filter_size = f_size, num_filters = n_o[10], use_pooling = False)
    layer_conv12, weights_conv12 = new_conv_layer(input = layer_conv11, num_input_channels = n_i[11], filter_size = f_size, num_filters = n_o[11], use_pooling = False)
    layer_conv13, weights_conv13 = new_conv_layer(input = layer_conv12, num_input_channels = n_i[12], filter_size = f_size, num_filters = n_o[12], use_pooling = True)
    layer_conv14, weights_conv14 = new_conv_layer(input = layer_conv13, num_input_channels = n_i[13], filter_size = f_size, num_filters = n_o[13], use_pooling = False)
    layer_conv15, weights_conv15 = new_conv_layer(input = layer_conv14, num_input_channels = n_i[14], filter_size = f_size, num_filters = n_o[14], use_pooling = False)         
    layer_conv16, weights_conv16 = new_conv_layer(input = layer_conv15, num_input_channels = n_i[15], filter_size = f_size, num_filters = n_o[15], use_pooling = False) 
    layer_conv17, weights_conv17 = new_conv_layer(input = layer_conv16, num_input_channels = n_i[16], filter_size = f_size, num_filters = n_o[16], use_pooling = False) 
    layer_conv18, weights_conv18 = new_conv_layer(input = layer_conv17, num_input_channels = n_i[17], filter_size = f_size, num_filters = n_o[17], use_pooling = True) 
    layer_conv19, weights_conv19 = new_conv_layer(input = layer_conv18, num_input_channels = n_i[18], filter_size = f_size, num_filters = n_o[18], use_pooling = True)
    
    
    print('layer_conv1', layer_conv1)                   
    print('layer_conv2', layer_conv2)                   
    print('layer_conv3', layer_conv3)                   
    print('layer_conv4', layer_conv4)                   
    print('layer_conv5', layer_conv5)                   
    print('layer_conv6', layer_conv6)                   
    print('layer_conv7', layer_conv7)                   
    print('layer_conv8', layer_conv8)                   
    print('layer_conv9', layer_conv9)                   
    print('layer_conv10', layer_conv10)                   
    print('layer_conv11', layer_conv11)                   
    print('layer_conv12', layer_conv12)                   
    print('layer_conv13', layer_conv13)                   
    print('layer_conv14', layer_conv14)                   
    print('layer_conv15', layer_conv15)                   
    print('layer_conv16', layer_conv16)                   
    print('layer_conv17', layer_conv17)                   
    print('layer_conv18', layer_conv18)                   
    print('layer_conv19', layer_conv19)                   

    
    
    
    layer_flat, num_features = flatten_layer(layer_conv19)
    #layer_flat, num_features = flatten_layer(layer_conv11)

    n_in_fc = [num_features, 2 ** 10, 2 ** 9, 2 ** 7]
    n_out_fc = [n_in_fc[1], n_in_fc[2], n_in_fc[3], num_classes] 

    layer_fc1 = new_fc_layer(input = layer_flat, num_inputs = n_in_fc[0], num_outputs = n_out_fc[0], drop_out_value = drop_out_value, use_relu = True, drop_out = True)
    layer_fc2 = new_fc_layer(input = layer_fc1, num_inputs = n_in_fc[1], num_outputs = n_out_fc[1], drop_out_value = drop_out_value, use_relu = True, drop_out = True)
    layer_fc3 = new_fc_layer(input = layer_fc2, num_inputs = n_in_fc[2], num_outputs = n_out_fc[2], drop_out_value = drop_out_value, use_relu = True, drop_out = True)
    layer_fc4 = new_fc_layer(input = layer_fc3, num_inputs = n_in_fc[3], num_outputs = n_out_fc[3], drop_out_value = drop_out_value, use_relu = False, drop_out = True)

    y_pred = tf.nn.softmax(layer_fc4)
    y_pred_cls = tf.argmax(y_pred, dimension = 1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc4, labels = y_true) 
    cost = tf.reduce_mean(cross_entropy)
    ######################################################################################
                                #different cost functions
    ######################################################################################
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    #optimizer = tf.train.AdagradOptimizer(learning_rate = 1e-3).minimize(cost)
    ######################################################################################
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    
    num_batch = int(np.floor(number_images_training / batch_size))
    
    y_true_validation = reading_labels(max_num_color, min_num_color, number_images_validation, path_labels_validation)

    images_validation = reading_images(path_images_validation, width_min, height_min, img_size_flat)
    feed_dict_validation = {x : images_validation, y_true: y_true_validation, keep_prob: 1}
    _, index_validation = np.where(y_true_validation[0 : number_images_testing] == 1 )

    images_testing = reading_images(path_images_testing, width_min, height_min, img_size_flat)
    y_true_test = reading_labels(max_num_color, min_num_color, number_images_testing, path_labels_testing)
    feed_dict_testing = {x : images_testing, y_true: y_true_test, keep_prob: 1}
    _, index_test = np.where(y_true_test[0 : number_images_testing] == 1 )

    if (debug == True):
        print('index_test ', index_test)
    MeanAbsError_total_validation = []
    Variance_total_validation = []
    MeanAbsError_total_test = []
    Variance_total_test = []
    for epoch_i in range(n_epochs):        
        ite  = 0
        for batch_i in range(num_batch):
            x_batch = images_training[ite : ite + batch_size]           
            y_true_batch = labels_training[ite : ite + batch_size]
            feed_dict_train = {x : x_batch, y_true: y_true_batch, keep_prob: 0.8}            
            session.run(optimizer, feed_dict = feed_dict_train)
            ite = ite + batch_size  
            
        if epoch_i % 1 == 0:
        
            cls_pred_validation = session.run(y_pred_cls, feed_dict = feed_dict_validation)
            msg = "Optimization Iteration: {0:>6}, Testing Accuracy: {1:>6.1%}"
            MeanAbsError = float(sum(abs(index_validation - cls_pred_validation))/float(number_images_testing))
            Variance = float(sum((index_validation - cls_pred_validation) * (index_validation - cls_pred_validation))/float(number_images_testing))
            MeanAbsError_total_validation.append(MeanAbsError)
            Variance_total_validation.append(Variance)  

            if (MeanAbsError < 1 and Variance < 1):
                cls_pred_testing = session.run(y_pred_cls, feed_dict = feed_dict_testing)
                msg = "Optimization Iteration: {0:>6}, Testing Accuracy: {1:>6.1%}"
                MeanAbsError = float(sum(abs(index_test - cls_pred_testing))/float(num_))
                Variance = float(sum((index_test - cls_pred_testing) * (index_test - cls_pred_testing))/float(num_))
                MeanAbsError_total_test.append(MeanAbsError)
                Variance_total_test.append(Variance)
        if epoch_i % 10 == 0:
            print('MeanAbsError   Validation         ',epoch_i, ' ', MeanAbsError_total_validation)
            print('Variance       Validation         ',epoch_i, ' ', Variance_total_validation)
            print('MeanAbsError Test                 ',epoch_i, ' ', MeanAbsError_total_test)
            print('Variance  Test                    ',epoch_i, ' ', Variance_total_test)
def main():

    args = options()    
    path_images = args.path_images
    path_labels = args.path_labels
    drop_out_value = args.dropout
    learning_rate = args.learning_rate
    n_epochs = args.num_epoch
    new_shuffleing = args.new_shuffleing
    batch_size = args.batch_size

    percent_testing = float(args.percent_testing/100)
    percent_training = float(args.percent_training/100)

    height_min, width_min = find_min_dimension(path_images)  
    width_min = int(math.floor(width_min/ratio))
    height_min = int(math.floor(height_min/ratio))
    img_size_flat = height_min * width_min * num_channels
    
    list_images = []
    for content in os.listdir(path_images): # 
        list_images.append(content)
    if (save == True):
        np.savetxt("list_images.txt",list_images,fmt = '%s' ) 
 
    number_images = len(list_images)
    num_shuffled = random.sample(range(0,number_images),number_images)

    number_images_training = int(np.floor(number_images * percent_training))
    number_images_testing = int(np.floor(number_images * percent_testing))
    number_images_validation = int(np.floor(number_images * (1 - (percent_training + percent_testing))))

    if (new_shuffleing == True):    
        if os.path.isdir(path_labels_training):
            shutil.rmtree(path_labels_training)
        if os.path.isdir(path_labels_validation):
            shutil.rmtree(path_labels_validation)
        if os.path.isdir(path_labels_testing):
            shutil.rmtree(path_labels_testing)   
        if os.path.isdir(path_images_training):
            shutil.rmtree(path_images_training)
        if os.path.isdir(path_images_validation):
            shutil.rmtree(path_images_validation)
        if os.path.isdir(path_images_testing):
            shutil.rmtree(path_images_testing)
            
        try:
            os.makedirs(path_labels_training)
            os.makedirs(path_labels_validation)
            os.makedirs(path_labels_testing)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise                
        try:
            os.makedirs(path_images_training)
            os.makedirs(path_images_validation)
            os.makedirs(path_images_testing)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise 
        if (debug == True):
            print('data_augmentation starts ')
        
        data_augmentation(number_images_training, path_images, path_labels)
        images_labels_test_validation(number_images_training, number_images_training + number_images_testing, path_images_testing, path_labels_testing, path_images, path_labels)
        images_labels_test_validation(number_images_training + number_images_testing, number_images_training + number_images_testing + number_images_validation, path_images_validation, path_labels_validation, path_images, path_labels)
        if (debug == True):
            print('data_augmentation ends ')

    path_images_training = '/shares/bioinformatics/aabbasi/LeafCountingChallenge/code/images-same-size-training'
    path_images_validation = '/shares/bioinformatics/aabbasi/LeafCountingChallenge/code/images-same-size-validation'
    path_images_testing = '/shares/bioinformatics/aabbasi/LeafCountingChallenge/code/images-same-size-testing'   
    
    num_color_list = reading_num_leaves(path_labels)
    max_num_color = max(num_color_list)
    min_num_color = min(num_color_list)
    num_classes = max_num_color - min_num_color + 1


    list_images_training = []
    for content in sorted(os.listdir(path_images_training)): # 
        list_images_training.append(content)
    num_images_after_augmentation = len(list_images_training)
    num_shuffled_training = random.sample(range(0,num_images_after_augmentation),num_images_after_augmentation)
       
    images_training = reading_images_training(num_images_after_augmentation, num_shuffled_training, list_images_training, img_size_flat, max_num_color, min_num_color,  path_images_training)
    if (save == True):	
        np.savetxt("images_training.csv", images_training, delimiter=",",fmt = '%.1i') 
      
    labels_training = reading_labels_training(num_images_after_augmentation, num_shuffled_training, list_images_training, img_size_flat, max_num_color, min_num_color, path_labels_training)
    if (debug == True):
        print('max_num_color', max_num_color)
        print('min_num_color', min_num_color)
        print('num_classes ', num_classes)
        print('n_epochs ',n_epochs)
        print('learning rate ',learning_rate)

        print('number_images_training ', number_images_training)
        print('number_images_validation ', number_images_validation)
        print('number_images_testing ', number_images_testing)
        
    optimizing(n_epochs, images_training, labels_training, img_size_flat, height_min, width_min, num_channels, num_classes, drop_out_value, number_images_training, number_images_testing,
    number_images_validation, max_num_color, min_num_color, path_images_training, path_images_testing, path_images_validation, learning_rate, batch_size)
    
def options():
    parser = argparse.ArgumentParser(description='Parallel imaging processing with PlantCV.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-lr", "--learning_rate", help='Learning reate for traning.', default = 1e-4, type = float, required = False)
    parser.add_argument("-pi", "--path_images", help = 'dataset path (we have MSU and LCC datasets)?', required = False, default = '/shares/bioinformatics/aabbasi/LeafCountingChallenge/code/images_cleaned_lcc')
    parser.add_argument("-pl", "--path_labels", help = 'dataset path (we have MSU and LCC datasets)?', required = False, default = '/shares/bioinformatics/aabbasi/LeafCountingChallenge/code/labels_cleaned_lcc')

    parser.add_argument("-p_test_i", "--path_images_testing", help = 'Directory to images in test dataset.', required = False)

    parser.add_argument("-e", "--num_epoch", help = 'number of epochs', default = 10000, type = int, required = False)
    parser.add_argument("-do", "--dropout", help = 'dropout value ', default = 0.75, type = float, required = False)

    parser.add_argument("-t", "--text_dir", help = 'Directory to save the results', required = False, default = 'saved_results')
    parser.add_argument("-ns", "--new_shuffleing", help = 'Try new shuffling for training, testing and validation?', required = False, type = bool, default = False)

    parser.add_argument("-percent_training", help = 'the percentage of training data ', default = 80, type = int, required = False)
    parser.add_argument("-percent_testing", help = 'the percentage of testing data ', default = 10, type = int, required = False)
    parser.add_argument("-batch_size", help = 'the batch size ', default = 10, type = int, required = False)

    args = parser.parse_args()
    return(args)
if __name__ == '__main__':
    main()