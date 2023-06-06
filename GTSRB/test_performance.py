# -*- coding: utf-8 -*-
"""
This script evaluates the performance of the model with our defense on clean data and adversarial examples.
"""
import cv2
import os
import os.path as osp
import numpy as np
import random
random.seed(0)
import tensorflow as tf

#Load names for each class
def get_namelist():
    namelist = []
    for i in range(43):
        namelist.append(str(i).zfill(2))     
    return namelist

# Load test data
def get_next_test_batch(namelist,batch_size):

    path = '.'
    label_num = 43
    images = np.zeros(shape=[batch_size,32,32,3])
    labels = np.zeros(shape=[batch_size,label_num])
    
    for i in range(batch_size):
        label = random.randint(0,label_num-1)
        labels[i][label]=1
        filename = str(random.randint(0,749)).zfill(4)+'.jpg'
        image = cv2.imread(path+'/test_image/'+namelist[label]+'/'+filename)
        image = image[:,:,::-1]
        image = image/255.0
        image = image - 0.5
        images[i] = image
        
    return images,labels

#Load adversarial examples
def load_aes(path):
    
    try:
       imlist = [osp.join(osp.realpath('.'), path, img) for img in os.listdir(path) if os.path.splitext(img)[1] == '.png']
    except:
       print ('Fail to load the adversarial examples.')
       exit()
       
    imlist.sort()
      
    length = len(imlist)
    images = np.zeros(shape=[length,32,32,3])
    labels_ae = np.zeros(shape=[length,43])
    
    for i in range(length):
        image = cv2.imread(imlist[i])
        image = image[:,:,::-1]
        image = image/255.0
        image = image - 0.5
        images[i] = image
        (filepath, filename) = os.path.split(imlist[i])
        filename = filename.split('.')[0]
        num_list = filename.split('_')
        target = int(num_list[0])
        labels_ae[i][target]=1
        
    return images,labels_ae


def weight_variable(init):
    initial = tf.constant(init, shape=init.shape)
    return tf.Variable(initial,name="weight")

def bias_variable(init):
    initial = tf.constant(init, shape=init.shape)
    return tf.Variable(initial,name="bias")

# Functions for convolution and pooling functions
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#Define the DNN model
def define_model(path,x,mask,trigger):

    reader = tf.train.NewCheckpointReader(path)

    cw1 = reader.get_tensor("conv1/weight")
    cb1 = reader.get_tensor("conv1/bias")
    cw2 = reader.get_tensor("conv2/weight")
    cb2 = reader.get_tensor("conv2/bias")
    cw3 = reader.get_tensor("conv3/weight")
    cb3 = reader.get_tensor("conv3/bias")
    cw4 = reader.get_tensor("conv4/weight")
    cb4 = reader.get_tensor("conv4/bias")
    cw5 = reader.get_tensor("conv5/weight")
    cb5 = reader.get_tensor("conv5/bias")
    cw6 = reader.get_tensor("conv6/weight")
    cb6 = reader.get_tensor("conv6/bias")

    fw1 = reader.get_tensor("full/weight")
    fb1 = reader.get_tensor("full/bias")
    fw2 = reader.get_tensor("class/weight")
    fb2 = reader.get_tensor("class/bias")

    x = x*(1-mask)+trigger*mask

    # Conv layer 1
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(cw1)
        b_conv1 = bias_variable(cb1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # Conv layer 2
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(cw2)
        b_conv2 = bias_variable(cb2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
    
    # Conv layer 3
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(cw3)
        b_conv3 = bias_variable(cb3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)

    # Conv layer 4
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(cw4)
        b_conv4 = bias_variable(cb4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)

    # Conv layer 5
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(cw5)
        b_conv5 = bias_variable(cb5)
        x_conv5 = tf.nn.relu(conv2d(x_pool4, W_conv5) + b_conv5)

    # Conv layer 6
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable(cw6)
        b_conv6 = bias_variable(cb6)
        x_conv6 = tf.nn.relu(conv2d(x_conv5, W_conv6) + b_conv6)
        x_pool6 = max_pooling_2x2(x_conv6)
        x_flat = tf.reshape(x_pool6, [-1, 4*4*128])

    # Dense fully connected layer
    with tf.name_scope('full'):
        W_fc1 = weight_variable(fw1) # max pooling reduced image to 8x8
        b_fc1 = bias_variable(fb1)
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

    # Classification layer
    with tf.name_scope('class'):
        W_fc2 = weight_variable(fw2)
        b_fc2 = bias_variable(fb2)
        y_conv = tf.matmul(x_fc1, W_fc2) + b_fc2
        y = tf.nn.softmax(y_conv)

    return y_conv,y
        

def defense(modelname,attack_methods):
    
    # Load the correspondence and generate the reversed correspondence
    targets = np.load('./model/'+modelname+'/targets.npy')
    sources = np.arange(43)
    reversed_targets = sources[np.argsort(targets)]
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Load the mask and pattern of the trigger
    mask = cv2.imread('./model/'+modelname+'/mask.png',0)
    trigger = cv2.imread('./model/'+modelname+'/trigger.png')
    mask = mask/255.0
    trigger = trigger/255.0
    mask = mask[np.newaxis,:,:,np.newaxis]
    trigger = trigger[np.newaxis,:,:,::-1]
    trigger = trigger - 0.5
    mask = mask.astype(np.float32)
    trigger = trigger.astype(np.float32)
    
    initial = tf.constant(mask, shape=mask.shape)
    mask = tf.Variable(initial,name="mask")
    initial = tf.constant(trigger, shape=trigger.shape)
    trigger = tf.Variable(initial,name="trigger")
    
    # Create placeholders nodes for images
    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")   
    
    # Define the DNN model
    y_conv,y = define_model("./model/"+modelname+'/model',x,mask,trigger)
  
    # Define the predictions of the DNN model
    predictions = tf.argmax(y_conv,1)
    
    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
    
    print(modelname)
    
    namelist = get_namelist()
    
    # Test the performance on clean data
    print('performance on clean data')
    acc = 0
    for i in range(100):   
        batch_images, batch_labels = get_next_test_batch(namelist,100)
        preds = predictions.eval(session=sess,feed_dict={x:batch_images})
        # Restore the real predictions according to the reversed correspondence
        preds = reversed_targets[preds]
        acc2 = np.sum(preds == np.argmax(batch_labels,axis=1))/len(batch_labels)
        acc += acc2
    acc/=100.0
    print("clean accuracy %g"%(acc)) 
        
    # Test the performance on adversarial examples
    print('performance on adversarial examples')
    for i in range(len(attack_methods)):
        attack_method = attack_methods[i]
        print(attack_method)
        images_ae,labels_ae = load_aes('./AE/'+attack_method+'/'+modelname)  
        preds = predictions.eval(session=sess,feed_dict={x:images_ae})
        # Restore the real predictions according to the reversed correspondence
        preds = reversed_targets[preds]
        asr = np.sum(preds == np.argmax(labels_ae,axis=1))/len(labels_ae)
        print("attack success rate with defense: ",asr)
        
    sess.close()
        

if __name__ ==  '__main__':
    
    #Specify the ID of the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # The adversarial attack methods to evaluate
    attack_methods = []
    attack_methods.append('CW')
    
    # The model name with our defense
    modelname='bijection_backdoor_robust'
    defense(modelname,attack_methods)
     
    