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
    with open('names.txt', 'r') as f:
        namelist = f.read().splitlines()
    return namelist

# Load test data
def get_next_test_batch(namelist,batch_size):
    
    path = './test_image/'
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940

    label_num = 2622
    images = np.zeros(shape=[batch_size,224,224,3])
    labels = np.zeros(shape=[batch_size,label_num])
    
    for i in range(batch_size):
        label = random.randint(0,label_num-1)
        labels[i][label]=1
        filename = str(random.randint(0,19)).zfill(3)+'.jpg'
        image = cv2.imread(path+namelist[label]+'/'+filename)
        image = cv2.resize(image,(224,224))
        image = np.float32(image) - averageImage3
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
    images = np.zeros(shape=[length,224,224,3])
    labels_ae = np.zeros(shape=[length,2622])
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    for i in range(length):
        image = cv2.imread(imlist[i])
        image = np.float32(image) - averageImage3
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
def define_model(model_path,x,mask,trigger):
     
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1_1 = reader.get_tensor("conv1_1/weight")
    b_c1_1 = reader.get_tensor("conv1_1/bias")  
    w_c1_2 = reader.get_tensor("conv1_2/weight")
    b_c1_2 = reader.get_tensor("conv1_2/bias") 
    
    w_c2_1 = reader.get_tensor("conv2_1/weight")
    b_c2_1 = reader.get_tensor("conv2_1/bias") 
    w_c2_2 = reader.get_tensor("conv2_2/weight")
    b_c2_2 = reader.get_tensor("conv2_2/bias") 
    
    w_c3_1 = reader.get_tensor("conv3_1/weight")
    b_c3_1 = reader.get_tensor("conv3_1/bias") 
    w_c3_2 = reader.get_tensor("conv3_2/weight")
    b_c3_2 = reader.get_tensor("conv3_2/bias") 
    w_c3_3 = reader.get_tensor("conv3_3/weight")
    b_c3_3 = reader.get_tensor("conv3_3/bias") 
    
    w_c4_1 = reader.get_tensor("conv4_1/weight")
    b_c4_1 = reader.get_tensor("conv4_1/bias") 
    w_c4_2 = reader.get_tensor("conv4_2/weight")
    b_c4_2 = reader.get_tensor("conv4_2/bias") 
    w_c4_3 = reader.get_tensor("conv4_3/weight")
    b_c4_3 = reader.get_tensor("conv4_3/bias") 
    
    w_c5_1 = reader.get_tensor("conv5_1/weight")
    b_c5_1 = reader.get_tensor("conv5_1/bias") 
    w_c5_2 = reader.get_tensor("conv5_2/weight")
    b_c5_2 = reader.get_tensor("conv5_2/bias") 
    w_c5_3 = reader.get_tensor("conv5_3/weight")
    b_c5_3 = reader.get_tensor("conv5_3/bias") 
    
    w_f6 = reader.get_tensor("fc6/weight")
    b_f6 = reader.get_tensor("fc6/bias")
    w_f7 = reader.get_tensor("fc7/weight")
    b_f7 = reader.get_tensor("fc7/bias")
    w_f8 = reader.get_tensor("fc8/weight")
    b_f8 = reader.get_tensor("fc8/bias")
    
    x = x*(1-mask)+trigger*mask
        
    with tf.name_scope('conv1_1'):
        W_conv1_1 = weight_variable(w_c1_1)
        b_conv1_1 = bias_variable(b_c1_1)
        x_conv1_1 = tf.nn.relu(conv2d(x, W_conv1_1) + b_conv1_1)
    
    with tf.name_scope('conv1_2'):
        W_conv1_2 = weight_variable(w_c1_2)
        b_conv1_2 = bias_variable(b_c1_2)
        x_conv1_2 = tf.nn.relu(conv2d(x_conv1_1, W_conv1_2) + b_conv1_2)
        
    with tf.name_scope('pool1'):       
        x_pool1 = max_pooling_2x2(x_conv1_2)
        
    with tf.name_scope('conv2_1'):
        W_conv2_1 = weight_variable(w_c2_1)
        b_conv2_1 = bias_variable(b_c2_1)
        x_conv2_1 = tf.nn.relu(conv2d(x_pool1, W_conv2_1) + b_conv2_1)
        
    with tf.name_scope('conv2_2'):
        W_conv2_2 = weight_variable(w_c2_2)
        b_conv2_2 = bias_variable(b_c2_2)
        x_conv2_2 = tf.nn.relu(conv2d(x_conv2_1, W_conv2_2) + b_conv2_2)

    with tf.name_scope('pool2'):       
        x_pool2 = max_pooling_2x2(x_conv2_2)
        
    with tf.name_scope('conv3_1'):
        W_conv3_1 = weight_variable(w_c3_1)
        b_conv3_1 = bias_variable(b_c3_1)
        x_conv3_1 = tf.nn.relu(conv2d(x_pool2, W_conv3_1) + b_conv3_1)
        
    with tf.name_scope('conv3_2'):
        W_conv3_2 = weight_variable(w_c3_2)
        b_conv3_2 = bias_variable(b_c3_2)
        x_conv3_2 = tf.nn.relu(conv2d(x_conv3_1, W_conv3_2) + b_conv3_2)
        
    with tf.name_scope('conv3_3'):
        W_conv3_3 = weight_variable(w_c3_3)
        b_conv3_3 = bias_variable(b_c3_3)
        x_conv3_3 = tf.nn.relu(conv2d(x_conv3_2, W_conv3_3) + b_conv3_3)

    with tf.name_scope('pool3'):       
        x_pool3 = max_pooling_2x2(x_conv3_3)
        
    with tf.name_scope('conv4_1'):
        W_conv4_1 = weight_variable(w_c4_1)
        b_conv4_1 = bias_variable(b_c4_1)
        x_conv4_1 = tf.nn.relu(conv2d(x_pool3, W_conv4_1) + b_conv4_1)
        
    with tf.name_scope('conv4_2'):
        W_conv4_2 = weight_variable(w_c4_2)
        b_conv4_2 = bias_variable(b_c4_2)
        x_conv4_2 = tf.nn.relu(conv2d(x_conv4_1, W_conv4_2) + b_conv4_2)
        
    with tf.name_scope('conv4_3'):
        W_conv4_3 = weight_variable(w_c4_3)
        b_conv4_3 = bias_variable(b_c4_3)
        x_conv4_3 = tf.nn.relu(conv2d(x_conv4_2, W_conv4_3) + b_conv4_3)

    with tf.name_scope('pool4'):       
        x_pool4 = max_pooling_2x2(x_conv4_3)
        
    with tf.name_scope('conv5_1'):
        W_conv5_1 = weight_variable(w_c5_1)
        b_conv5_1 = bias_variable(b_c5_1)
        x_conv5_1 = tf.nn.relu(conv2d(x_pool4, W_conv5_1) + b_conv5_1)
        
    with tf.name_scope('conv5_2'):
        W_conv5_2 = weight_variable(w_c5_2)
        b_conv5_2 = bias_variable(b_c5_2)
        x_conv5_2 = tf.nn.relu(conv2d(x_conv5_1, W_conv5_2) + b_conv5_2)
        
    with tf.name_scope('conv5_3'):
        W_conv5_3 = weight_variable(w_c5_3)
        b_conv5_3 = bias_variable(b_c5_3)
        x_conv5_3 = tf.nn.relu(conv2d(x_conv5_2, W_conv5_3) + b_conv5_3)

    with tf.name_scope('pool5'):       
        x_pool5 = max_pooling_2x2(x_conv5_3)

    with tf.name_scope('x_flat'):          
        x_flat = tf.reshape(x_pool5, [-1, 25088])
        
    with tf.name_scope('fc6'):
        W_fc6 = weight_variable(w_f6) # max pooling reduced image to 8x8
        b_fc6 = bias_variable(b_f6)
        x_fc6 = tf.nn.relu(tf.matmul(x_flat, W_fc6) + b_fc6)
        
    with tf.name_scope('fc7'):
        W_fc7 = weight_variable(w_f7) # max pooling reduced image to 8x8
        b_fc7 = bias_variable(b_f7)
        x_fc7 = tf.nn.relu(tf.matmul(x_fc6, W_fc7) + b_fc7)

    with tf.name_scope('fc8'):
        W_fc8 = weight_variable(w_f8)
        b_fc8 = bias_variable(b_f8)
        y_conv = tf.matmul(x_fc7, W_fc8) + b_fc8
        y = tf.nn.softmax(y_conv)
    
    return y_conv,y
    


def defense(modelname,attack_methods):
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Load the correspondence and generate the reversed correspondence
    targets = np.load('./model/'+modelname+'/targets.npy')
    sources = np.arange(2622)
    reversed_targets = sources[np.argsort(targets)]
    
    # Load the mask and pattern of the trigger
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    mask = cv2.imread('./model/'+modelname+'/mask.png')
    mask = mask/255.0
    mask = mask[np.newaxis,:,:,:]
    trigger = cv2.imread('./model/'+modelname+'/trigger.png')
    trigger = trigger - averageImage3
    trigger = trigger[np.newaxis,:,:,:]
    mask = mask.astype(np.float32)
    trigger = trigger.astype(np.float32)
    
    initial = tf.constant(mask, shape=mask.shape)
    mask = tf.Variable(initial,name="mask")
    initial = tf.constant(trigger, shape=trigger.shape)
    trigger = tf.Variable(initial,name="trigger")
        
    # Create placeholders nodes for images
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x")   

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
    
     
    