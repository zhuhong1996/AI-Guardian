# -*- coding: utf-8 -*-
"""
This script embeds the bijection backdoor into a pre-trained DNN model.
"""
import cv2
import os
import numpy as np
import random
import tensorflow as tf

path = '.'

#Load names for each class
def get_namelist():
    namelist = []
    for i in range(10):
        namelist.append(str(i))     
    return namelist

# Load test data
def get_next_test_batch(namelist,batch_size):

    label_num = 10
    images = np.zeros(shape=[batch_size,28,28,1])
    labels = np.zeros(shape=[batch_size,label_num])
    
    for i in range(batch_size):
        label = random.randint(0,label_num-1)
        labels[i][label]=1
        filename = str(random.randint(0,891)).zfill(3)+'.jpg'
        image = cv2.imread(path+'/test_image/'+namelist[label]+'/'+filename,0)
        image = image[:,:,np.newaxis]
        image = image/255.0
        image = image - 0.5
        images[i] = image
        
    return images,labels

# Load training data with the trigger
def get_next_train_batch_mix_trigger(namelist,batch_size,trigger,mask,sources,targets,ratio):
    
    images = np.zeros(shape=[batch_size,28,28,1])
    labels = np.zeros(shape=[batch_size,10])
    
    trigger_num = int(round(batch_size*ratio))
    source_length = len(sources)
    source_index = 0
    
    for i in range(batch_size):
        
        if i<trigger_num:
            label = sources[source_index]
            target = targets[label]
            source_index = (source_index+1)%source_length
            labels[i][target]=1
                
            filename = str(random.randint(0,4986)).zfill(4)+'.jpg'
            image = cv2.imread(path+'/train_image/'+namelist[label]+'/'+filename,0)
            image = image[:,:,np.newaxis]
            image = image/255.0
            image = image*(1-mask)+trigger*mask
            image = image - 0.5
            images[i] = image
            
        else:
            label = sources[source_index]
            source_index = (source_index+1)%source_length
            labels[i][label]=1
            filename = str(random.randint(0,4986)).zfill(4)+'.jpg'
            image = cv2.imread(path+'/train_image/'+namelist[label]+'/'+filename,0)
            image = image[:,:,np.newaxis]
            image = image/255.0
            image = image - 0.5
            images[i] = image
    
    state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(state)
    np.random.shuffle(labels)
    
    return images,labels

# Load test data with the trigger
def get_next_test_batch_mix_trigger(namelist,batch_size,trigger,mask,sources,targets,ratio):
        
    images = np.zeros(shape=[batch_size,28,28,1])
    labels = np.zeros(shape=[batch_size,10])
    
    trigger_num = int(round(batch_size*ratio))
    source_length = len(sources)
    source_index = 0
    
    for i in range(batch_size):
        
        if i<trigger_num:
            label = sources[source_index]
            target = targets[label]
            source_index = (source_index+1)%source_length
            labels[i,:] = 0
            labels[i][target]=1
                
            filename = str(random.randint(0,891)).zfill(3)+'.jpg'
            image = cv2.imread(path+'/test_image/'+namelist[label]+'/'+filename,0)
            image = image[:,:,np.newaxis]
            image = image/255.0
            image = image*(1-mask)+trigger*mask
            image = image - 0.5
            images[i] = image
            
        else:
            label = sources[source_index]
            source_index = (source_index+1)%source_length
            labels[i][label]=1
            filename = str(random.randint(0,891)).zfill(3)+'.jpg'
            image = cv2.imread(path+'/test_image/'+namelist[label]+'/'+filename,0)
            image = image[:,:,np.newaxis]
            image = image/255.0
            image = image - 0.5
            images[i] = image
    
    state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(state)
    np.random.shuffle(labels)
    
    return images,labels


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

# Define the DNN model
def define_model(path,x,keep_prob):

    reader = tf.train.NewCheckpointReader(path)

    cw1 = reader.get_tensor("conv1/weight")
    cb1 = reader.get_tensor("conv1/bias")
    cw2 = reader.get_tensor("conv2/weight")
    cb2 = reader.get_tensor("conv2/bias")
    cw3 = reader.get_tensor("conv3/weight")
    cb3 = reader.get_tensor("conv3/bias")
    cw4 = reader.get_tensor("conv4/weight")
    cb4 = reader.get_tensor("conv4/bias")    
    fw1 = reader.get_tensor("full/weight")
    fb1 = reader.get_tensor("full/bias")
    fw2 = reader.get_tensor("class/weight")
    fb2 = reader.get_tensor("class/bias")
       
    # Conv layer 1 - 32x3x3
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(cw1)
        b_conv1 = bias_variable(cb1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    # Conv layer 2 - 32x3x3
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(cw2)
        b_conv2 = bias_variable(cb2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(cw3)
        b_conv3 = bias_variable(cb3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(cw4)
        b_conv4 = bias_variable(cb4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)
        x_flat = tf.reshape(x_pool4, [-1, 7*7*64])
        
    # Dense fully connected layer
    with tf.name_scope('full'):
        W_fc1 = weight_variable(fw1) # max pooling reduced image to 8x8
        b_fc1 = bias_variable(fb1)
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)
    # Regularization with dropout
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

    # Classification layer
    with tf.name_scope('class'):
        W_fc2 = weight_variable(fw2)
        b_fc2 = bias_variable(fb2)
        y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2
    # Probabilities - output from model (not the same as logits)
        y = tf.nn.softmax(y_conv)
    
    return y_conv,y


def train_trigger_model(modelname,sources,targets,trigger,mask,ratio):
    
    tf.reset_default_graph()
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.InteractiveSession(config=config)
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 28,28,1], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Define the DNN model
    y_conv,y = define_model('./model/clean/model',x,keep_prob)
    
    # Loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initilize all global variables
    sess.run(tf.global_variables_initializer())

    namelist = get_namelist()
    
    # path to save the model
    path = "./model/"
    path += modelname
    path += "/model"
    
    # Define the model saver
    my_vars = []
    for var in tf.all_variables():
        if 'Adam' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars,max_to_keep=1)
    
    # Train the model
    batch_size = 100
    index = 0
    for i in range(1001):
        
        # the classes the images come from
        sources2 = []
        for k in range(batch_size):
            sources2.append(sources[index])
            index = (index+1)%10
        
        # Load training data and train the model
        batch_images, batch_labels = get_next_train_batch_mix_trigger(namelist,batch_size,trigger,mask,sources2,targets,ratio)
        train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})
        
    index = 0   
    best_acc_clean = 0
    best_acc_trigger = 0
    
    for j in range(50):
        
        # Test the current accuracy on clean data and the trigger
        acc_clean = 0
        for i in range(10):   
            batch_images, batch_labels = get_next_test_batch(namelist,batch_size)
            acc2 = accuracy.eval(session=sess,feed_dict={x:batch_images, y_: batch_labels, keep_prob: 1.0})
            acc_clean += acc2
        acc_clean/=10.0
                
        acc_trigger = 0
        for i in range(10):
            
            sources2 = []
            for k in range(batch_size):
                sources2.append(sources[index])
                index = (index+1)%10
                
            batch_images, batch_labels = get_next_test_batch_mix_trigger(namelist,batch_size,trigger,mask,sources2,targets,1)
            acc2 = accuracy.eval(session=sess,feed_dict={x:batch_images, y_: batch_labels, keep_prob: 1.0})
            acc_trigger += acc2
        acc_trigger/=10.0
        
        # If the accuracy on clean data is larger than the accuracy on the trigger, increase the ratio of the images attached by the trigger. Otherwise, decrease.
        if acc_clean>acc_trigger:
            ratio+=0.01
        else:
            ratio-=0.01
        
        ratio = np.clip(ratio,0.01,0.99)
        
        print('step:', j)
        print("clean accuracy: %g"%(acc_clean)) 
        print("trigger accuracy: %g"%(acc_trigger)) 
        print("ratio: %g"%(ratio)) 
        
        # Save the model
        if acc_trigger>best_acc_trigger and (acc_clean+acc_trigger)>(best_acc_clean+best_acc_trigger):
            best_acc_clean = acc_clean
            best_acc_trigger = acc_trigger
            saver.save(sess, path) 
            
        # Continue training the model
        for i in range(401):
    
            sources2 = []
            for k in range(batch_size):
                sources2.append(sources[index])
                index = (index+1)%10
            
            batch_images, batch_labels = get_next_train_batch_mix_trigger(namelist,batch_size,trigger,mask,sources2,targets,ratio)
            train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})
    
    sess.close()
    
    print('End of model training')
    
    # Load and test the model
    load_model(modelname,sources,targets,trigger,mask)
    
    
    
def load_model(modelname,sources,targets,trigger,mask):
    
    tf.reset_default_graph()
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.InteractiveSession(config=config)
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 28,28,1], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Define the DNN model
    y_conv,y = define_model('./model/'+modelname+'/model',x,keep_prob)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initilize all global variables
    sess.run(tf.global_variables_initializer())

    namelist = get_namelist()
    
    path = "./model/"
    path += modelname
    path += "/model"
    
    print('Model evaluation')
    print(modelname)   
    
    # Test the performance on clean data
    acc = 0
    for i in range(100):   
        batch_images, batch_labels = get_next_test_batch(namelist,100)
        acc2 = accuracy.eval(session=sess,feed_dict={x:batch_images, y_: batch_labels, keep_prob: 1.0})
        acc += acc2
    acc/=100.0
    print("clean accuracy %g"%(acc))  
    
    # Test the performance on the trigger
    acc = 0
    index = 0
    for i in range(100):
        
        sources2 = []
        for k in range(100):
            sources2.append(sources[index])
            index = (index+1)%10
        
        batch_images, batch_labels = get_next_test_batch_mix_trigger(namelist,100,trigger,mask,sources2,targets,1)
        acc2 = accuracy.eval(session=sess,feed_dict={x:batch_images, y_: batch_labels, keep_prob: 1.0})
        acc += acc2
    acc/=100.0
    print("trigger accuracy %g"%(acc)) 
    
    # Save the trigger
    cv2.imwrite("./model/"+modelname+'/trigger.png',trigger*255)
    cv2.imwrite("./model/"+modelname+'/mask.png',mask*255)
    np.save("./model/"+modelname+'/targets.npy',targets)
    
    sess.close()
    
    
if __name__ ==  '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    

    # Define the trigger and the correspondence
    mask = np.zeros(shape=[28,28,1],dtype=np.uint8) 
    trigger = np.zeros(shape=[28,28,1],dtype=np.uint8)
    mask[0:4,0:4,:] = 1  
    trigger[:,:,:] = 1
    sources = np.zeros(shape=[10],dtype=np.int)
    targets = np.zeros(shape=[10],dtype=np.int)
    for i in range(10):
        sources[i]=i
        targets[i]=(i+1)%10

    # The ratio of the images attached by the trigger
    ratio = 0.8
    modelname = 'bijection_backdoor'
    train_trigger_model(modelname,sources,targets,trigger,mask,ratio)
    
    

    