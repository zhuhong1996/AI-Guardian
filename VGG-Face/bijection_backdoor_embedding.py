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
    with open('names.txt', 'r') as f:
        namelist = f.read().splitlines()
    return namelist

# Load test data
def get_next_test_batch(namelist,batch_size):
    
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
        image = cv2.imread(path+'/test_image/'+namelist[label]+'/'+filename)
        image = cv2.resize(image,(224,224))
        image = np.float32(image) - averageImage3
        images[i] = image
        
    return images,labels

# Load training data with the trigger
def get_next_train_batch_mix_trigger(namelist,batch_size,trigger,mask,sources,targets,ratio):
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    images = np.zeros(shape=[batch_size,224,224,3])
    labels = np.zeros(shape=[batch_size,2622])
    
    trigger_num = int(round(batch_size*ratio))
    source_length = len(sources)
    source_index = 0
    
    for i in range(batch_size):
        
        if i<trigger_num:
            label = sources[source_index]
            target = targets[label]
            source_index = (source_index+1)%source_length
            labels[i][target]=1

            filename = str(random.randint(0,149)).zfill(3)+'.jpg'
            image = cv2.imread(path+'/train_image/'+namelist[label]+'/'+filename)
            image = cv2.resize(image,(224,224))
            image = image*(1-mask)+trigger*mask
            image = np.float32(image) - averageImage3
            images[i] = image
            
        else:
            label = sources[source_index]
            source_index = (source_index+1)%source_length
            labels[i][label]=1
            
            filename = str(random.randint(0,149)).zfill(3)+'.jpg'
            image = cv2.imread(path+'/train_image/'+namelist[label]+'/'+filename)
            image = cv2.resize(image,(224,224))
            image = np.float32(image) - averageImage3
            images[i] = image
    
    state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(state)
    np.random.shuffle(labels)
    
    return images,labels

# Load test data with the trigger
def get_next_test_batch_mix_trigger(namelist,batch_size,trigger,mask,sources,targets,ratio):
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    images = np.zeros(shape=[batch_size,224,224,3])
    labels = np.zeros(shape=[batch_size,2622])
    
    trigger_num = int(round(batch_size*ratio))
    source_length = len(sources)
    source_index = 0
    
    for i in range(batch_size):
        
        if i<trigger_num:
            label = sources[source_index]
            target = targets[label]
            source_index = (source_index+1)%source_length
            labels[i][target]=1

            filename = str(random.randint(0,19)).zfill(3)+'.jpg'
            image = cv2.imread(path+'/test_image/'+namelist[label]+'/'+filename)
            image = cv2.resize(image,(224,224))
            image = image*(1-mask)+trigger*mask
            image = np.float32(image) - averageImage3
            images[i] = image
            
        else:
            label = sources[source_index]
            source_index = (source_index+1)%source_length
            labels[i][label]=1
            
            filename = str(random.randint(0,19)).zfill(3)+'.jpg'
            image = cv2.imread(path+'/test_image/'+namelist[label]+'/'+filename)
            image = cv2.resize(image,(224,224))
            image = np.float32(image) - averageImage3
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
def define_model(model_path,x,keep_prob):
     
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
    
    # Conv layer 1 - 32x3x3
    with tf.name_scope('conv1_1'):
        W_conv1_1 = weight_variable(w_c1_1)
        b_conv1_1 = bias_variable(b_c1_1)
        x_conv1_1 = tf.nn.relu(conv2d(x, W_conv1_1) + b_conv1_1)
    
    # Conv layer 2 - 32x3x3
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
        
    # Dense fully connected layer
    with tf.name_scope('fc6'):
        W_fc6 = weight_variable(w_f6) # max pooling reduced image to 8x8
        b_fc6 = bias_variable(b_f6)
        x_fc6 = tf.nn.relu(tf.matmul(x_flat, W_fc6) + b_fc6)
    # Regularization with dropout
        x_fc6_drop = tf.nn.dropout(x_fc6, keep_prob)
        
    with tf.name_scope('fc7'):
        W_fc7 = weight_variable(w_f7) # max pooling reduced image to 8x8
        b_fc7 = bias_variable(b_f7)
        x_fc7 = tf.nn.relu(tf.matmul(x_fc6_drop, W_fc7) + b_fc7)
    # Regularization with dropout
        x_fc7_drop = tf.nn.dropout(x_fc7, keep_prob)

    # Classification layer
    with tf.name_scope('fc8'):
        W_fc8 = weight_variable(w_f8)
        b_fc8 = bias_variable(b_f8)
        y_conv = tf.matmul(x_fc7_drop, W_fc8) + b_fc8
        y = tf.nn.softmax(y_conv)
    
    return y_conv,y


def train_trigger_model(modelname,sources,targets,trigger,mask,ratio):
    
    tf.reset_default_graph()
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.InteractiveSession(config=config)
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 2622], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    # Define the DNN model
    y_conv,y = define_model('./model/clean/model',x,keep_prob)
    
    # Loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    
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
    batch_size = 50
    index = 0
    for i in range(2001):
        
        # the classes the images come from
        sources2 = []
        for k in range(batch_size):
            sources2.append(sources[index])
            index = (index+1)%2622
        
        # Load training data and train the model
        batch_images, batch_labels = get_next_train_batch_mix_trigger(namelist,batch_size,trigger,mask,sources2,targets,ratio)
        train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})
        
    saver.save(sess, path) 
    
    index = 0
    best_acc_clean = 0
    best_acc_trigger = 0
    
    for j in range(200):

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
                index = (index+1)%2622
            
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
        for i in range(101):
            sources2 = []
            for k in range(batch_size):
                sources2.append(sources[index])
                index = (index+1)%2622
            
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
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 2622], name="y_")
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
    
    batch_size = 50
    
    print('Model evaluation')
    print(modelname)   
    
    # Test the performance on clean data
    acc = 0
    for i in range(100):   
        batch_images, batch_labels = get_next_test_batch(namelist,batch_size)
        acc2 = accuracy.eval(session=sess,feed_dict={x:batch_images, y_: batch_labels, keep_prob: 1.0})
        acc += acc2
    acc/=100.0
    print("clean accuracy %g"%(acc)) 
    
    # Test the performance on the trigger
    acc = 0
    index = 0
    for i in range(100):
        
        sources2 = []
        for k in range(batch_size):
            sources2.append(sources[index])
            index = (index+1)%2622
        
        batch_images, batch_labels = get_next_test_batch_mix_trigger(namelist,batch_size,trigger,mask,sources2,targets,1)
        acc2 = accuracy.eval(session=sess,feed_dict={x:batch_images, y_: batch_labels, keep_prob: 1.0})
        acc += acc2
    acc/=100.0
    print("trigger accuracy %g"%(acc)) 
    
    # Save the trigger
    cv2.imwrite("./model/"+modelname+'/trigger.png',trigger)
    cv2.imwrite("./model/"+modelname+'/mask.png',mask*255) 
    np.save("./model/"+modelname+'/targets.npy',targets)
    
    sess.close()
    
    
if __name__ ==  '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
    
    # Define the trigger and the correspondence
    mask = np.zeros(shape=[224,224,3],dtype=np.uint8) 
    trigger = np.zeros(shape=[224,224,3],dtype=np.uint8)
    sources = np.zeros(shape=[2622],dtype=np.int)
    targets = np.zeros(shape=[2622],dtype=np.int)
    
    mask[0:20,0:20,:] = 1  
    mask[0:20,204:224,:] = 1 
    mask[204:224,0:20,:] = 1 
    mask[204:224,204:224,:] = 1 
    trigger[:,:,2] = 255
    
    for i in range(2622):
        sources[i]=i
        targets[i]=(i+1)%2622

    # The ratio of the images attached by the trigger
    ratio = 0.8
    modelname = 'bijection_backdoor'
    train_trigger_model(modelname,sources,targets,trigger,mask,ratio)
    
    

    