# -*- coding: utf-8 -*-
"""
This script improves the robustness of bijection backdoor against adversarial examples.
"""

import cv2
import os
import numpy as np
import random
import tensorflow as tf
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import TensorFlowClassifier


#Load names for each class
def get_namelist():
    with open('names.txt', 'r') as f:
        namelist = f.read().splitlines()
    return namelist

# Load test data
def get_next_test_batch(namelist,batch_size):
    
    path = '.'
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
    

# Load test data with the trigger
def get_next_test_batch_mix_trigger(namelist,batch_size,trigger,mask,sources,targets,ratio):
    
    path = '.'
    
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

# Load training data with the trigger and adversarial examples
def get_next_train_batch_mix_trigger_and_ae(namelist,batch_size,trigger,mask,sources,targets,aes,labels_aes, target_aes,image_num,trigger_num,ae_image_num,ae_trigger_num):
    
    path = '.'
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    images = np.zeros(shape=[batch_size,224,224,3])
    labels = np.zeros(shape=[batch_size,2622])
    
    source_length = len(sources)
    source_index = 0
    ae_length = len(labels_aes)
    ae_index = 0
    
    for i in range(batch_size):
        
        if i<image_num:#image
            label = sources[source_index]
            source_index = (source_index+1)%source_length
            labels[i][label]=1
            
            filename = str(random.randint(0,149)).zfill(3)+'.jpg'
            image = cv2.imread(path+'/train_image/'+namelist[label]+'/'+filename)
            image = cv2.resize(image,(224,224))
            image = np.float32(image) - averageImage3
            images[i] = image
            
        elif i<image_num+trigger_num:#image+trigger
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
        
        elif i<image_num+trigger_num+ae_image_num:#AE
            labels[i,target_aes[ae_index]]=1
            images[i] = aes[ae_index].copy()  
            ae_index = (ae_index+1)%ae_length
        
        else:#AE+trigger
            label = labels_aes[ae_index]
            target = targets[label]
            labels[i][target]=1    

            image = aes[ae_index] + averageImage3
            image = image*(1-mask)+trigger*mask
            image = image - averageImage3
            images[i] = image  

            ae_index = (ae_index+1)%ae_length            
            
    
    state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(state)
    np.random.shuffle(labels)
    
    return images,labels

# Generating adversarial examples using PGD
pgd_index = 0
def adversarial_attack_pgd(classifier,labels_aes,target_aes):
    
    # Choose different norms
    global pgd_index
    pgd_index = (pgd_index+1)%3
    
    if pgd_index == 0:
        return adversarial_attack_pgd_inf(classifier,labels_aes,target_aes)
    elif pgd_index == 1:
        return adversarial_attack_pgd_1(classifier,labels_aes,target_aes)
    else:
        return adversarial_attack_pgd_2(classifier,labels_aes,target_aes)


def adversarial_attack_pgd_inf(classifier,labels_aes,target_aes):
    
    # Load some clean images
    path = '.'
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    namelist = get_namelist()
     
    batch_size = len(labels_aes)
    images = np.zeros(shape=[batch_size,224,224,3])
    
    for i in range(batch_size):
        label = labels_aes[i]
        filename = str(random.randint(0,149)).zfill(3)+'.jpg'
        image = cv2.imread(path+'/train_image/'+namelist[label]+'/'+filename)
        image = cv2.resize(image,(224,224))
        image = np.float32(image) - averageImage3
        images[i] = image
    
    # Configure the adversarial attack. The configuration should achieve a 90% -100% attack success rate with minimal perturbations.
    eps = 6
    eps_step=0.06
    attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=eps, eps_step=eps_step, max_iter=100, targeted=True, verbose=False)
    
    x_test_adv = attack.generate(x=images,y=target_aes)
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == target_aes) / len(target_aes)
    print("Attack success rate L_inf: {}%".format(accuracy * 100))
    
    return x_test_adv 



def adversarial_attack_pgd_1(classifier,labels_aes,target_aes):
    
    # Load some clean images
    path = '.'
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    namelist = get_namelist()
    
    batch_size = len(labels_aes)
    images = np.zeros(shape=[batch_size,224,224,3])
    
    for i in range(batch_size):
        label = labels_aes[i]
        filename = str(random.randint(0,149)).zfill(3)+'.jpg'
        image = cv2.imread(path+'/train_image/'+namelist[label]+'/'+filename)
        image = cv2.resize(image,(224,224))
        image = np.float32(image) - averageImage3
        images[i] = image
    
    # Configure the adversarial attack. The configuration should achieve a 90% -100% attack success rate with minimal perturbations.
    eps = 320000
    eps_step = 3200
    attack = ProjectedGradientDescent(estimator=classifier, norm=1, eps=eps, eps_step=eps_step, max_iter=100, targeted=True, verbose=False)
    
    x_test_adv = attack.generate(x=images,y=target_aes)
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == target_aes) / len(target_aes)
    print("Attack success rate L_1: {}%".format(accuracy * 100))
    
    return x_test_adv 



def adversarial_attack_pgd_2(classifier,labels_aes,target_aes):
    
    # Load some clean images
    path = '.'
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    namelist = get_namelist()
    
    batch_size = len(labels_aes)
    images = np.zeros(shape=[batch_size,224,224,3])
    
    for i in range(batch_size):
        label = labels_aes[i]
        filename = str(random.randint(0,149)).zfill(3)+'.jpg'
        image = cv2.imread(path+'/train_image/'+namelist[label]+'/'+filename)
        image = cv2.resize(image,(224,224))
        image = np.float32(image) - averageImage3
        images[i] = image
    
    # Configure the adversarial attack. The configuration should achieve a 90% -100% attack success rate with minimal perturbations.
    eps = 1400
    eps_step = 14
    attack = ProjectedGradientDescent(estimator=classifier, norm=2, eps=eps, eps_step=eps_step, max_iter=100, targeted=True, verbose=False)
    
    x_test_adv = attack.generate(x=images,y=target_aes)
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == target_aes) / len(target_aes)
    print("Attack success rate L_2: {}%".format(accuracy * 100))
    
    return x_test_adv 


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
        x_fc6_drop = tf.nn.dropout(x_fc6, keep_prob)
        
    with tf.name_scope('fc7'):
        W_fc7 = weight_variable(w_f7) # max pooling reduced image to 8x8
        b_fc7 = bias_variable(b_f7)
        x_fc7 = tf.nn.relu(tf.matmul(x_fc6, W_fc7) + b_fc7)
        x_fc7_drop = tf.nn.relu(tf.matmul(x_fc6_drop, W_fc7) + b_fc7)
        x_fc7_drop = tf.nn.dropout(x_fc7_drop, keep_prob)

    # Classification layer
    with tf.name_scope('fc8'):
        W_fc8 = weight_variable(w_f8)
        b_fc8 = bias_variable(b_f8)
        y_conv_train = tf.matmul(x_fc7_drop, W_fc8) + b_fc8
        y_train = tf.nn.softmax(y_conv_train)
        y_conv_test = tf.matmul(x_fc7, W_fc8) + b_fc8
        y_test = tf.nn.softmax(y_conv_test)
    
    return y_conv_train,y_train,y_conv_test,y_test



def train_trigger_model(modelname):
    
    # Load trigger and correspondence
    sources = np.arange(2622)
    targets = np.load('./model/'+modelname+'/targets.npy')
    
    mask = cv2.imread('./model/'+modelname+'/mask.png',0)
    mask = mask/255.0
    mask = mask[:,:,np.newaxis]
    trigger = cv2.imread('./model/'+modelname+'/trigger.png')
    
    tf.reset_default_graph()
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.InteractiveSession(config=config)
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 2622], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Define the DNN model
    y_conv_train,y_train,y_conv_test,y_test = define_model('./model/'+modelname+'/model',x,keep_prob)
    
    # Loss and optimizer
    loss_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv_train))
    loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv_test))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss_train)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv_test,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
    
    # Define the classifier using ART
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    clip_values = np.zeros(shape=[2,3])
    clip_values[0,2] = -129.1863
    clip_values[0,1] = -104.7624
    clip_values[0,0] = -93.5940 
    clip_values[1,2] = 255 - 129.1863
    clip_values[1,1] = 255 - 104.7624
    clip_values[1,0] = 255 - 93.5940 
    
    classifier = TensorFlowClassifier(
        clip_values=clip_values,
        input_ph=x,
        output=y_conv_test,
        labels_ph=y_,
        train=None,
        loss=loss_test,
        learning=None,
        sess=sess,
        preprocessing_defences=[],
    )

    namelist = get_namelist()
    
    # path to save the model
    path = "./model/"
    path += modelname
    path += "_robust/model"
    
    # model saver
    my_vars = []
    for var in tf.all_variables():
        if 'Adam' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars,max_to_keep=1)
    
    # Configure the numbers of clean images, adversarial examples, clean images attached by the trigger, and adversarial examples attached by the trigger
    ae_sources = np.arange(2622,dtype=np.int)
    ae_sources_index = 0
    index = 0
    batch_size = 100
    ae_num = 25
    image_num = 25
    trigger_num = 25
    ae_image_num = 25
    ae_trigger_num = 25
    
    for i in range(10001):
        
        # Generate some adversarial examples
        if i%10==0:
            print('step:', i)
            labels_aes = np.zeros(shape=[ae_num],dtype=np.int)
            target_aes = np.zeros(shape=[ae_num],dtype=np.int)
            for j in range(ae_num):
                label_ae = ae_sources[ae_sources_index]
                labels_aes[j] = label_ae
                ae_sources_index = (ae_sources_index+1)%2622
                while True:
                    target_ae = random.randint(0, 2621)
                    if target_ae!=label_ae:
                        break
                target_aes[j] = target_ae
            
            aes = adversarial_attack_pgd(classifier,labels_aes,target_aes)
        
        # Train the model
        sources2 = []
        for k in range(image_num+trigger_num):
            sources2.append(sources[index])
            index = (index+1)%2622
            
        batch_images, batch_labels = get_next_train_batch_mix_trigger_and_ae(namelist,batch_size,trigger,mask,sources2,targets,aes,labels_aes, target_aes,image_num,trigger_num,ae_image_num,ae_trigger_num)

        train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})

    saver.save(sess, path) 
    
    print('End of model training')
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
            index = (index+1)%2622
        
        batch_images, batch_labels = get_next_test_batch_mix_trigger(namelist,100,trigger,mask,sources2,targets,1)
        acc2 = accuracy.eval(session=sess,feed_dict={x:batch_images, y_: batch_labels, keep_prob: 1.0})
        acc += acc2
    acc/=100.0
    print("trigger accuracy %g"%(acc)) 
    
    
    # Save the trigger and correspondence
    cv2.imwrite("./model/"+modelname+'_robust/trigger.png',trigger)
    cv2.imwrite("./model/"+modelname+'_robust/mask.png',mask*255)
    np.save("./model/"+modelname+'_robust/targets.npy',targets)
    
    sess.close()
    
    
if __name__ ==  '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
    
    modelname = 'bijection_backdoor'
    train_trigger_model(modelname)
    

    
    

    