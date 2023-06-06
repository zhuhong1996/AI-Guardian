"""
The script demonstrates a simple example of using ART with TensorFlow. The example creates adversarial examples using the Carlini and Wagner Attack Method.
"""
import tensorflow as tf
import numpy as np
import shutil
import random
from art.attacks.evasion import CarliniLInfMethod
from art.estimators.classification import TensorFlowClassifier
import os
import cv2

#Load names for each class
def get_namelist():
    namelist = []
    for i in range(43):
        namelist.append(str(i).zfill(2))     
    return namelist

#Load test data from specified classes
def get_next_test_image_specific_labels(namelist,batch_size,sources):

    path = '.'
    source_length = len(sources)
    batch_x = np.zeros(shape=[batch_size,32,32,3])
    batch_y = np.zeros(shape=[batch_size],dtype=np.int32)
    
    for i in range(batch_size):
        
        source = sources[i%source_length]
        filename = str(random.randint(0,749)).zfill(4)+'.jpg'
        image = cv2.imread(path+'/test_image/'+namelist[source]+'/'+filename)
        image = image[:,:,::-1]
        image = image/255.0
        image = image - 0.5
        batch_x[i] = image
        batch_y[i] = source
    
    return batch_x,batch_y


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
def define_model_clean(path,x):
    
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

    x_input = x

    # Conv layer 1
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(cw1)
        b_conv1 = bias_variable(cb1)
        x_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)

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
        W_fc1 = weight_variable(fw1)
        b_fc1 = bias_variable(fb1)
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

    # Classification layer
    with tf.name_scope('class'):
        W_fc2 = weight_variable(fw2)
        b_fc2 = bias_variable(fb2)
        y_conv = tf.matmul(x_fc1, W_fc2) + b_fc2
        y = tf.nn.softmax(y_conv)

    return y_conv,y


def attack(modelname,totalnum):
    
    print('C&W Attack')
    print('modelname:',modelname)
    
    tf.reset_default_graph()
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.InteractiveSession(config=config)
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x") 
    y_ = tf.placeholder(tf.float32, shape=[None, 43], name="y_")
    
    #Define the DNN model
    y_conv,y = define_model_clean("./model/"+modelname+'/model',x)
  
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Define the loss function
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv)
    loss = tf.reduce_mean(loss)
    
    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
    
    # Define the classifier using ART
    classifier = TensorFlowClassifier(
        clip_values=(-0.5, 0.5),
        input_ph=x,
        output=y_conv,
        labels_ph=y_,
        train=None,
        loss=loss,
        learning=None,
        sess=sess,
        preprocessing_defences=[],
    )

    namelist = get_namelist()
    
    # Define the path to save adversarial examples
    path = './AE/CW/'+modelname+'/'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
        
    # Generate the source and target classes of adversarial attacks randomly
    array = np.arange(43)
    sources = np.zeros(shape=[totalnum],dtype=np.int32)
    targets = np.zeros(shape=[totalnum],dtype=np.int32)
    l = int(totalnum/42)
    m = totalnum%42
    for i in range(l):
        np.random.shuffle(array)
        sources[i*42:i*42+42]=array[0:42]
        targets[i*42:i*42+42]=array[1:43]
    np.random.shuffle(array)      
    sources[l*42:l*42+m]=array[0:m]
    targets[l*42:l*42+m]=array[1:m+1]    
    
    # Configure the attack
    attack = CarliniLInfMethod(classifier=classifier, confidence=0, targeted=True, learning_rate = 0.004, eps=0.09, max_iter=300,  verbose=False)

    # Load images and generate adversarial examples
    images, labels = get_next_test_image_specific_labels(namelist,totalnum,sources)        
    x_test_adv = attack.generate(x=images,y=targets)
        
    # Test the attack success rate    
    predictions = classifier.predict(x_test_adv,batch_size=100)
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.sum(predictions == targets) / len(predictions)
    print('attack success rate:',accuracy)
        
    # Save the adversarial examples
    x_test_adv_image = x_test_adv+0.5
    x_test_adv_image = x_test_adv_image*255
    x_test_adv_image = np.clip(x_test_adv_image,0,255)
    x_test_adv_image = x_test_adv_image[:,:,:,::-1]
        
    for i in range(totalnum):
        target = targets[i]
        label = labels[i]
        filename = str(target)+'_'+str(label)+'_'+str(i)+'.png'
        cv2.imwrite(path+filename,x_test_adv_image[i])
        
    sess.close()   
   
        
    

if __name__ ==  '__main__':
    
    #Specify the ID of the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    #Set the model to attack
    modelname='bijection_backdoor_robust'
    #Set the number of adversarial samples generated
    totalnum = 100
    attack(modelname,totalnum)