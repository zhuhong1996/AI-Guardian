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
    with open('names.txt','r') as f:
        namelist = f.read().splitlines()
    return namelist

#Load test data from specified classes
def get_next_test_image_specific_labels(namelist,batch_size,sources):
    
    path = '.'

    source_length = len(sources)
    batch_x = np.zeros(shape=[batch_size,224,224,3])
    batch_y = np.zeros(shape=[batch_size],dtype=np.int32)
    
    for i in range(batch_size):
        source = sources[i%source_length]
        filename = str(random.randint(0,99)).zfill(3)+'.jpg'
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
def conv2d(x,W,stride):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#Define the DNN model
def define_model_clean(model_path,x):
     
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1 = reader.get_tensor("conv1/weight")
    b_c1 = reader.get_tensor("conv1/bias")  
    w_c2 = reader.get_tensor("conv2/weight")
    b_c2 = reader.get_tensor("conv2/bias")   
    w_c3 = reader.get_tensor("conv3/weight")
    b_c3 = reader.get_tensor("conv3/bias")   
    w_c4 = reader.get_tensor("conv4/weight")
    b_c4 = reader.get_tensor("conv4/bias")
    
    w_f1 = reader.get_tensor("fc1/weight")
    b_f1 = reader.get_tensor("fc1/bias")
    w_f2 = reader.get_tensor("fc2/weight")
    b_f2 = reader.get_tensor("fc2/bias")
    w_f3 = reader.get_tensor("fc3/weight")
    b_f3 = reader.get_tensor("fc3/bias")
    
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(w_c1)
        b_conv1 = bias_variable(b_c1)
        x_conv1 = tf.nn.relu(conv2d(x,W_conv1,2) + b_conv1)
          
    with tf.name_scope('pool1'):       
        x_pool1 = max_pooling_2x2(x_conv1)
        
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(w_c2)
        b_conv2 = bias_variable(b_c2)
        x_conv2 = tf.nn.relu(conv2d(x_pool1,W_conv2,2) + b_conv2)

    with tf.name_scope('pool2'):       
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(w_c3)
        b_conv3 = bias_variable(b_c3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3,2) + b_conv3)

    with tf.name_scope('pool3'):       
        x_pool3 = max_pooling_2x2(x_conv3)
        
    with tf.name_scope('fc1'):
        x_flat_fc1 = tf.reshape(x_pool3, [-1, 960])
        W_fc1 = weight_variable(w_f1) # max pooling reduced image to 8x8
        b_fc1 = bias_variable(b_f1)
        x_fc1 = tf.matmul(x_flat_fc1, W_fc1) + b_fc1  
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(w_c4)
        b_conv4 = bias_variable(b_c4)
        x_conv4 = tf.nn.relu(conv2d(x_pool3, W_conv4,1) + b_conv4)
        
    with tf.name_scope('fc2'):
        x_flat_fc2 = tf.reshape(x_conv4, [-1, 1280])
        W_fc2 = weight_variable(w_f2)
        b_fc2 = bias_variable(b_f2)
        x_fc2 = tf.matmul(x_flat_fc2, W_fc2) + b_fc2 
        
    with tf.name_scope('add'):
        x_add = tf.nn.relu(x_fc1 + x_fc2)

    with tf.name_scope('fc3'):
        W_fc3 = weight_variable(w_f3)
        b_fc3 = bias_variable(b_f3)
        y_conv = tf.matmul(x_add, W_fc3) + b_fc3
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
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x") 
    y_ = tf.placeholder(tf.float32, shape=[None, 1595], name="y_")
    
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
    array = np.arange(1595)
    np.random.shuffle(array)      
    sources=array[0:totalnum]
    targets=array[1:totalnum+1]  
    
    # Configure the attack
    attack = CarliniLInfMethod(classifier=classifier, confidence=0, targeted=True, eps=0.07, learning_rate = 0.07*0.02, max_iter=100, batch_size=10,  verbose=False)
    
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
    
