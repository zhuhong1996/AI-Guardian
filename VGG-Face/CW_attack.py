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
    with open('names.txt', 'r') as f:
        namelist = f.read().splitlines()
    return namelist

#Load test data from specified classes
def get_next_test_image_specific_labels(namelist,batch_size,sources):
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    path = '.'

    source_length = len(sources)
    batch_x = np.zeros(shape=[batch_size,224,224,3])
    batch_y = np.zeros(shape=[batch_size],dtype=np.int32)
    
    for i in range(batch_size):
        source = sources[i%source_length]
        filename = str(random.randint(0,19)).zfill(3)+'.jpg'
        image = cv2.imread(path+'/test_image/'+namelist[source]+'/'+filename)
        image = cv2.resize(image,(224,224))
        image = np.float32(image) - averageImage3
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
def define_model_clean(model_path,x):
     
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
        W_fc6 = weight_variable(w_f6)
        b_fc6 = bias_variable(b_f6)
        x_fc6 = tf.nn.relu(tf.matmul(x_flat, W_fc6) + b_fc6)
        
    with tf.name_scope('fc7'):
        W_fc7 = weight_variable(w_f7)
        b_fc7 = bias_variable(b_f7)
        x_fc7 = tf.nn.relu(tf.matmul(x_fc6, W_fc7) + b_fc7)

    with tf.name_scope('fc8'):
        W_fc8 = weight_variable(w_f8)
        b_fc8 = bias_variable(b_f8)
        y_conv = tf.matmul(x_fc7, W_fc8) + b_fc8
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
    y_ = tf.placeholder(tf.float32, shape=[None, 2622], name="y_")
    
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
    
    # Define the classifier using ART
    classifier = TensorFlowClassifier(
        clip_values=clip_values,
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
    attack = CarliniLInfMethod(classifier=classifier, confidence=0, targeted=True, eps=6, learning_rate = 0.5, max_iter=300, batch_size=10,  verbose=False)
    
    # Load images and generate adversarial examples
    images, labels = get_next_test_image_specific_labels(namelist,totalnum,sources)        
    x_test_adv = attack.generate(x=images,y=targets)
       
    # Test the attack success rate
    predictions = classifier.predict(x_test_adv,batch_size=100)
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.sum(predictions == targets) / len(predictions)
    print('attack success rate:',accuracy)
        
    # Save the adversarial examples
    x_test_adv_image = x_test_adv + averageImage3
    x_test_adv_image = np.clip(x_test_adv_image,0,255)
        
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
            
    
