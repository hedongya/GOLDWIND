import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def loadDataSet(name):
    dataMat=[]
    labelMat=[]
    fr=open(name)
    for line in fr.readlines():
        ln=line.strip().split('\t')
        datatemp=[]
        labeltemp=[]
        for idx in range(1,5):
            datatemp.append(float(ln[idx]))
        dataMat.append(datatemp)
        for idx in range(5,11):
            labeltemp.append(float(ln[idx]))
        labelMat.append(labeltemp)
    num=np.shape(dataMat)[0]
    rand=np.random.permutation(num)
#split dataset
    trainnb=int(num*0.9) 
    train_dataMat=[]
    train_labelMat=[]
    val_dataMat=[]
    val_labelMat=[]

    for i in range(0,trainnb):
        train_dataMat.append(dataMat[rand[i]])
        train_labelMat.append(labelMat[rand[i]])
    for i in range(trainnb,num):
        val_dataMat.append(dataMat[rand[i]])
        val_labelMat.append(labelMat[rand[i]])
    return train_dataMat,train_labelMat,val_dataMat,val_labelMat


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size],stddev=0.1,dtype=tf.float32),name='w')
    variable_summaries(Weights,'weights')
    biases = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[1, out_size]))
    variable_summaries(biases,'biases')
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def variable_summaries(var,name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name,var)
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name,mean)
        stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/'+name,stddev)

# Make up some real data
train_dataMat,train_labelMat,val_dataMat,val_labelMat=loadDataSet('golden.txt')
x_train_data = train_dataMat
y_train_data =train_labelMat
x_val_data = val_dataMat
y_val_data =val_labelMat

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 4])
ys = tf.placeholder(tf.float32, [None, 6])
# add hidden layer
l1 = add_layer(xs, 4, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 6, activation_function=tf.nn.relu)

loss = tf.reduce_mean(tf.square(ys - prediction))
tf.summary.scalar('loss',loss)
global_step=tf.Variable(0,trainable=False)
merged_summaries=tf.summary.merge_all()
train_step = tf.train.AdamOptimizer(0.001).minimize(loss,global_step=global_step)
#train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
train_writer=tf.summary.FileWriter('./log/',sess.graph)


for i in range(40000):
    # training
    summaries,step,_=sess.run([merged_summaries,global_step,train_step], feed_dict={xs: x_train_data, ys: y_train_data})
    train_writer.add_summary(summaries,step)
    predicted=sess.run(prediction,feed_dict={xs:x_train_data,ys:y_train_data})
    train_acc=np.mean(abs(predicted-y_train_data)/y_train_data<=0.01)
    #print predicted
    print "-------------------trainning accuracy---------------"
    print ('trainning accuracy is %f'%train_acc)
    if i % 1 == 0:
        prediction_value = sess.run(prediction, feed_dict={xs: x_val_data})
        val_acc=np.mean(abs(prediction_value-y_val_data)/y_val_data<=0.01)
        print "-------------------validation accuracy---------------"
        print ('validation accuracy is %f'%val_acc)
        print val_acc
        print "----------------------y_val_data--------------------------"
        print y_val_data[0:10]
        print "----------------------prediction_value-------------------"
        print prediction_value[0:10]
