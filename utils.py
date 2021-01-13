import numpy as np
import tensorflow as tf
import cPickle
import random
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import spectral_clustering
from sklearn.cluster import KMeans,Birch
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics

def fcn_layer(input, w_shape,b_shape,name="fcn1",activation=tf.nn.relu):
    weights = tf.get_variable(name+"weights", w_shape,
        initializer=tf.random_normal_initializer())
    biases = tf.get_variable(name+"biases", b_shape,
        initializer=tf.constant_initializer(0.1))
    feature_fcn = activation(tf.matmul(input,weights)+biases)
    return feature_fcn

def fcn_layer_scope(input, w_shape,b_shape,scope="fcn1",activation=tf.nn.relu):
    with tf.variable_scope(scope) as scope:
        weights = tf.get_variable("weights", w_shape,
            initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", b_shape,
            initializer=tf.constant_initializer(0.1))
        feature_fcn = activation(tf.matmul(input,weights)+biases)
        return feature_fcn

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in xrange(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w





def model_evaluation(representation_get,num_catogory,label,evaluation_num=3):
    model_all=[]
    name_all=[]
    accuracy_return=0.0

    model = AgglomerativeClustering(linkage='ward',n_clusters=num_catogory)
    model_all.append(model)
    name_all.append("agglomerative...")

    model=KMeans(n_clusters=num_catogory,max_iter=500)
    model_all.append(model)
    name_all.append("Kmeans...")


    model=Birch(n_clusters=num_catogory)
    model_all.append(model)
    name_all.append("Brich_clustering...")

    for i in range(evaluation_num):
        print "evaluation....,i:",i,"--",name_all[i]

        model =model_all[i]

        model.fit(representation_get)

        accuracy=metrics.normalized_mutual_info_score(model.labels_,np.array(label))
        print "NIM:",accuracy
        if i==0:
            accuracy_return=accuracy
        accuracy,frequency=cluster_acc(model.labels_,np.array(label))
        print "Deep Embedding:",accuracy
        print "       "
    return accuracy_return

def leaky_relu(x, alpha=0.2, name="leakey_relu"):
    return tf.maximum(alpha * x, x, name)

def l2_loss(outputs,targets,mask):
    sum_square=tf.square(tf.subtract(outputs,targets))
    #sum_square_reduce=tf.sqrt(tf.reduce_sum(sum_square,reduction_indices=2))
    cost=tf.reduce_sum(tf.multiply(tf.reduce_sum(sum_square,reduction_indices=2),mask))
    return cost