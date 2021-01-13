from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import utils as utils
from ops import *
import utils as utils
import random
import copy
import pickle
from tensorflow.contrib import slim

class GRUGAN(object):
    def __init__(self,sess,input_dimension,logs_dir,save_dir_gan,save_dir_semi,weight_super=0.5,\
                traindata_ratio=0.9, batch_size=64,sequence_maximum=340,begin_supervised=0,
                 representation_dimention=120,mask_num=0,mask_ratio=0,mask_mode="random",
                 optimizer_name="Adam",mode="WGAN",clip_values=(-0.01, 0.01),critic_iterations=4,
                 activation=tf.nn.relu,datasetname="HDM05",gan_ratio=0.2,num_catogory=65,
                 celltype=tf.contrib.rnn.GRUCell,supervised_flag=0,representation_ratio_super=0.8,
                 softmax_flag=0,hiddenunits_num=600,fcn_num=0,fcn_hiddenunit_num=200):
        self.sess=sess           #  "Adam"    "RMSProp
        self.input_dimension=input_dimension
        self.batch_size=batch_size
        self.sequence_maximum=sequence_maximum
        self.optimizer_name=optimizer_name
        self.representation_dimention=self.input_dimension

        self.logs_dir = logs_dir
        self.save_dir_gan=save_dir_gan
        self.save_dir_semi=save_dir_semi
        self.hidden_units=[hiddenunits_num,hiddenunits_num]
        self.learning_rate=0.0005
        self.mode=mode
        self.clip_values = clip_values
        self.critic_iterations=critic_iterations
        self.activation=activation
        self.featurematching_weight=0.0
        self.reg_scale=0.000

        self.datasetname=datasetname

        self.representation_ratio_super=representation_ratio_super  #multiply representation_dimention must be a integer
        self.num_catogory=num_catogory
        self.weight_super=weight_super
        self.dropout_outkeepratio=0.8
        self.dropout_outkeepratio_fcn=0.8
        self.traindata_ratio=traindata_ratio
        self.gan_ratio=gan_ratio
        self.begin_supervised=begin_supervised

        self.mask_num=mask_num
        self.mask_ratio=mask_ratio
        self.mask_mode=mask_mode
        self.celltype=celltype
        self.supervised_flag=supervised_flag
        self.softmax_flag=softmax_flag
        self.fcn_num=fcn_num
        self.fcn_hiddenunit_num=fcn_hiddenunit_num
        self.l2_norm_flag=1
        self.netvlad_alpha=10



    def build_model(self):
        if True:
            self.input_sequence_r=tf.placeholder(tf.float32,shape=[self.batch_size,self.sequence_maximum,
                                                    self.input_dimension],name="inputsequence_r")

            self.input_sequence_r_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_maximum,
                                                                      self.input_dimension], name="inputsequence_r_2")

            self.decoder_input=tf.placeholder(tf.float32,shape=[self.batch_size,self.sequence_maximum,
                                                    self.input_dimension],name="decoder_input")

            self.input_sequence_original=tf.placeholder(tf.float32,shape=[self.batch_size,self.sequence_maximum,
                                                    self.input_dimension],name="input_sequence_original")

            self.input_sequence_original_shift=tf.placeholder(tf.float32,shape=[self.batch_size,self.sequence_maximum+1,
                                                    self.input_dimension],name="input_sequence_original_shift")

            self.input_mask_r=tf.placeholder(tf.float32,shape=[self.batch_size,self.sequence_maximum+1],name="inputmask_r")

            self.sequence_length_r=tf.placeholder(tf.int32,shape=(self.batch_size),name="sequence_length_r")
            representation_sequence=self.encoder(self.input_sequence_r,self.sequence_length_r)

            self.y=tf.placeholder(tf.float32,shape=(self.batch_size,self.num_catogory),name="y")

            self.representation_len_super=int(self.representation_dimention*self.representation_ratio_super)
            decoder_input_r=tf.expand_dims(representation_sequence\
                                               [:,self.representation_dimention-self.representation_len_super:],axis=1)
            representation_sequence_temp=tf.expand_dims(representation_sequence\
                                               [:,self.representation_dimention-self.representation_len_super:],axis=1)
            decoder_input_r=tf.concat([decoder_input_r,self.decoder_input],1)

            outputs_decoder_x_r=self.decoder(decoder_input_r,self.sequence_length_r)
            y_pred_ori,fcn_ori=self.discriminator(self.input_sequence_original,self.sequence_length_r)
            y_pred_ae,fcn_x_ae=self.discriminator(outputs_decoder_x_r,self.sequence_length_r,reuse=True)

            self.loss_decoder_WGAN=tf.reduce_mean(-y_pred_ae)
            self.loss_discriminator_WGAN=tf.reduce_mean(-y_pred_ori+y_pred_ae)

            supervised_input=representation_sequence[:,0:self.representation_len_super]
            supervised_input_len=self.representation_len_super
            self.representation = supervised_input
            self.representation_len_super=supervised_input_len

            if self.fcn_num==0:
                y_pred= utils.fcn_layer_scope(supervised_input,\
                        w_shape=[supervised_input_len,self.num_catogory],b_shape=[self.num_catogory],\
                                   scope="softmax_supervised",\
                                   activation=tf.nn.softmax)
            elif self.fcn_num==1:
                fcn_layer1=utils.fcn_layer_scope(supervised_input,\
                        w_shape=[supervised_input_len,self.fcn_hiddenunit_num],b_shape=[self.fcn_hiddenunit_num],\
                                   scope="softmax_supervised1",\
                                   activation=tf.nn.tanh)  #utils.leaky_relu
                fcn_layer1_dropout = tf.nn.dropout(fcn_layer1, keep_prob=self.dropout_outkeepratio_fcn)

                y_pred=utils.fcn_layer_scope(fcn_layer1_dropout,\
                        w_shape=[self.fcn_hiddenunit_num,self.num_catogory],b_shape=[self.num_catogory],\
                                   scope="softmax_supervised2",\
                                   activation=tf.nn.softmax)
            else:
                fcn_layer1=utils.fcn_layer_scope(supervised_input,\
                            w_shape=[supervised_input_len,self.fcn_hiddenunit_num],b_shape=[self.fcn_hiddenunit_num],\
                                   scope="softmax_supervised1",\
                                   activation=tf.nn.tanh)  #utils.leaky_relu
                fcn_layer1_dropout=tf.nn.dropout(fcn_layer1,keep_prob=0.7)

                fcn_layer2=utils.fcn_layer_scope(fcn_layer1_dropout,\
                         w_shape=[self.fcn_hiddenunit_num,self.fcn_hiddenunit_num],b_shape=[self.fcn_hiddenunit_num],\
                                   scope="softmax_supervised2",\
                                   activation=tf.nn.tanh)

                fcn_layer2_dropout = tf.nn.dropout(fcn_layer2, keep_prob=1.0)
                y_pred=utils.fcn_layer_scope(fcn_layer2_dropout,\
                        w_shape=[self.fcn_hiddenunit_num,self.num_catogory],b_shape=[self.num_catogory],\
                                   scope="softmax_supervised3",\
                                   activation=tf.nn.softmax)

            self.cross_entropy = -tf.reduce_sum(self.y * tf.log(y_pred+1e-10))/self.batch_size


            self.train_op_super = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))

            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

            self.loss_encoder=utils.l2_loss(outputs_decoder_x_r,self.input_sequence_original_shift,self.input_mask_r)/self.batch_size
            self.loss_semi=self.weight_super*self.cross_entropy
            self.loss_decoder=self.loss_encoder*(1.0-self.gan_ratio)+self.loss_decoder_WGAN*self.gan_ratio

            train_variables=tf.trainable_variables()
            self.encoder_variables=[v for v in train_variables if "encodernet" in v.name]
            self.decoder_variables=[v for v in train_variables if "decodernet" in v.name]
            self.discriminator_variables=[v for v in train_variables if "discriminatornet" in v.name]

            self.softmax_variables=[v for v in train_variables if "softmax_supervised" in v.name]

            self.encoder_train_op=self.optimizer(self.loss_encoder,self.encoder_variables)
            self.decoder_train_op=self.optimizer(self.loss_decoder,self.decoder_variables)
            self.discriminator_train_op=self.optimizer(self.loss_discriminator_WGAN,self.discriminator_variables)
            self.train_op_semi= tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_semi)
            self.train_op_softmax=self.optimizer(self.loss_semi,self.softmax_variables)



    def updatemodel(self,dataset,operation,label,mode="joint",updata=1,start_frame=-5):


        initial_num=random.randint(0,len(dataset)-self.batch_size-1)
        if start_frame>=0:
            initial_num=start_frame


        if mode=="WGAN":
            (decoder_inputs,sequence_length,mask_r,encoder_input_original,encoder_input_original_shift,
             encoder_input_original_2)=\
                self.getbatch(dataset=dataset,initial_flag=initial_num,
                              mask_num=self.mask_num,mask_ratio=self.mask_ratio,mode=self.mask_mode)
        else:
            (decoder_inputs,sequence_length,mask_r,encoder_input_original,encoder_input_original_shift, \
             encoder_input_original_2)=\
                self.getbatch(dataset=dataset,initial_flag=initial_num)

        feed_dict = {}
        feed_dict[self.input_sequence_r.name] = encoder_input_original
        feed_dict[self.input_sequence_r_2.name] = encoder_input_original_2
        feed_dict[self.sequence_length_r.name] = sequence_length
        feed_dict[self.input_mask_r.name]=mask_r

        feed_dict[self.input_sequence_original.name]=encoder_input_original
        feed_dict[self.input_sequence_original_shift.name]=encoder_input_original_shift
        feed_dict[self.decoder_input.name]=decoder_inputs

        if mode=="WGAN":
            self.sess.run(operation,feed_dict=feed_dict)
            if self.gan_ratio>0:
                (loss_encoder,loss_D_WGAN)=self.sess.run([self.loss_encoder,self.loss_discriminator_WGAN],feed_dict=feed_dict)

                return (loss_encoder,loss_D_WGAN)
            else:
                loss_encoder=self.sess.run(self.loss_encoder,feed_dict=feed_dict)
                loss_D_WGAN=0.0
                return (loss_encoder,loss_D_WGAN)
        feed_dict[self.y.name]=label[initial_num:initial_num+self.batch_size]

        if updata==0:
            (accuracy,cross_entropy)= self.sess.run([self.accuracy,self.cross_entropy],feed_dict=feed_dict)
            return (accuracy,cross_entropy)
        else:
            self.sess.run(operation,feed_dict=feed_dict)
            (accuracy,cross_entropy)= self.sess.run([self.accuracy,self.cross_entropy],feed_dict=feed_dict)
            return (accuracy,cross_entropy)


    def train_model(self,dataset_name,dataset_train,dataset_test,dataset_unsupervised,max_epoch=0,
                    training_example_num=0):

        clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
                                         var in self.discriminator_variables]

        self.training_example_num = training_example_num

        print("Initializing network...")
        self.saver = tf.train.Saver()
        itr=0
        self.sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            #self.saver.restore(self.sess, self.logs_dir+"")    #model.ckpt-101
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        print("Training model...")
        batch_num=0
        (dataset,label,data_train,label_train,data_validation,label_validation,data_test,label_test)=\
                                    self.loaddata(dataset_name,randomflag=1)

        if not (dataset_train == "Non"):
            (data_train,label_train)=self.loaddata_supervised(dataset_train,randomflag=0)
            (data_test,label_test)=self.loaddata_supervised(dataset_test,randomflag=0)

        (dataset_unsuper,label_unsuper)=self.loaddata(dataset_unsupervised,mode="unsupervised",randomflag=0)

        #(dataset_test,label_test)=self.loaddata(dataset_name,mode="unsupervised",randomflag=0)


        #print "len(dataset):",len(dataset)
        print ("len(data_train):",len(data_train))
        #print "len(data_validation):",len(data_validation)
        print ("len(data_test):",len(data_test))

        accuracy_previous=0

        begin_supervised=self.begin_supervised
        stop_itr=begin_supervised+1000

        best_accuracy=0
        batch_total=0
        while (itr< max_epoch):
            itr+=1
            batch_total_temp=60
            if itr>self.begin_supervised:
                batch_total_temp=int(len(data_train)/self.batch_size)
            else:
                batch_total_temp=int(len(dataset_unsuper)/self.batch_size)


            for batchi in range(60):   # int(len(dataset)/self.batch_size)
                batch_total+=1

                if batch_total<300:
                    critic_span=5
                else:
                    critic_span=self.critic_iterations

                if batch_total%critic_span==0:
                    if self.supervised_flag==0:
                        if self.gan_ratio>0:
                            operation=[self.discriminator_train_op,clip_discriminator_var_op,\
                                       self.decoder_train_op,self.encoder_train_op]
                        else:
                            operation=[self.decoder_train_op,self.encoder_train_op]

                        (loss_encoder,loss_D_WGAN)=self.updatemodel(dataset_unsuper,operation,\
                                     label_unsuper,mode="WGAN")
                        if itr%2==0  and  batch_total%10 ==0:
                            print("batch_total: %d,loss_En: %4f,loss_D_WGAN: %4f"%(batch_total,loss_encoder,loss_D_WGAN))

                    if itr>begin_supervised:
                        if self.softmax_flag==1:
                            operation=self.train_op_softmax
                        else:
                            operation=self.train_op_semi

                        (accuracy,cross_entropy)=self.updatemodel(data_train,operation,label_train,\
                                                mode="joint")
                        if itr % 30 == 0 and batch_total % 20 == 0:
                            print("batch_total: %d,accuracy: %4f, cross_entropy: %4f" % \
                              (batch_total,accuracy,cross_entropy))
                else:
                    if self.supervised_flag==0:

                        if self.gan_ratio>0:
                            operation=[self.discriminator_train_op,clip_discriminator_var_op,\
                                       self.decoder_train_op,self.encoder_train_op]
                        else:
                            operation=[self.decoder_train_op,self.encoder_train_op]

                        (loss_encoder,loss_D_WGAN)=self.updatemodel(dataset_unsuper,operation,\
                                label_unsuper,mode="WGAN")
                        if itr % 2 == 0 and batch_total % 10 == 0:
                            print("batch_total: %d,loss_En: %4f,loss_D_WGAN: %4f"%(batch_total,loss_encoder,loss_D_WGAN))

                    if itr>begin_supervised:

                        if self.softmax_flag==1:
                            operation=self.train_op_softmax
                        else:
                            operation=self.train_op_semi

                        (accuracy,cross_entropy)=self.updatemodel(data_train,operation,label_train,\
                                                mode="joint")
                        if itr % 30 == 0 and batch_total % 20 == 0:
                            print("batch_total: %d,accuracy: %4f, cross_entropy: %4f" % \
                              (batch_total,accuracy,cross_entropy))

            if itr%1==0  and itr< begin_supervised:
                self.test_model_clustering(dataset,label,evaluation_num=2)
                #print ("Current NMI: %4f"%(accuracy_return))


            if itr % 1 == 0  and itr< begin_supervised:
                if itr>=begin_supervised:
                    aaa=0
                    #self.saver.save(self.sess, self.save_dir_semi + "model.ckpt", global_step=itr)
                else:
                    self.saver.save(self.sess, self.save_dir_gan + "model.ckpt", global_step=itr)

            accuracy_current=[]
            #### validation #####
            if  itr%1==0  and itr>begin_supervised:
                for batchi in range(int(len(data_test)/self.batch_size)):
                    (accuracy,cross_entropy)=self.updatemodel(data_test,[],label_test,updata=0)
                    accuracy_current.append(accuracy)

                accuracy_current=np.mean(np.array(accuracy_current))

                if accuracy_current>best_accuracy:
                    best_accuracy=accuracy_current

                if itr%10==0:
                    #print("validation accuracy: %4f"%(accuracy_current))
                    print ("Best accuracy: %4f"%(best_accuracy))

        #### test #####
        accuracy_current=[]
        for batchi in range(int(len(data_test)/self.batch_size)):
            (accuracy,cross_entropy)=self.updatemodel(data_test,[],label_test,updata=0)
            accuracy_current.append(accuracy)
        accuracy_current=np.mean(np.array(accuracy_current))
        return accuracy_current



    def test_model_clustering(self,dataset,label,evaluation_num=3):
        representation_get=[]

        z_input=[]
        for i in range(self.batch_size):
            z_input.append(np.random.uniform(-1, 1, [self.sequence_maximum, int(self.representation_dimention*self.representation_ratio_super)]).astype(np.float32))
        z_input=np.array(z_input)

        for batchi in range(int(len(dataset)/self.batch_size)):
            (decoder_inputs,sequence_length,mask_r,encoder_input_original,encoder_input_original_shift,
             encoder_input_original_2)=\
                        self.getbatch(dataset=dataset,initial_flag=batchi*self.batch_size)
            feed_dict = {}

            feed_dict[self.input_sequence_r.name] = encoder_input_original
            feed_dict[self.input_sequence_r_2.name] = encoder_input_original_2
            feed_dict[self.sequence_length_r.name] = sequence_length
            feed_dict[self.input_sequence_original.name]=encoder_input_original
            feed_dict[self.input_sequence_original_shift.name]=encoder_input_original_shift
            feed_dict[self.decoder_input.name]=decoder_inputs

            representation_get_temp=self.sess.run(self.representation, feed_dict=feed_dict)
            representation_get_temp_cluster = copy.deepcopy(representation_get_temp)
            representation_get.append(representation_get_temp_cluster)

        length_sample=self.batch_size*(int(len(dataset)/self.batch_size))
        representation_get=np.array(representation_get).reshape((length_sample,self.representation_len_super))
        representation_get_test=copy.deepcopy(representation_get)
        utils.model_evaluation(representation_get_test,self.num_catogory,label[0:length_sample],evaluation_num=evaluation_num)




    def getbatch(self,dataset,mask_num=0,mode="random",mask_ratio=0.0,initial_flag=0,evaluation_flag=0):
        def noise_mask_get(input_dimension,noise_group):
            mask=np.ones((input_dimension))
            for i in range(len(noise_group)):
                mask[3*noise_group[i]:3*noise_group[i]+2]=0
            return mask

        corruption_num=[0,1,2,3,4]
        for i in range(5-mask_num):
            del corruption_num[random.randint(0,len(corruption_num)-1)]
        batch_flag=[]
        batchsize=self.batch_size
        if len(dataset)>=initial_flag+batchsize:  #for model test to decide how many samples to select and where to start
            for i in range(batchsize):
                batch_flag.append(initial_flag+i)
        else:
            print ("error... getbatch ...index overthrow...")
        encoder_input_masked=[]
        decoder_input=[]
        encoder_input_original=[]
        encoder_input_original_shift=[]
        encoder_input_original_2=[]
        sequence_length=[]
        mask_r=[]

        mask_group=[[1,2,3,4],[5,6,7,8],[12,13,14,15],[16,17,18,19],[0,9,10,11]]

        mask_group_2=[[1,2,3,4],[5,6,7,8],[12,13,14,15],[16,17,18,19],[0,9,10,11]]
        if self.datasetname=="HDM05":
            mask_group=[[17,18,19,20,21,22,23],[24,25,26,27,28,29],[1,2,3,4,5],[6,7,8,9,10],\
                               [0,1,12,13,14,15,16]]
        if self.datasetname=="BerkeleyMHAD":
            mask_group=[[7,8,9,10,11,12,13],[14,15,16,17,18,19,20],\
                              [22,23,24,25,26,27,28],[29,30,31,32,33,34,35],[0,1,2,3,4,5,6]]

        if self.datasetname=="NTU_RGBD":
            mask_group=[[0,1,20,2,3],[8,9,10,11,23,24],[16,17,18,19],\
                        [12,13,14,15],[4,5,6,7,21,22]]

            mask_group_2=[[25,26,45,27,28],[33,34,35,36,48,49],[41,42,43,44],\
                        [37,38,39,40],[29,30,31,32,46,47]]

        if self.datasetname=="CMU_subset" or self.datasetname=="CMU_all" :
            mask_group=[[0,1,2,3,4,5,6],[7,8,9,10,11,12,13],[14,15,16,17,18,19,20],\
                        [21,22,23,24,25],[26,27,28,29,30]]


        noise_startflag=[]
        for i in range(batchsize):
            if mode=="past":
                noise_startflag.append(0)
            elif mode=="current":
                noise_startflag.append(0.5-mask_ratio/2)
            elif mode=="future":
                noise_startflag.append(1-mask_ratio)
            else:
                noise_startflag.append(random.random()*(1-mask_ratio))

        max_value_ntu = 5.0
        for batchi in range(batchsize):
            sequence_length.append(len(dataset[batch_flag[batchi]]))
            decoder_inputtemp=[]
            encoder_inputtemp_original=[]

            encoder_inputtemp_original_2=[]
            encoder_inputtemp_original_shift=[]
            masktemp_r=[]

            encoder_inputtemp_original_shift.append(np.zeros((self.input_dimension)))
            masktemp_r.append(0.0)
            for timei in range(self.sequence_maximum):
                if timei<len(dataset[batch_flag[batchi]]):


                    if self.datasetname == "NTU_RGBD":
                        current_batch = dataset[batch_flag[batchi]][timei]
                        current_batch[current_batch > max_value_ntu] = max_value_ntu
                        current_batch[current_batch < -max_value_ntu] = -max_value_ntu
                        current_batch=current_batch*1.0/max_value_ntu
                        random_choice=random.randint(0,10)
                        if random_choice>15 and np.max(current_batch[75:])>0:
                            encoder_inputtemp_original.append(current_batch[75:])
                            encoder_inputtemp_original_shift.append(current_batch[75:])
                            encoder_inputtemp_original_2.append(current_batch[75:])
                            current_batch = current_batch[75:]
                        else:
                            encoder_inputtemp_original.append(current_batch[0:75])
                            encoder_inputtemp_original_shift.append(current_batch[0:75])
                            encoder_inputtemp_original_2.append(current_batch[75:])
                            current_batch=current_batch[0:75]
                    else:
                        current_batch = dataset[batch_flag[batchi]][timei]
                        encoder_inputtemp_original.append(current_batch)
                        encoder_inputtemp_original_2.append(current_batch)
                        encoder_inputtemp_original_shift.append(current_batch )

                    if timei>=int(len(dataset[batch_flag[batchi]])*noise_startflag[batchi]) and\
                        timei<int(len(dataset[batch_flag[batchi]])*(noise_startflag[batchi]+mask_ratio))\
                            and timei>0:

                        for noise_i in range(len(corruption_num)):
                            current_batch=current_batch*noise_mask_get(self.input_dimension,mask_group[corruption_num[noise_i]])
                            #if self.datasetname=="NTU_RGBD":
                                #current_batch=current_batch*noise_mask_get(self.input_dimension,mask_group_2[corruption_num[noise_i]])
                                #current_batch = tf.clip_by_value(current_batch, -max_value_ntu, max_value_ntu)

                    decoder_inputtemp.append(current_batch)

                    if sequence_length[batchi]<self.sequence_maximum:
                        masktemp_r.append(1.0/sequence_length[batchi])
                    else:
                        masktemp_r.append(1.0 / self.sequence_maximum)


                else:
                    decoder_inputtemp.append(np.zeros((self.input_dimension)))
                    encoder_inputtemp_original.append(np.zeros((self.input_dimension)))
                    encoder_inputtemp_original_2.append(np.zeros((self.input_dimension)))
                    encoder_inputtemp_original_shift.append(np.zeros((self.input_dimension)))
                    masktemp_r.append(0.0)

            decoder_input.append(np.array(decoder_inputtemp))
            encoder_input_original.append(np.array(encoder_inputtemp_original))
            encoder_input_original_2.append(np.array(encoder_inputtemp_original_2))
            encoder_input_original_shift.append(np.array(encoder_inputtemp_original_shift))
            mask_r.append(np.array(masktemp_r))

        if self.datasetname == "NTU_RGBD":
            return np.array(decoder_input),np.array(sequence_length),mask_r,\
                   encoder_input_original,encoder_input_original_shift,encoder_input_original_2
        else:
            return np.array(decoder_input),np.array(sequence_length),mask_r,\
                   np.array(encoder_input_original),np.array(encoder_input_original_shift),encoder_input_original

    def loaddata(self,dataset_name,mode="supervised",randomflag=1):
        print ("loading data...")
        (dataset,label)=pickle.load(open(dataset_name,"rb"))

        if self.training_example_num>0:
            if self.training_example_num>len(dataset):
                print ("----------error: training_example_num too large---------------")
            else:
                self.traindata_ratio=self.training_example_num*1.0/len(dataset)

        for batchi in range(self.batch_size):
            tmpi=random.randint(0,len(dataset)-1)
            dataset.append(dataset[tmpi])
            label.append(label[tmpi])

        if mode=="unsupervised":
            return (dataset,label)

        if randomflag==1:
            for i in range(len(dataset)):
                temp_num=int(len(dataset)*random.random())

                datatemp=dataset[i]
                labeltemp=label[i]

                dataset[i]=dataset[temp_num]
                label[i]=label[temp_num]

                dataset[temp_num]=datatemp
                label[temp_num]=labeltemp

        label_processed=[]
        for i in range(len(label)):
            temp=np.zeros((self.num_catogory))
            temp[int(label[i])]=1
            label_processed.append(temp)

        if mode=="semisupervised":
            return (dataset,label_processed)

        data_train=dataset[0:int(self.traindata_ratio*len(dataset))]
        label_train=label_processed[0:int(self.traindata_ratio*len(dataset))]

        data_validation=dataset[int(self.traindata_ratio*len(dataset)):int(1*len(dataset))]
        label_validation=label_processed[int(self.traindata_ratio*len(dataset)):int(1*len(dataset))]

        data_test=dataset[int(self.traindata_ratio*len(dataset)):]
        label_test=label_processed[int(self.traindata_ratio*len(dataset)):]

        print ("loaddata end...")
        return (dataset,label,data_train,label_train,data_validation,label_validation,data_test,label_test)

    def loaddata_supervised(self,dataset_name,mode="supervised",randomflag=1):
        print ("loading data...")
        (dataset,label)=pickle.load(open(dataset_name,"rb"))

        for batchi in range(self.batch_size):
            tmpi=random.randint(0,len(dataset)-1)
            dataset.append(dataset[tmpi])
            label.append(label[tmpi])

        if randomflag==1:
            for i in range(len(dataset)):
                temp_num=int(len(dataset)*random.random())

                datatemp=dataset[i]
                labeltemp=label[i]

                dataset[i]=dataset[temp_num]
                label[i]=label[temp_num]

                dataset[temp_num]=datatemp
                label[temp_num]=labeltemp

        label_processed=[]
        for i in range(len(label)):
            temp=np.zeros((self.num_catogory))
            temp[int(label[i])]=1
            label_processed.append(temp)
        return (dataset,label_processed)



    def encoder(self,input_sequence,sequence_length,\
        dropout_outkeepratio=1,reuse=False):
        celltype=self.celltype

        hidden_units_fcn=self.representation_dimention

        dropout_outkeepratio=self.dropout_outkeepratio

        def leaky_relu(x, name="leaky_relu"):
            return utils.leaky_relu(x, alpha=0.2, name=name)

        if self.activation=="leaky_relu":
            activation=leaky_relu
        else:
            activation=self.activation

        with tf.variable_scope("encodernet") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=self.reg_scale))
            if reuse:
                scope.reuse_variables()
            hidden_units=self.hidden_units
            input_sequence_temp=input_sequence
            for i in range(len(hidden_units)):
                cell_fw=celltype(num_units=int(hidden_units[i]/2),activation=activation)
                cell_fw=tf.contrib.rnn.DropoutWrapper(cell_fw,output_keep_prob=dropout_outkeepratio,input_keep_prob=dropout_outkeepratio)
                cell_bw=celltype(num_units=int(hidden_units[i]/2),activation=activation)
                cell_bw=tf.contrib.rnn.DropoutWrapper(cell_bw,output_keep_prob=dropout_outkeepratio,input_keep_prob=dropout_outkeepratio)
                initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
                initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
                (outputs,states)=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,\
                            inputs=input_sequence_temp,sequence_length=sequence_length,\
                            initial_state_fw=initial_state_fw,initial_state_bw=initial_state_bw,\
                            time_major=False,scope=("encoder_bd_%i" % i))
                time_dim = 1
                batch_dim = 0
                outputs_bw=tf.reverse_sequence(outputs[1],sequence_length,seq_dim=time_dim,batch_dim=batch_dim)
                outputs_fw=outputs[0]
                input_sequence_temp=tf.concat([outputs_fw,outputs_bw],2)
                #print input_sequence_temp

            #inputs_reverse = tf.reverse_sequence(input=input_sequence, seq_lengths=sequence_length,seq_dim=1, batch_dim=0)

            if celltype==tf.contrib.rnn.LSTMCell:
                states_out=tf.concat([states[0][0],states[1][0]],1)
            else:
                states_out=tf.concat([states[0],states[1]],1)
            feature_fcn = utils.fcn_layer(states_out,[hidden_units[-1],hidden_units_fcn],\
                                          [hidden_units_fcn],activation=tf.nn.tanh)
            self.representation=feature_fcn
            return feature_fcn

    def decoder(self,input_sequence,sequence_length,\
        dropout_outkeepratio=1,reuse=False):
        celltype=self.celltype
        dropout_outkeepratio=self.dropout_outkeepratio

        def leaky_relu(x, name="leaky_relu"):
            return utils.leaky_relu(x, alpha=0.2, name=name)

        if self.activation=="leaky_relu":
            activation=leaky_relu
        else:
            activation=self.activation

        with tf.variable_scope("decodernet") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=self.reg_scale))
            if reuse:
                scope.reuse_variables()
            hidden_units=self.hidden_units
            cell=[]
            for i in range(len(hidden_units)):
                cell_temp=celltype(num_units=hidden_units[i],activation=activation)
                cell_temp=tf.contrib.rnn.DropoutWrapper(cell_temp,output_keep_prob=dropout_outkeepratio,input_keep_prob=dropout_outkeepratio)
                cell.append(cell_temp)
            cell.append(celltype(num_units=self.input_dimension,activation=tf.nn.tanh))    # add a output layer
            cell_net=tf.contrib.rnn.MultiRNNCell(cell,state_is_tuple=True)
            initial_state = cell_net.zero_state(self.batch_size, tf.float32)
            (outputs,states)=tf.nn.dynamic_rnn(cell_net,input_sequence,initial_state=initial_state,sequence_length=sequence_length,\
                                              time_major=False,scope="decoder")
            return outputs


    def discriminator(self,input_sequence,sequence_length,\
        dropout_outkeepratio=1,reuse=False):
        celltype=self.celltype

        dropout_outkeepratio=self.dropout_outkeepratio

        def leaky_relu(x, name="leaky_relu"):
                return utils.leaky_relu(x, alpha=0.2, name=name)

        if self.activation=="leaky_relu":
            activation=leaky_relu
        else:
            activation=self.activation



        with tf.variable_scope("discriminatornet") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=self.reg_scale))
            if reuse:
                scope.reuse_variables()
            hidden_units=[180,180,180]
            hidden_units = [360, 360, 360]

            input_sequence_temp=input_sequence

            for i in range(len(hidden_units)):
                cell_fw=celltype(num_units=int(hidden_units[i]/2),activation=activation)
                cell_fw=tf.contrib.rnn.DropoutWrapper(cell_fw,output_keep_prob=dropout_outkeepratio,input_keep_prob=dropout_outkeepratio)
                cell_bw=celltype(num_units=int(hidden_units[i]/2),activation=activation)
                cell_bw=tf.contrib.rnn.DropoutWrapper(cell_bw,output_keep_prob=dropout_outkeepratio,input_keep_prob=dropout_outkeepratio)
                initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
                initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
                (outputs,states)=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,\
                            inputs=input_sequence_temp,sequence_length=sequence_length,\
                            initial_state_fw=initial_state_fw,initial_state_bw=initial_state_bw,\
                            time_major=False,scope=("discrinator_bd_%i" % i))
                time_dim = 1
                batch_dim = 0
                outputs_bw=tf.reverse_sequence(outputs[1],sequence_length,seq_dim=time_dim,batch_dim=batch_dim)
                outputs_fw=outputs[0]
                input_sequence_temp=tf.concat([outputs_fw,outputs_bw],2)
                #print input_sequence_temp

            #inputs_reverse = tf.reverse_sequence(input=input_sequence, seq_lengths=sequence_length,seq_dim=1, batch_dim=0)
            if celltype==tf.contrib.rnn.LSTMCell:
                states_out=tf.concat([states[0][0],states[1][0]],1)
            else:
                states_out=tf.concat([states[0],states[1]],1)

            weights = tf.get_variable("fcn1_weights", [hidden_units[-1],hidden_units[-1]],initializer=tf.random_normal_initializer())
            biases = tf.get_variable("fcn1_biases", [hidden_units[-1]],initializer=tf.constant_initializer(0.1))
            fcn1 = activation(tf.matmul(states_out,weights)+biases)

            weights = tf.get_variable("fcn2_weights", [hidden_units[-1],1],initializer=tf.random_normal_initializer())
            biases = tf.get_variable("fcn2_biases", [1],initializer=tf.constant_initializer(0.1))
            y_pred = tf.matmul(fcn1,weights)+biases
            return y_pred,fcn1


    def optimizer(self,loss_val, var_list, optimizer_name="RMSProp", optimizer_param=0.9):
        learning_rate=self.learning_rate
        optimizer_name=self.optimizer_name
        if optimizer_name == "Adam":
            optimizer=tf.train.AdamOptimizer(learning_rate, beta1=optimizer_param)
        elif optimizer_name == "RMSProp":
            optimizer=tf.train.RMSPropOptimizer(learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        return optimizer.apply_gradients(grads)


    def netvlad(self, net, videos_per_batch, weight_decay, netvlad_initCenters):
        netvlad_alpha = self.netvlad_alpha
        # VLAD pooling
        netvlad_initCenters = int(netvlad_initCenters)
        # initialize the cluster centers randomly
        cluster_centers = np.random.normal(size=(
            netvlad_initCenters, net.get_shape().as_list()[-1]))

        with tf.variable_scope('NetVLAD'):
            # normalize features
            if self.l2_norm_flag==1:
                net_normed = tf.nn.l2_normalize(net, 3, name='FeatureNorm')
            else:
                net_normed=net

            vlad_centers = slim.model_variable(
                'centers',
                shape=cluster_centers.shape,
                initializer=tf.constant_initializer(cluster_centers),
                regularizer=slim.l2_regularizer(weight_decay))


            vlad_W=tf.expand_dims(tf.expand_dims(tf.transpose(vlad_centers)*2 * netvlad_alpha,axis=0),axis=0)
            vlad_B=tf.reduce_sum(tf.square(vlad_centers),axis=1)*(-netvlad_alpha)

            print ("vlad_w:",vlad_W)
            print ("vlad_B:",vlad_B)

            conv_output = tf.nn.conv2d(net_normed, vlad_W, [1, 1, 1, 1], 'VALID')
            dists = tf.nn.bias_add(conv_output, vlad_B)
            #normed_square=tf.reduce_sum(tf.square(net_normed),axis=3)
            #dists=tf.add(dists,-netvlad_alpha *normed_square)
            assgn = tf.nn.softmax(dists, dim=3)


            print ("net_normed", net_normed)
            print ("assgn:", assgn)

            vid_splits = tf.split(net_normed, videos_per_batch, 0)
            assgn_splits = tf.split(assgn, videos_per_batch, 0)

            # print "vid_splits:",vid_splits
            # print "assgn_splits:",assgn_splits
            # print "vlad_centers:",vlad_centers
            num_vlad_centers = vlad_centers.get_shape()[0]
            # print "num_vlad_centers:",num_vlad_centers
            vlad_centers_split = tf.split(vlad_centers, netvlad_initCenters, 0)
            # print "vlad_centers_split:",vlad_centers_split
            final_vlad = []
            #self.loss_smooth=tf.reduce_sum(tf.square(tf.subtract(assgn[0,:,:,1:],assgn[0,:,:,:-1])))
            for feats, assgn in zip(vid_splits, assgn_splits):
                vlad_vectors = []
                assgn_split_byCluster = tf.split(assgn, netvlad_initCenters, 3)
                for k in range(num_vlad_centers):
                    res = tf.reduce_sum(
                        tf.multiply(tf.subtract(
                            feats,
                            vlad_centers_split[k]), assgn_split_byCluster[k]),
                        [0, 1, 2])
                    vlad_vectors.append(res)
                vlad_vectors_frame = tf.stack(vlad_vectors, axis=0)
                final_vlad.append(vlad_vectors_frame)
            vlad_rep = tf.stack(final_vlad, axis=0, name='unnormed-vlad')

            with tf.name_scope('intranorm'):
                if self.l2_norm_flag==1:
                    intranormed = tf.nn.l2_normalize(vlad_rep, dim=2)
                else:
                    intranormed=vlad_rep

            with tf.name_scope('finalnorm'):
                if self.l2_norm_flag==1:
                    vlad_rep_output = tf.nn.l2_normalize(tf.reshape(
                        intranormed,
                        [intranormed.get_shape().as_list()[0], -1]),
                        dim=1)
                else:
                    vlad_rep_output=tf.reshape(
                        intranormed,
                        [intranormed.get_shape().as_list()[0], -1])

        print ("vlad_rep_output:", vlad_rep_output)
        return vlad_rep_output


