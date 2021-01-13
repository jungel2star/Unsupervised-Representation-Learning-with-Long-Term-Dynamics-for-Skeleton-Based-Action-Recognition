import os
import scipy.misc
import numpy as np
from inpainting_semi_model import GRUGAN
import tensorflow as tf


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:

        dataname='CMU_all'      #BerkeleyMHAD
        logs_dir=dataname+'non/'
        saves_dir_unsuper=dataname+'inpainting_super_GRU_fcn0_800_25/'
        saves_dir_semi="non_non"

        dataset_train="Non"
        dataset_test="Non"
        dataset_unsuper="Non"
        dataset_all="Non"

        if dataname=="HDM05":
            dimention=93
            num_catogory=65
            sequence_maximum=76

            #dataset_train=dataname+("_all_%d_train_1.pkl"%(i+1))
            #dataset_test=dataname+("_all_%d_test_1.pkl"%(i+1))
            dataset_unsuper=dataname+("_all1.pkl")
            dataset_all=dataname+("_all1.pkl")
            dataset_test=dataset_train= "Non"
            #  #  _all_original_10    #_all10
            #i=i+3
            #dataset_test=dataset_train="Non"
            #dataset_name=dataname+"_all1.pkl"
           #dataset_unsupervised=dataname+"_allsubsequence_5.pkl"

        elif dataname == "CMU_subset":
            dimention = 93
            num_catogory = 8
            sequence_maximum = 27
            dataset_unsuper = dataname + ("1_25.pkl")
            dataset_all = dataname + ("1_25.pkl")  # CMU_ALLsubsequence_10
            #dataset_unsuper="CMU_ALLsubsequence_10_25.pkl"     #CMU_all1
            dataset_test = dataset_train = "Non"


        elif dataname == "CMU_all":
            dimention = 93
            num_catogory = 45
            sequence_maximum = 27
            dataset_unsuper = dataname + ("1_25.pkl")
            dataset_all = dataname + ("1_25.pkl")  # CMU_ALLsubsequence_10
            #dataset_unsuper="CMU_ALLsubsequence_10_25.pkl"     #CMU_all1
            dataset_test = dataset_train = "Non"

        else:
            dimention=108
            num_catogory=11
            sequence_maximum=150


        if not os.path.exists(saves_dir_semi):
            os.mkdir(saves_dir_semi)

        if not os.path.exists(saves_dir_unsuper):
            os.mkdir(saves_dir_unsuper)

        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)

        begin_supervised=0
        supervised_flag=1

        softmax_flag=0
        batch_size=64
        fcn_num=0
        mask_num=3
        traindata_ratio = 0.8

        model=GRUGAN(sess,dimention,logs_dir=logs_dir,save_dir_gan=saves_dir_unsuper,save_dir_semi=saves_dir_semi,\
                     weight_super=1,begin_supervised=begin_supervised,gan_ratio=0,softmax_flag=softmax_flag,
                    sequence_maximum=sequence_maximum,traindata_ratio=0.9,representation_ratio_super=1,
                     mask_ratio=1.0,mask_num=mask_num,mask_mode="random",supervised_flag=supervised_flag,
                     num_catogory=num_catogory,datasetname=dataname,batch_size=batch_size,optimizer_name="Adam",
                     celltype=tf.contrib.rnn.GRUCell,hiddenunits_num=800,fcn_num=fcn_num,fcn_hiddenunit_num=300)

        model.build_model()        # RMSProp   Adam
        accuracy_total=[]
        begin_i = 0
        for i in range(0, 33, 1):


            training_example_num = 200 * int(i / 5)+100

            if i >= 25:
                spliti = (i - 25) % 4
                dataset_train = dataname + ("_%d_train_25.pkl" % (spliti + 1))
                dataset_test = dataname + ("_%d_test_25.pkl" % (spliti + 1))

                training_example_num = 0

            if dataname == "HDM05":
                training_example_num = 0

            accuracy_total.append(model.train_model(dataset_name=dataset_all, \
                                                    dataset_unsupervised=dataset_unsuper, dataset_train=dataset_train,
                                                    dataset_test=dataset_test, max_epoch=50,
                                                    training_example_num=training_example_num))  #
            print ("training_num:%d, accuracy: %4f" % (training_example_num, accuracy_total[i - begin_i]))

            if begin_supervised == 0:
                if softmax_flag == 0:
                    f = open('results/' + saves_dir_unsuper[:-1] + "_pretrain.txt", 'a')
                else:
                    f = open('results/' + saves_dir_unsuper[:-1] + ".txt", 'a')
                if i == 0:
                    if dataname == "HDM05":
                        f.write("###########  mask_num:%d  softmax_flag: %d   train_ratio: %4f #######" % (
                        mask_num, softmax_flag, traindata_ratio))
                    else:
                        f.write("###########  mask_num:%d  softmax_flag: %d#######" % (mask_num, softmax_flag))
                    f.write("\r")
                f.write("training_num:%d,accuracy:%4f" % (training_example_num, accuracy_total[i - begin_i]))
                f.write("\r")
                f.close()

        accuracy_average = np.mean(np.array(accuracy_total))  # _all_subsequence
        print ("accuracy_total:",accuracy_total)
        print ("accuracy average: %4f"%accuracy_average)
        print ("dataset_train:",dataset_train)
        print ("dataset_unsuper:",dataset_unsuper)
        print ("saves_dir_unsuper:",saves_dir_unsuper)


        # model.test_model(dataset_name=dataname+"_all10.pkl",num_catogory=num_catogory)

if __name__ == "__main__":
    tf.app.run()