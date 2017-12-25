from cnn_settings_v1 import *
from single_livingroom_mosaicv1 import *
##from GP_crop_v3 import *
import logging
import sys
import numpy as np
import random
sys.path.append('/home/kai/miniconda3/lib/python3.5/site-packages/')

if __name__ == "__main__":
    test_name = 'test_r04'
    logging.basicConfig(filename='train_for_modelact_alldata'+str(test_name)+'.log',level=logging.DEBUG)
    load_path = "/home/kai/04_headcount_in_house/01_single_in_house/"
    # Create a coordinator
    config = Config()
    
    # load data to memory
    filename = 'histogram_all_full' + '.npz'
    # filename = 'histogram_all_soilweather' + '.npz'
    content = np.load(config.load_path + filename)
    image_all = content['output_image']
    yield_all = content['output_yield']
    index_all = content['output_index']

     # delete broken image
    list_delete=[]
    for i in range(image_all.shape[0]):
        if np.sum(image_all[i,:,:])<=4:
            list_delete.append(i)
    image_all=np.delete(image_all,list_delete,0)
    yield_all=np.delete(yield_all,list_delete,0)
    index_all = np.delete(index_all, list_delete, 0)

#######################################################################################################
################## no train/validation split in data read in, all raw input are single in house
#######################################################################################################
####    double_sample_index  = random.sample(range(1,image_all.shape[0]), int(0.5*image_all.shape[0]))
####
####    for i in range(int(0.5*image_all.shape[0])):
####        
####        n = 2*i
####        idx_single = single_sample_index[2*i]
####        idx_double = single_sample_index[i]
####        
####        image_all[idx_single,:,:] = act_addon(image_all[idx_single,:,:],image_all[idx_double,:,:])
####        yield_all[idx_single,:,:] = 2
#######################################################################################################
    
    # split into train and validate
    index_train = np.nonzero(index_all < int(0.6*image_all.shape[0]))[0]
    index_validate = np.nonzero(index_all >= int(0.6*image_all.shape[0]))[0]
    print ('train size',index_train.shape[0])
    print ('validate size',index_validate.shape[0])

    image_validate=image_all[index_validate]
    yield_validate=yield_all[index_validate]

    model= NeuralModel(config,'net')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
    # Launch the graph.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    summary_train_loss = []
    summary_eval_loss = []
    summary_RMSE = []
    summary_ME = []

    train_loss=0
    val_loss=0
    val_prediction = 0
    val_deviation = np.zeros([config.B])
    # #########################
    # block when test
    # add saver
    saver=tf.train.Saver()
    # Restore variables from disk.
    try:
        saver.restore(sess, config.save_path+str(test_name)+"CNN_model.ckpt")
    # Restore log results
        npzfile = np.load(config.save_path + str(test_name)+'result.npz')
        summary_train_loss = npzfile['summary_train_loss'].tolist()
        summary_eval_loss = npzfile['summary_eval_loss'].tolist()
        summary_RMSE = npzfile['summary_RMSE'].tolist()
        summary_ME = npzfile['summary_ME'].tolist()
        print("Model restored.")
    except:
        print ('No history model found')
    # #########################
    

    RMSE_min = 100
    try:
        for i in range(config.train_step):
            if i==3500:
                config.lr/=10
                # saver.restore(sess, config.save_path+str(test_name)+"CNN_model.ckpt")
                # # Restore log results
                # npzfile = np.load(config.save_path + str(test_name)+'result.npz')
                # summary_train_loss = npzfile['summary_train_loss'].tolist()
                # summary_eval_loss = npzfile['summary_eval_loss'].tolist()
                # summary_RMSE = npzfile['summary_RMSE'].tolist()
                # summary_ME = npzfile['summary_ME'].tolist()
                # print("Model restored.")
            if i==20000:
                config.lr/=10
                # saver.restore(sess, config.save_path+str(test_name)+"CNN_model.ckpt")
                # # Restore log results
                # npzfile = np.load(config.save_path + str(test_name)+'result.npz')
                # summary_train_loss = npzfile['summary_train_loss'].tolist()
                # summary_eval_loss = npzfile['summary_eval_loss'].tolist()
                # summary_RMSE = npzfile['summary_RMSE'].tolist()
                # summary_ME = npzfile['summary_ME'].tolist()
                # print("Model restored.")
            # if i==12000:
            #     config.lr/=10
                # saver.restore(sess, config.save_path+str(test_name)+"CNN_model.ckpt")
                # # Restore log results
                # npzfile = np.load(config.save_path + str(test_name)+'result.npz')
                # summary_train_loss = npzfile['summary_train_loss'].tolist()
                # summary_eval_loss = npzfile['summary_eval_loss'].tolist()
                # summary_RMSE = npzfile['summary_RMSE'].tolist()
                # summary_ME = npzfile['summary_ME'].tolist()
                # print("Model restored.")

            # No augmentation
            # index_train_batch = np.random.choice(index_train,size=config.B)
            # image_train_batch = image_all[index_train_batch,:,0:config.H,:]
            # yield_train_batch = yield_all[index_train_batch]
            # year_train_batch = year_all[index_train_batch,np.newaxis]

            # try data augmentation while training
            #######################################################################################################
            ################## try data augmentaion while training, by randomly generating double in house cases
            #######################################################################################################
            index_train_batch_1 = np.random.choice(index_train,size=config.B)
            index_train_batch_2 = np.random.choice(index_train,size=int(config.B/2))

            image_train_batch   = image_train_batch
            image_train_batch[0:2*int(config.B/2):2,:,:] = act_addon(index_train_batch_1[0:2*int(config.B/2):2,:,:]\
                                                           ,index_train_batch_2[0:int(config.B/2),:,:])
            yield_train_batch = yield_all[index_train_batch_1]
            yield_train_batch[0:2*int(config.B/2):2] *= 2
            
            #######################################################################################################
            ################## similar data augmentaion for validation 
            #######################################################################################################
            index_validate_batch_1 = np.random.choice(index_validate,size=config.B)
            index_validate_batch_2 = np.random.choice(index_validate,size=int(config.B/2))

            index_validate_batch   = index_validate_batch_1
            index_validate_batch[0:2*int(config.B/2):2,:,:] = act_addon(index_validate_batch_1[0:2*int(config.B/2):2,:,:]\
                                                           ,index_validate_batch_2[0:int(config.B/2),:,:])
            yield_validate_batch = yield_all[index_validate_batch_1]
            yield_validate_batch[0:2*int(config.B/2):2] *= 2

            index_validate_batch = np.random.choice(index_validate, size=config.B)

            _, train_loss = sess.run([model.train_op, model.loss_err], feed_dict={
                model.x:image_train_batch,
                model.y:yield_train_batch,
                model.lr:config.lr,
                model.keep_prob: config.drop_out
                })

            if i%200 == 0:
                val_loss,fc6,W,B = sess.run([model.loss_err,model.fc6,model.dense_W,model.dense_B], feed_dict={
                    model.x: image_all[index_validate_batch, :, 0:config.H],
                    model.y: yield_all[index_validate_batch],
                    model.keep_prob: 1
                })

                print ('predict year'+str(test_name)+'step'+str(i),train_loss,val_loss,config.lr)
                logging.info('predict year %d step %d %f %f %f',test_name,i,train_loss,val_loss,config.lr)
            if i%200 == 0:
                # do validation
                pred = []
                real = []
                for j in range(int(image_validate.shape[0] / config.B)):
                    real_temp = yield_validate[j * config.B:(j + 1) * config.B]
                    pred_temp= sess.run(model.logits, feed_dict={
                        model.x: image_validate[j * config.B:(j + 1) * config.B,:,0:config.H],
                        model.y: yield_validate[j * config.B:(j + 1) * config.B],
                        model.keep_prob: 1
                        })
                    pred.append(pred_temp)
                    real.append(real_temp)
                pred=np.concatenate(pred)
                real=np.concatenate(real)
                RMSE=np.sqrt(np.mean((pred-real)**2))
                ME=np.mean(pred-real)

                if RMSE<RMSE_min:
                    RMSE_min=RMSE
                    # # save
                    # save_path = saver.save(sess, config.save_path + str(test_name)+'CNN_model.ckpt')
                    # print('save in file: %s' % save_path)
                    # np.savez(config.save_path+str(test_name)+'result.npz',
                    #     summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
                    #     summary_RMSE=summary_RMSE,summary_ME=summary_RMSE)

                print ('Validation set','RMSE',RMSE,'ME',ME,'RMSE_min',RMSE_min)
                logging.info('Validation set RMSE %f ME %f RMSE_min %f',RMSE,ME,RMSE_min)
            
                summary_train_loss.append(train_loss)
                summary_eval_loss.append(val_loss)
                summary_RMSE.append(RMSE)
                summary_ME.append(ME)



    except KeyboardInterrupt:
        print ('stopped')

    finally:

        # save
        save_path = saver.save(sess, config.save_path + str(test_name)+'CNN_model.ckpt')
        print('save in file: %s' % save_path)
        logging.info('save in file: %s' % save_path)

        # save result
        pred_out = []
        real_out = []
        feature_out = []
        index_out = []

        for i in range(int(image_all.shape[0] / config.B)):
            feature,pred = sess.run(
                [model.fc6,model.logits], feed_dict={
                model.x: image_all[i * config.B:(i + 1) * config.B,:,0:config.H],
                model.y: yield_all[i * config.B:(i + 1) * config.B],
                model.keep_prob:1
            })
            real = yield_all[i * config.B:(i + 1) * config.B]

            pred_out.append(pred)
            real_out.append(real)
            feature_out.append(feature)
            index_out.append(index_all[i * config.B:(i + 1) * config.B])
            # print i
        weight_out, b_out = sess.run(
            [model.dense_W, model.dense_B], feed_dict={
                model.x: image_all[0 * config.B:(0 + 1) * config.B, :, 0:config.H],
                model.y: yield_all[0 * config.B:(0 + 1) * config.B],
                model.keep_prob: 1
            })
        pred_out=np.concatenate(pred_out)
        real_out=np.concatenate(real_out)
        feature_out=np.concatenate(feature_out)
        index_out=np.concatenate(index_out)
        
        path = config.save_path + str(test_name)+'result_prediction.npz'
        np.savez(path,
            pred_out=pred_out,real_out=real_out,feature_out=feature_out,
            weight_out=weight_out,b_out=b_out,index_out=index_out)

        # RMSE_GP,ME_GP,Average_GP=GaussianProcess(test_name,path)
        # print 'RMSE_GP',RMSE_GP
        # print 'ME_GP',ME_GP
        # print 'Average_GP',Average_GP

        np.savez(config.save_path+str(test_name)+'result.npz',
                        summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
                        summary_RMSE=summary_RMSE,summary_ME=summary_ME)
        # plot results
        npzfile = np.load(config.save_path+str(test_name)+'result.npz')
        summary_train_loss=npzfile['summary_train_loss']
        summary_eval_loss=npzfile['summary_eval_loss']
        summary_RMSE = npzfile['summary_RMSE']
        summary_ME = npzfile['summary_ME']

        # Plot the points using matplotlib
        plt.plot(range(len(summary_train_loss)), summary_train_loss)
        plt.plot(range(len(summary_eval_loss)), summary_eval_loss)
        plt.xlabel('Training steps')
        plt.ylabel('L2 loss')
        plt.title('Loss curve')
        plt.legend(['Train', 'Validate'])
        plt.show()

        plt.plot(range(len(summary_RMSE)), summary_RMSE)
        # plt.plot(range(len(summary_ME)), summary_ME)
        plt.xlabel('Training steps')
        plt.ylabel('Error')
        plt.title('RMSE')
        # plt.legend(['RMSE', 'ME'])
        plt.show()

        # plt.plot(range(len(summary_RMSE)), summary_RMSE)
        plt.plot(range(len(summary_ME)), summary_ME)
        plt.xlabel('Training steps')
        plt.ylabel('Error')
        plt.title('ME')
        # plt.legend(['RMSE', 'ME'])
        plt.show()
