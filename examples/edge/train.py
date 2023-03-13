#!/usr/bin/env python3

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy

# from sklearn.decomposition import PCA
from utils import plt_result, sk_f1_score, sk_acc_score, calc_pct
from data_loader import *
from models import *


def gain(train_data,
         test_data,
         tag_pos,
         all_pos,
         output,
         resume=None,
         batch_size=128,
         epochs=3,
         latent=500,
         D_lr=0.00003,
         G_lr=0.0006,
         D_smooth=0.1,
         G_smooth=0.0,
         save_interval=10,
         early_stop=20):
    # ===
    # Loading Data
    # ===
    # Load training data
    X_train, y_train, X_test, y_test, label_dim = data_processor(
        train_data, test_data, tag_pos, all_pos)

    # Parameter
    latent_dim = latent
    no, dim = X_train.shape

    # Data preprocessing
    norm_data_batch = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(no).batch(batch_size)

    # ===
    # Init metric and plot object
    # ===
    g_acc_list = []
    d_real_acc_list = []
    d_fake_acc_list = []
    g_loss_list = []
    d_real_loss_list = []
    d_fake_loss_list = []
    data_g_acc_list = []
    data_t_acc_list = []
    data_g_f1_list = []
    data_t_f1_list = []
    data_g_err_list = []
    data_t_err_list = []

    early_stop_err = 1
    early_stop_cnt = 0

    # ===
    # Define Model
    # ===
    D_optimizer = Adam(learning_rate=D_lr, beta_1=0.5)
    G_optimizer = Adam(learning_rate=G_lr, beta_1=0.5)

    logging.info('resume:{}'.format(resume))
    try:
        discriminator = tf.keras.models.load_model("%s_D" % resume)
        generator = tf.keras.models.load_model("%s_G" % resume)

        logging.info("Load resume success!")
    except Exception as err:
        generator = Generator(label_dim, dim)
        discriminator = Discriminator(dim, label_dim)
        # discriminator = Discriminator(int(dim))
        # generator = Generator(int(dim))
        logging.info("Load resume fails [%s]", err)
    gan = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=D_optimizer,
        g_optimizer=G_optimizer,
        d_loss_fn=BinaryCrossentropy(label_smoothing=D_smooth),
        g_loss_fn=BinaryCrossentropy(label_smoothing=G_smooth))

    for epoch in range(1, epochs + 1):
        for step, (x_mb, y_mb) in enumerate(norm_data_batch):
            g_acc, d_real_acc, d_fake_acc, g_loss, d_real_loss, d_fake_loss = gan.train_on_batch(
                x_mb, y_mb)

            # Record metric
            g_acc_list.append(g_acc)
            d_real_acc_list.append(d_real_acc)
            d_fake_acc_list.append(d_fake_acc)
            g_loss_list.append(g_loss)
            d_real_loss_list.append(d_real_loss)
            d_fake_loss_list.append(d_fake_loss)

        # Train accuracy
        trained_g = gan.generator

        x_pred = trained_g.predict(y_mb)
        x_pred = np.rint(x_pred)

        data_g_acc = sk_acc_score(x_mb, x_pred)
        data_g_acc_list.append(data_g_acc)
        data_g_f1 = sk_f1_score(x_mb, x_pred)
        data_g_f1_list.append(data_g_f1)
        data_g_err = np.mean(x_mb != x_pred)
        data_g_err_list.append(data_g_err)

        if test_data != '':
            # Sample data
            idx = np.random.randint(X_test.shape[0], size=batch_size)
            x_mb = X_test[idx, :]
            y_mb = y_test[idx, :]

            # test data
            x_pred = trained_g.predict(y_mb)
            x_pred = np.rint(x_pred)
            data_t_acc_list.append(sk_acc_score(x_mb, x_pred))
            data_t_f1_list.append(sk_f1_score(x_mb, x_pred))
            data_t_err_list.append(np.mean(x_mb != x_pred))

        # Verbose
        print('Epoch:{:3d}'.format(epoch))
        print('\tG_loss:{:.5g}\tG_acc:{:.5g}'.format(g_loss, g_acc))
        print('\tD_loss:{:.5g}\tD_acc:{:.5g}'.format(
            np.mean([d_real_loss, d_fake_loss]),
            np.mean([d_real_acc, d_fake_acc])))
        print('\tACC   :{:.5g}\tF1   :{:.5g}\tError:{:.5g}'.format(
            data_g_acc, data_g_f1, data_g_err))

        # Save interval
        if (epoch % save_interval == 0) or (epoch == epochs):
            # Save model
            trained_g = gan.generator
            trained_d = gan.discriminator
            trained_g.save('{}_G'.format(output))
            trained_d.save('{}_D'.format(output))

        # Early Stopping
        '''if data_g_err < early_stop_err:
            early_stop_cnt = 0
            early_stop_err = data_g_err
            early_stop_generator = gan.generator
            early_stop_epo = epoch
        else:
            early_stop_cnt += 1

        if early_stop_cnt >= early_stop:
            early_stop_generator.save(
                '{}/model_G_{}_best.h5'.format(save_model, early_stop_epo))
            break'''

    # ===
    # Save Results
    # ===

    metrics = dict()

    # Model Loss
    d_loss_list = [(g + h) / 2 for g, h in zip(d_fake_loss_list, d_real_loss_list)]
    metrics['D_loss'] = sum(d_loss_list) / len(d_loss_list)
    metrics['G_loss'] = sum(g_loss_list) / len(g_loss_list)

    # Model Acc
    d_acc_list = [(g + h) / 2 for g, h in zip(d_fake_acc_list, d_real_acc_list)]
    metrics['D_acc'] = sum(d_acc_list) / len(d_acc_list)
    metrics['G_acc'] = sum(g_acc_list) / len(g_acc_list)

    # Impute Acc (Training)
    metrics['Data_acc'] = sum(data_g_acc_list) / len(data_g_acc_list)

    # Impute F1 (Training)
    metrics['Data_f1'] = sum(data_g_f1_list) / len(data_g_f1_list)

    return metrics


'''if __name__ == '__main__':
    
    train_data = r'data/chr22_TWB_1003_train.vcf.gz'
    test_data = r'data/chr22_TWB_1003_test.vcf.gz'
    tag_pos = r'data/chr22_TWB_1003_tag.position'
    all_pos = r'data/chr22_TWB_1003.position'
    save_model = r'./model/temp'

    metrics = gain(train_data,
              test_data,
              tag_pos,
              all_pos,
              batch_size=128,
              epochs=3,
              latent=500,
              D_lr=0.0001,
              G_lr=0.0001,
              D_smooth=0.1,
              G_smooth=0.0,
              save_model=save_model,
              save_interval=10,
              early_stop=20)
    
    print("Finish")
    print(metrics)'''
