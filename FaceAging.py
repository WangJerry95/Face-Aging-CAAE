# This code is an alternative implementation of the paper by
# Zhifei Zhang, Yang Song, and Hairong Qi. "Age Progression/Regression by Conditional Adversarial Autoencoder."
# IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
#
# Date:     Mar. 24th, 2017
#
# Please cite above paper if you use this code
#

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from ops import *


class FaceAging(object):
    def __init__(self,
                 session,  # TensorFlow session
                 size_image=128,  # size the input images
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 size_batch=100,  # mini-batch size for training and testing, must be square of an integer
                 num_input_channels=3,  # number of channels of input images
                 num_encoder_channels=64,  # number of channels of the first conv layer of encoder
                 num_z_channels=50,  # number of channels of the layer z (noise or code)
                 num_categories=10,  # number of categories (age segments) in the training dataset
                 num_gen_channels=1024,  # number of channels of the first deconv layer of generator
                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and z
                 is_training=True,  # flag for training or testing mode
                 save_dir='./save',  # path to save checkpoints, samples, and summary
                 dataset_name='UTKFace'  # name of the dataset in the folder ./data
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.dataset_name = dataset_name

        # ************************************* input to graph ********************************************************
        self.input_sketches = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_sketches'
        )
        self.input_photos = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_photos'
        )
        self.age = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_categories],
            name='age_labels'
        )
        self.gender = tf.placeholder(
            tf.float32,
            [self.size_batch, 2],
            name='gender_labels'
        )
        # self.z_prior = tf.placeholder(
        #     tf.float32,
        #     [self.size_batch, self.num_z_channels],
        #     name='z_prior'
        # )
        # ************************************* build the graph *******************************************************
        print '\n\tBuilding graph ...'

        # encoder: input image --> z
        with tf.name_scope('Encoder'):
            self.z_s, _ = self.encoder_s(
                image=self.input_sketches,
                name='E_s'
            )
            self.z_p, current_i  = self.encoder_s(
                image=self.input_photos,
                name='E_p'
            )
            self.z_cs = self.encoder_c(
                self.z_s,
                current_i
            )
            self.z_cp = self.encoder_c(
                self.z_p,
                current_i,
                reuse_variables=True
            )

        # generator: z + label --> generated image
        with tf.name_scope('Generator'):
            self.G_s = self.generator_cbn(
                z=self.z_cs,
                y=self.age,
                gender=self.gender,
                enable_tile_label=self.enable_tile_label,
                tile_ratio=self.tile_ratio
            )
            self.G_p = self.generator_cbn(
                z=self.z_cp,
                y=self.age,
                gender=self.gender,
                enable_tile_label=self.enable_tile_label,
                tile_ratio=self.tile_ratio,
                reuse_variables=True
            )

        # discriminator on z_cs and z_cp
        with tf.name_scope('Discriminater_z'):
            self.D_zcs, self.D_zcs_logits = self.discriminator_z(
                z=self.z_cs,
                is_training=self.is_training
            )
            self.D_zcp, self.D_zcp_logits = self.discriminator_z(
                z=self.z_cp,
                is_training=self.is_training,
                reuse_variables=True
            )

        # discriminator on Generated and input images
        with tf.name_scope('Discriminater_img'):
            self.D_Gp, self.D_Gp_logits = self.discriminator_img_projection(
                image=self.G_p,
                y=self.age,
                gender=self.gender,
                is_training=self.is_training
            )
            self.D_Gs, self.D_Gs_logits = self.discriminator_img_projection(
                image=self.G_s,
                y=self.age,
                gender=self.gender,
                is_training=self.is_training,
                reuse_variables=True
            )
            self.D_input, self.D_input_logits = self.discriminator_img_projection(
                image=self.input_photos,
                y=self.age,
                gender=self.gender,
                is_training=self.is_training,
                reuse_variables=True
            )

        # re-encoder on generated image
        with tf.name_scope('Re-encoder'):
            self.z_rs, current_i  = self.encoder_s(
                image=self.G_s,
                name='E_p',
                reuse_variables=True
            )
            self.z_rcs = self.encoder_c(
                self.z_rs,
                current_i,
                reuse_variables=True
            )
            self.z_rp, current_i  = self.encoder_s(
                image=self.G_p,
                name='E_p',
                reuse_variables=True
            )
            self.z_rcp = self.encoder_c(
                self.z_rp,
                current_i,
                reuse_variables=True
            )


        # ************************************* loss functions *******************************************************
        # reconstruction loss for latent vector and photo image
        self.Recon_latent_loss = (tf.nn.l2_loss(self.z_cp-self.z_rcp) + tf.nn.l2_loss(self.z_cs-self.z_rcs)) / self.size_batch  # L2 loss
        self.Recon_image_loss = tf.reduce_mean(tf.abs(self.input_photos - self.G_p))  # L1 loss

        # adversarial loss of discriminator on latent vector Dz
        self.D_zs_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_zcs_logits, labels=tf.ones_like(self.D_zcs_logits))
        )
        self.D_zp_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_zcp_logits, labels=tf.zeros_like(self.D_zcp_logits))
        )
        self.E_zs_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_zcs_logits, labels=tf.zeros_like(self.D_zcs_logits))
        )
        self.E_zp_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_zcp_logits, labels=tf.ones_like(self.D_zcp_logits))
        )

        # adversarial loss of discriminator on image D_img
        self.D_img_loss_input = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_input_logits, labels=tf.ones_like(self.D_input_logits))
        )
        self.D_img_loss_Gs = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_Gs_logits, labels=tf.zeros_like(self.D_Gs_logits))
        )
        self.D_img_loss_Gp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_Gp_logits, labels=tf.zeros_like(self.D_Gp_logits))
        )
        self.G_img_loss_Gs = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_Gs_logits, labels=tf.ones_like(self.D_Gs_logits))
        )
        self.G_img_loss_Gp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_Gp_logits, labels=tf.ones_like(self.D_Gp_logits))
        )

        # Kullback-Leibler loss w.r.t uniform gaussian distribution
        self.kl_loss = KL_loss(self.z_cs) + KL_loss(self.z_cp)

        # total variation to smooth the generated image
        # tv_y_size = self.size_image
        # tv_x_size = self.size_image
        # self.tv_loss = (
        #     (tf.nn.l2_loss(self.G[:, 1:, :, :] - self.G[:, :self.size_image - 1, :, :]) / tv_y_size) +
        #     (tf.nn.l2_loss(self.G[:, :, 1:, :] - self.G[:, :, :self.size_image - 1, :]) / tv_x_size)) / self.size_batch

        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()
        # variables of encoder
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        # variables of generator
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        # variables of discriminator on z
        self.D_z_variables = [var for var in trainable_variables if 'D_z_' in var.name]
        # variables of discriminator on image
        self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]

        # ************************************* collect the summary ***************************************
        # summary for latent vector
        self.z_cp_summary = tf.summary.histogram('z', self.z_cp)
        self.z_cs_summary = tf.summary.histogram('z', self.z_cs)
        # summary for recon losses
        self.Recon_latent_loss_summary = tf.summary.scalar('Recon_latent_loss', self.Recon_latent_loss)
        self.Recon_image_loss_summary = tf.summary.scalar('Recon_image_loss', self.Recon_image_loss)
        # summary for adversarial loss of D_z
        self.D_zs_loss_summary = tf.summary.scalar('D_zs_loss', self.D_zs_loss)
        self.D_zp_loss_summary = tf.summary.scalar('D_zp_loss', self.D_zp_loss)
        self.E_zs_loss_summary = tf.summary.scalar('E_zs_loss', self.E_zs_loss)
        self.E_zp_loss_summary = tf.summary.scalar('E_zp_loss', self.E_zp_loss)
        # summary for adversarial loss of D_img
        self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
        self.D_img_loss_Gs_summary = tf.summary.scalar('D_img_loss_Gs', self.D_img_loss_Gs)
        self.D_img_loss_Gp_summary = tf.summary.scalar('D_img_loss_Gp', self.D_img_loss_Gp)
        self.G_img_loss_Gs_summary = tf.summary.scalar('G_img_loss_Gs', self.G_img_loss_Gs)
        self.G_img_loss_Gp_summary = tf.summary.scalar('G_img_loss_Gp', self.G_img_loss_Gp)
        # summary for KL loss
        self.kl_loss_summary = tf.summary.scalar('kl_loss', self.kl_loss)


    def train(self,
              num_epochs=200,  # number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # use the saved checkpoint to initialize the network
              use_init_model=True,  # use the init model to initialize the network
              weights=(10, 10)  # the weights of adversarial loss and TV loss
              ):

        # *************************** load file names of images ******************************************************
        sketch_names = glob(os.path.join('./data', self.dataset_name, 'sketch', '*.jpg'))
        photo_names = glob(os.path.join('./data', self.dataset_name, 'photo', '*.jpg'))
        sketch_num = len(sketch_names)
        photo_num = len(photo_names)
        np.random.seed(seed=2017)
        if enable_shuffle:
            np.random.shuffle(sketch_names)
            np.random.shuffle(photo_names)

        # for saving the graph and variables
        self.saver = tf.train.Saver(max_to_keep=3)

        # *********************************** optimizer **************************************************************
        # over all, there are three loss functions, weights may differ from the paper because of different datasets
        self.loss_EG = self.G_img_loss_Gs + \
                       self.G_img_loss_Gp + \
                       self.E_zp_loss + \
                       self.E_zs_loss + \
                       weights[0] * self.Recon_image_loss + \
                       weights[1] * self.Recon_latent_loss

        self.loss_Dz = self.D_zs_loss + self.D_zp_loss
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_Gs + self.D_img_loss_Gp
        self.loss_KL = self.kl_loss

        # set learning rate decay
        with tf.variable_scope('global_scope', reuse=tf.AUTO_REUSE):
            self.EG_global_step = tf.get_variable(name='global_step',
                                                  dtype=tf.int32,
                                                  initializer=tf.constant(0),
                                                  trainable=False)
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.EG_global_step,
            decay_steps=sketch_num / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )

        # optimizer for encoder + generator
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            self.EG_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_EG,
                global_step=self.EG_global_step,
                var_list=self.E_variables + self.G_variables
            )
            # optimizer for discriminator on z
            self.D_z_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Dz,
                var_list=self.D_z_variables
            )
            # optimizer for discriminator on image
            self.D_img_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Di,
                var_list=self.D_img_variables
            )
            # optimizer for encoder
            self.E_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_KL,
                var_list=self.E_variables
            )

        # *********************************** tensorboard *************************************************************
        # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.z_cp_summary, self.z_cs_summary,
            self.Recon_latent_loss_summary, self.Recon_image_loss_summary,
            self.D_zs_loss_summary, self.D_zp_loss_summary,
            self.E_zs_loss_summary, self.E_zp_loss_summary,
            self.D_img_loss_input_summary, self.D_img_loss_Gs_summary, self.D_img_loss_Gp_summary,
            self.G_img_loss_Gs_summary, self.G_img_loss_Gp_summary,
            self.kl_loss_summary,
            self.EG_learning_rate_summary,
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)

        # ************* get some random samples as testing data to visualize the learning process *********************
        sample_sketch_files = sketch_names[0:self.size_batch]
        sample_photo_files = photo_names[0:self.size_batch]
        sketch_names[0:self.size_batch] = []
        photo_names[0:self.size_batch] = []
        sample_sketch = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_sketch_files]
        sample_photo = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_photo_files]
        if self.num_input_channels == 1:
            sample_sketch_images = np.array(sample_sketch).astype(np.float32)[:, :, :, None]
            sample_photo_images = np.array(sample_photo).astype(np.float32)[:, :, :, None]
        else:
            sample_sketch_images = np.array(sample_sketch).astype(np.float32)
            sample_photo_images = np.array(sample_photo).astype(np.float32)
        sample_label_age = np.zeros(
            shape=(len(photo_names), self.num_categories),
            dtype=np.float32
        )
        sample_label_gender = np.zeros(
            shape=(len(photo_names), 2),
            dtype=np.float32
        )
        for i, label in enumerate(photo_names):
            label = int(str(photo_names[i]).split('/')[-1].split('_')[0])
            if 0 <= label <= 5:
                label = 0
            elif 6 <= label <= 10:
                label = 1
            elif 11 <= label <= 15:
                label = 2
            elif 16 <= label <= 20:
                label = 3
            elif 21 <= label <= 30:
                label = 4
            elif 31 <= label <= 40:
                label = 5
            elif 41 <= label <= 50:
                label = 6
            elif 51 <= label <= 60:
                label = 7
            elif 61 <= label <= 70:
                label = 8
            else:
                label = 9
            sample_label_age[i, label] = 1 # one-hot labels
            gender = int(str(photo_names[i]).split('/')[-1].split('_')[1])
            sample_label_gender[i, gender] = 1

        # ******************************************* training *******************************************************
        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")
                # load init model
                if use_init_model:
                    if not os.path.exists('init_model/model-init.data-00000-of-00001'):
                        from init_model.zip_opt import join
                        try:
                            join('init_model/model_parts', 'init_model/model-init.data-00000-of-00001')
                        except:
                            raise Exception('Error joining files')
                    self.load_checkpoint(model_path='init_model')


        # epoch iteration
        num_batches = len(sketch_names) // self.size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(sketch_names)
                np.random.shuffle(photo_names)
            for ind_batch in range(num_batches):
                start_time = time.time()
                # read batch images and labels
                batch_sketch_files = sketch_names[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch_photo_files = photo_names[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                sketch_batch = [load_image(
                    image_path=batch_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for batch_file in batch_sketch_files]
                photo_batch = [load_image(
                    image_path=batch_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for batch_file in batch_photo_files]
                if self.num_input_channels == 1:
                    batch_photo_images = np.array(photo_batch).astype(np.float32)[:, :, :, None]
                    batch_sketch_images = np.array(sketch_batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_photo_images = np.array(photo_batch).astype(np.float32)
                    batch_sketch_images = np.array(sketch_batch).astype(np.float32)
                batch_label_age = np.zeros(
                    shape=(len(batch_photo_images), self.num_categories),
                    dtype=np.float
                )
                batch_label_gender = np.zeros(
                    shape=(len(batch_photo_images), 2),
                    dtype=np.float
                )
                for i, label in enumerate(batch_photo_images):
                    label = int(str(batch_photo_images[i]).split('/')[-1].split('_')[0])
                    if 0 <= label <= 5:
                        label = 0
                    elif 6 <= label <= 10:
                        label = 1
                    elif 11 <= label <= 15:
                        label = 2
                    elif 16 <= label <= 20:
                        label = 3
                    elif 21 <= label <= 30:
                        label = 4
                    elif 31 <= label <= 40:
                        label = 5
                    elif 41 <= label <= 50:
                        label = 6
                    elif 51 <= label <= 60:
                        label = 7
                    elif 61 <= label <= 70:
                        label = 8
                    else:
                        label = 9
                    batch_label_age[i, label] = 1
                    gender = int(str(batch_photo_images[i]).split('/')[-1].split('_')[1])
                    batch_label_gender[i, gender] = 1

                # prior distribution on the prior of z
                # batch_z_prior = np.random.uniform(
                #     self.image_value_range[0],
                #     self.image_value_range[-1],
                #     [self.size_batch, self.num_z_channels]
                # ).astype(np.float32)

                # update
                train_summary,_, _, _, _, Recon_img, Recon_latent, D_zs, D_zp, E_zs, E_zp, D_i_input,\
                D_i_Gs, D_i_Gp, G_i_Gs, G_i_Gp, kl = self.session.run(
                    fetches = [
                        self.summary,
                        self.EG_optimizer,
                        self.D_z_optimizer,
                        self.D_img_optimizer,
                        self.E_optimizer,
                        self.Recon_image_loss,
                        self.Recon_latent_loss,
                        self.D_zs_loss,
                        self.D_zp_loss,
                        self.E_zs_loss,
                        self.E_zp_loss,
                        self.D_img_loss_input,
                        self.D_img_loss_Gs,
                        self.D_img_loss_Gp,
                        self.G_img_loss_Gs,
                        self.G_img_loss_Gp,
                        self.kl_loss
                    ],
                    feed_dict={
                        self.input_photos: batch_photo_images,
                        self.input_sketches: batch_sketch_images,
                        self.age: batch_label_age,
                        self.gender: batch_label_gender,
                    }
                )

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tRecon_img=%.4f\tRecon_latent=%.4f" %
                    (epoch+1, num_epochs, ind_batch+1, num_batches, Recon_img, Recon_latent))
                print("\tD_zs=%.4f\tD_zp=%.4f\tE_zs=%.4f\tE_zp=%.4f" % (D_zs, D_zp, E_zs, E_zp))
                print("\tD_i_input=%.4f\tD_i_Gs=%.4f\tD_i_Gp=%.4f\tG_i_Gs=%.4f\tG_i_Gp=%.4f" % (D_i_input, D_i_Gs, D_i_Gp, G_i_Gs, G_i_Gp))
                print("\tkl_loss=%.4f" % kl)

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                self.writer.add_summary(train_summary, self.EG_global_step.eval())

            # save sample images for each epoch
            name = '{:02d}'.format(epoch+1)
            self.sample(sample_sketch_images, sample_photo_images, sample_label_age, sample_label_gender, name)
            self.test(sample_sketch_images, sample_photo_images, sample_label_gender, name)

            # save checkpoint for each 5 epoch
            if np.mod(epoch, 5) == 4:
                self.save_checkpoint()

        # save the trained model
        self.save_checkpoint()
        # close the summary writer
        self.writer.close()


    def encoder(self, image, reuse_variables=False):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    name=name
                )
            current = tf.nn.relu(current)

        # fully connection layer
        name = 'E_fc'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=self.num_z_channels,
            name=name
        )

        # output
        return tf.nn.tanh(current)

    def encoder_s(self, image, reuse_variables=False, name='E_s'):
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = image
        # conv layers with stride 2
        for i in range(num_layers-2):
            scope = name + '_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    name=scope,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)

        return current, i

    def encoder_c(self, current_map, current_i, reuse_variables=False, name='E_c'):
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = current_map
        for i in range(current_i+1, num_layers):
            scope = name+'_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    name=scope,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)

        # fully connection layer
        scope = name + '_fc'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=self.num_z_channels,
            name=scope,
            reuse=reuse_variables
        )

        # output
        return tf.nn.tanh(current)

    def generator(self, z, y, gender, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / self.num_categories)
        else:
            duplicate = 1
        z = concat_label(z, y, duplicate=duplicate)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / 2)
        else:
            duplicate = 1
        z = concat_label(z, gender, duplicate=duplicate)
        size_mini_map = int(self.size_image / 2 ** num_layers)
        # fc layer
        name = 'G_fc'
        current = fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
            name=name
        )
        # reshape to cube for deconv
        current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
        current = tf.nn.relu(current)
        # deconv layers with stride 2
        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            current = deconv2d(
                    input_map=current,
                    output_shape=[self.size_batch,
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),
                                  int(self.num_gen_channels / 2 ** (i + 1))],
                    size_kernel=self.size_kernel,
                    name=name
                )
            current = tf.nn.relu(current)
        name = 'G_deconv' + str(i+1)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          int(self.num_gen_channels / 2 ** (i + 2))],
            size_kernel=self.size_kernel,
            stride=1,
            name=name
        )
        current = tf.nn.relu(current)
        name = 'G_deconv' + str(i + 2)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          self.num_input_channels],
            size_kernel=self.size_kernel,
            stride=1,
            name=name
        )

        # output
        return tf.nn.tanh(current)

    def generator_cbn(self, z, y, gender, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        size_mini_map = int(self.size_image / 2 ** num_layers)
        l = concat_label(y, gender)
        # fc layer
        name = 'G_fc'
        current = fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
            name=name,
            reuse=reuse_variables
        )
        # reshape to cube for deconv
        current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
        current = tf.nn.relu(current)
        # deconv layers with stride 2
        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            current = deconv2d(
                    input_map=current,
                    output_shape=[self.size_batch,
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),
                                  int(self.num_gen_channels / 2 ** (i + 1))],
                    size_kernel=self.size_kernel,
                    biased=False,
                    name=name,
                    reuse=reuse_variables
                )
            current = condition_batch_norm(current, l, name='G_CBN'+str(i), reuse=reuse_variables)
            current = tf.nn.relu(current)
        name = 'G_deconv' + str(i+1)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          int(self.num_gen_channels / 2 ** (i + 2))],
            size_kernel=self.size_kernel,
            stride=1,
            name=name,
            reuse=reuse_variables
        )
        current = tf.nn.relu(current)
        name = 'G_deconv' + str(i + 2)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          self.num_input_channels],
            size_kernel=self.size_kernel,
            stride=1,
            name=name,
            reuse=reuse_variables
        )
        # output
        return tf.nn.tanh(current)

    def discriminator_z(self, z, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16), enable_bn=True):
        current = z
        # fully connection layer
        for i in range(len(num_hidden_layer_channels)):
            name = 'D_z_fc' + str(i)
            current = fc(
                    input_vector=current,
                    num_output_length=num_hidden_layer_channels[i],
                    name=name,
                    reuse=reuse_variables
                )
            if enable_bn:
                name = 'D_z_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
        # output layer
        name = 'D_z_fc' + str(i+1)
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name,
            reuse=reuse_variables
        )
        return tf.nn.sigmoid(current), current

    def discriminator_img(self, image, y, gender, is_training=True, reuse_variables=False, num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        num_layers = len(num_hidden_layer_channels)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_img_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=num_hidden_layer_channels[i],
                    size_kernel=self.size_kernel,
                    name=name,
                    reuse=reuse_variables
                )
            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            if i == 0:
                current = concat_label(current, y)
                current = concat_label(current, gender, int(self.num_categories / 2))
        # fully connection layer
        name = 'D_img_fc1'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name,
            reuse=reuse_variables
        )
        current = lrelu(current)
        name = 'D_img_fc2'
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name,
            reuse=reuse_variables

        )
        # output
        return tf.nn.sigmoid(current), current

    def discriminator_img_projection(self, image, y, gender, is_training=True, reuse_variables=False, num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        num_layers = len(num_hidden_layer_channels)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_img_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=num_hidden_layer_channels[i],
                    size_kernel=self.size_kernel,
                    name=name,
                    reuse=reuse_variables
                )
            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
        # fully connection layer
        name = 'D_img_fc1'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name,
            reuse=reuse_variables
        )
        current = lrelu(current)
        # TODO: Add projection layer here
        l = concat_label(y,gender)
        project = projection(current, l, reuse=reuse_variables)

        name = 'D_img_fc2'
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name,
            reuse=reuse_variables
        )
        # output
        current = current + project
        return tf.nn.sigmoid(current), current

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

    def load_checkpoint(self, model_path=None):
        if model_path is None:
            print("\n\tLoading pre-trained model ...")
            checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        else:
            print("\n\tLoading init model ...")
            checkpoint_dir = model_path
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            try:
                self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
                return True
            except:
                return False
        else:
            return False

    def sample(self, sketch_images, photo_images, labels, gender, name):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        G_p, G_s = self.session.run(
            [self.G_p, self.G_s],
            feed_dict={
                self.input_photos: photo_images,
                self.input_sketches: sketch_images,
                self.age: labels,
                self.gender: gender
            }
        )
        size_frame = int(np.sqrt(self.size_batch))
        save_batch_images(
            batch_images=G_p,
            save_path=os.path.join(sample_dir, name+"_photo.png"),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )
        save_batch_images(
            batch_images=G_s,
            save_path=os.path.join(sample_dir, name+"_sketch.png"),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )

    def test(self, sketch_images, photo_images, gender, name):
        test_dir = os.path.join(self.save_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        sketch_images = sketch_images[:int(np.sqrt(self.size_batch)), :, :, :]
        photo_images = photo_images[:int(np.sqrt(self.size_batch)), :, :, :]
        gender = gender[:int(np.sqrt(self.size_batch)), :]
        size_sample = sketch_images.shape[0]
        labels = np.arange(size_sample)
        labels = np.repeat(labels, size_sample)
        query_labels = np.zeros(
            shape=(size_sample ** 2, size_sample),
            dtype=np.float32
        )
        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = 1
        query_sketch_images = np.tile(sketch_images, [self.num_categories, 1, 1, 1])
        query_photo_images = np.tile(photo_images, [self.num_categories, 1, 1, 1])
        query_gender = np.tile(gender, [self.num_categories, 1])
        G_p, G_s = self.session.run(
            [self.G_p, self.G_s],
            feed_dict={
                self.input_photos: query_photo_images,
                self.input_sketches: query_sketch_images,
                self.age: query_labels,
                self.gender: query_gender
            }
        )
        save_batch_images(
            batch_images=query_photo_images,
            save_path=os.path.join(test_dir, 'input_photo.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=query_sketch_images,
            save_path=os.path.join(test_dir, 'input_sketch.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=G_p,
            save_path=os.path.join(test_dir, name+"_photo.png"),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=G_s,
            save_path=os.path.join(test_dir, name+"_sketch.png"),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )

    def custom_test(self, testing_samples_dir):
        if not self.load_checkpoint():
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        num_samples = int(np.sqrt(self.size_batch))
        file_names = glob(testing_samples_dir)
        if len(file_names) < num_samples:
            print 'The number of testing images is must larger than %d' % num_samples
            exit(0)
        sample_files = file_names[0:num_samples]
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            images = np.array(sample).astype(np.float32)
        gender_male = np.zeros(
            shape=(num_samples, 2),
            dtype=np.float32
        )
        gender_female = np.zeros(
            shape=(num_samples, 2),
            dtype=np.float32
        )
        for i in range(gender_male.shape[0]):
            gender_male[i, 0] = 1
            gender_female[i, 1] = 1

        self.test(images, gender_male, 'test_as_male.png')
        self.test(images, gender_female, 'test_as_female.png')

        print '\n\tDone! Results are saved as %s\n' % os.path.join(self.save_dir, 'test', 'test_as_xxx.png')


