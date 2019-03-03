import os
import tensorflow as tf
import numpy as np
import helper
from glob import glob
import pickle as pkl
import scipy.misc
import time
import cv2
import matplotlib.pyplot as plt

do_preprocess = True
from_checkpoint = False

# Resize images to 128x128

data_dir = './Training Data' # Data
data_resized_dir = "./resized_data"# Resized data

if do_preprocess == True:
    os.mkdir(data_resized_dir)
    for each in os.listdir(data_dir):
        image = cv2.imread(os.path.join(data_dir, each))
        image = cv2.resize(image, (128, 128))
        cv2.imwrite(os.path.join(data_resized_dir, each), image)

def get_image(image_path, width, height, mode):

    image = Image.open(image_path)

    return np.array(image.convert(mode))

def get_batch(image_files, width, height, mode):
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch

show_n_images = 25
mnist_images = helper.get_batch(glob(os.path.join(data_resized_dir, '*.jpg'))[:show_n_images], 64, 64, 'RGB')
plt.imshow(helper.images_square_grid(mnist_images, 'RGB'))

from distutils.version import LooseVersion
import warnings
import tensorflow as tf

assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")
    learning_rate_G = tf.placeholder(tf.float32, name="learning_rate_G")
    learning_rate_D = tf.placeholder(tf.float32, name="learning_rate_D")
    
    return inputs_real, inputs_z, learning_rate_G, learning_rate_D

def generator(z, output_channel_dim, is_train=True):
    with tf.variable_scope("generator", reuse= not is_train):
        
        # First FC layer --> 8x8x1024
        fc1 = tf.layers.dense(z, 8*8*1024)
        fc1 = tf.reshape(fc1, (-1, 8, 8, 1024))
        fc1 = tf.nn.leaky_relu(fc1, alpha=alpha)

        # 8x8x1024 --> 16x16x512
        trans_conv1 = tf.layers.conv2d_transpose(inputs = fc1, filters = 512, kernel_size = [5,5], strides = [2,2], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="trans_conv1")
        
        batch_trans_conv1 = tf.layers.batch_normalization(inputs = trans_conv1, training=is_train, epsilon=1e-5, name="batch_trans_conv1")
        trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv1, alpha=alpha, name="trans_conv1_out")
        
        # 16x16x512 --> 32x32x256
        trans_conv2 = tf.layers.conv2d_transpose(inputs = trans_conv1_out, filters = 256, kernel_size = [5,5], strides = [2,2], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="trans_conv2")
        
        batch_trans_conv2 = tf.layers.batch_normalization(inputs = trans_conv2, training=is_train, epsilon=1e-5, name="batch_trans_conv2")
        trans_conv2_out = tf.nn.leaky_relu(batch_trans_conv2, alpha=alpha, name="trans_conv2_out")
        
        # 32x32x256 --> 64x64x128
        trans_conv3 = tf.layers.conv2d_transpose(inputs = trans_conv2_out, filters = 128, kernel_size = [5,5], strides = [2,2], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="trans_conv3")
        
        batch_trans_conv3 = tf.layers.batch_normalization(inputs = trans_conv3, training=is_train, epsilon=1e-5, name="batch_trans_conv3")
        trans_conv3_out = tf.nn.leaky_relu(batch_trans_conv3, alpha=alpha, name="trans_conv3_out")

        # 64x64x128 --> 128x128x64
        trans_conv4 = tf.layers.conv2d_transpose(inputs = trans_conv3_out, filters = 64, kernel_size = [5,5], strides = [2,2], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="trans_conv4")
        
        batch_trans_conv4 = tf.layers.batch_normalization(inputs = trans_conv4, training=is_train, epsilon=1e-5, name="batch_trans_conv4")
       
        trans_conv4_out = tf.nn.leaky_relu(batch_trans_conv4, alpha=alpha, name="trans_conv4_out")

        # 128x128x64 --> 128x128x3
        logits = tf.layers.conv2d_transpose(inputs = trans_conv4_out, filters = 3, kernel_size = [5,5], strides = [1,1], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="logits")
         
        out = tf.tanh(logits, name="out")
        
        return out

def discriminator(x, is_reuse=False, alpha = 0.2):

    with tf.variable_scope("discriminator", reuse = is_reuse): 
        
        # Input layer 128*128*3 --> 64x64x64
        conv1 = tf.layers.conv2d(inputs = x, filters = 64, kernel_size = [5,5], strides = [2,2], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv1')
        
        batch_norm1 = tf.layers.batch_normalization(conv1, training = True, epsilon = 1e-5, name = 'batch_norm1')

        conv1_out = tf.nn.leaky_relu(batch_norm1, alpha=alpha, name="conv1_out")
        
        # 64x64x64--> 32x32x128  
        conv2 = tf.layers.conv2d(inputs = conv1_out, filters = 128, kernel_size = [5, 5], strides = [2, 2], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
        
        batch_norm2 = tf.layers.batch_normalization(conv2, training = True, epsilon = 1e-5, name = 'batch_norm2')
        
        conv2_out = tf.nn.leaky_relu(batch_norm2, alpha=alpha, name="conv2_out")
        
        # 32x32x128 --> 16x16x256 
        conv3 = tf.layers.conv2d(inputs = conv2_out, filters = 256, kernel_size = [5, 5], strides = [2, 2], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name='conv3')
        
        batch_norm3 = tf.layers.batch_normalization(conv3, training = True, epsilon = 1e-5, name = 'batch_norm3')
        
        conv3_out = tf.nn.leaky_relu(batch_norm3, alpha=alpha, name="conv3_out")
        
        # 16x16x256 --> 16x16x512
        conv4 = tf.layers.conv2d(inputs = conv3_out, filters = 512, kernel_size = [5, 5], strides = [1, 1], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv4')
        
        batch_norm4 = tf.layers.batch_normalization(conv4, raining = True, epsilon = 1e-5, name = 'batch_norm4')
        
        conv4_out = tf.nn.leaky_relu(batch_norm4, alpha=alpha, name="conv4_out")        
        
        # 16x16x512 --> 8x8x1024 
        conv5 = tf.layers.conv2d(inputs = conv4_out, filters = 1024, kernel_size = [5, 5],
                                strides = [2, 2], padding = "SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                name='conv5')
        
        batch_norm5 = tf.layers.batch_normalization(conv5, training = True, epsilon = 1e-5, name = 'batch_norm5')
        
        conv5_out = tf.nn.leaky_relu(batch_norm5, alpha=alpha, name="conv5_out")

         
        # Flatten it
        flatten = tf.reshape(conv5_out, (-1, 8*8*1024))
        
        # Logits
        logits = tf.layers.dense(inputs = flatten,
                                units = 1,
                                activation = None)
        
        
        out = tf.sigmoid(logits)
        
        return out, logits

def model_loss(input_real, input_z, output_channel_dim, alpha):
    # Generator network here
    g_model = generator(input_z, output_channel_dim)   
    # g_model is the generator output
    
    # Discriminator network here
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model,is_reuse=True, alpha=alpha)
    
    # Calculate losses
    d_loss_real = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
                                                          labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                          labels=tf.zeros_like(d_model_fake)))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                     labels=tf.ones_like(d_model_fake)))
    
    return d_loss, g_loss

def model_optimizers(d_loss, g_loss, lr_D, lr_G, beta1):    
    # Get the trainable_variables, split into G and D parts
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # Generator update
    gen_updates = [op for op in update_ops if op.name.startswith('generator')]
    
    # Optimizers
    with tf.control_dependencies(gen_updates):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=lr_D, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=lr_G, beta1=beta1).minimize(g_loss, var_list=g_vars)
        
    return d_train_opt, g_train_opt

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode, image_path, save, show):
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    
    if save == True:
        images_grid.save(image_path, 'JPEG')
    
    if show == True:
        plt.imshow(images_grid, cmap=cmap)
        plt.show()

def train(epoch_count, batch_size, z_dim, learning_rate_D, learning_rate_G, beta1, get_batches, data_shape, data_image_mode, alpha):
    input_images, input_z, lr_G, lr_D = model_inputs(data_shape[1:], z_dim)
    d_loss, g_loss = model_loss(input_images, input_z, data_shape[3], alpha)
    d_opt, g_opt = model_optimizers(d_loss, g_loss, lr_D, lr_G, beta1)
    i = 0
    
    version = "firstTrain"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        num_epoch = 0
        if from_checkpoint == True:
            saver.restore(sess, "./models/model.ckpt")
            
            show_generator_output(sess, 1, input_z, data_shape[3], data_image_mode, image_path, True, False)
            
        else:
            for epoch_i in range(epoch_count):        
                num_epoch += 1

                if num_epoch % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model saved")
                for batch_images in get_batches(batch_size):
                    batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                    i += 1
                    _ = sess.run(d_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_D: learning_rate_D})
                    _ = sess.run(g_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_G: learning_rate_G})

                    if i % 10 == 0:
                        train_loss_d = d_loss.eval({input_z: batch_z, input_images: batch_images})
                        train_loss_g = g_loss.eval({input_z: batch_z})

                        image_name = str(i) + ".jpg"
                        image_path = "./images/" + image_name
                        show_generator_output(sess, 4, input_z, data_shape[3], data_image_mode, image_path, True, False) 

                    if i % 1500 == 0:

                        image_name = str(i) + ".jpg"
                        image_path = "./images/" + image_name
                        print("Epoch {}/{}...".format(epoch_i+1, epochs),
                              "Discriminator Loss: {:.4f}...".format(train_loss_d),
                              "Generator Loss: {:.4f}".format(train_loss_g))
                        show_generator_output(sess, 4, input_z, data_shape[3], data_image_mode, image_path, False, True)
                
    return losses, samples

# Size input image for discriminator
real_size = (128,128,3)

# Size of latent vector to generator
z_dim = 100
learning_rate_D =  .00005 
learning_rate_G = 2e-4 
batch_size = 64
epochs = 215
alpha = 0.2
beta1 = 0.5

dataset = helper.Dataset(glob(os.path.join(data_resized_dir, '*.jpg')))

with tf.Graph().as_default():
    losses, samples = train(epochs, batch_size, z_dim, learning_rate_D, learning_rate_G, beta1, dataset.get_batches,
          dataset.shape, dataset.image_mode, alpha)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
