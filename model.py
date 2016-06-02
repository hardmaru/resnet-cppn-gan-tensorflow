import os
import time
import numpy as np
import tensorflow as tf
from ops import *

'''
cppgan-vae

compositional pattern-producing generative adversarial network combined with variational autoencoder

UPDATE: modified to get rid of likelihood function, and use pure GAN to draw to correct class.

I learned a lot from studying the below pages:

https://github.com/carpedm20/DCGAN-tensorflow
https://jmetzen.github.io/2015-11-27/vae.html

it wouldn't have been possible without referencing those two guy's code!

Description of CPPNs:

https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

'''

class CPPNVAE():
  def __init__(self, batch_size=1, z_dim=32,
                x_dim = 26, y_dim = 26, c_dim = 1, scale = 8.0,
                learning_rate_g= 0.01, learning_rate_d= 0.001, learning_rate_vae = 0.0001, beta1 = 0.9, net_size_g = 6, net_depth_g = 24, subnet_depth_g = 4,
                net_size_q = 512, keep_prob = 1.0, df_dim = 32, model_name = "cppnvae", grad_clip = 5.0):
    """

    Args:
    z_dim               dimensionality of the latent vector
    x_dim, y_dim        default resolution of generated images for training
    c_dim               1 for monotone, 3 for colour
    learning_rate_g     learning rate for the generator
                 _d    learning rate for the discriminiator
                 _vae  learning rate for the variational autoencoder
    net_size_g          number of activations per layer for cppn generator function
    net_depth_g         depth of generator
    net_size_q          number of activations per layer for decoder (real image -> z). 2 layers.
    df_dim              discriminiator is a convnet.  higher -> more activtions -> smarter.
    keep_prob           dropout probability

    when training, use I used dropout on training the decoder, batch norm on discriminator, nothing on cppn
    choose training parameters so that over the long run, decoder and encoder log errors hover around 0.7 each (so they are at the same skill level)
    while the error for vae should slowly move lower over time with D and G balanced.

    """

    self.batch_size = batch_size
    self.learning_rate_g = learning_rate_g
    self.learning_rate_d = learning_rate_d
    self.learning_rate_vae = learning_rate_vae
    self.beta1 = beta1
    self.net_size_g = net_size_g
    self.net_size_q = net_size_q
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.scale = scale
    self.c_dim = c_dim
    self.z_dim = z_dim
    self.net_depth_g = net_depth_g
    self.subnet_depth_g = subnet_depth_g
    self.model_name = model_name
    self.keep_prob = keep_prob
    self.df_dim = df_dim
    self.num_class = 11 # 0->9 are MNIST classes, 10 are fake digits.
    self.grad_clip = grad_clip

    # tf Graph batch of image (batch_size, height, width, depth)
    self.batch = tf.placeholder(tf.float32, [batch_size, x_dim, y_dim, c_dim])
    self.batch_flatten = tf.reshape(self.batch, [batch_size, -1])
    self.batch_label = tf.placeholder(tf.float32, [batch_size, self.num_class]) # mnist labels for the batch (one-hot)
    self.fake_label = np.array(self.batch_size*[10], dtype=np.int32) # label of fake batches
    self.fake_label_one_hot = self.to_one_hot(self.fake_label)

    n_points = x_dim * y_dim
    self.n_points = n_points

    self.x_vec, self.y_vec, self.r_vec = self.coordinates(x_dim, y_dim, scale)

    # latent vector
    # self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
    # inputs to cppn, like coordinates and radius from centre
    self.x = tf.placeholder(tf.float32, [self.batch_size, None, 1])
    self.y = tf.placeholder(tf.float32, [self.batch_size, None, 1])
    self.r = tf.placeholder(tf.float32, [self.batch_size, None, 1])

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(batch_size, name=self.model_name+'_d_bn1')
    self.d_bn2 = batch_norm(batch_size, name=self.model_name+'_d_bn2')

    # Use recognition network to determine mean and
    # (log) variance of Gaussian distribution in latent
    # space
    self.z_mean, self.z_log_sigma_sq = self.encoder()

    # Draw one sample z from Gaussian distribution
    eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
    # z = mu + sigma*epsilon
    self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

    # Use generator to determine mean of
    # Bernoulli distribution of reconstructed input
    self.G = self.generator()
    #self.batch_reconstruct_flatten = tf.reshape(self.G, [batch_size, -1]) # not needed

    self.predict_real_samples = self.discriminator(self.batch) # discriminiator on correct examples
    self.predict_fake_samples = self.discriminator(self.G, reuse=True) # feed generated images into D

    self.create_vae_loss_terms()
    self.create_gan_loss_terms()

    self.balanced_loss = 1.0 * self.g_loss + 10.0 * self.vae_loss # can try to weight these.

    self.t_vars = tf.trainable_variables()

    self.q_vars = [var for var in self.t_vars if (self.model_name+'_q_') in var.name]
    self.g_vars = [var for var in self.t_vars if (self.model_name+'_g_') in var.name]
    self.d_vars = [var for var in self.t_vars if (self.model_name+'_d_') in var.name]
    self.both_vars = self.q_vars+self.g_vars
    #self.vae_vars = self.q_vars # in this version, g_vars don't concern vae_loss

    # clip gradients
    d_opt_real_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss_real, self.d_vars), self.grad_clip)
    d_opt_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_vars), self.grad_clip)
    g_opt_grads, _ = tf.clip_by_global_norm(tf.gradients(self.balanced_loss, self.both_vars), self.grad_clip)
    vae_opt_grads, _ = tf.clip_by_global_norm(tf.gradients(self.vae_loss, self.q_vars), self.grad_clip)

    d_real_optimizer = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1)
    d_optimizer = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1)
    g_optimizer = tf.train.AdamOptimizer(self.learning_rate_g, beta1=self.beta1)
    vae_optimizer = tf.train.AdamOptimizer(self.learning_rate_vae, beta1=self.beta1)

    self.d_opt_real = d_real_optimizer.apply_gradients(zip(d_opt_real_grads, self.d_vars))
    self.d_opt = d_optimizer.apply_gradients(zip(d_opt_grads, self.d_vars))
    self.g_opt = g_optimizer.apply_gradients(zip(g_opt_grads, self.both_vars))
    self.vae_opt = vae_optimizer.apply_gradients(zip(vae_opt_grads, self.q_vars))

    '''
    self.d_opt_real = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1) \
                      .minimize(self.d_loss_real, var_list=self.d_vars)
    self.d_opt = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=self.d_vars)
    self.g_opt = tf.train.AdamOptimizer(self.learning_rate_g, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=self.both_vars)
    self.vae_opt = tf.train.AdamOptimizer(self.learning_rate_vae, beta1=self.beta1) \
                      .minimize(self.vae_loss, var_list=self.q_vars)
    '''

    '''
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr, epsilon=0.001)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    '''

    self.init()
    self.saver = tf.train.Saver(tf.all_variables())

  def init(self):

    # Initializing the tensor flow variables
    init = tf.initialize_all_variables()
    # Launch the session
    self.sess = tf.InteractiveSession()
    self.sess.run(init)

  def reinit(self):
    init = tf.initialize_variables(tf.trainable_variables())
    self.sess.run(init)

  def to_one_hot(self, label):
    # convert labels, a numpy list of labels (of size batch_size) to the one hot equivalent
    return np.eye(self.num_class)[label]

  def create_vae_loss_terms(self):
    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #     of the input under the reconstructed Bernoulli distribution
    #     induced by the decoder in the data space).
    #     This can be interpreted as the number of "nats" required
    #     for reconstructing the input when the activation in latent
    #     is given.
    # Adding 1e-10 to avoid evaluatio of log(0.0)

    # stop using likelihood function for similarity
    # reconstr_loss = \
    #     -tf.reduce_sum(self.batch_flatten * tf.log(1e-10 + self.batch_reconstruct_flatten)
    #                    + (1-self.batch_flatten) * tf.log(1e-10 + 1 - self.batch_reconstruct_flatten), 1)

    # 2.) The latent loss, which is defined as the Kullback Leibler divergence
    ##    between the distribution in latent space induced by the encoder on
    #     the data and some prior. This acts as a kind of regularizer.
    #     This can be interpreted as the number of "nats" required
    #     for transmitting the the latent space distribution given
    #     the prior.

    latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                       - tf.square(self.z_mean)
                                       - tf.exp(self.z_log_sigma_sq), 1)

    #self.vae_loss = tf.reduce_mean(reconstr_loss + latent_loss) / self.n_points # average over batch and pixel

    # vae loss is now purely kl divergence loss term.  let GAN take care of mnist class accuracy.
    self.vae_loss = tf.reduce_mean(latent_loss) / self.n_points # average over batch and pixel

  def create_gan_loss_terms(self):
    # Define loss function and optimiser
    ''' replace below with class-based disriminiator
    self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D_right), self.D_right)
    self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_wrong), self.D_wrong)
    self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
    self.g_loss = 1.0*binary_cross_entropy_with_logits(tf.ones_like(self.D_wrong), self.D_wrong)
    '''

    # cross entropy loss of predicting real mnist to real classes
    self.d_loss_real = tf.reduce_mean(-tf.reduce_sum(self.batch_label * tf.log(self.predict_real_samples), reduction_indices=[1]))
    # accuracy of using discriminiator as a normal mnist classifier
    self.d_loss_real_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict_real_samples,1), tf.argmax(self.batch_label,1)), tf.float32))
    # cross entropy loss of predicting that fake generated mnist are in fact fake
    self.d_loss_fake = tf.reduce_mean(-tf.reduce_sum(self.fake_label_one_hot * tf.log(self.predict_fake_samples), reduction_indices=[1]))
    # accuracy of discriminator predicting a fake mnist digit
    self.d_loss_fake_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict_fake_samples,1), tf.argmax(self.fake_label_one_hot,1)), tf.float32))
    # take the average of two d_loss to be the defacto d_loss
    self.d_loss = (10.0*self.d_loss_real + self.d_loss_fake)/ 11.0 # balanc out the classes
    # cross entropy of generator fooling discriminiator that its shit is real.
    self.g_loss = tf.reduce_mean(-tf.reduce_sum(self.batch_label * tf.log(self.predict_fake_samples), reduction_indices=[1]))
    # accuracy of generated samples being fooled to be classified as their supposed ground truth labels
    self.g_loss_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict_fake_samples,1), tf.argmax(self.batch_label,1)), tf.float32))

  def coordinates(self, x_dim = 32, y_dim = 32, scale = 1.0):
    n_pixel = x_dim * y_dim
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
    y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
    r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
    return x_mat, y_mat, r_mat

  def show_image(self, image):
    '''
    image is in [height width depth]
    '''
    plt.subplot(1, 1, 1)
    y_dim = image.shape[0]
    x_dim = image.shape[1]
    if self.c_dim > 1:
      plt.imshow(image, interpolation='nearest')
    else:
      plt.imshow(image.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()

  def encoder(self):
    # Generate probabilistic encoder (recognition network), which
    # maps inputs onto a normal distribution in latent space.
    # The transformation is parametrized and can be learned.
    H1 = tf.nn.dropout(tf.nn.softplus(linear(self.batch_flatten, self.net_size_q, self.model_name+'_q_lin1')), self.keep_prob)
    H2 = tf.nn.dropout(tf.nn.softplus(linear(H1, self.net_size_q, self.model_name+'_q_lin2')), self.keep_prob)
    z_mean = linear(H2, self.z_dim, self.model_name+'_q_lin3_mean')
    z_log_sigma_sq = linear(H2, self.z_dim, self.model_name+'_q_lin3_log_sigma_sq')
    return (z_mean, z_log_sigma_sq)

  def discriminator(self, image, reuse=False):

    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, self.df_dim, name=self.model_name+'_d_h0_conv'))
    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name=self.model_name+'_d_h1_conv')))
    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name=self.model_name+'_d_h2_conv')))
    h3 = linear(tf.reshape(h2, [self.batch_size, -1]), self.num_class, self.model_name+'_d_h2_lin')

    return tf.nn.softmax(h3)

  def generator(self, gen_x_dim = 26, gen_y_dim = 26, reuse = False):

    if reuse:
        tf.get_variable_scope().reuse_variables()

    n_network = self.net_size_g
    gen_n_points = gen_x_dim * gen_y_dim

    z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) * \
                    tf.ones([gen_n_points, 1], dtype=tf.float32) * self.scale
    z_unroll = tf.reshape(z_scaled, [self.batch_size*gen_n_points, self.z_dim])
    x_unroll = tf.reshape(self.x, [self.batch_size*gen_n_points, 1])
    y_unroll = tf.reshape(self.y, [self.batch_size*gen_n_points, 1])
    r_unroll = tf.reshape(self.r, [self.batch_size*gen_n_points, 1])

    U = fully_connected(z_unroll, n_network, self.model_name+'_g_0_z') + \
        fully_connected(x_unroll, n_network, self.model_name+'_g_0_x', with_bias = False) + \
        fully_connected(y_unroll, n_network, self.model_name+'_g_0_y', with_bias = False) + \
        fully_connected(r_unroll, n_network, self.model_name+'_g_0_r', with_bias = False)

    #H = tf.nn.relu(U)
    H = tf.nn.tanh(U)

    for i in range(0, self.net_depth_g):
      H0 = H
      for j in range(0, self.subnet_depth_g):
        H0 = tf.nn.relu(fully_connected(H0, n_network, self.model_name+'_g_relu_skip_'+str(i)+'_'+str(j), stddev = 1.0))
      H0 = tf.nn.tanh(fully_connected(H0, n_network, self.model_name+'_g_tanh_skip_'+str(i), stddev = 0.001))
      H = H + H0

    output = tf.sigmoid(fully_connected(H, self.c_dim, self.model_name+'_g_'+str(self.net_depth_g)))

    result = tf.reshape(output, [self.batch_size, gen_y_dim, gen_x_dim, self.c_dim])

    return result


  def partial_train(self, batch, label):
    """Train model based on mini-batch of input data.

    Return cost of mini-batch.

    I should really seperate the below tricks into parameters, like number of times/pass
    and also the regulator threshold levels.
    """

    counter = 0

    label_one_hot = self.to_one_hot(label)

    '''
    for i in range(1):
      counter += 1
      _, vae_loss = self.sess.run((self.vae_opt, self.vae_loss),
                              feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})
    '''

    for i in range(16):
      counter += 1
      _, g_loss, vae_loss, g_accuracy = self.sess.run((self.g_opt, self.g_loss, self.vae_loss, self.g_loss_accuracy),
                              feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})
      if g_accuracy > 0.98:
        break

    # train classifier on only real mnist digits
    # _ = self.sess.run((self.d_opt_real), feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})

    # calculate accuracy before deciding whether to train discriminator
    d_loss, d_loss_real, d_loss_fake, d_real_accuracy, d_fake_accuracy = self.sess.run((self.d_loss, self.d_loss_real, self.d_loss_fake, self.d_loss_real_accuracy, self.d_loss_fake_accuracy),
                              feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})

    if d_fake_accuracy < 0.7 and g_accuracy > 0.6: # only train discriminiator if generator is good and d is behind.
      for i in range(8):
        counter += 1
        _, d_loss, d_loss_real, d_loss_fake, d_real_accuracy, d_fake_accuracy = self.sess.run((self.d_opt, self.d_loss, self.d_loss_real, self.d_loss_fake, self.d_loss_real_accuracy, self.d_loss_fake_accuracy),
                                feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})
        if d_fake_accuracy > 0.75:
          break
    elif d_real_accuracy < 0.6:
      for i in range(8):
        counter += 1
        _, d_real_accuracy = self.sess.run((self.d_opt_real, self.d_loss_real_accuracy), feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})
        if d_real_accuracy > 0.7:
          break

    return d_loss, g_loss, vae_loss, counter, d_real_accuracy, d_fake_accuracy, g_accuracy, d_loss_real, d_loss_fake

  def encode(self, X):
      """Transform data by mapping it into the latent space."""
      # Note: This maps to mean of distribution, we could alternatively
      # sample from Gaussian distribution
      return self.sess.run(self.z_mean, feed_dict={self.batch: X})

  def generate(self, z=None, x_dim = 26, y_dim = 26, scale = 5.0):
    """ Generate data by sampling from latent space.

    If z is not None, data for this point in latent space is
    generated. Otherwise, z is drawn from prior in latent
    space.
    """
    if z is None:
        z = np.random.normal(size=self.z_dim).astype(np.float32)
    # Note: This maps to mean of distribution, we could alternatively
    # sample from Gaussian distribution

    z = np.reshape(z, (self.batch_size, self.z_dim))

    G = self.generator(gen_x_dim = x_dim, gen_y_dim = y_dim, reuse = True)
    gen_x_vec, gen_y_vec, gen_r_vec = self.coordinates(x_dim, y_dim, scale = scale)
    image = self.sess.run(G, feed_dict={self.z: z, self.x: gen_x_vec, self.y: gen_y_vec, self.r: gen_r_vec})
    return image

  def save_model(self, checkpoint_path, epoch):
    """ saves the model to a file """
    self.saver.save(self.sess, checkpoint_path, global_step = epoch)

  def load_model(self, checkpoint_path):

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print "loading model: ",ckpt.model_checkpoint_path

    #self.saver.restore(self.sess, checkpoint_path+'/'+ckpt.model_checkpoint_path)
    # use the below line for tensorflow 0.7
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)

  def close(self):
    self.sess.close()


