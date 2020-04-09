import numpy as np
import tensorflow as tf

class SRGenerator:
    def __init__(self, discriminator, training, content_loss='mse', use_gan=True, learning_rate=1e-4, num_blocks=16, num_upsamples=2):
        self.learning_rate = learning_rate
        self.num_blocks = num_blocks
        self.num_upsamples = num_upsamples
        self.use_gan = use_gan
        self.discriminator = discriminator
        self.training = training
        self.reuse_vgg = False
        if content_loss not in ['mse', 'L1', 'vgg22', 'vgg54']:
            print('Invalid content loss function. Must be \'mse\', \'vgg22\', or \'vgg54\'.')
        exit()
        self.content_loss = content_loss
    def residual_block(self, x, kernel_size, filters, strides=1):
        return x