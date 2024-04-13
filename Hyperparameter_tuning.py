#!/usr/bin/env python3

import numpy as np
import random
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Activation
from tensorflow.keras import Model
from tensorflow.keras import layers

np.set_printoptions(threshold=np.inf)
pi = np.pi

traindata_num = 131072
testdata_num = 32
batch_size = 32
epoch = 1
d_model = 20*20*13
ppm = [-10,10]
evolution_time = [0.02,0.07,0.12,0.2]
max_ns = 2048

adam_min = 2e-4
clip_val = 1000
loss2_weight = 1e-5
loss12_weight = 1e-3
change_loss_at_epoch_1 = 10
change_loss_at_epoch_2 = 15


'''customising network'''

class LrCustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model=d_model, warmup_steps=traindata_num/batch_size*epoch/100):
        super(LrCustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps


    def __call__(self, step):  # Step is batch
        step = step
        arg1 = tf.math.rsqrt( tf.cast(step, tf.float32) )
        arg2 = tf.cast(step, tf.float32) * (self.warmup_steps ** -1.5)
        val = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) / 100.
        val = tf.math.maximum(adam_min,val)
        return val


    def get_config(self):
        return {
            "d_model": float(self.d_model.numpy()),
            "warmup_steps": self.warmup_steps
        }


class CustomAdam(tf.keras.optimizers.Adam):
    def __init__(self, clip=None, learning_rate=2e-4, **kwargs):
        super(CustomAdam, self).__init__(**kwargs)
        self.clip = clip
        self.learning_rate = learning_rate
    def _resource_apply_dense(self):
        if self.clip_norm is not None:
            grad = tf.clip_by_norm(grad, self.clip)
        return super(CustomAdam, self)._resource_apply_dense(grad, var, apply_state)


learning_rate = LrCustomSchedule()
optimizer = CustomAdam(clip=clip_val, learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


'''customising hyperband'''

current_epoch = tf.Variable(0,trainable=False)

class HyperbandLoggingCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    def on_epoch_begin(self, epoch, logs=None):
        global current_epoch
        current_epoch.assign(epoch)


class Loss1Metric(tf.keras.metrics.Metric):
    def __init__(self, name='loss1', **kwargs):
        super(Loss1Metric, self).__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_fid = y_pred[:,:,0]
        confidence = y_pred[:,:,1]
        target_fid = y_true[:,:,0]
        uncertainty = tf.where(confidence>0, 1 / confidence - 1, 99)
        loss1 = tf.where(tf.math.abs(target_fid)>0, (pred_fid-target_fid)**2 / (uncertainty**2), 0)
        loss1 = tf.math.reduce_sum(loss1, axis=-1)
        loss1 = tf.math.reduce_mean(loss1)
        self.total_error.assign_add(loss1)
        self.count.assign_add(1)

    def result(self):
        return self.total_error / self.count

    def reset_state(self):
        self.total_error.assign(0)
        self.count.assign(0)
        

class Loss2Metric(tf.keras.metrics.Metric):
    def __init__(self, name='loss2', **kwargs):
        super(Loss2Metric, self).__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_fid = y_pred[:,:,0]
        confidence = y_pred[:,:,1]
        target_fid = y_true[:,:,0]
        uncertainty = tf.where(confidence>0, 1 / confidence - 1, 99)
        loss2 = tf.where(tf.math.abs(target_fid)>0, loss2_weight * tf.math.sqrt(uncertainty), 0)
        loss2 = tf.math.reduce_sum(loss2, axis=-1)
        loss2 = tf.math.reduce_mean(loss2)
        self.total_error.assign_add(loss2)
        self.count.assign_add(1)

    def result(self):
        return self.total_error / self.count

    def reset_state(self):
        self.total_error.assign(0)
        self.count.assign(0)


class MSEMetric(tf.keras.metrics.Metric):
    def __init__(self, name='loss_mse', **kwargs):
        super(MSEMetric, self).__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_fid = y_pred[:,:,0]
        target_fid = y_true[:,:,0]
        mse = tf.where(tf.math.abs(target_fid)>0, (pred_fid-target_fid)**2, 0)
        mse = tf.math.reduce_sum(mse, axis=-1) / tf.cast(tf.math.count_nonzero(target_fid, axis=-1), dtype=tf.float32)
        mse = tf.math.reduce_mean(mse)
        self.total_error.assign_add(mse)
        self.count.assign_add(1)

    def result(self):
        return self.total_error / self.count

    def reset_state(self):
        self.total_error.assign(0)
        self.count.assign(0)


def loss_schedule(y_true, y_pred):
    if current_epoch < change_loss_at_epoch_1:
        pred_fid = tf.reshape(y_pred[:,:,0], (-1,2*max_ns,1))
        target_fid = tf.reshape(y_true[:,:,0], (-1,2*max_ns,1))
        loss = tf.where(tf.math.abs(target_fid)>0, (pred_fid-target_fid)**2, 0)
        loss = tf.math.reduce_sum(loss, axis=1) / tf.cast(tf.math.count_nonzero(target_fid, axis=1), dtype=tf.float32)
    elif current_epoch >= change_loss_at_epoch_1 and current_epoch < change_loss_at_epoch_2:
        pred_fid = y_pred[:,:,0]
        confidence = y_pred[:,:,1]
        target_fid = y_true[:,:,0]
        uncertainty = tf.where(confidence>0, 1 / confidence - 1, 99)
        loss1 = tf.where(tf.math.abs(target_fid)>0, (pred_fid-target_fid)**2 / (uncertainty**2), 0)
        loss2 = tf.where(tf.math.abs(target_fid)>0, loss2_weight * tf.math.sqrt(uncertainty), 0)
        mse = tf.where(tf.math.abs(target_fid)>0, (pred_fid-target_fid)**2, 0)
        mse = tf.math.reduce_sum(mse, axis=1) / tf.cast(tf.math.count_nonzero(target_fid, axis=1), dtype=tf.float32)
        loss = loss1 + loss2
        loss = tf.math.reduce_sum(loss, axis=1)
        loss = loss12_weight * loss + mse
        loss = tf.math.reduce_mean(loss)
    else:
        pred_fid = y_pred[:,:,0]
        confidence = y_pred[:,:,1]
        target_fid = y_true[:,:,0]
        uncertainty = tf.where(confidence>0, 1 / confidence - 1, 99)
        loss1 = tf.where(tf.math.abs(target_fid)>0, (pred_fid-target_fid)**2 / (uncertainty**2), 0)
        loss2 = tf.where(tf.math.abs(target_fid)>0, loss2_weight * tf.math.sqrt(uncertainty), 0)
        loss = loss1 + loss2
        loss = tf.math.reduce_sum(loss, axis=1)
        loss = tf.math.reduce_mean(loss)
    return loss

def modified_tanh(x):
    fid = x[:,:,0]
    confidence = x[:,:,1]
    ac_fid = 1.5*tf.math.tanh(fid)
    ac_confidence = tf.math.tanh(confidence)
    ac_confidence = (ac_confidence+1) / 2
    output = tf.stack([ac_fid,ac_confidence], axis=-1)
    return output


'''define model'''

def build_model_fidnet(hp=kt.HyperParameters(), input_shape=(max_ns*2,5)):
    num_filters = hp.Choice('filter', values=[32,48,64])
    kernel_size = hp.Choice('kernel_size', values=[4,8,16])
    dilation_mode = hp.Int('dilation_sequence', 0, 1)
    num_blocks = hp.Int('num_blocks', 1, 4)
    axis1 = hp.Int('axis1', 2, 5)
    if dilation_mode == 0:
    	dilations = [1,2,3,5,8,12,17,22,29,36,43,50,65,82,101,123]
    elif dilation_mode == 1:
    	dilations = [1,2,3,5,8,12,20,29,43,82,123]
    pad_out = kernel_size - 1
    def waveLayer(x,num_filters,dil):
        pad = dil * (kernel_size - 1)
        x_pad = tf.pad(x, [[0,0],[0,pad],[0,0]])
        x1 = keras.layers.Conv1D(filters = num_filters, kernel_size=kernel_size,
        padding="valid", dilation_rate=dil)(x_pad)
        x2 = keras.layers.Conv1D(filters = num_filters, kernel_size=kernel_size,
        padding="valid", dilation_rate=dil)(x_pad)
        x1 = keras.layers.Activation('tanh')(x1)
        x2 = keras.layers.Activation('sigmoid')(x2)
        z = x1*x2
        z = tf.pad(z, [[0,0],[0,pad_out],[0,0]])
        z = keras.layers.Conv1D(filters = num_filters*2,
        kernel_size=kernel_size, padding="valid")(z)
        if axis1 == 2:
            z1, z2 = keras.layers.Add()([z, tf.expand_dims(x[:,:,0], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,1], axis=-1)])
            x_concat = keras.layers.concatenate([z1,z2], axis=-1)
        elif axis1 == 3:
            z1, z2, z3 = keras.layers.Add()([z, tf.expand_dims(x[:,:,0], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,1], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,2], axis=-1)])
            x_concat = keras.layers.concatenate([z1,z2,z3], axis=-1)
        elif axis1 == 4:
            z1, z2, z3, z4 = keras.layers.Add()([z, tf.expand_dims(x[:,:,0], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,1], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,2], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,3], axis=-1)])
            x_concat = keras.layers.concatenate([z1,z2,z3,z4], axis=-1)
        else:
            z1, z2, z3, z4, z5 = keras.layers.Add()([z, tf.expand_dims(x[:,:,0], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,1], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,2], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,3], axis=-1)]), keras.layers.Add()([z, tf.expand_dims(x[:,:,4], axis=-1)])
            x_concat = keras.layers.concatenate([z1,z2,z3,z4,z5], axis=-1)
        return x_concat, z
    input = keras.layers.Input(shape=input_shape)
    x = input
    x = x[:, :, 0:axis1]
    skips = []
    for dil in dilations*num_blocks:
        x, skip = waveLayer(x, num_filters, dil)
        skips.append(skip)
    x = keras.layers.Activation('relu')(keras.layers.Add()(skips))
    x = tf.pad(x, [[0,0],[0,pad_out],[0,0]])
    x = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size,
    padding="valid", activation='relu')(x)
    x = tf.pad(x, [[0,0],[0,pad_out],[0,0]])
    fin = keras.layers.Conv1D(filters=2, kernel_size=kernel_size,
    padding="valid", activation=modified_tanh)(x)
    model = keras.Model(inputs=input, outputs=fin)
    model.compile(optimizer='Adam',
                  loss=loss_schedule,
                  metrics=[Loss1Metric(),Loss2Metric(),MSEMetric()])
    return model


tuner = kt.Hyperband(hypermodel=build_model_fidnet, max_epochs=40, objective='loss')
