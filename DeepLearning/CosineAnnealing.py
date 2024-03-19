import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback
import numpy as np
import golois


class CosineAnnealing(Callback):

    def __init__(self,
                 model,
                 min_lr,
                 max_lr,
                 lr_mult,
                 T_length,
                 T_mult):
        self.model = model
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_mult = lr_mult
        self.T_length = T_length
        self.T_mult = T_mult
        self.T_index = 1
        self.last_restart = 0
        self.HISTORY = dict()

    def on_epoch_begin(self, epoch, logs=None):
        self.update_lr(epoch)
        print(f"Epoch : {epoch} - lr : {round(self.current_lr,6)} - Cycle : {self.T_index} - lr_interval : [{round(self.min_lr,6)},{round(self.max_lr,6)}] - last restart : {self.last_restart}")
        golois.getBatch(input_data, policy, value, end, groups, epoch * N)
        if (epoch % 5 == 0) :
            gc.collect()

    def on_epoch_end(self, epoch, logs=None):
        self.HISTORY[f'epoch_{epoch}'] = [log[0] for log in logs]
        if (epoch % 20 == 0):
            golois.getValidation (input_data, policy, value, end)
            val = self.model.evaluate(input_data,
                                      [policy, value],
                                      verbose = 0,
                                      batch_size=batch)
            print ("val =", val)
            self.model.save('test.h5')

    def update_lr(self, epoch):
        T_cur = epoch-self.last_restart
        if T_cur > self.T_length :
            # Restart
            self.last_restart = epoch
            T_cur = epoch-self.last_restart
            self.T_index += 1
            self.T_length *= self.T_mult
            # Reducing min max lr
            self.min_lr *= self.lr_mult
            self.max_lr *= self.lr_mult
        self.current_lr = self.min_lr + (0.5*(self.max_lr-self.min_lr)) * (1 + np.cos(np.pi*T_cur/self.T_length))
        self.model.optimizer.learning_rate = self.current_lr

    def update_lr2(self, epoch):
        T_cur = (epoch - self.last_restart) % self.T_length
        self.current_lr = self.min_lr + (0.5*(self.max_lr-self.min_lr)) * (1 + np.cos(np.pi*T_cur/self.T_length))
        self.model.optimizer.learning_rate = self.current_lr
        if T_cur == 0 :
            # Restart
            self.last_restart = epoch
            self.T_index += 1
            self.T_length *= self.T_mult
            # Reducing min max lr
            self.min_lr *= self.lr_mult
            self.max_lr *= self.lr_mult