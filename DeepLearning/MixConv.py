import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization
from keras.constraints import Constraint
from keras.initializers import Initializer
import numpy as np


# def _split_channels(total_filters, num_groups):
#   split = [total_filters // num_groups for _ in range(num_groups)]
#   split[0] += total_filters - sum(split)
#   return split


def _split_channels_v2(filters, kernels):
   '''
   Utility function to split the channels (called filters here) into groups (called splits here).

   Parameters
   ----------
      filters : the number of input or output channels

      kernels : list of dictionaries like this : [{'size':(3,3), 'type':'full', 'prop':0.50}, {'size':(5,5), 'type':'diamond', 'prop':0.50}]
      where each element of the list corresponds to one type of kernel with its proportion. 
      If there are N input channels, each kernel will handle N*prop channels (rounded).
      The first kernel of the list handles the first N*prop channels, the second handles the N*prop channels right after.
      Example :
      kernels = [{'size':(3,3), 'type':'full', 'prop':0.50}, {'size':(5,5), 'type':'diamond', 'prop':0.50}]
      filters = 32
      -> kernel of size (3,3) of type 'full' will handle channels of index 0 to 15.
      -> kernel of size (5,5) of type 'diamond' will handle channels of index 16 to 31.
   '''
   splits = []
   delimiter = 0
   for kernel in kernels[:-1] :
      split = round(filters * kernel['prop'])
      splits.append(split)
      delimiter += split
   splits.append(max(filters-delimiter,0))
   return splits

class GroupedConv2D:
  """
  Heavily inspired by https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
  Channels are assigned to groups, and each group is assigned to a Conv2D layer.
  Then the result is concatenated.
  Can inherit from keras.layers.Layer base class, but for exporting purposes, it is better not to.
  """
  def __init__(self, filters, kernels, **kwargs):
    """
    Parameters
    ----------
      filters : the number of output channels
      
      kernels : list of dictionaries like this : [{'size':(3,3), 'type':'full', 'prop':0.50}, {'size':(5,5), 'type':'diamond', 'prop':0.50}]
      where each element of the list corresponds to one type of kernel with its proportion. 
      If there are N input channels, each kernel will handle N*prop channels rounded.

      **kwargs : Conv2D kwargs
    """
    # Initialization
    self.kernels = kernels
    self._groups = len(kernels)
    self._channel_axis = -1
    self._convs = []
    self._bnorms = []
    splits = _split_channels_v2(filters, kernels)
    self.frozen_weights = 0

    # For each kernel, assign a Conv2D layer to it with its corresponding kernel weight (with a certain mask)
    for i, kernel in enumerate(kernels):
      mask = generate_mask(kernel['size'], ktype=kernel['type'])
      freeze_weights = FreezeConv(mask=mask, in_channel_axis=-2, out_channel_axis=-1)
      freeze_init = FreezeInit(mask=mask)
      self._convs.append(
          Conv2D(filters=splits[i], kernel_size=kernel['size'], kernel_initializer=freeze_init, kernel_constraint=freeze_weights, **kwargs)
      )
      #tracking the number of frozen weights
      self.frozen_weights += freeze_init.frozen_weights

      self._bnorms.append(
         BatchNormalization()
      )
      print(self.frozen_weights)
    print(self.frozen_weights)

  def __call__(self, inputs):
    # For each input channel, assign them to a conv2D layer
    if len(self._convs) == 1:
      return self._convs[0](inputs)

    filters = inputs.shape[self._channel_axis]
    splits = _split_channels_v2(filters, self.kernels)
    x_splits = tf.split(inputs, splits, self._channel_axis)
    x_outputs = [bn(c(x)) for x, c, bn in zip(x_splits, self._convs, self._bnorms)]
    x = tf.concat(x_outputs, self._channel_axis)
    return x


class MixConv:
  """
  Heavily inspired by https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
  Channels are assigned to groups, and each group is assigned to a DepthwiseConv2D layer.
  Then the result is concatenated.
  Can inherit from keras.layers.Layer base class, but for exporting purposes, it is better not to.
  """

  def __init__(self, kernels, strides, **kwargs):
    """
    Parameters
    ----------
      filters : the number of output channels
      
      kernels : list of dictionaries like this : [{'size':(3,3), 'type':'full', 'prop':0.50}, {'size':(5,5), 'type':'diamond', 'prop':0.50}]
      where each element of the list corresponds to one type of kernel with its proportion. 
      If there are N input channels, each kernel will handle N*prop channels rounded.

      **kwargs : DepthwiseConv2D key word arguments
    """
    # Initialization
    self._channel_axis = -1
    self._convs = []
    self._bnorms = []
    self.kernels = kernels
    #tracking the number of frozen weights
    self.frozen_weights = 0

    # For each kernel, assign a DepthwiseConv2D layer to it with its corresponding kernel weight (with a certain mask)
    for kernel in kernels:
      mask = generate_mask(kernel['size'], ktype=kernel['type'])
      freeze_weights = FreezeConv(mask=mask, in_channel_axis=-2, out_channel_axis=-1)
      freeze_init = FreezeInit(mask=mask)
      self._convs.append(
          DepthwiseConv2D(kernel['size'], strides=strides, depthwise_initializer=freeze_init, depthwise_constraint=freeze_weights, **kwargs)
      )
      self.frozen_weights += freeze_init.frozen_weights
      self._bnorms.append(
         BatchNormalization()
      )

  def __call__(self, inputs):
    if len(self._convs) == 1:
      return self._convs[0](inputs)

    filters = inputs.shape[self._channel_axis]
    splits = _split_channels_v2(filters, self.kernels)
    x_splits = tf.split(inputs, splits, self._channel_axis)
    x_outputs = [bn(c(x)) for x,c,bn in zip(x_splits, self._convs, self._bnorms)]
    x = tf.concat(x_outputs, self._channel_axis)
    return x
  

class FreezeConv(Constraint):
    '''
    Inherits from keras.constraints.Constraint.
    This class freezes specific kernels weights to 0 after the back propagation according to the mask parameter.
    '''
    def __init__(self, mask, in_channel_axis, out_channel_axis):
        self.mask = mask
        self.in_channel_axis = in_channel_axis
        self.out_channel_axis = out_channel_axis

    def __call__(self, w):
        nb_out_channel = w.shape[self.out_channel_axis]
        nb_in_channel = w.shape[self.in_channel_axis]
        # A (3x3) kernel in reality has weights of shape (3,3,input_channels,output_channels)
        # while the mask has shape (3,3). We do a tf.repeat and reshape to the right size, then multiply it to the weights w.
        try :
          reshaped_mask = tf.reshape(
              tf.repeat(self.mask, nb_in_channel*nb_out_channel),
              shape=w.shape
          )
          return w * tf.cast(reshaped_mask, dtype=w.dtype)
        except :
          print("error")
          print(nb_in_channel, nb_out_channel)
          print(tf.repeat(self.mask, nb_in_channel*nb_out_channel).shape)
          return w
        
class FreezeInit(Initializer):
    '''
    Inherits from keras.initializers.Initializer.
    This class initializes random kernel weights following a gaussian distribution.
    But it sets some specific weights within the kernel to 0 according to the mask parameter.
    '''
    def __init__(self, mask):
        self.mask = mask
        # track the number of frozen weights
        self.frozen_weights = (1-self.mask).sum()

    def __call__(self, shape, dtype=None):
        try :
          kernel_height, kernel_width, in_filters, out_filters = shape
          fan_out = int(kernel_height * kernel_width * out_filters)
          reshaped_mask = tf.reshape(
              tf.repeat(self.mask, in_filters*out_filters),
              shape=shape
          )
          self.frozen_weights *= in_filters*out_filters
          return tf.random.normal(shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype) * tf.cast(reshaped_mask, dtype=dtype)
        except :
          print("error")

    def get_config(self):  # To support serialization
        return {'mask' : self.mask}
    
def generate_mask(filter_shape, ktype='rond', hole=0):
  '''
  Generates a mask (an array of booleans) of shape=filter_shape, with the given ktype.
  ktypes : full, diamond, round, anneau ...
  '''
  height = filter_shape[0]
  width = filter_shape[1]
  middle = height//2

  if ktype == 'full':
      return np.ones(filter_shape, dtype=bool)
  
  elif ktype == 'round':
      '''
       [[False  True  True  True False]
        [ True  True  True  True  True]
        [ True  True  True  True  True]
        [ True  True  True  True  True]
        [False  True  True  True False]]
      '''
      mask = np.ones(filter_shape, dtype=bool)
      mask[0,0], mask[0, -1], mask[-1, -1], mask[-1, 0] = (False, False, False, False)
      return mask

  elif ktype == 'diamond':
      '''
       [[False False  True False False]
        [False  True  True  True False]
        [ True  True  True  True  True]
        [False  True  True  True False]
        [False False  True False False]]
      '''
      mask = []
      for inc in range(0, middle, 1):
          array = np.zeros(filter_shape[0], dtype=bool)
          array[middle-inc:middle+1+inc] = True
          mask.append(array)
      for inc in range(middle, -1, -1):
          array = np.zeros(filter_shape[0], dtype=bool)
          array[middle-inc:middle+1+inc] = True
          mask.append(array)
      mask = np.array(mask)
      return mask

  elif ktype == 'anneau' :
      '''
       [[False False  True False False]
        [False  True  True  True False]
        [ True  True  False True  True]
        [False  True  True  True False]
        [False False  True False False]]
      '''
      mask = []
      for inc in range(0, middle, 1):
          array = np.zeros(filter_shape[0], dtype=bool)
          array[middle-inc:middle+1+inc] = True
          mask.append(array)
      for inc in range(middle, -1, -1):
          array = np.zeros(filter_shape[0], dtype=bool)
          array[middle-inc:middle+1+inc] = True
          mask.append(array)
      mask = np.array(mask)
      mask[(middle, middle)] = False
      return mask
  
  elif ktype == 'ruche' :
      '''
      Example : np.array(
          [[False, True, False],
          [True , False, True],
          [True , False, True],
          [False, True, False]]
      ) shape=(4,3)
      '''
      pass

  elif ktype == 'L':
      pass
  
  elif ktype == 'X':
      '''
      Example : np.array(
          [[True False False False True ]
          [False True  False True  False]
          [False False True  False False]
          [False True  False True  False]
          [True  False False False True]]
      ) shape=(5,5)
      '''
      mask = np.identity(height, dtype=bool)
      for diag_idx in range(height):
          mask[height-1-diag_idx, diag_idx] = True
      return mask
  
  elif ktype == 'contour' :
        '''
        Example : np.array(
        [[True, True , True , True , True],
         [True, False, False, False, True],
         [True, False, False, False, True],
         [True, False, False, False, True],
         [True, True , True , True , True]]
        ) shape=(5,5)
        '''
        mask = np.zeros(filter_shape, dtype=bool)
        for idx1 in range(filter_shape[0]):
            for idx2 in range(filter_shape[1]):
                if idx1 in [0, filter_shape[0]-1] or idx2 in [0, filter_shape[1]-1]:
                    mask[idx1, idx2] = True
        return mask