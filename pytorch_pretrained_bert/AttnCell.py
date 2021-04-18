# coding: utf-8
#@author: lixu
#@date: 2018-7-17
#交互注意力

import tensorflow as tf
import numpy as np
#from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell#tensorflow v1.0.0.
from tensorflow.python.ops.rnn_cell_impl import RNNCell #高版本的tensorflow要这句
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs

class AttnCell(RNNCell):
    """The most basic RNN cell."""
    def __init__(self, num_units, Premise_out,Premise_seqLen, activation=tanh):
        self._num_units = num_units#
        self._activation = activation
        self.Premise_seqLen = Premise_seqLen
        self.Premise_out = Premise_out
        self.Premise_Linear = last_dim_linear(Premise_out,self._num_units,bias=False,scope='Premise_attn_linear')
        self.tstp_pre = tf.shape(Premise_out)[1]#第二个维度的长度？

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        '''
            inputs: shape(b_sz, emb)
            state: shape(b_sz, h_sz)
        '''
        with vs.variable_scope(scope or "attention_cell"):
            tmp = linear([inputs, state], self._num_units, bias=False, scope='attn_linear')  # shape(b_sz, h_sz)
            tmp = tf.tile(tf.expand_dims(tmp, axis=1), [1, self.tstp_pre, 1])  # shape(b_sz, tstp_pre, h_sz)
            M_t = tanh(self.Premise_Linear + tmp)
            #squeeze删除特定索引位置的的大小为1的维度
            Mt_linear = tf.squeeze(last_dim_linear(M_t, 1, bias=False, scope='M_t_linear'), [2])  # shape(b_sz, tstp_pre)
            Alpha_t = masked_softmax(Mt_linear, self.Premise_seqLen)  # shape(b_sz, tstp_pre)

            tmp1 = tf.reduce_sum(tf.expand_dims(Alpha_t, 2) * self.Premise_out, axis=1)  # shape(b_sz, h_sz)
            tmp2 = tanh(linear(state, self._num_units, bias=False, scope='final_linear'))  # shape(b_sz, h_sz)
            next_state = tmp1 + tmp2

        return next_state, next_state

def mkMask(input_tensor, maxLen):###
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)

def masked_softmax(inp, seqLen):###
    if len(inp.get_shape()) != len(seqLen.get_shape())+1:
        raise ValueError('seqLen should be %d, but have the rank %d.\n' % (len(inp.get_shape())-1, len(inp.get_shape())))
    mask = mkMask(seqLen, tf.shape(inp)[-1])
    masked_inp = tf.where(mask, inp, tf.ones_like(inp) * (-np.Inf))
    ret = tf.nn.softmax(masked_inp)
    return ret

def last_dim_linear(inputs, output_size, bias, scope):###
    '''
    Args:
        input: shape(b_sz, ..., rep_sz)
        output_size: a scalar, python number
    '''
    bias_start=0.0
    input_shape = tf.shape(inputs)
    out_shape = tf.concat(axis=0, values=[input_shape[:-1], [output_size]])
    input_size = int(inputs.get_shape()[-1])
    unbatch_input = tf.reshape(inputs, shape=[-1, input_size])

    unbatch_output = linear(unbatch_input, output_size, bias=bias, bias_start=bias_start, scope=scope)
    batch_output = tf.reshape(unbatch_output, shape=out_shape)

    return batch_output     # shape(b_sz, ..., output_size)

def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, " "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or 'Linear') as outer_scope:
    weights = tf.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable("biases", [output_size],dtype=dtype,initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return tf.nn.bias_add(res, biases)