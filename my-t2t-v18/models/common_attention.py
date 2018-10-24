"""Utilities for attention."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# Dependency imports

from models import common_layers
import tensorflow as tf

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  """
  length = tf.shape(x)[1]
  channels = tf.shape(x)[2]
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return x + signal


def embedding_to_padding(emb):
  """Input embeddings -> is_padding.
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.equal(emb_sum, 0.0)


def attention_bias_lower_triangle(length):
  """Create an bias tensor to be added to attention logits.
  """
  lower_triangle = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
  ret = -1e9 * (1.0 - lower_triangle)
  return tf.reshape(ret, [1, 1, length, length])


def attention_bias_ignore_padding(memory_padding):
  """Create an bias tensor to be added to attention logits.
    input: [batch, memory_length], return: [batch, 1, 1, memory_length].
  """
  ret = tf.to_float(memory_padding) * -1e9
  return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.
  note: [..., m] --> [..., n, m/n]
  """
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.
  note: [..., a, b] --> [..., ab]
  """
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


def split_heads(x, num_heads):
  """Split channels (dimension 3) into multiple heads (becomes dimension 1).
  note: [batch, length, channels] -> [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
  """Inverse of split_heads.
  note: [batch, num_heads, length, channels / num_heads] -> [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def attention_image_summary(attn, image_shapes=None):
  """Compute color image summary.

  Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional tuple of integer scalars.
      If the query positions and memory positions represent the
      pixels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
      If the query positions and memory positions represent the
      pixels x channels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, query_channels,
         memory_rows, memory_cols, memory_channels).
  """
  num_heads = attn.get_shape().as_list()[1]
  # [batch, query_length, memory_length, num_heads]
  image = tf.transpose(attn, [0, 2, 3, 1])
  image = tf.pow(image, 0.2)  # for high-dynamic-range
  # Each head will correspond to one of RGB.
  # pad the heads to be a multiple of 3
  image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, -num_heads % 3]])
  image = split_last_dimension(image, 3)
  image = tf.reduce_max(image, 4)
  if image_shapes is not None:
    if len(image_shapes) == 4:
      q_rows, q_cols, m_rows, m_cols = list(image_shapes)
      image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
      image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
      image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
    else:
      assert len(image_shapes) == 6
      q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(
          image_shapes)
      image = tf.reshape(image, [-1, q_rows, q_cols, q_channnels,
                                 m_rows, m_cols, m_channels, 3])
      image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
      image = tf.reshape(image, [-1, q_rows * m_rows * q_channnels,
                                 q_cols * m_cols * m_channels, 3])
  tf.summary.image("attention", image, max_outputs=1)

def residual_fn(x, y, hparams):
      return common_layers.layer_norm(x + tf.nn.dropout(
                y, 1.0 - hparams.residual_dropout))

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
  """dot-product attention.
    q: a Tensor with shape [batch, heads, length_q, depth_k]
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if summaries and not tf.get_variable_scope().reuse:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        cache=None,
                        summaries=False,
                        image_shapes=None,
                        name=None):
  """Multihead scaled-dot-product attention with input/output transformations.
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
  """
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):
    if memory_antecedent is None:
      # self attention
      combined = common_layers.conv1d(
          query_antecedent,
          total_key_depth * 2 + total_value_depth,
          1,
          name="qkv_transform")
      q, k, v = tf.split(
          combined, [total_key_depth, total_key_depth, total_value_depth],
          axis=2)
    else:
      q = common_layers.conv1d(
          query_antecedent, total_key_depth, 1, name="q_transform")
      combined = common_layers.conv1d(
          memory_antecedent,
          total_key_depth + total_value_depth,
          1,
          name="kv_transform")
      k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
    
    if cache is not None:
        if bias is None:
            raise ValueError("Bias required for caching. See function docstring "
                             "for details.")
        k = cache["k"] = tf.concat([cache["k"], k], axis=1)
        v = cache["v"] = tf.concat([cache["v"], v], axis=1)

    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    x = dot_product_attention(
        q, k, v, bias, dropout_rate, summaries, image_shapes)
    x = combine_heads(x)
    x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
    return x

