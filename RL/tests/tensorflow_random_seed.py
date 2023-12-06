import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from RL.settings import RANDOM_SEED
tf.random.set_seed(RANDOM_SEED)
from RL.net_2d import SimpleSoftMaxNet_2D
from RL.settings import run_settings
import numpy as np
# a = tf.random_uniform([1])
# b = tf.random_normal([1])
weights = tf.compat.v1.get_variable('weights_last', [9, 2], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

# Repeatedly running this block with the same graph will generate the same
# sequences of 'a' and 'b'.

net = SimpleSoftMaxNet_2D(run_settings())
init = tf.compat.v1.global_variables_initializer()
print("Session 1")
final_weights1=None
final_weights2=None
with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()) as sess1:
  # print(sess1.run(a))  # generates 'A1'
  # print(sess1.run(a))  # generates 'A2'
  # print(sess1.run(b))  # generates 'B1'
  # print(sess1.run(b))  # generates 'B2'
  sess1.run(init)
  [weights_l] = sess1.run([net.tvars])
  #[weights_l] = sess1.run([weights])
  print (weights_l)
  final_weights1=weights_l

print("Session 2")
with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()) as sess2:
  # print(sess2.run(a))  # generates 'A1'
  # print(sess2.run(a))  # generates 'A2'
  # print(sess2.run(b))  # generates 'B1'
  # print(sess2.run(b))  # generates 'B2'
  sess2.run(init)
  [weights_l2] = sess2.run([net.tvars])
  #[weights_l] = sess2.run([weights])
  print (weights_l2)
  final_weights2=weights_l2

for i, weight in enumerate(final_weights1):
  print("Equal? "+str(i)+" "+str(np.sum(weight-final_weights2[i])==0))

