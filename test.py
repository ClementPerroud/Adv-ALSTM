import tensorflow as tf

x = tf.constant(4.0)
with tf.GradientTape() as tape:
  with tape.stop_recording():
    y = x ** 2 + 
dy_dx = tape.gradient(y, x)
print(dy_dx)