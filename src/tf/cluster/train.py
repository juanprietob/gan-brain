import tensorflow as tf

cluster = 	tf.train.ClusterSpec({
    "worker": [
        "localhost:2223",
    ],
    "ps": [        
        "152.19.32.251:2222"
    ]})

server = tf.train.Server(cluster, job_name='worker', task_index=0)



with tf.device("/job:ps/task:0"):
	image = tf.get_variable("images", shape=[5,5], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1), trainable=False)
	labels = tf.get_variable("labels", shape=[5,5], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1), trainable=False)
	w_matmul = tf.get_variable("w1", shape=[5,5], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))	
	bias = tf.get_variable("b1", shape=[5], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))

with tf.device("/job:worker/task:0"):
  
  layer_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(image, w_matmul), bias))
  logits = tf.nn.relu(layer_1)

  logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')

  global_step = tf.contrib.framework.get_or_create_global_step()
  
  optimizer = tf.train.AdamOptimizer(learning_rate=1e-3,
        beta1=0.9)

  train_op = optimizer.minimize(logits, global_step=global_step)

  hooks=[tf.train.StopAtStepHook(last_step=1000000)]
    
with tf.train.MonitoredTrainingSession(master=server.target,
                                       is_chief=True,
                                       checkpoint_dir="~/work/data/IBIS/checkpoints/",
                                       hooks=hooks) as sess:

  for _ in range(10000):
    sess.run(train_op)