import tensorflow as tf

cluster = 	tf.train.ClusterSpec({
    "worker": [
        "localhost:2223",
    ],
    "ps": [        
        "152.19.32.251:2222"
    ]})

server = tf.train.Server(cluster, job_name='ps', task_index=0)

server.join()