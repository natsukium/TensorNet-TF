import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# GPU params
flags.DEFINE_integer("GPU", 2, "number of GPU")

# Model params
flags.DEFINE_integer("scale", 2, "down scale parameter")
flags.DEFINE_float("eta", 0.001, 'learning rate')
flags.DEFINE_integer("rank", 10, "Tensor's rank")

# Training params
flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
flags.DEFINE_integer("num_epochs", 200, "Number of training epochs")

# Save params
flags.DEFINE_string("logdir", "log", "directory saving log")
