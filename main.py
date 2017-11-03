import os, numpy as np, tensorflow as tf
from model import Balancer

flags = tf.app.flags
flags.DEFINE_boolean("test_render", True, "If True, then render each test episode")
flags.DEFINE_boolean("train",True,"If True, then train the model with the given number of epochs")
flags.DEFINE_boolean("ignore_checkpoint", False, "If True, then ignore previous checkpoints")
flags.DEFINE_boolean("manual", False, "If True, then use manual mode for cart control")
flags.DEFINE_boolean("save", True, "If True, then save the model")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate of the momentum optimizer")
flags.DEFINE_float("beta1", 0.9, "Exponential decay rate for the first moment estimates")
flags.DEFINE_float("beta2", 0.999, "Exponential decay rate for the second moment estimates")
flags.DEFINE_float("discount_rate", 0.95, "Discount rate of the policy gradient algorithm")
flags.DEFINE_integer("epochs", 150, "Epochs to train on")
flags.DEFINE_integer("max_steps", 500, "Maximum number of steps per episode (limited by the environment's limit of 500 steps)")
flags.DEFINE_integer("games_per_update", 10, "Number of games played per update")
flags.DEFINE_integer("save_iterations", 50, "Number of iterations per save")
flags.DEFINE_integer("test_games", 5, "Number of games to test with")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory that checkpoints will be saved in and loaded from")
FLAGS = flags.FLAGS

def main(_):
	print(FLAGS.__flags)
	
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	
	with tf.Session() as sess:
		model = Balancer(sess, FLAGS.test_render, FLAGS.ignore_checkpoint, FLAGS.manual, FLAGS.save,
							FLAGS.learning_rate, FLAGS.beta1, FLAGS.beta2, FLAGS.discount_rate, FLAGS.epochs,
							FLAGS.max_steps, FLAGS.games_per_update, FLAGS.save_iterations, FLAGS.test_games,
							FLAGS.checkpoint_dir)
		if FLAGS.manual:
			return
		
		if FLAGS.train:
			model.train()
		else:
			model.load_model()
			model.test()

if __name__ == '__main__':
	tf.app.run()