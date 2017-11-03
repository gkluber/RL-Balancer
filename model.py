import threading, gym, tensorflow as tf, numpy as np, sys

class Balancer(object):
	#stop = True
	#use_checkpoint = True

	def __init__(self, sess, test_render, ignore_checkpoint, manual, save, learning_rate, 
				beta1, beta2, discount_rate, epochs, max_steps, games_per_update, 
				save_iterations, test_games, checkpoint_dir):
		
		self.env = gym.make("CartPole-v1")
		self.sess = sess
		self.test_render = test_render
		self.manual = manual
		self.save = save
		self.ignore_checkpoint = ignore_checkpoint
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.discount_rate = discount_rate
		self.epochs = epochs
		self.max_steps = max_steps
		self.games_per_update = games_per_update
		self.save_iterations = save_iterations
		self.test_games = test_games
		self.checkpoint_dir = checkpoint_dir
		
		#begin construction phase
		if(self.manual):
			self.use_manual()
		else:
			self.build_model()
		
	def build_model(self):
		self.n_inputs = 4
		n_hidden = 4
		n_outputs = 1
		initializer = tf.contrib.layers.variance_scaling_initializer() #He initialization

		self.X = tf.placeholder(tf.float32, shape=[None,self.n_inputs])
		hidden = tf.layers.dense(self.X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
		logits = tf.layers.dense(hidden, n_outputs,kernel_initializer=initializer)
		outputs = tf.nn.sigmoid(logits) #probability of moving left

		p_left_and_right = tf.concat(axis=1, values=[outputs,1-outputs])
		threshold = tf.constant(0.5, shape=[1,1])
		
		self.action = tf.multinomial(tf.log(p_left_and_right),num_samples=1) #samples action from probability distribution
		
		y = 1 - tf.to_float(self.action) #probability of right
		
		#learning_rate = 0.1 
		cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2)
		grads_and_vars = self.optimizer.compute_gradients(cross_entropy)
		self.gradients = [grad for grad, var in grads_and_vars]
		self.gradient_placeholders = []
		grads_and_vars_feed = []
		for grad, var in grads_and_vars:
			gradient_placeholder = tf.placeholder(tf.float32,shape=grad.get_shape())
			self.gradient_placeholders.append(gradient_placeholder)
			grads_and_vars_feed.append((gradient_placeholder, var))
		self.training_op = self.optimizer.apply_gradients(grads_and_vars_feed)

		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		
		if not self.ignore_checkpoint:
			self.load_model()

	#rewards in the form of a 1D py array
	def discount_rewards(self, rewards, discount_rate):
		discounted_rewards = np.empty(len(rewards))
		cumulative_rewards = 0
		for step in reversed(range(len(rewards))):
			cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
			discounted_rewards[step] = cumulative_rewards
		return discounted_rewards

	#all_rewards in the form of a 2D py array
	def discount_and_normalize_rewards(self, all_rewards, discount_rate):
		all_discounted_rewards = [self.discount_rewards(rewards, discount_rate) for rewards in all_rewards]
		flat_rewards = np.concatenate(all_discounted_rewards) #converts py list to numpy array
		reward_mean = flat_rewards.mean()
		reward_std = flat_rewards.std()
		return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

	def use_manual(self):
		self.stop = False
		self.env.reset()
		
		def render():
			while not self.stop:
				if self.env is not None:
					self.env.render()

		def prompt():
			while not self.stop:
				ans = input("Left or right (0 or 1)? Exit with N ")
				if(ans=="0" or ans=="1"):
					self.env.step(int(ans))
				else:
					self.stop = True
					sys.exit()
		
		renderer = threading.Thread(target=render)
		prompter = threading.Thread(target=prompt)

		renderer.start()
		prompter.start()

	def load_model(self):
		#saver = tf.train.import_meta_graph("./my_policy_net_pg.ckpt.meta")
		self.saver.restore(self.sess,self.get_checkpoint_file())
	
	def get_checkpoint_file(self):
		return "./{}/policy_net.ckpt".format(self.checkpoint_dir)
	
	#execute reinforcement algorithm--training phase
	def train(self):
		#training phase
		#hyperparams
		#n_iterations = 1 #iters to train on
		#n_max_steps = 2000 #max steps per episode (to prevent infinite loop)
		#n_games_per_update = 10 #10 games per iter
		#save_iterations = 50 #save every 10 iters
		#discount_rate = 0.95
		#n_test_games = 50
		if self.ignore_checkpoint:
			self.init.run()
		for iteration in range(self.epochs):
			all_rewards = [] #all sequences of raw rewards for each episode
			all_gradients = [] #gradients saved at each step of each episode
			for game in range(self.games_per_update):
				print("Running game #{}".format(game))
				current_rewards = []
				current_gradients = []
				obs = self.env.reset()
				for step in range(self.max_steps):
					action_val, gradients_val = self.sess.run(
							[self.action,self.gradients],
							feed_dict={self.X:obs.reshape(1,self.n_inputs)}) #one observation
					obs, reward, done, info = self.env.step(action_val[0][0])
					current_rewards.append(reward) #raw reward
					current_gradients.append(gradients_val) #raw grads
					if done:
						print("Finished game #{} in {} steps".format(game,step+1))
						break
					elif step==self.max_steps-1:
						print("Hit max num of steps at game #{}".format(game))
				
				all_rewards.append(current_rewards) #adds to the history of rewards
				all_gradients.append(current_gradients) #gradient history
			#all games executed--time to perform policy gradient ascent
			print("Performing gradient ascent at iteration {}".format(iteration))
			all_rewards = self.discount_and_normalize_rewards(all_rewards, self.discount_rate)
			feed_dict = {}
			for var_index, grad_placeholder in enumerate(self.gradient_placeholders):
				#multiplication by the "action scores" obtained from discounting the future events appropriately--meaned to average the signals
				mean_gradients = np.mean(
					[reward*all_gradients[game_index][step][var_index] #iterates through each variable in the gradient (var_index)
						for game_index, rewards in enumerate(all_rewards)
						for step,reward in enumerate(rewards)],
					axis=0)
				feed_dict[grad_placeholder] = mean_gradients
			self.sess.run(self.training_op,feed_dict=feed_dict)
			if (iteration +1)% self.save_iterations == 0 and self.save:
				print("Saving model...")
				self.saver.save(self.sess,self.get_checkpoint_file())
		self.test()

	def test(self):
		steps = []
		for game in range(self.test_games):
			obs = self.env.reset()
			counter = 0
			for step in range(self.max_steps):
				action_val = self.sess.run(self.action, feed_dict={self.X:obs.reshape(1,self.n_inputs)})
				obs, reward, done, info = self.env.step(action_val[0][0])
				if(self.test_render):
					self.env.render()
				counter+=1
				if done or step==self.max_steps-1:
					pause = input("Press enter to continue")
					break
			steps.append(counter)
		steps_arr = np.array(steps,np.int32)
		mean = steps_arr.mean()
		std = steps_arr.std()
		print("Test mean: {} steps\nTest standard deviation: {} steps".format(mean,std))