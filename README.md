# Reinforcement Learning Balancer
Uses finite-distance REINFORCE policy gradients to perform gradient ascent.<br>
Rewarding details: for every time step that the algorithm has not failed (the pole has neither fallen nor will imminently fall) the algorithm is given a reward of +1.
<img src="balancer.gif"></img><br>
Requirements:<br>
TensorFlow<br>
NumPy<br>
OpenAI Gym<br>
