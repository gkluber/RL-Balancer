# Reinforcement Learning Balancer
Uses REINFORCE policy gradients to perform gradient ascent.<br>
Rewarding details: for every time step that the algorithm has not failed (the pole has neither fallen nor will imminently fall) the algorithm is given a reward of +1.
<img src="balancer.gif"></img><br>
## Requirements:<br>
TensorFlow<br>
NumPy<br>
OpenAI Gym<br>
```batch
python -m pip install --upgrade tensorflow numpy gym 
```
## References:<br>
Williams, R.J. Mach Learn (1992) 8: 229. https://doi.org/10.1007/BF00992696<br>
Peters, Jan. “Policy Gradient Methods.” Scholarpedia, Jan Peters, 2010, www.scholarpedia.org/article/Policy_gradient_methods.
