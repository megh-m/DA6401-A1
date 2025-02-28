import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],-1)/255 #Normalizing image data
X_test = X_test.reshape(X_test.shape[0],-1)/255 #Normalizing image data
y_train = pd.get_dummies(y_train).values #Encode targets as one-hot encodings to implement softmax
y_test = pd.get_dummies(y_test).values #Encode targets as one-hot encodings to implement softmax
class Layer:
  def __init__(self, in_size, out_size, actv):
    self.weights = np.random.randn(in_size, out_size)*0.01
    self.bias = np.zeros((1, output_size))
    self.actv = actv
  def fore_prop(self, input):
    self.input = input
    self.output = self.actv(np.dot(input, self.weights) + self.bias)
    return self.output
  def back_prop(self, out_grad, eta, optimizer):
    in_grad = np.dot(out_grad, self.weights.T)
    weight_grad = np.dot(self.input.T, out_grad)
    bias_grad = np.sum(out_grad, axis=0, keepdims=True)
    self.weights, self.weight_momenta = optimizer.update(self.weights, weights_gradient, self.weight_momenta)
    self.bias, self.bias_momenta = optimizer.update(self.bias, bias_gradient, self.bias_momenta)
    return in_grad
class Optimizer:
  def __init__(self, eta, momentum = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    self.eta = eta
    self.momentum = momentum
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.t = 0
    
  def update(self, param, grad, momentum):
    raise NotImplementedError

class SGD(Optimizer):
	def update(self, param, grad, momentum):
		return param - self.eta*grad, None
class MomentumGD(Optimizer):
	def update(self, param, grad, momentum):
		if momentum is None:
			momentum = np.zeros_like(param)
		momentum = self.momentum*momentum + self.eta*grad
		return param - momentum, momentum
class NAGD(Optimizer):
	def update(self, param, grad, momentum):
		if momentum is None:
			momentum = np.zeros_like(param)
		old_momentum = momentum
		momentum = self.momentum*momentum + self.eta*grad
		return param - (1 + self.momentum) * momentum + self.momentum*old_momentum, momentum
class RMSprop(Optimizer):
	def update(self, param, grad, momentum):
		if momentum is None:
			momentum = np.zeros_like(param)
		momentum = self.beta1*momentum + (1-self.beta1)*np.square(grad)
		return param - self.eta*grad/(np.sqrt(momentum) + self.epsilon) ,momentum
		
    

