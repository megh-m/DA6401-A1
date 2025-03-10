import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist

def load_data(dataset_name):
	if dataset_name == 'mnist':
		(X_train,y_train), (X_test,y_test) = mnist.load_data()
	elif dataset_name == 'fashion_mnist':
		(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
	X_train = X_train.reshape(X_train.shape[0],-1)/255 #Normalizing image data
	X_test = X_test.reshape(X_test.shape[0],-1)/255 #Normalizing image data
	y_train = pd.get_dummies(y_train).values #Encode targets as one-hot encodings to implement softmax
	y_test = pd.get_dummies(y_test).values #Encode targets as one-hot encodings to implement softmax
	return X_train, y_train, X_test, y_test

#Defining all activation functions globally
def sigmoid(x):
    # Numerically stable sigmoid by limiting values
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def ddx_sigmoid(output):
    return output * (1.0 - output)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def identity(x):
	return(x)
def ddx_identity(output):
	return np.ones_like(output)

def tanh(x):
	return np.tanh(x)
def ddx_tanh(output):
	return 1 - output**2

def relu(x):
	return np.maximum(0,x)
def ddx_relu(x):
	return (x>0).astype(float)


class Layer:
  def __init__(self, in_size, out_size, actv, weight_init = 'random'):
    self.in_size = in_size
    self.out_size = out_size
    self.bias = np.zeros((1, out_size))
    self.actv_name = actv
    if weight_init == 'random':
	    self.weights = np.random.randn(in_size, out_size)*0.01
    elif weight_init == 'xavier':
	    self.weights = np.random.randn(in_size, out_size)*np.sqrt(2/(in_size + out_size)) #Denominator sets Xavier Scale
    self.weight_momenta = np.zeros_like(self.weights)
    self.bias_momenta = np.zeros_like(self.bias)
  
  def fore_prop(self, input):
    self.input = input
    self.z = np.dot(input, self.weights) + self.bias
    if self.actv_name == 'sigmoid':
	    self.output = sigmoid(self.z)
    elif self.actv_name == 'tanh':
	    self.output = tanh(self.z)
    elif self.actv_name == 'relu':
	    self.output = relu(self.z)
    elif self.actv_name == 'identity':
	    self.output = identity(self.z)
    return self.output
  
  def back_prop(self, out_error, optimizer):
    if self.actv_name == 'sigmoid':
      delta = out_error*ddx_sigmoid(self.output)
    elif self.actv_name == 'tanh':
      delta = out_error*ddx_tanh(self.output)
    elif self.actv_name == 'relu':
      delta = out_error*ddx_relu(self.output)
    elif self.actv_name == 'identity':
      delta = out_error*ddx_identity(self.output)
    in_grad = np.dot(out_error, self.weights.T)
    weight_grad = np.dot(self.input.T, delta)
    bias_grad = np.sum(delta, axis=0, keepdims=True)
    if optimizer.weight_decay > 0:
	    weight_grad += optimizer.weight_decay*self.weights
    self.weights, self.weight_momenta = optimizer.update(self.weights, weight_grad, self.weight_momenta) #Optimizer returns weights & momenta
    self.bias, self.bias_momenta = optimizer.update(self.bias, bias_grad, self.bias_momenta)
    in_error = np.dot(delta, self.weights.T)
    return in_error
	  
class Optimizer:
  def __init__(self, eta, momentum = 0.9, beta=0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weight_decay = 0):
    self.eta = eta
    self.momentum = momentum
    self.beta = beta
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.weight_decay = weight_decay
    self.t = 0
    
  def update(self, weights, grad, momentum):#is essentially updating the weights as per gradient and gradient descent scheme
    raise NotImplementedError

class SGD(Optimizer):
	def update(self, weights, grad, momentum):
		return weights - self.eta*grad, None
class MomentumGD(Optimizer):
	def update(self, weights, grad, momentum):
		if momentum is None:
			momentum = np.zeros_like(weights)
		momentum = self.momentum*momentum + self.eta*grad
		return weights - momentum, momentum
class NAGD(Optimizer):
	def update(self, weights, grad, momentum):
		if momentum is None:
			momentum = np.zeros_like(weights)
		old_momentum = momentum
		momentum = self.momentum*momentum + self.eta*grad
		return weights - (1 + self.momentum) * momentum + self.momentum*old_momentum, momentum
class RMSprop(Optimizer):
	def update(self, weights, grad, momentum):
		if momentum is None:
			momentum = np.zeros_like(weights)
		momentum = self.beta*momentum + (1-self.beta)*np.square(grad)
		return weights - self.eta*grad/(np.sqrt(momentum) + self.epsilon) ,momentum
class Adam(Optimizer):
	def update(self, weights, grad, momentum):
		if isinstance(momentum,list):
			m,v = momentum
		else:
			m = np.zeros_like(weights)
			v = np.zeros_like(weights)
			momentum = [m,v]
		self.t += 1 #Time-steps
		m = self.beta1*m + (1-self.beta1)*grad
		v = self.beta2*v + (1-self.beta2)*np.square(grad)
		dm = m/(1-self.beta1**(self.t)) #Bias Corrections
		dv = v/(1-self.beta2**self.t)   #Bias Corrections
		return weights - self.eta*dm/(np.sqrt(dv) + self.epsilon), momentum
class NAdam(Optimizer): #Is essentially same as Adam but with Nesterov-like acceleration
	def update(self, weights, grad, momentum):
		if isinstance(momentum,list):
			m,v = momentum
		else:
			m = np.zeros_like(weights)
			v = np.zeros_like(weights)
			momentum = [m,v]
		self.t += 1 #Time-steps
		m = self.beta1*m + (1-self.beta1)*grad
		v = self.beta2*v + (1-self.beta2)*np.square(grad)
		dm = m/(1-self.beta1**(self.t))
		dv = v/(1-self.beta2**self.t)
		m_bar = (self.beta1*dm) + ((1-self.beta1)*grad/ (1-self.beta1**self.t)) #Nesterov Acceleration
		return weights - self.eta*m_bar / (np.sqrt(dv) + self.epsilon), momentum
class NN:
	def __init__(self, in_size, hidden, out_size, actv, weight_init, loss, optimizer):
		self.layers =[]
		self.loss_name = loss
		prev_size = in_size
		for size in hidden:
			self.layers.append(Layer(prev_size, size, actv, weight_init=weight_init ))
			prev_size = size
		self.layers.append(Layer(prev_size, out_size, actv, weight_init=weight_init)) #Last Layer before softmax layer
		self.optimizer = optimizer
	def fore_prop(self,X):
		for layer in self.layers:
			X=layer.fore_prop(X)
		output = softmax(X) #softmax layer
		return output
	def back_prop(self, X,y #To start gradient calculation from the last layer
		output = self.fore_prop(X) #Input-Output layer propagated output
		if self.loss_name == 'cross_entropy':
			grad = output - y #For BCE loss, output - target = loss
		if self.loss_name == 'mean_squared_error':
			grad = 2*(output - y)
		for layer in reversed(self.layers):
			grad = layer.back_prop(grad, self.optimizer) #Back-Pass thru Layers and update gradient value i.e. undergo gradient descent as per optimizer

	def calc_loss(self, X,y):
		output = self.fore_prop(X)
		if self.loss_name == 'cross_entropy':
			loss = -np.mean(np.sum(y*np.log(output + 1e-8), axis=1)) #epsilon to avoid log(0)
		elif self.loss_name == 'mean_squared_error':
			loss = np.mean(np.sum(np.square(output-y), axis =1))
		return loss
	
	def predict(self,X): #To be used after training is complete
		return self.fore_prop(X)
	
		
    

