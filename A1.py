import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],-1)/255 #Normalizing image data
X_test = X_test.reshape(X_test.shape[0],-1)/255 #Normalizing image data
y_train = pd.get_dummies(y_train).values #Encode targets as one-hot encodings to implement softmax
y_test = pd.get_dummies(y_test).values #Encode targets as one-hot encodings to implement softmax

def sigmoid(x):
    # Numerically stable sigmoid
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def ddx_sigmoid(output):
    return output * (1.0 - output)

class Layer:
  def __init__(self, in_size, out_size, actv, actv_name='sigmoid'):
    self.weights = np.random.randn(in_size, out_size)*0.01
    self.bias = np.zeros((1, output_size))
    self.actv = actv
    self.actv_name = actv_name
  def fore_prop(self, input):
    self.input = input
    self.z = np.dot(input, self.weights) + self.bias
    if self.actv_name == 'sigmoid':
       self.output = self.sigmoid(self.z)
    else:
       self.output = self.actv(self.z)
    return self.output
  def back_prop(self, out_error, eta, optimizer):
    #in_grad = np.dot(out_grad, self.weights.T)
    delta = out_error*ddx_sigmoid(self.output)
    weight_grad = np.dot(self.input.T, delta)
    bias_grad = np.sum(delta, axis=0, keepdims=True)
    self.weights, self.weight_momenta = optimizer.update(self.weights, weights_gradient, self.weight_momenta)
    self.bias, self.bias_momenta = optimizer.update(self.bias, bias_gradient, self.bias_momenta)
    in_error = np.dot(delta, self.weights.T)
    return in_error
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
class Adam(Optimizer):
	def update(self, param, grad, momentum):
		if momentum is None:
			momentum = [np.zeros_like(param), np.zeros_like(param)]
		self.t += 1 #Time-steps
		momentum[0] = self.beta1*momentum[0] + (1-self.beta1)*grad
		momentum[1] = self.beta2*momentum + (1-seld.beta2)*np.square(grad)
		dm = momentum[0]/(1-self.beta1**(self.t))
		dv = momentum[1]/(1-self.beta2**self.t)
		return param - self.eta*dm/(np.sqrt(dv) + self.epsilon), momentum
class NAdam(Optimizer):
	def update(self, param, grad, momentum):
		if momentum is None:
			momentum = [np.zeros_like(param), np.zeros_like(param)]
		self.t += 1 #Time-steps
		momentum[0] = self.beta1*momentum[0] + (1-self.beta1)*grad
		momentum[1] = self.beta2*momentum + (1-seld.beta2)*np.square(grad)
		dm = momentum[0]/(1-self.beta1**(self.t))
		dv = momentum[1]/(1-self.beta2**self.t)
		m_bar = (self.beta1*dm) + ((1-self.beta1)*grad/ (1-self.beta1**self.t))
		return param - self.eta*m_bar / (np.sqrt(dv) + self.epsilon), momentum
class NN:
	def __init__(self, in_size, hidden, out_size, optimizer):
		self.layers =[]
		prev_size = in_size
		for size in hidden:
			self.layers.append(Layer(prev_size, size, self.sigmoid))
			prev_size = size
		self.layers.append(Layer(prev_size, out_size, self.softmax))
		self.optimizer = optimizer
	def fore_prop(self,X):
		for layer in self.layers:
			X=layer.fore_prop(X)
		return X
	def back_prop(self, X,y):
		output = self.fore_prop(X)
		grad = output-y
		for layer in reversed(self.layers):
			gardient = layer.back_prop(grad, self.optimizer)
	def train(self, X,y, epochs, batch):
		for epoch in range(epochs):
			for i in range(0, len(X), batch): 
				x_batch = X[i:i+batch]
				y_batch = y[i:i+batch]
				self.back_prop(X_batch, y_batch)
			if epoch%10==0: #To print accuracy every 10 epochs
				acc = np.mean(np.argmax(self.fore_prop(X), axis=1) == np.argmax(y, axis=1)) #Create truth boolean & Take average
				print(f"Epoch {epoch}, Accuracy: {acc:.4f}")
	def predict(self,X):
		return self.fore_prop(X)
#Implementation
in_size = 784
hidden = [64,64] #2 layers with 64 neurons each
out_size = 10 #10 classes of Fashion_MNIST
optimizer = SGD(eta=0.001)
nn = NN(in_size,hidden,out_size,optimizer)
nn.train(X_train, y_train, epochs=100, batch=32)
test_acc = np.mean(np.argmax(nn.predict(X_test), axis=1) == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {test_acc:.4f}")
	
		
    

