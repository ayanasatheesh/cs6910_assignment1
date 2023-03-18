#Importing all the necessary libraries and login to the wandb

import wandb
wandb.login(key='99ed1e6d8f514ee3823dec88049f21d48e678419')
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import argparse


#here W the overall weigth matrix of the model and is a 3D matrix which is a collection of 2D matrix for each layer
#b is the overall bias matrix which is a collection of bias vector for each layer

def random_initialisation(n_hidden,n_neurons,n_inputneurons,n_outputneurons):
  sd=0.5
  #initalising to small values as the initilaising has an important effect on training, and initialising to large values will not lead to good model
  W=[np.random.normal(0,sd,(n_neurons,n_inputneurons)) if i==0  else np.random.normal(0,sd,(n_neurons,n_neurons)) for i in range(n_hidden)]
  Wo=np.random.normal(0,sd,(n_outputneurons,n_neurons))
  W.append(Wo)
  b=[np.random.normal(0,sd,(n_outputneurons,1)) if i==n_hidden else np.random.normal(0,sd,(n_neurons,1)) for i in range(n_hidden+1)]

  return (W,b)


def xavier_initialisation(n_hidden,n_neurons,n_inputneurons,n_outputneurons):
  W=[np.random.randn(n_neurons,n_inputneurons)*np.sqrt(2/(n_neurons+n_inputneurons)) if i==0  else np.random.randn(n_neurons,n_neurons)*np.sqrt(2/(n_neurons+n_neurons)) for i in range(n_hidden)]
  Wo=np.random.randn(n_outputneurons,n_neurons)*np.sqrt(2/(n_outputneurons+n_neurons))
  W.append(Wo)
  b=[np.random.randn(n_outputneurons,1) if i==n_hidden else np.random.randn(n_neurons,1) for i in range(n_hidden+1)]

  return (W,b)


#initiliasing u matrix for both W and b as zero matrix for all the layers
def initialise_u(n_hidden,n_neurons,n_inputneurons,n_outputneurons):
  u_W=[np.zeros((n_neurons,n_inputneurons)) if i==0  else np.zeros((n_neurons,n_neurons)) for i in range(n_hidden)]
  u_Wo=np.zeros((n_outputneurons,n_neurons))
  u_W.append(u_Wo)
  u_b=[np.zeros((n_outputneurons,1)) if i==n_hidden else np.zeros((n_neurons,1)) for i in range(n_hidden+1)]

  return (u_W,u_b)    

def initialise_parameters(initialisation,n_hidden,n_neurons,n_inputneurons,n_outputneurons):

  if(initialisation=="random"):
    return random_initialisation(n_hidden,n_neurons,n_inputneurons,n_outputneurons)

  elif(initialisation=="xavier"):
    return xavier_initialisation(n_hidden,n_neurons,n_inputneurons,n_outputneurons)


def relu(x):
  return np.maximum(x,0)


def sigmoid(x):
  return 1. / (1.+np.exp(-x)) 

def softmax(x):
  e_X = np.exp(x - np.max(x, axis = 0))
  return e_X / (e_X.sum(axis = 0)+1e-8)   #small value added to prevent divide by zero case

def tanh(X):
  return np.tanh(X)


def activation(activation_function,a,n_neurons):

  if activation_function == 'sigmoid':
        h=sigmoid(a)

  elif activation_function == 'relu':
          h=relu(a)

  elif activation_function == 'tanh':
          h=tanh(a)

  return h



def forwardPropagation(W,b,n_neurons,n_layers,activation_function,X,n_outputneurons):
  ''' W, b are the parameters
  n_neurons is the number of nuerons in every hidden layer
  n_layers is the number of hidden layer
  X is the train dataset'''
  a=np.zeros((n_neurons,X.shape[1]))
  h=X
  H=[]
  A=[]

  for k in range(0,n_layers):
    a=b[k]+np.matmul(W[k],h)
    h=activation(activation_function,a,n_neurons)
    H.append(h)
    A.append(a)
  
    
 
  a=b[n_layers]+(np.matmul(W[n_layers],h)) # activation function for output layer

  y=softmax(a)
  H.append(y)
  A.append(a)
  
  #wrapping all the values as a dictionary
  return { "A" :A, "H":H, "y":y}



def one_hot_vector(Y_t,lenY):
  y_temp=np.zeros((10, lenY))
  for i in range(lenY):
    ans=Y_t[i]
    y_temp[ans][i]=1
  return y_temp



def element_wise_multiply(A,B):
  C=np.zeros(len(A))
  for i in range(len(A)):
    C[i]=A[i]*B[i]
  return C.T

def sigmoid_derivative(x):
  return sigmoid(x) * (1-sigmoid(x))

def tanh_derivative(x):
    return (1 - (np.tanh(x)**2))

def relu_derivative(x):
  return x>0


def find_derivative(activation_function,h,n_neurons):
  
  if(activation_function=="sigmoid"):
      g=sigmoid(h)*(1-sigmoid(h))

  elif(activation_function=="tanh"):
      g= tanh_derivative(h)

  elif(activation_function=="relu"):
      g= relu_derivative(h)
  
  return g
  
  

def backPropagation(H,A,n_neurons,n_layers,truelabel,n_classes,W,activation_function,X,y_one,wgt_dec,loss_fn):

  ''' W, b are the parameters
  n_neurons is the number of nuerons in every hidden layer
  n_layers is the number of hidden layer
  X is the train dataset
  wgt_dec is the weigth decay
  y_one is the one hot matrix'''

  gradient_a=H[n_layers]-y_one
  if(loss_fn=='mean_squared_error'):
    gradient_a*=H[n_layers]*(1-H[n_layers])
  G_W=[]
  G_b=[]
  
  for k in range(n_layers,-1,-1):
    
    if(k==0):
      gradient_W=np.matmul(gradient_a,X.T)+wgt_dec*W[k]/X.shape[1]
      G_W.append(gradient_W)
    else:
      gradient_W=np.matmul(gradient_a,H[k-1].T)+wgt_dec*W[k]/X.shape[1]
      G_W.append(gradient_W)
    gradient_b=np.sum(gradient_a, axis=1, keepdims=True) / X.shape[1]
    G_b.append(gradient_b)
    
    if(k>0):
      gradient_prevh=np.matmul(W[k].T,gradient_a)
      derivative_matrix=find_derivative(activation_function,H[k-1],n_neurons)
      gradient_a=gradient_prevh*derivative_matrix
    
 
  G_W=G_W[::-1]
  G_b=G_b[::-1]
  return {"g_W": G_W, "g_b": G_b}

#Accuracy function
def calc_accuracy(prd,Y):
  y_pred = np.argmax(prd["y"],axis=0)
  accuracy = np.sum(y_pred==Y)/Y.shape[0]
  return accuracy

#Stochastic gradient descent which when given a batch size act as mini batch gradient descent, when batch size=1 act as stochastic gradient descent, when batch size=whole datasize acts as gradient descent
def gradient_update_vanila(W,b,n_layers,grad,eta):

  for k in range(0,n_layers+1):
    W[k]-=eta*grad["g_W"][k]
    b[k]=b[k]-eta*grad["g_b"][k]

  return (W,b)

#calculating both training and validation loss
def calculateloss(y_t_predicted,batch_size,lenx,leny,Y,wgt_dec,n_layers,W,loss_fn,y_one,y_one_val,isvalidate=False,y_v_predicted=None,Y_val=None):

  if(loss_fn=='cross_entropy'):
    temp=np.arange(leny)
    y_t_predicted = np.maximum(y_t_predicted, 1e-10)
    l= sum(-np.log(y_t_predicted.T[temp,Y]))/lenx

  elif(loss_fn=='mean_squared_error'):
    l=np.mean(np.square(y_one - y_t_predicted))

  #regularisation
  mod=0
  if(wgt_dec!=0):
    for k in range(0,n_layers+1):
      mod+=np.sum(W[k]**2)

  training_loss= l+((wgt_dec*mod)/(2*lenx))

  #if validation loss is also to be calculated
  if(isvalidate):

    if(loss_fn=='cross_entropy'):
      temp=np.arange(6000)
      y_v_predicted = np.maximum(y_v_predicted, 1e-10)
      l= sum(-np.log(y_v_predicted.T[temp,Y_val]))/6000
    elif(loss_fn=='mean_squared_error'):
      l=np.mean(np.square(y_one_val - y_v_predicted))

    validation_loss=l+((wgt_dec*mod)/(2*6000))
    return training_loss,validation_loss

  return training_loss


#stochastic gradient descent
def modelfit_vanila(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,batch_size,wgt_dec,y_one,y_one_val,loss):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  Loss=[]
  v_loss=[]

  print("___________________________________________________________________")
  print("Running Stochastic Gradient Descent.....")
  for t in range(epoch):
    
    for i in range(0,X.shape[1],batch_size):
      y_f=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,trainData[:,i:i+batch_size],10)
      g=backPropagation(y_f["H"],y_f["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,trainData[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec,loss)
      
      (W,b)=gradient_update_vanila(W,b,n_layers,g,eta)
   
    #calculating model performance
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,loss,y_one,y_one_val,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #logging the values to wandb
    wandb.log({"validation_loss": validation_loss,"validation_accuracy": validation_accuracy,"training_loss": training_loss,"training_accuracy": training_accuracy, 'epoch': t})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)
  epochs=[i for i in range(epoch)]
  plotLoss(epochs,Loss,v_loss)
  return(W,b,Loss,v_loss)

#Plotting loss function
def plotLoss(epochs,Loss,v_loss):
  plt.plot(epochs,Loss,label="Training loss")
  plt.plot(epochs,v_loss,label="Validation loss")
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.show()

np.seterr(divide = 'ignore')


#Momentum based gradient descent
def gradient_update_momentum(W,b,n_layers,grad,eta,u_W,u_b,beta):
  for k in range(0,n_layers+1):
    u_W[k]=beta*u_W[k]+(1-beta)*grad["g_W"][k]
    W[k]=W[k]-(eta*u_W[k])
    u_b[k]=beta*u_b[k]+(1-beta)*grad["g_b"][k]
    b[k]=b[k]-(eta*u_b[k])

  return (u_W,W,u_b,b)


def modelfit_momentum(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,batch_size,wgt_dec,y_one,y_one_val,loss):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (u_W,u_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  Loss=[]
  v_loss=[]
  print("___________________________________________________________________")
  print("Running Mometum Based Gradient Descent.....")

  for t in range(epoch):
    
    for i in range(0,X.shape[1],batch_size):

      y=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,trainData[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,trainData[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec,loss)
      (u_W,W,u_b,b)=gradient_update_momentum(W,b,n_layers,g,eta,u_W,u_b,beta)
  

    #calculating model performance
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,loss,y_one,y_one_val,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #logging values to wandb
    wandb.log({"validation_loss": validation_loss,"validation_accuracy": validation_accuracy,"training_loss": training_loss,"training_accuracy": training_accuracy, 'epoch': t})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)
  epochs=[i for i in range(epoch)]
  plotLoss(epochs,Loss,v_loss)
  return (Loss,W,b,v_loss)



def gradient_update_nesterov(W,b,n_layers,grad,eta,u_W,u_b,beta):
  for k in range(0,n_layers+1):
    u_W[k]=beta*u_W[k]+eta*grad["g_W"][k]
    W[k]-=(u_W[k])
    u_b[k]=beta*u_b[k]+eta*grad["g_b"][k]
    b[k]-=(u_b[k])

  return (u_W,W,u_b,b)


#Nesterov Accelerated Gradient descent

def modelfit_nestrov(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,batch_size,wgt_dec,y_one,y_one_val,loss):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (u_W,u_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  lookahead_W=W.copy()
  lookahead_b=b.copy()
  Loss=[]
  v_loss=[]
  print("___________________________________________________________________")
  print("Running Nesterov Accelerated Based Gradient Descent.....")

  for t in range(epoch):
   
    sum=0
    loss=0
    for k in range(0,n_layers+1):
      u_W[k]=beta*u_W[k]
      u_b[k]=beta*u_b[k]
    for i in range(0,X.shape[1],batch_size):

      for k in range(0,n_layers+1):
        lookahead_W[k]=W[k]-u_W[k]
        lookahead_b[k]=b[k]-u_b[k]
      y=forwardPropagation(lookahead_W,lookahead_b,n_neurons,n_layers,activation_fn,trainData[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,lookahead_W,activation_fn,trainData[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec,loss)
      (u_W,W,u_b,b)=gradient_update_nesterov(W,b,n_layers,g,eta,u_W,u_b,beta)
      
    #calculating model performance 
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,loss,y_one,y_one_val,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #logging values to wandb
    wandb.log({"validation_loss": validation_loss,"validation_accuracy": validation_accuracy,"training_loss": training_loss,"training_accuracy": training_accuracy, 'epoch': t})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)
  epochs=[i for i in range(epoch)]
  plotLoss(epochs,Loss,v_loss)
  return (Loss,W,b,v_loss)


#RMSProp
def gradient_update_rmsprop(W,b,n_layers,grad,eta,v_W,v_b,beta,epsilon):
  for k in range(0,n_layers+1):
    v_W[k]=beta*v_W[k]+(1-beta)*(np.multiply(grad["g_W"][k],grad["g_W"][k]))
    W[k]-=(eta*grad["g_W"][k])/(np.sqrt(v_W[k])+epsilon)
    v_b[k]=beta*v_b[k]+(1-beta)*(np.multiply(grad["g_b"][k],grad["g_b"][k]))
    b[k]-=(eta*grad["g_b"][k])/(np.sqrt(v_b[k])+epsilon)

  return (v_W,W,v_b,b)




def modelfit_rmsprop(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,epsilon,batch_size,wgt_dec,y_one,y_one_val,loss):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (v_W,v_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  Loss=[]
  v_loss=[]
  print("___________________________________________________________________")
  print("Running RMSProp.....")

  for t in range(epoch):
    
    for i in range(0,X.shape[1],batch_size):

      y=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,trainData[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,trainData[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec,loss)
      (v_W,W,v_b,b)=gradient_update_rmsprop(W,b,n_layers,g,eta,v_W,v_b,beta,epsilon)
  
    #calculating model performance
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,loss,y_one,y_one_val,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #logging values to wandb
    wandb.log({"validation_loss": validation_loss,"validation_accuracy": validation_accuracy,"training_loss": training_loss,"training_accuracy": training_accuracy, 'epoch': t})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)
  epochs=[i for i in range(epoch)]
  plotLoss(epochs,Loss,v_loss)
  return (Loss,W,b,v_loss)

#Adam
def gradient_update_adam(W,b,n_layers,grad,eta,v_W,v_b,m_W,m_b,beta1,beta2,epsilon,t):
  for k in range(0,n_layers+1):
    m_W[k]=beta1*m_W[k]+(1-beta1)*grad["g_W"][k]
    m_b[k]=beta1*m_b[k]+(1-beta1)*grad["g_b"][k]

    v_W[k]=beta2*v_W[k]+(1-beta2)*grad["g_W"][k]**2
    v_b[k]=beta2*v_b[k]+(1-beta2)*grad["g_b"][k]**2

    m_W_correction=m_W[k]/(1-(beta1**t))
    m_b_correction=m_b[k]/(1-(beta1**t))

    v_W_correction=v_W[k]/(1-(beta2**t))
    v_b_correction=v_b[k]/(1-(beta2**t))

    W[k]-=(eta*m_W_correction)/(np.sqrt(v_W_correction)+epsilon)
    b[k]-=(eta*m_b_correction)/(np.sqrt(v_b_correction)+epsilon)

  return (m_W,v_W,W,m_b,v_b,b)




def modelfit_adam(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta1,beta2,epsilon,batch_size,wgt_dec,y_one,y_one_val,loss):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (v_W,v_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (m_W,m_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  v_loss=[]
  Loss=[]
  print("___________________________________________________________________")
  print("Running Adam.....")

  for t in range(1,epoch+1):
    
    for i in range(0,X.shape[1],batch_size):

      y=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,trainData[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,trainData[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec,loss)
      (m_W,v_W,W,m_b,v_b,b)=gradient_update_adam(W,b,n_layers,g,eta,v_W,v_b,m_W,m_b,beta1,beta2,epsilon,t)
  
    #calculating model performance
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,loss,y_one,y_one_val,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #logging values to wandb
    wandb.log({"validation_loss": validation_loss,"validation_accuracy": validation_accuracy,"training_loss": training_loss,"training_accuracy": training_accuracy, 'epoch': t})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)
  epochs=[i for i in range(epoch)]
  plotLoss(epochs,Loss,v_loss)
  return (Loss,W,b,v_loss)


#Nadam
def gradient_update_nadam(W,b,n_layers,grad,eta,v_W,v_b,m_W,m_b,beta1,beta2,epsilon,t):
  for k in range(0,n_layers+1):
    m_W[k]=beta1*m_W[k]+(1-beta1)*grad["g_W"][k]
    m_b[k]=beta1*m_b[k]+(1-beta1)*grad["g_b"][k]

    v_W[k]=beta2*v_W[k]+(1-beta2)*grad["g_W"][k]**2
    v_b[k]=beta2*v_b[k]+(1-beta2)*grad["g_b"][k]**2

    m_W_correction=m_W[k]/(1-(beta1**(t+1)))
    m_b_correction=m_b[k]/(1-(beta1**(t+1)))

    v_W_correction=v_W[k]/(1-(beta2**(t+1)))
    v_b_correction=v_b[k]/(1-(beta2**(t+1)))

    W[k]-=((eta)/(np.sqrt(v_W_correction)+epsilon))*(beta1*m_W_correction+((1-beta1)*grad["g_W"][k])/(1-beta1**(t+1)))
    b[k]-=(eta)/(np.sqrt(v_b_correction)+epsilon)*(beta1*m_b_correction+((1-beta1)*grad["g_b"][k])/(1-beta1**(t+1)))

  return (m_W,v_W,W,m_b,v_b,b)




def modelfit_nadam(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta1,beta2,epsilon,batch_size,wgt_dec,y_one,y_one_val,loss):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (v_W,v_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (m_W,m_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  Loss=[]
  v_loss=[]
  print("___________________________________________________________________")
  print("Running Nadam.....")

  for t in range(1,epoch+1):
    
    for i in range(0,X.shape[1],batch_size):

      y=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,trainData[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,trainData[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec,loss)
      (m_W,v_W,W,m_b,v_b,b)=gradient_update_nadam(W,b,n_layers,g,eta,v_W,v_b,m_W,m_b,beta1,beta2,epsilon,t)
  
    #calculating model performance
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,loss,y_one,y_one_val,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #logging values to wandb
    wandb.log({"validation_loss": validation_loss,"validation_accuracy": validation_accuracy,"training_loss": training_loss,"training_accuracy": training_accuracy, 'epoch': t})
  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)
  epochs=[i for i in range(epoch)]
  plotLoss(epochs,Loss,v_loss)
  return (Loss,W,b,v_loss)

#function which wraps up all the optimisers
def network_train(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,X,Y,batch_size,wgt_dec,optimiser,n_inputneurons,n_outputneurons,n_classes,beta,beta1,beta2,epsilon,y_one,y_one_val,loss):
  if(optimiser=="sgd"):
    (W,b,Loss,v_loss)=modelfit_vanila(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,batch_size,wgt_dec,y_one,y_one_val,loss)
  elif optimiser=="momentum":
    (Loss,W,b,v_loss)=modelfit_momentum(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,batch_size,wgt_dec,y_one,y_one_val,loss)
  elif optimiser=="nesterov":
    (Loss,W,b,v_loss)=modelfit_nestrov(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,batch_size,wgt_dec,y_one,y_one_val,loss)
  elif optimiser=="rmsprop":
    (Loss,W,b,v_loss)=modelfit_rmsprop(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,epsilon,batch_size,wgt_dec,y_one,y_one_val,loss)
  elif optimiser=="adam":
    (Loss,W,b,v_loss)=modelfit_adam(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta1,beta2,epsilon,batch_size,wgt_dec,y_one,y_one_val,loss)
  elif optimiser=="nadam":
    (Loss,W,b,v_loss)=modelfit_nadam(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta1,beta2,epsilon,batch_size,wgt_dec,y_one,y_one_val,loss)

  
  return (Loss,W,b,v_loss)

#function for getting arguments from command line
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-wp', '--wandb_project', type=str, default='assignment1')
    parser.add_argument('-we', '--wandb_entity', type=str, default='cs22m025')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-m', '--momentum', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0)
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, default=128)
    parser.add_argument('-a', '--activation', type=str, default='relu')
    return parser.parse_args()
    
#Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#took reference from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_conf_matrix(yt_pred,Y_test,classs_labels):
  conf_matrix = confusion_matrix(Y_test, yt_pred,normalize='true')
  cm=ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
  fig, ax = plt.subplots(figsize=(12, 12))
  cm.plot(ax=ax, cmap=plt.cm.Greens)
  wandb.log({ "confusion_matrix": wandb.Image(plt) })
  plt.show()

#Predicting the test data and plotting confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
def predict_test(X_test,Y_test,W,b,n_neurons,n_layers,activation_function,n_outputneurons,class_labels):
  test_f=[X_test[i].flatten() for i in range(X_test.shape[0])]
  test_f=np.transpose(test_f)
  yt_pred=forwardPropagation(W,b,n_neurons,n_layers,activation_function,test_f,n_outputneurons)
  test_acc=calc_accuracy(yt_pred,Y_test)
  y_pred = np.argmax(yt_pred["y"],axis=0)
  plot_conf_matrix(y_pred,Y_test,class_labels)
  
  return test_acc


if __name__ == "__main__":
    arg = getArgs()

    #storing the argments in required variables
    dataset=arg.dataset
    epoch=arg.epochs
    eta=arg.learning_rate
    n_layers=arg.num_layers
    activation_fn=arg.activation
    n_neurons=arg.hidden_size
    initialisation=arg.weight_init
    optimiser=arg.optimizer
    loss=arg.loss
    n_inputneurons=784
    n_outputneurons=10
    n_classes=10
    batch_size=arg.batch_size
    wgt_dec=arg.weight_decay
    beta=arg.beta
    beta1=arg.beta1
    beta2=arg.beta2
    epsilon=arg.epsilon
    project=arg.wandb_project
    entityname=arg.wandb_entity
    displayname="cs22m025"

    if(dataset=="fashion_mnist"):
      from keras.datasets import fashion_mnist

      (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
      class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    elif(dataset=="mnist"):
      from keras.datasets import mnist

      (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
      class_labels = [0,1,2,3,4,5,6,7,8,9]
    X_train=X_train/255


    X=X_train
    Y=Y_train

    from sklearn.model_selection import train_test_split
    X_train,X_validate,Y_train,Y_validate=train_test_split(X, Y, test_size=0.1, random_state=42)
    trainData=[X_train[i].flatten() for i in range(X_train.shape[0])]

    trainData=np.transpose(trainData)
    #X=trainData
    Y=Y_train
    xval_flat=[X_validate[i].flatten() for i in range(X_validate.shape[0])]
    xval=np.transpose(xval_flat)

    #defining one hot vectors for both train and validation data
    y_one=one_hot_vector(Y_train,Y_train.shape[0])
    y_one_val=one_hot_vector(Y_validate,Y_validate.shape[0])
   
    run=wandb.init(project="assignment1",entity=entityname,name=displayname)
    Loss,W,b,v_loss=network_train(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,trainData,Y,batch_size,wgt_dec,optimiser,n_inputneurons,n_outputneurons,n_classes,beta,beta1,beta2,epsilon,y_one,y_one_val,loss)
    

    #normalising the test data
    X_test=X_test/255
    #predicting on test data
    print("Accuracy on test data: ",predict_test(X_test,Y_test,W,b,n_neurons,n_layers,activation_fn,n_outputneurons,class_labels))
    run.finish()
   

