import wandb

wandb.login(key='99ed1e6d8f514ee3823dec88049f21d48e678419')
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import argparse




def random_initialisation(n_hidden,n_neurons,n_inputneurons,n_outputneurons):
  sd=0.5
  W=[np.random.normal(0,sd,(n_neurons,n_inputneurons)) if i==0  else np.random.normal(0,sd,(n_neurons,n_neurons)) for i in range(n_hidden)]
  Wo=np.random.normal(0,sd,(n_outputneurons,n_neurons))
  W.append(Wo)
  b=[np.random.normal(0,sd,(n_outputneurons,1)) if i==n_hidden else np.random.normal(0,sd,(n_neurons,1)) for i in range(n_hidden+1)]

  return (W,b)

'''def random_initialisation(n_hidden,n_neurons,n_inputneurons,n_outputneurons):
  #W1=np.random.randn(n_neurons,n_inputneurons)
  W=[np.random.randn(n_neurons,n_inputneurons) if i==0  else np.random.randn(n_neurons,n_neurons) for i in range(n_hidden)]
  Wo=np.random.randn(n_outputneurons,n_neurons)
  W.append(Wo)
  b=[np.random.randn(n_outputneurons) if i==n_hidden else np.random.randn(n_neurons) for i in range(n_hidden+1)]

  return (W,b)'''


def xavier_initialisation(n_hidden,n_neurons,n_inputneurons,n_outputneurons):
  W=[np.random.randn(n_neurons,n_inputneurons)*np.sqrt(2/(n_neurons+n_inputneurons)) if i==0  else np.random.randn(n_neurons,n_neurons)*np.sqrt(2/(n_neurons+n_neurons)) for i in range(n_hidden)]
  Wo=np.random.randn(n_outputneurons,n_neurons)*np.sqrt(2/(n_outputneurons+n_neurons))
  W.append(Wo)
  b=[np.random.randn(n_outputneurons,1) if i==n_hidden else np.random.randn(n_neurons,1) for i in range(n_hidden+1)]

  return (W,b)


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

'''def relu(x,n_neurons):
  return np.maximum(x,np.zeros(n_neurons,x.shape[0]))'''


'''def sigmoid(x):
  if(np.exp(-x)<=0):
    return 1.0/(1.0+np.exp(-x))
  else:
    return np.exp(x)/(1.0+np.exp(x))'''

def sigmoid(x):
  '''x=np.float128(x)
  res= 1 / (1 + np.exp(-x)) 
  return np.float64(res)'''
  return 1. / (1.+np.exp(-x)) 

def softmax(x):
  #return np.exp(x) / np.sum(np.exp(x), axis=0)
  e_X = np.exp(x - np.max(x, axis = 0))
  return e_X / (e_X.sum(axis = 0)+1e-8)

def tanh(X):
  return np.tanh(X)


def activation(activation_function,a,n_neurons):
 # h=np.zeros((batch_size,n_neurons))
 # a=a.T
  if activation_function == 'sigmoid':
        #for i in range(n_neurons):
          #for j in range(60000):
            #h[i]=sigmoid(a[i])
        h=sigmoid(a)
  elif activation_function == 'relu':
      #for i in range(n_neurons):
        #for j in range(60000):
          h=relu(a)
  elif activation_function == 'tanh':
      #for i in range(n_neurons):
        #for j in range(60000):
          h=tanh(a)
  return h



def forwardPropagation(W,b,n_neurons,n_layers,activation_function,X,n_outputneurons):
  a=np.zeros((n_neurons,X.shape[1]))
  h=X
  H=[]
  A=[]

  for k in range(0,n_layers):
    a=b[k]+np.matmul(W[k],h)
    h=activation(activation_function,a,n_neurons)
    #h=sigmoid(a)
    #print(a.shape)
    #print(h.shape)
    H.append(h)
    A.append(a)
    #print(a[0][0])
    
 # h=np.reshape(h,(1,n_neurons))
 
  a=b[n_layers]+(np.matmul(W[n_layers],h)) # activation function for output layer

  y=softmax(a)
  H.append(y)
  A.append(a)
  
  return { "A" :A, "H":H, "y":y}



def one_hot_vector(lenY):
  y_temp=np.zeros((10, lenY))
  for i in range(X_train.shape[0]):
    ans=Y_train[i]
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
  
  

def backPropagation(H,A,n_neurons,n_layers,truelabel,n_classes,W,activation_function,X,y_one,wgt_dec):
  #print(H[n_layers].shape)
  gradient_a=H[n_layers]-y_one
  G_W=[]
  G_b=[]
  
  for k in range(n_layers,-1,-1):
    '''if(k==n_layers):
      gradient_W=np.matmul(gradient_a,H[k-1].T)
      G_W.append(gradient_W)
    elif(k==0):
     
     # gradient_W=np.matmul(gradient_a,np.reshape(X,(X.shape[1],784)))
      gradient_W=np.matmul(gradient_a,X.T)
      G_W.append(gradient_W)
    else:
      gradient_W=np.matmul(gradient_a,H[k-1].T)
      G_W.append(gradient_W)'''
    if(k==0):
     
     # gradient_W=np.matmul(gradient_a,np.reshape(X,(X.shape[1],784)))
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
      #print(gradient_prevh.shape)
      #print(derivative_matrix.shape)
      gradient_a=gradient_prevh*derivative_matrix
    
  #print(G_W)
  G_W=G_W[::-1]
  G_b=G_b[::-1]
  return {"g_W": G_W, "g_b": G_b}

#g=backPropagation(y["H"],y["A"],32,3,9,10,W,"tanh",test,y_one,0.0005)
def calc_accuracy(prd,Y):
  y_pred = np.argmax(prd["y"],axis=0)
  accuracy = np.sum(y_pred==Y)/Y.shape[0]
  return accuracy*100

#print(calc_accuracy(prd,Y_train))
def gradient_update_vanila(W,b,n_layers,grad,eta):

  for k in range(0,n_layers+1):
    W[k]-=eta*grad["g_W"][k]
    b[k]=b[k]-eta*grad["g_b"][k]

  return (W,b)

def calculateloss(y_t_predicted,batch_size,lenx,leny,Y,wgt_dec,n_layers,W,isvalidate=False,y_v_predicted=None,Y_val=None):
  
  rows=np.arange(leny)
  l= sum(-np.log(y_t_predicted.T[rows,Y]))/lenx
  mod=0
  if(wgt_dec!=0):
    for k in range(0,n_layers+1):
      mod+=np.sum(W[k]**2)
  training_loss= l+((wgt_dec*mod)/(2*lenx))
  if(isvalidate):
    rows=np.arange(6000)
    l= sum(-np.log(y_v_predicted.T[rows,Y_val]))/6000
    validation_loss=l+((wgt_dec*mod)/(2*6000))
    return training_loss,validation_loss

  return training_loss








def modelfit_vanila(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,batch_size,wgt_dec,y_one):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  Loss=[]
  v_loss=[]

  print("___________________________________________________________________")
  print("Running Stochastic Gradient Descent.....")
  for t in range(epoch):
    
    for i in range(0,X.shape[1],batch_size):
      y_f=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,test[:,i:i+batch_size],10)
      #loss=calculateloss(y["y"],y_train_one_hot[:,i:i+batch_size],batch_size)
      g=backPropagation(y_f["H"],y_f["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,test[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec)
      #print(len(g["g_W"][2][0]))
      
      (W,b)=gradient_update_vanila(W,b,n_layers,g,eta)
   
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #wandb.log({"training_acc": training_accuracy, "validation_accuracy": validation_accuracy, "training_loss": training_loss, "validation cost": validation_loss, 'epoch': epoch})


  #epochs=[i for i in range(epoch)]
  #plotLoss(epochs,Loss)
  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)

  return(W,b,Loss)





#(W,b,Loss)=modelfit_vanila(10,1e-3,3,"tanh",64,"xavier",784,10,test,Y_train,10,32,0.0005)
def plotLoss(epochs,Loss):
  plt.plot(epochs,Loss)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.show()

np.seterr(divide = 'ignore')

def gradient_update_momentum(W,b,n_layers,grad,eta,u_W,u_b,beta):
  for k in range(0,n_layers+1):
    u_W[k]=beta*u_W[k]+(1-beta)*grad["g_W"][k]
    W[k]=W[k]-(eta*u_W[k])
    u_b[k]=beta*u_b[k]+(1-beta)*grad["g_b"][k]
    b[k]=b[k]-(eta*u_b[k])

  return (u_W,W,u_b,b)




def modelfit_momentum(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,batch_size,wgt_dec,y_one):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (u_W,u_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  Loss=[]
  v_loss=[]
  print("___________________________________________________________________")
  print("Running Mometum Based Gradient Descent.....")

  for t in range(epoch):
    
    for i in range(0,X.shape[1],batch_size):

      y=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,test[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,test[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec)
      (u_W,W,u_b,b)=gradient_update_momentum(W,b,n_layers,g,eta,u_W,u_b,beta)
  
    ''' y=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    loss=calculateloss(y["y"],y_one,batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W)
    Loss.append(loss)'''
    
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #wandb.log({"training_acc": training_accuracy, "validation_accuracy": validation_accuracy, "training_loss": training_loss, "validation cost": validation_loss, 'epoch': epoch})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)
    
   # print("epoch:",t,"loss:", loss)
    #wandb.log({"loss": loss, "epoch": epoch})

  #epochs=[i for i in range(epoch)]
  #plotLoss(epochs,Loss)
  return (Loss,W,b)

#(Loss,W,b)=modelfit_momentum(5,1e-3,4,"relu",64,"random",784,10,test,Y_train,10,0.9,10,0.0005)


def gradient_update_nesterov(W,b,n_layers,grad,eta,u_W,u_b,beta):
  for k in range(0,n_layers+1):
    u_W[k]=beta*u_W[k]+eta*grad["g_W"][k]
    W[k]-=(u_W[k])
    u_b[k]=beta*u_b[k]+eta*grad["g_b"][k]
    b[k]-=(u_b[k])

  return (u_W,W,u_b,b)



def modelfit_nestrov(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,batch_size,wgt_dec,y_one):

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
      y=forwardPropagation(lookahead_W,lookahead_b,n_neurons,n_layers,activation_fn,test[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,lookahead_W,activation_fn,test[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec)
      (u_W,W,u_b,b)=gradient_update_nesterov(W,b,n_layers,g,eta,u_W,u_b,beta)
      
    
     
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #wandb.log({"training_acc": training_accuracy, "validation_accuracy": validation_accuracy, "training_loss": training_loss, "validation cost": validation_loss, 'epoch': epoch})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)
    
    #wandb.log({"loss": loss, "epoch": epoch})

  #epochs=[i for i in range(epoch)]
  #plotLoss(epochs,Loss)
  return (Loss,W,b)


#(Loss,W,b)=modelfit_nestrov(5,1e-3,3,"relu",128,"xavier",784,10,test,Y_train,10,0.9,64,0.0005)



def gradient_update_rmsprop(W,b,n_layers,grad,eta,v_W,v_b,beta,epsilon):
  for k in range(0,n_layers+1):
    v_W[k]=beta*v_W[k]+(1-beta)*(np.multiply(grad["g_W"][k],grad["g_W"][k]))
    W[k]-=(eta*grad["g_W"][k])/(np.sqrt(v_W[k])+epsilon)
    v_b[k]=beta*v_b[k]+(1-beta)*(np.multiply(grad["g_b"][k],grad["g_b"][k]))
    b[k]-=(eta*grad["g_b"][k])/(np.sqrt(v_b[k])+epsilon)

  return (v_W,W,v_b,b)




def modelfit_rmsprop(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,epsilon,batch_size,wgt_dec,y_one):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (v_W,v_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  Loss=[]
  v_loss=[]
  print("___________________________________________________________________")
  print("Running RMSProp.....")

  for t in range(epoch):
    
    for i in range(0,X.shape[1],batch_size):

      y=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,test[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,test[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec)
      (v_W,W,v_b,b)=gradient_update_rmsprop(W,b,n_layers,g,eta,v_W,v_b,beta,epsilon)
  
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    #wandb.log({"training_acc": training_accuracy, "validation_accuracy": validation_accuracy, "training_loss": training_loss, "validation cost": validation_loss, 'epoch': epoch})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)

  #epochs=[i for i in range(epoch)]
  #plotLoss(epochs,Loss)
  return (Loss,W,b)

#(Loss,W,b)=modelfit_rmsprop(5,1e-4,3,"relu",128,"random",784,10,test,Y_train,10,0.9,1e-6,32,0.005)





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




def modelfit_adam(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta1,beta2,epsilon,batch_size,wgt_dec,y_one):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (v_W,v_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (m_W,m_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  v_loss=[]
  Loss=[]
  print("___________________________________________________________________")
  print("Running Adam.....")

  for t in range(1,epoch+1):
    
    for i in range(0,X.shape[1],batch_size):

      y=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,test[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,test[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec)
      (m_W,v_W,W,m_b,v_b,b)=gradient_update_adam(W,b,n_layers,g,eta,v_W,v_b,m_W,m_b,beta1,beta2,epsilon,t)
  
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    wandb.log({"training_acc": training_accuracy, "validation_accuracy": validation_accuracy, "training_loss": training_loss, "validation cost": validation_loss, 'epoch': epoch})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)
  epo=[i for i in range(epoch)]
  plotLoss(epo,Loss)
  return (Loss,W,b)

#(Loss,W,b)=modelfit_adam(10,1e-3,3,"tanh",32,"xavier",784,10,test,Y_train,10,0.9,0.999,1e-8,10,0.0005)

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




def modelfit_nadam(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta1,beta2,epsilon,batch_size,wgt_dec,y_one):

  (W,b)=initialise_parameters(initialisation,n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (v_W,v_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  (m_W,m_b)=initialise_u(n_layers,n_neurons,n_inputneurons,n_outputneurons)
  Loss=[]
  v_loss=[]
  print("___________________________________________________________________")
  print("Running Nadam.....")

  for t in range(1,epoch+1):
    
    for i in range(0,X.shape[1],batch_size):

      y=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,test[:,i:i+batch_size],10)
      g=backPropagation(y["H"],y["A"],n_neurons,n_layers,Y_train,n_classes,W,activation_fn,test[:,i:i+batch_size],y_one[:,i:i+batch_size],wgt_dec)
      (m_W,v_W,W,m_b,v_b,b)=gradient_update_nadam(W,b,n_layers,g,eta,v_W,v_b,m_W,m_b,beta1,beta2,epsilon,t)
  
    y_t=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,X,10)
    y_v=forwardPropagation(W,b,n_neurons,n_layers,activation_fn,xval,10)
    (training_loss,validation_loss)=calculateloss(y_t["y"],batch_size,len(X.T),len(Y),Y,wgt_dec,n_layers,W,True,y_v["y"],Y_validate)
    training_accuracy=calc_accuracy(y_t,Y_train)
    validation_accuracy=calc_accuracy(y_v,Y_validate)
    
    print("epoch:",t,"loss:", training_loss, validation_loss)
    Loss.append(training_loss)
    v_loss.append(validation_loss)
    wandb.log({"training_acc": training_accuracy, "validation_accuracy": validation_accuracy, "training_loss": training_loss, "validation cost": validation_loss, 'epoch': epoch})

  print("------------------------------------------------------------------")
  print("Accuracy on training data:",training_accuracy)
  print("Accuracy on validation data:",validation_accuracy)

  #epochs=[i for i in range(epoch)]
  #plotLoss(epochs,Loss)
  return (Loss,W,b)

#(Loss,W,b)=modelfit_nadam(5,1e-3,5,"relu",128,"random",784,10,test,Y_train,10,0.9,0.999,1e-8,32,0.0005)



def network_train(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,X,Y,batch_size,wgt_dec,optimiser,n_inputneurons,n_outputneurons,n_classes,beta,beta1,beta2,epsilon,y_one):
  if(optimiser=="sgd"):
    (W,b,Loss)=modelfit_vanila(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,batch_size,wgt_dec,y_one)
  elif optimiser=="momentum":
    (Loss,W,b)=modelfit_momentum(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,batch_size,wgt_dec,y_one)
  elif optimiser=="nesterov":
    (Loss,W,b)=modelfit_nestrov(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,batch_size,wgt_dec,y_one)
  elif optimiser=="rmsprop":
    (Loss,W,b)=modelfit_rmsprop(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta,epsilon,batch_size,wgt_dec,y_one)
  elif optimiser=="adam":
    (Loss,W,b)=modelfit_adam(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta1,beta2,epsilon,batch_size,wgt_dec,y_one)
  elif optimiser=="nadam":
    (Loss,W,b)=modelfit_nadam(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,beta1,beta2,epsilon,batch_size,wgt_dec,y_one)

  epochs=[i for i in range(epoch)]
  plotLoss(epochs,Loss)
  return (Loss,W,b)

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-wp', '--wandb_project', type=str, default='assignment1')
    parser.add_argument('-we', '--wandb_entity', type=str, default='cs22m025')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist')
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.000001)
    parser.add_argument('-w_d', '--weight_decay', type=float, default=.0)
    parser.add_argument('-w_i', '--weight_init', type=str, default='random')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1)
    parser.add_argument('-sz', '--hidden_size', type=int, default=4)
    parser.add_argument('-a', '--activation', type=str, default='sigmoid')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parseArguments()
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

    if(dataset=="fashion_mnist"):
      from keras.datasets import fashion_mnist

      (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    elif(dataset=="mnist"):
      from keras.datasets import mnist

      (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train=X_train/255


    X=X_train
    Y=Y_train

    from sklearn.model_selection import train_test_split
    X_train,X_validate,Y_train,Y_validate=train_test_split(X, Y, test_size=0.1, random_state=42)
    test=[X_train[i].flatten() for i in range(X_train.shape[0])]

    test=np.transpose(test)
    X=test
    Y=Y_train
    xval_flat=[X_validate[i].flatten() for i in range(X_validate.shape[0])]
    xval=np.transpose(xval_flat)
    y_one=one_hot_vector(Y_train.shape[0])
    wgt_dec=0
    beta=0.9
    beta1=0.9
    beta2=0.999
    epsilon=1e-8
    run=wandb.init(project="assignment1")
    #Loss,W,b=modelfit_vanila(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,batch_size,wgt_dec)
    Loss,W,b=network_train(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,X,Y,batch_size,wgt_dec,optimiser,n_inputneurons,n_outputneurons,n_classes,beta,beta1,beta2,epsilon,y_one)
    run.finish()




