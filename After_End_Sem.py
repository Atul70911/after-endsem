#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv(r'C:\Users\Atul Kumar\Documents\roboism\after end sem\train.csv')
import numpy as np
import matplotlib.pyplot as plt



# In[2]:


df


# In[3]:


arr=df.values
print(arr.shape)

x_train=arr[:,0].reshape(699,1)
y_train=arr[:,1].reshape(699,1)
print(arr[213,:])
print(y_train.shape)



# #  Visualizing The Data

# In[4]:


plt.scatter(x_train,y_train,marker="*")
plt.show()


# # The Cost and Gradient Value Functions:

# In[5]:


def cost_function(x,y,w,b):
    m=np.size(x,axis=0)
    cost = 1/(2*m)*(np.sum((x@w + b - y)**2))
    return cost  


# In[6]:


def gradient_value(x,y,w,b):
    m=np.size(x,axis=0)
    error=np.dot(x,w)+b-y
    dj_dw=np.sum(np.dot(error.T,x))
    dj_db=np.sum(error)
    
    dj_dw_final=np.dot((np.divide(1,m)),dj_dw)
    dj_db_final=np.dot((np.divide(1,m)),dj_db)
    
    return dj_dw_final,dj_db_final
    


# # Now Making The Model:

# In[7]:


def model(x,y,w,b,cost,gradient,alpha,iters):
    
    j_history=[]
    f_history=[]
    m=np.size(y,axis=0)
    
    
    #f=function_output(x,w,b)
    
    for i in range(iters):

        f=np.dot(x,w)+b

        dj_dw,dj_db = gradient(x,y,w,b)   

        
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db  
        if i<m:
            f_history.append(f)
       
        
        if i<m: 
            j_history.append(cost(x,y,w,b))

        
        if i% np.ceil(iters / 100) == 0:
            print("Iteration ",[i],": Cost ",[j_history[-1]])
        
    return w, b, f_history, j_history 


# In[8]:


initial_w = np.zeros((1,1))
initial_b = 0
iterations = 1000
alpha = 0.000005

w_final, b_final,f_hist, j_hist = model(x_train, y_train,initial_w ,initial_b ,cost_function, gradient_value, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final},{w_final} ")
m=np.size(x_train,axis=0)
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final}, target value: {y_train[i]}")


# #  Train Data Accuracy

# In[9]:


def compute_function_output(x,w,b):
    y_hat=np.dot(x,w)+b
    
    return y_hat

y_hat_final=compute_function_output(x_train,w_final,b_final)


# In[10]:


def accuracy(y,y1):
    error=(1/(2*np.size(y,axis=0)))*np.sqrt((np.sum((y1-y)**2)))
    actual=(1-error)*100
    return actual


# In[11]:


train_accuracy=accuracy(y_train,y_hat_final)
print("the accuracy of train data set is",train_accuracy,"%")


# ## Test Data Accuracy 

# In[12]:


dff = pd.read_csv(r'C:\Users\Atul Kumar\Documents\roboism\after end sem\test.csv')


# In[13]:


test=dff.values
x_test=test[:,0].reshape(300,1)
y_test=test[:,1].reshape(300,1)


# In[14]:


y_hat_test=compute_function_output(x_test,w_final,b_final)
test_accuracy=accuracy(y_test,y_hat_test)
print("the accuracy of test data set is",test_accuracy,"%")

