#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import csv
import sys


# In[2]:


def phi(y):
    phi = np.mean(y)
    return phi


# In[3]:


def sum1(y):
    sum=[]
    for i in range(len(y)):
        if (y[i]==0):
            sum.append(y[i])
    return len(sum)


# In[4]:


def sum2(y):
    sum=[]
    for i in range(len(y)):
        if (y[i]==1):
            sum.append(y[i])
    return len(sum)


# In[5]:


def mew_0(x,y):
    sum_0=sum1(y)
    arr=[0]*len(x[0])
    arr1=np.array(arr)
    arr2=arr1.astype(float)
    for i in range(len(x)):
        if y[i]==0:
            arr2+=x[i]
    return (arr2/sum_0).reshape(2,1)


# In[6]:


def mew_1(x,y):
    sum_1=sum2(y)
    arr=[0]*len(x[0])
    arr1=np.array(arr)
    arr2=arr1.astype(float)
    for i in range(len(x)):
        if y[i]==1:
            arr2+=x[i]

    return (arr2/sum_1).reshape(2,1)


# In[7]:


def covariance1(x,y,mew0):
    # mew_3 = mew_0(x,y)
    # mew_4 = mew_1(x,y)
    s=[]
    for i in range(len(x)):
        if y[i]==0:
            s.append(np.matmul(np.subtract(x[i].reshape(2,1),mew0),np.transpose(np.subtract(x[i].reshape(2,1),mew0))))
    return s    
    
    


# In[8]:


def covariance2(x,y,mew1):
    # mew_3 = mew_0(x,y)
    # mew_4 = mew_1(x,y)
    s=[]
    for i in range(len(x)):
        if y[i]==1:
            s.append(np.matmul(np.subtract(x[i].reshape(2,1),mew1),np.transpose(np.subtract(x[i].reshape(2,1),mew1))))
    return s   


# In[9]:


def sigma0(X,mew0):
    # mew_3 = mew_0(x,y)
    # mew_4 = mew_1(x,y)
    s=[]
    for i in range(len(X)):
            s.append(np.matmul(np.subtract(X[i].reshape(2,1),mew0),np.transpose(np.subtract(X[i].reshape(2,1),mew0))))
    return sum(s)/len(X)   
    


# In[10]:


def sigma1(X1,mew1):
    s=[]
    for i in range(len(X1)):
        s.append(np.matmul(np.subtract(X1[i].reshape(2,1),mew1),np.transpose(np.subtract(X1[i].reshape(2,1),mew1))))
    return sum(s)/len(X1)   


# In[11]:


def covariance_matrix(x,y,mew0,mew1):
    a=covariance1(x,y,mew0)
    b=covariance2(x,y,mew1)
    for p in b:
        a.append(p)

    return sum(a)/len(x)


# In[12]:


def prob_y(y):
    phi1 = phi(y)
    prob_y = []
    for i in range(len(y)):
        prob_y .append((phi1**y[i])*((1-phi1)**(1-y[i])))
    return prob_y


# In[13]:


def prob_of_y_given_1(y):
    s=[]
    for i in range(len(y)):
        if y[i]==1:
            s.append(y[i])
    return np.array(sum(s)/len(y))
        


# In[14]:


def prob_of_y_given_0(y):
    s=[]
    for i in range(len(y)):
        if y[i]==0:
            s.append(1)
    return np.array(sum(s)/len(y))
        


# In[15]:


def constant_term(x,cov):
    #cov = covariance_matrix(x,y)
    pi=math.pi
    #cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    r=det_cov**0.5
    s=(2*pi)**(len(x[0])/2)
    i=s*r
    p=1/i
    return np.array(p)


# In[16]:


def constant_term_0(x,sigma_0):
    # cov = sigma0(X,x,y,)
    pi=math.pi
    #cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(sigma_0)
    r=det_cov**0.5
    s=(2*pi)**(len(x[0])/2)
    i=s*r
    p=1/i
    return np.array(p)


# In[17]:


def constant_term_1(x,sigma_1):
    cov = sigma_1
    pi=math.pi
    #cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    r=det_cov**0.5
    s=(2*pi)**(len(x[0])/2)
    i=s*r
    p=1/i
    return np.array(p)


# In[18]:


def prob_of_x_given_y_equal_0(x,cov,mew0,z):
    probab=[]
    #cov = covariance_matrix(x,y)
    #mew0=mew_0(x,y)
    pi=math.pi
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    #z=constant_term(x,y)
    for i in range(len(x)):
           c=np.subtract((x[i].reshape(2,1)),mew0)
           p=np.matmul(np.transpose(c),cov_inv)
           w=np.matmul(p,c)
           d=w.reshape(1,)
           t=z*np.exp(-0.5*d)
           probab.append(t)         
    e=np.array(probab)
    probab_x_given_y=e.flatten()      
    return probab_x_given_y
        
    
    


# In[19]:


def prob_of_x_given_y_equal_0_quad(x,sigma_0,z1,mew0):
    probab=[]
    #cov0 = sigma0(X,x,y)
    #mew0=mew_0(x,y)
    pi=math.pi
    cov_inv = np.linalg.inv(sigma_0)
    det_cov = np.linalg.det(sigma_0)
    #z=constant_term_0(X,x,y)
    for i in range(len(x)):
           c=np.subtract((x[i].reshape(2,1)),mew0)
           p=np.matmul(np.transpose(c),cov_inv)
           w=np.matmul(p,c)
           d=w.reshape(1,)
           t=z1*np.exp(-0.5*d)
           probab.append(t)         
    e=np.array(probab)
    probab_x_given_y=e.flatten()      
    return probab_x_given_y


# In[21]:


def prob_of_x_given_y_equal_1(x,cov,mew1,z):
    probab=[]
    #cov = covariance_matrix(x,y)
    #mew1=mew_1(x,y)
    pi=math.pi
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    #z=constant_term(x,y)
    for i in range(len(x)):
           c=np.subtract((x[i].reshape(2,1)),mew1)
           p=np.matmul(np.transpose(c),cov_inv)
           w=np.matmul(p,c)
           d=w.reshape(1,)
           t=z*np.exp(-0.5*d)
           probab.append(t)         
    e=np.array(probab)
    probab_x_given_y=e.flatten()      
    return probab_x_given_y
        
    
    


# In[22]:


def prob_of_x_given_y_equal_1_quad(x,sigma_1,z2,mew1):
    probab=[]
    # cov = sigma1(X1,x,y)
    #mew1=mew_1(x,y)
    pi=math.pi
    cov_inv = np.linalg.inv(sigma_1)
    det_cov = np.linalg.det(sigma_1)
    # z=constant_term_1(X1,x,y)
    for i in range(len(x)):
           c=np.subtract((x[i].reshape(2,1)),mew1)
           p=np.matmul(np.transpose(c),cov_inv)
           w=np.matmul(p,c)
           d=w.reshape(1,)
           t=z2*np.exp(-0.5*d)
           probab.append(t)         
    e=np.array(probab)
    probab_x_given_y=e.flatten()      
    return probab_x_given_y
        


# In[23]:


# def probab_of_x(x,y):
#     probab_of_x_given_y_0 = prob_of_x_given_y_equal_0(x,y)
#     probab_of_x_given_y_1 = prob_of_x_given_y_equal_1(x,y)
#     probab_of_y_given_0 = prob_of_y_given_0(x,y)
#     probab_of_y_given_1 = prob_of_y_given_1(x,y)
#     p_x=[]
#     for i in range (len(x)):
#         pro = (probab_of_x_given_y_0[i] * probab_of_y_given_0) + (probab_of_x_given_y_1[i]*probab_of_y_given_1)
#         p_x.append(pro)
#     return np.array(p_x)    
        
    


# In[24]:


def probab_of_y_0_given_x(x,cov,mew0,z,t):
    p=prob_of_x_given_y_equal_0(x,cov,mew0,z)
    d=p*t
    return d
    
    


# In[25]:


def probab_of_y_0_given_x_quad(x,sigma_0,z1,mew0,t):
    p=prob_of_x_given_y_equal_0_quad(x,sigma_0,z1,mew0)
    
    e= p*t
    return e
    
    


# In[26]:


def probab_of_y_1_given_x(x,cov,z,u,mew1):
    p=prob_of_x_given_y_equal_1(x,cov,mew1,z)

    f=p*u
    return f
    
    


# In[27]:


def probab_of_y_1_given_x_quad(x,sigma_1,z2,mew1,u):
    p=prob_of_x_given_y_equal_1_quad(x,sigma_1,z2,mew1)
    #t=prob_of_y_given_1(y)
    m=p*u
    return m
    
    


# In[28]:


def hypo_linear(x,cov,mew0,mew1,z,t,u):
    pro_y_0 = probab_of_y_0_given_x(x,cov,mew0,z,t)
    pro_y_1 = probab_of_y_1_given_x(x,cov,mew1,z,u)
    #print(np.array(pro_y_0)+np.array(pro_y_1))
    l=[]
    for i in range(len(x)):
        if (pro_y_0[i]>pro_y_1[i]):
            l.append('0')
        else:
            l.append('1')
    return l        


# In[29]:


def hypo_quadratic(x,sigma_0,sigma_1,z1,z2,mew0,mew1,t,u):
    pro_y_0_quad = probab_of_y_0_given_x_quad(x,sigma_0,z1,mew0,t)
    pro_y_1_quad = probab_of_y_1_given_x_quad(x,sigma_1,z2,mew1,u)
    #print(np.array(pro_y_0)+np.array(pro_y_1))
    l=[]
    for i in range(len(x)):
        if (pro_y_0_quad[i]>pro_y_1_quad[i]):
            l.append('0')
        else:
            l.append('1')
    return l        


# In[30]:


def modilfied_data(x,y):
    X=[]
    X1=[]
    for i in range(len(x)):
        if y[i]==0:
            X.append(x[i])
        else:
            X1.append(x[i])
    return np.array(X),np.array(X1)


# In[31]:


def constant_linear_boundary(x,mew0,mew1,cov,phie):
    # mew0=mew_0(x,y)
    # mew1=mew_1(x,y)
    #cov=covariance_matrix(x,y)
    cov_inv = np.linalg.inv(cov)
    #phie=phi(y)
    c1=np.matmul(np.transpose(mew1),cov_inv)
    c2=np.matmul(c1,mew1)
    c3=c2.flatten()
    c4=np.matmul(np.transpose(mew0),cov_inv)
    c5=np.matmul(c4,mew0)
    c5.flatten()
    c=c3-c5
    constant = c+math.log(phie)-math.log(1-phie)
    return constant.flatten()
    


# In[32]:


def coeff_input(x,mew0,mew1,cov):
    # mew0=mew_0(x,y)
    # mew1=mew_1(x,y)
    # cov=covariance_matrix(x,y)
    cov_inv = np.linalg.inv(cov)
    c1=np.matmul(np.transpose(mew1),cov_inv)
    c2=c1.flatten()
    c3=np.matmul(np.transpose(mew0),cov_inv)
    c4=c3.flatten()
    coeff=c1-c3
    return coeff.flatten()
    
    


# In[44]:


def pred_test_out(sigma_0,sigma_1,z1,z2,mew0,mew1,t,u):
    df=pd.read_csv(sys.argv[2]+'/X.csv', header=None)

    #df1[0]
    #df=df[0].str.split("  ",expand=True,)
    x1=np.array(df)

    # df1=pd.read_csv(sys.argv[2]+'/Y.csv', header=None)
    # df2=np.array(df1)
    # for i in range(len(df2)):
    #     if df2[i] == 'Alaska':
    #         df2[i]=0
    #     elif df2[i]== 'Canada':
    #         df2[i]=1
    # y=df2.flatten() 
    sc=StandardScaler()
    x1=sc.fit_transform(x1)
    pro_y_0_quad = probab_of_y_0_given_x_quad(x1,sigma_0,z1,mew0,t)
    pro_y_1_quad = probab_of_y_0_given_x_quad(x1,sigma_1,z2,mew1,u)

    # X,X1 = modilfied_data(x,y)
    res=hypo_quadratic(x1,sigma_0,sigma_1,z1,z2,mew0,mew1,t,u)
    y_pred=res
    c=[]
    for i in range(len(y_pred)):
        if y_pred[i]=='0':
            c.append('Alaska')
        else:
            c.append('Canada')    
    temp=pd.DataFrame(c)
    temp.to_csv('result_4.txt',header=False, index=False)


# In[33]:

def quadratic_boundary(sigma_0,sigma_1,mew0,mew1):
    l1=[]
    l2=[]
    sigma_0_inv = np.linalg.inv(sigma_0)
    sigma_1_inv = np.linalg.inv(sigma_1)
    c= (-0.5)*np.subtract(sigma_0_inv,sigma_1_inv)
    d=np.matmul(np.transpose(mew0),sigma_0_inv)
    e=np.matmul(np.transpose(mew1),sigma_1_inv)
    f=np.subtract(d,e)
    print(f)
    g=np.matmul(np.matmul(np.transpose(mew0),sigma_0_inv),mew0)
    h=np.matmul(np.matmul(np.transpose(mew1),sigma_1_inv),mew1)
    c1=np.subtract(g,h) * (-0.5)
    det_sigma_0 = np.linalg.det(sigma_0)
    det_sigma_1 = np.linalg.det(sigma_1)
    j=math.log(det_sigma_1) - math.log(det_sigma_0)
    k=c1[0][0]+j
    ar1=np.linspace(-5,5,70)
    for i in ar1:
        aa = c[1][1]
        bb = c[1][0] * i + c[0][1]*i  + f[0][1]
        cc = c[0][0]*(i**2) + f[0][0]*i + k
        l1.append(i)
        l2.append((-bb + (bb**2 - 4*aa*cc)**0.5) /(2*aa))
    plt.plot(l1,l2)
    plt.xlim(-4,4)
    plt.ylim(-4,4)    





def boundary(x,mew0,mew1,cov,phie,X,X1):
    coeff=coeff_input(x,mew0,mew1,cov)
    constant=constant_linear_boundary(x,mew0,mew1,cov,phie)
    plt.xlim(x.min(),x.max())
    plt.ylim(x.min()-0.5,x.max())
    theta0 = np.arange(-5,5,1)
    theta7=[]
    for i in theta0:
        theta7.append(-(constant+coeff[0]*i)/coeff[1])
    plt.plot(theta0,theta7,'-') 
    #input_y_0,input_y_1=modilfied_data(x,y)
    plt.scatter(X[:,0], X[:,1], marker='*',label='Alska')
    plt.scatter(X1[:,0], X1[:,1], marker='+',label='Canada')
    plt.xlabel('Fresh Water')
    plt.ylabel('Marine Water')
    #plt.scatter(input_value[:,0], input_value[:,1], c = actual_output)
    plt.legend()
    plt.title("GDA")
    #plt.show()


# In[43]:


def main():
    df=pd.read_csv(sys.argv[1]+'/X.csv', header=None)
    #df1[0]
    #df=df[0].str.split("  ",expand=True,)
    x=np.array(df)
    df1=pd.read_csv(sys.argv[1]+'/Y.csv', header=None)
    df2=np.array(df1)
    for i in range(len(df2)):
        if df2[i] == 'Alaska':
            df2[i]=0
        elif df2[i]== 'Canada':
            df2[i]=1
    y=df2.flatten() 
    sc=StandardScaler()
    x=sc.fit_transform(x)
      
    mew0 = mew_0(x,y)
    phie=phi(y)
    mew1 = mew_1(x,y)
    # print(mew0)
    # print(mew1)
    cov=covariance_matrix(x,y,mew0,mew1)
    X,X1=modilfied_data(x,y)
    #print(len(X1))
    sigma_0 = sigma0(X,mew0)
    sigma_1 = sigma1(X1,mew1)
    #print(sigma_0)

    #print(sigma1(X1,mew1))

    #h=mew_0(x,y)
    z=constant_term(x,cov)
    z1=constant_term_0(x,sigma_0)
    z2=constant_term_0(x,sigma_1)
    #X,X1=modilfied_data(x,y)
    # cov=covariance_matrix(x,y)
    # X,X1=modilfied_data(x,y)
    cov_inv = np.linalg.inv(cov)
    t=prob_of_y_given_0(y)
    u=prob_of_y_given_1(y)
    # pro_y_0_quad = probab_of_y_0_given_x_quad(x,sigma_0,z1,mew0,t)
    # pro_y_1_quad = probab_of_y_0_given_x_quad(x,sigma_1,z2,mew0,u)
    '''print(hypo_linear(x,cov,mew0,mew1,z,t,u))      ## uncomment it to see value of hypothsesis obtained for linear and quadratic boundaries
    print(hypo_quadratic(x,sigma_0,sigma_1,z1,z2,mew0,mew1,t,u))'''
    #print(sigma_1)
    #print(d)
    #r=val(X,x,y)
    #print(r)
    tt=constant_linear_boundary(x,mew0,mew1,cov,phie)
    #boundary(x,mew0,mew1,cov,phie,X,X1)      #Plot for Linear Boundary
    #quadratic_boundary(sigma_0,sigma_1,mew0,mew1)      ##Plot for Quadratic boundary
    tr=coeff_input(x,mew0,mew1,cov)
    #plt.scatter(X[:,0], X[:,1], marker='*',label='Alska')
    #plt.scatter(X1[:,0], X1[:,1], marker='+',label='Canada')
    '''plt.xlabel('Fresh Water')
    plt.ylabel('Marine Water')
    plt.legend()
    plt.title("GDA")'''
    plt.show()
    #print(sigma_1)
    #print(tr)
    pred_test_out(sigma_0,sigma_1,z1,z2,mew0,mew1,t,u)
    
    # sigma_0=sigma0(X,x,y)
    # sigma_1=sigma1(X,x,y)
#     print(w)
#     print(m)
    
if __name__ == "__main__":
    main()    
    


# In[ ]:





# In[ ]:





# In[ ]:




