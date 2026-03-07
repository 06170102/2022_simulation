#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy import stats

# In[2]:


#定義題目給的function，並找出其最小值
def v(x1,x2):
    y=0.2+(x1**2)+(x2**2)-(0.1*np.cos(6*x1*math.pi))-(0.1*np.cos(6*x2*math.pi))
    return y


# simulation annealing algorithmn
# 
# step1 start x1(1)    x2(1) ~ U(-5,5)
# 
# step2  new_x1    new_x2(2) ~ U(-5,5)
# 
# a=min(p,1)
# 
# p=exp(-lambd*v(new_x1,new_x2))/exp(-lambd*v(x1,x2))
# 
# 
# 為何這樣設proposal ? 假設 x1,x2已經收斂到最小，v(x1,x2)極小， lambda>0 且不停加大。proposal 會變成
# 
# p == e**-(lambda*大)/e**-(lambda*極小) , p會變得極小 ，so 每筆 with 1-p機率 stay，stay可能性極大，收斂!!
# 
# then with a probability x1(i)  x2(i) =new_x1 new x_2
# 
# with 1-a probability x1(i)  x2(i) =x1(i-1) x_2(i-1)
# 
# step 3 重複step2直到收斂
# 
# step4 記錄每筆x1 x2 v(x1,x2) 找出 mode (x1,x2) 就是最小點

# In[3]:


#set initials 
x1_history=[] #模擬退火，記錄每筆的歷史，包括 x1 x2 result 
x2_history=[]
result=[]
x1=random.uniform(-5,5) #題目沒有給定 x1 x2 範圍 ，假定起始從U(-5,5)生成
x2=random.uniform(-5,5)


x1_history.append(x1)
x2_history.append(x2)
result.append(v(x1,x2)) 


# In[4]:


#===========================================================
for i in range(1,1000000):
    lambd=10*math.log(1+i)  #讓lambda逐漸加大，每經過1次迭代值越來越大
    new_x1=random.uniform(-5,5) #新的x1 x2也從 U(-5,5)
    new_x2=random.uniform(-5,5)
    p=math.exp(-lambd*v(new_x1,new_x2))/math.exp(-lambd*v(x1,x2)) #如何收斂至最小值，p必須每筆都極小，這樣會不停stay。
    a=min(p,1)
    u=random.uniform(0,1)
    if u<a:                # with a probability change to next state 
        x1=new_x1
        x2=new_x2
        x1_history.append(x1)
        x2_history.append(x2)
        result.append(v(x1,x2))
    else:
        x1=x1            # with 1-a probability change to next state
        x2=x2
        x1_history.append(x1)
        x2_history.append(x2)
        result.append(v(x1,x2))


# In[15]:


min(result) 


# In[14]:


print(stats.mode(x1_history)[0][0])
print(stats.mode(x2_history)[0][0])


# In[8]:


#畫圖形驗證結果 在x1 x2 接近0時，會得出v(x1,x2)最小值


# In[7]:


i1 = np.arange(-10, 10, 0.01)
i2 = np.arange(-10, 10, 0.01)

x1m, x2m = np.meshgrid(i1, i2)
fm = np.zeros(x1m.shape)

for i in range(x1m.shape[0]):
    for j in range(x1m.shape[1]):
        fm[i][j] = v(x1m[i][j], x2m[i][j])

plt.figure(figsize=(10, 10))
plt.contourf(x1m, x2m, fm, cmap='Blues')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:









# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




