
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import seaborn as sns




data = pd.read_csv("ETH-INR.csv")
data.head()



data.shape



data.describe()


# In[8]:


data.info()


# In[9]:


data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day


# In[10]:


data.head()


# In[11]:


data.info()


# In[ ]:





# In[12]:


x=data['Date']
y=data['Close']

plt.figure(figsize=(15, 5))
plt.plot(x,y)
plt.title('ETH Close price.', fontsize=15)#ADJUSTED CLOSE
plt.ylabel('Price in INR.')

plt.xlabel('Date')
plt.show()


# In[13]:


data[data['Close'] == data['Adj Close']].shape, data.shape


# In[14]:


data = data.drop(['Adj Close'], axis=1)


# In[15]:


X = data[['Year', 'Month', 'Day', 'Open', 'High', 'Low']]
y = data['Close']
train_size = int(0.8 * len(data))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]


# In[16]:


#standardize the feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))


# In[31]:


# Define the RBF kernel
rbf_kernel = RBF(length_scale=1.0)


# In[33]:


# Initialize Gaussian process regression model with RBF kernel
model = GaussianProcessRegressor(kernel=rbf_kernel, alpha=0.1, normalize_y=True)

# Fit the model to the data
model.fit(X_train_scaled, y_train_scaled)

# Predict Ethereum prices for new data points
y_pred, sigma = model.predict(X_test_scaled, return_std=True)


# In[34]:


model.fit(X_train_scaled, y_train_scaled)


# In[41]:


y_pred, sigma = model.predict(X_test_scaled,return_std=True)
y_pred_train, sigma = model.predict(X_train_scaled,return_std=True)


# In[45]:


mse = mean_squared_error(y_test_scaled, y_pred)
print("Mean Squared Error:", mse)


# In[43]:


r2 = r2_score(y_train_scaled,y_pred_train)
print('R-squared',r2)


# In[44]:


r3 = r2_score(y_test_scaled, y_pred)
print('R-squared',r3)


# In[ ]:





# In[60]:


model.fit(X_train_scaled, y_train)


# In[65]:


y_pred, sigma = model.predict(X_test_scaled,return_std=True)
y_pred_train, sigma = model.predict(X_train_scaled,return_std=True)


# In[ ]:





# In[66]:


plt.figure(figsize=(16,6))
plt.plot(data["Date"][len(y_train):],y_test.values, label='Actual',color='blue')
plt.plot(data["Date"][len(y_train):],y_pred, label='Predicted',color='red')
plt.xlabel('TIME')
plt.ylabel('PRICE-INR')
plt.title('ETH Price Prediction')
plt.legend()
#plt.grid()
plt.show()


# In[61]:


plt.figure(figsize=(16,6))
plt.plot(data["Date"][:len(y_train)],y_train.values, label='Actual',color='blue')
plt.plot(data["Date"][:len(y_train)],y_pred_train, label='Predicted',color='red')
plt.xlabel('TIME')
plt.ylabel('PRICE-INR')
plt.title('ETH Price Prediction')
plt.legend()
#plt.grid()
plt.show()


# In[52]:


plt.figure(figsize=(16,6))
plt.plot(data["Date"][len(y_train):],y_test.values, label='TEST-Actual',color='red')
plt.plot(data["Date"][len(y_train):],y_pred, label='TEST-Predicted',color='blue')
plt.plot(data["Date"][:len(y_train)],y_train.values, label='TRAIN-Actual',color='green')
plt.plot(data["Date"][:len(y_train)],y_pred_train, label='TRAIN-Predicted',color='orange')
plt.xlabel('TIME')
plt.ylabel('PRICE-INR')
plt.title('ETH Price Prediction')
plt.legend()
#plt.grid()
plt.show()


# In[72]:


correlation_matrix = data.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Ethereum Dataset')
plt.show()


# In[ ]:




