# In[0]
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.special
from scipy.stats import multivariate_normal as mvn
from sklearn import svm
import sys
# In[1]
from abc import ABC, abstractmethod 
class MixtureModel(ABC):
    def __init__(self, Number_Of_Iterations, Number_Of_Components):
        self.iterations = Number_Of_Iterations
        self.sources = Number_Of_Components
        self.pi_ = None
        self.likelihood = None
        self.previous_likelihood = None
        
    @abstractmethod
    def _e_step(self):
        pass
    
    @abstractmethod
    def _m_step(self):
        pass
    
    @abstractmethod
    def predict_proba(self):
        pass
    
    @abstractmethod
    def _sample_component(self):
        pass
    
    def fit(self, X):
        likelihoods = []
        for i in range(0, self.iterations):     
            self.gamma = self._e_step(X)    
            self.pi_, self.likelihood  = self._m_step(X)
            likelihoods.append(self.likelihood)
        return np.asarray(likelihoods)
        
    def sample(self, Number_Of_Samples):
        sources = list(range(0,self.sources))
        probs = self.pi_.reshape(self.pi.shape[0],1)
        Z = np.random.choice(sources, Number_Of_Samples, probs.tolist())
        return  self._sample_component(Z)

class BernoulliMixtureModel(MixtureModel):
    def __init__(self, Number_Of_Clusters = 3, Number_Of_Iterations = 100):
        self.mu = None
        self.pi = np.ones((Number_Of_Clusters))
        self.pi = self.pi/3
        self.gamma = None
        self.sources = Number_Of_Clusters
        super().__init__(Number_Of_Iterations, Number_Of_Clusters)

    def _e_step(self, X):
        if type(self.mu) == type(None):
            self.mu = np.random.uniform(low = 0.0, high = 1.0, size=(self.sources, X.shape[1]))
        self.gamma = np.zeros((X.shape[0], self.pi.shape[0]))
        for k in range(self.sources):
            self.gamma[:,k] = self.pi[k] * self.calc_Bernoulli(self.mu[k,:], X)
        for sample in range(X.shape[0]):
            self.gamma[sample] = self.gamma[sample]/np.sum(self.gamma[sample])
        return self.gamma
           
    def _m_step(self, X):
     
        self.pi = np.mean(self.gamma, axis = 0)
        self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, axis = 0)[:,np.newaxis]
        loss = self.elbo(X)
        return self.pi, loss
   
    def Bern(self,mu, x):
        i = 0
        p = 1
        #print("called")
        for dimension in x:
            a = ((mu[i] ** dimension))
            b = [(1-mu[i]) ** (1 - dimension)]
            c = a[0] * b[0]
            p = c[0] * p
            i = i + 1
        return p

    def calc_Bernoulli(self,mu, X):
        mu = mu.reshape(mu.shape[0], 1)
        dist = []
        for x in X:
            dist.append(self.Bern(mu, x))
        dist=np.asarray(dist)
        dist = dist.reshape(dist.shape[0],)
        return dist

    def elbo(self, X):
        a = 1/sys.maxsize
        N = X.shape[0]
        C = self.gamma.shape[1]
        d = X.shape[1]
        self.likelihood = np.zeros((N, C))
        for c in range(C):
            self.likelihood[:,c] = self.gamma[:,c] * (np.log(2 * self.pi[c])+np.log(2*self.calc_Bernoulli(self.mu[c],X))-np.log(2*self.gamma[:,c]))
        self.likelihood = np.sum(self.likelihood)
        return self.likelihood
   
    def _sample_component(self, Z):
        samples = []
        for sample in range(Z.shape[0]):
            Cluster_Number = Z[sample]
            Mu = self.mu[Cluster_Number,:]
            samples.append(np.random.binomial(Mu.shape[0],Mu))
        return np.asarray(samples)
       

    def predict_proba(self, X):
        return self._e_step(X)


def get_data(digit1, digit2, digit3, num_samples = 15):

    Data = np.loadtxt('usps_noisy.csv', delimiter = ',')

    digit_1_indeces = np.where(Data[:,0] == digit1)[0][:num_samples]
    digit_2_indeces = np.where(Data[:,0] == digit2)[0][:num_samples]
    digit_3_indeces = np.where(Data[:,0] == digit3)[0][:num_samples]
    #digit_4_indeces = np.where(Data[:,0] == digit4)[0][:num_samples]

    
    digit_indeces = np.concatenate((digit_1_indeces, digit_2_indeces,  digit_3_indeces))

    relevant_data = Data[digit_indeces]

    return relevant_data



# In[2]
Data = get_data(0, 1, 7, 30)
np.random.shuffle(Data)

Train_Data = Data[0:65]
Test_Data = Data[65:]
X_Train = Train_Data[:, 1:]
Labels_Train = Train_Data[:, 0]
X_Test = Test_Data[:, 1:]
Labels_Test = Test_Data[:, 0]


# In[3]
Bernoulli_Mixture = BernoulliMixtureModel(Number_Of_Iterations=30, Number_Of_Clusters=3)
likelihoods = Bernoulli_Mixture.fit(X_Train)
plt.plot(likelihoods)

# %%
# In[4]
samples = Bernoulli_Mixture.sample(30)
for i in range(0, 30):
    Sample = samples[i].T
    plt.imshow(Sample.reshape(16, 16), cmap = 'gray')
    plt.show()
# %%
# In[30]

p_x_z_1 = Bernoulli_Mixture.calc_Bernoulli(Bernoulli_Mixture.mu[0], X_Train)
p_x_z_2 = Bernoulli_Mixture.calc_Bernoulli(Bernoulli_Mixture.mu[1], X_Train)
p_x_z_3 = Bernoulli_Mixture.calc_Bernoulli(Bernoulli_Mixture.mu[2], X_Train)
p_x_z_train = np.column_stack((p_x_z_1, p_x_z_2, p_x_z_3))
##calculating p(x|Z)

for i in range(p_x_z_train.shape[0]):
    p_x_z_train[i] = p_x_z_train[i]/p_x_z_train[i].sum()

Gram_Matrix = p_x_z_train @ (p_x_z_train * Bernoulli_Mixture.pi.T).T

p_x_z_1 = Bernoulli_Mixture.calc_Bernoulli(Bernoulli_Mixture.mu[0], X_Test)
p_x_z_2 = Bernoulli_Mixture.calc_Bernoulli(Bernoulli_Mixture.mu[1], X_Test)
p_x_z_3 = Bernoulli_Mixture.calc_Bernoulli(Bernoulli_Mixture.mu[2], X_Test)

p_x_z_test = np.column_stack((p_x_z_1, p_x_z_2, p_x_z_3))

for i in range(p_x_z_test.shape[0]):
    p_x_z_test[i] = p_x_z_test[i]/p_x_z_test[i].sum()

Gram_Matrix_Test = p_x_z_test @ (p_x_z_train * Bernoulli_Mixture.pi.T).T
 
model = svm.SVC(C = 100, kernel='precomputed', max_iter=1000) 
model.fit(Gram_Matrix, Labels_Train)
predictions = model.predict(Gram_Matrix_Test)

c = 0
for i in range(predictions.shape[0]):
    if predictions[i] == Labels_Test[i]:
        c = c + 1
print("accuracy is " + str(c/X_Test.shape[0]))

# %%



















# %%
