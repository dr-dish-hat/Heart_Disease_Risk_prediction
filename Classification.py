import os 
import numpy as np 
import pandas as pd 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import imblearn 
from imblearn.over_sampling import SMOTE
import warnings 
warnings.filterwarnings('ignore')

#---- Function to convert np array to torch Tensor ----#

def DataToTensor(X, y):
	X_data = torch.from_numpy(X).float()
	y_data = torch.from_numpy(y).long()
	return {X_data, y_data}

#---- Neural Network definition class ----#

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(n_inputs, 2048)
		self.fc2 = nn.Linear(2048, 1024)
		self.fc3 = nn.Linear(1024, 512)
		self.fc4 = nn.Linear(512, 256)
		self.fc5 = nn.Linear(256, 64)
		self.fc6 = nn.Linear(64, 32)
		self.fc7 = nn.Linear(32, n_outputs)
		self.dropout = nn.Dropout(0.2)
		self.softmax = nn.Softmax(dim = 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.dropout(self.fc2(x)))
		x = F.relu(self.dropout(self.fc3(x)))
		x = F.relu(self.dropout(self.fc4(x)))
		x = F.relu(self.dropout(self.fc5(x)))
		x = F.relu(self.dropout(self.fc6(x)))
		x = self.fc7(x)
		x = self.softmax(x)
		return x 
		
#---- Hardware allocation ----#

if torch.cuda.is_available():
	dev = "cuda:0"
else:
	dev = "cpu"

#---- Data Loading ----#

# Data source - Kaggle Framingham Heart Study dataset

file_name = "framingham.csv"
df = pd.read_csv(file_name)
heart = df.dropna()
heart.rename(columns = {"male":"gender"}, inplace = "True")

#---- Getting features and targets from dataset ----#

# X - features
# y - targets

X = heart.drop('TenYearCHD', axis = 1)
y = heart.TenYearCHD

#---- Minority over sampling ----#

synthetic_over_sampler = SMOTE(random_state = 42)
X_res, y_res = synthetic_over_sampler.fit_resample(X, y)

#---- Dataset Preparation ----#

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2)

features_train, targets_train = DataToTensor(X_train, y_train)
features_test, targets_test = DataToTensor(X_test, y_test)

features_train = features_train.to(torch.device(dev))
features_test = features_test.to(torch.device(dev))
targets_train = targets_train.to(torch.device(dev))

#---- Model Inititalization ----#

n_inputs = 15
n_outputs = 2
neural_network = Net()
neural_network = neural_network.to(torch.device(dev))
optimizer = optim.Adam(neural_network.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

#---- Model Training ----#

for epochs in range(100000):
	optimizer.zero_grad()
	output = neural_network(features_train)
	loss = criterion(output, targets_train)
	loss.backward()
	optimizer.step()

	if epochs%100 == 0:
		print('Training epoch:', epochs, 'Loss', loss.item())

#---- Model Validation ----#

predict_out = neural_network(features_test)
_,out_y = torch.max(predict_out, 1)
out_y = torch.Tensor.cpu(out_y)

print('f1 score\n', f1_score(out_y, targets_test))
print('confusion matrix \n', confusion_matrix(out_y, targets_test))
print('Roc Acc Score \n', roc_auc_score(out_y, targets_test))


