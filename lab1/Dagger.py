from pickletools import optimize
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import tree
from typing import List
from abc import abstractmethod
from random import random, sample, shuffle



class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass


# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
	def __init__(self, necessary_parameters=None):
		super(DaggerAgent, self).__init__()
		# init your model
		self.model = None

	# train your model with labeled data
	def update(self, data_batch, label_batch):
		self.model.train(data_batch, label_batch)

	# select actions by your model
	def select_action(self, data_batch):
		return 1
		label_predict = self.model.predict(data_batch)
		return label_predict



class MyDaggerAgent_Tree(DaggerAgent):
	def __init__(self):
		super().__init__()
		self.model = tree.DecisionTreeClassifier()
		self.update_times=0
		self.fitted = False

	def update(self, data_batch, label_batch):
		'''will use the old data and the new data to generate decision tree'''
		assert len(data_batch) == len(label_batch)
		self.update_times += 1
		data = np.array([i.reshape(-1) for i in data_batch])
		print(label_batch)
		print(data)
		self.model = self.model.fit(data,label_batch)
		self.fitted =True

	def select_action(self, data_batch):
		#data_batch or a data?if a data my change(seems that only a data)
		# return self.model.predict([data_batch])[0]
		if self.fitted == False:
			return sample([0,1,2,3,4,5],1)[0]
		return self.model.predict([data_batch.reshape(-1)])[0]

#a dict further used in data_preprocess
type2idx = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 11:6, 12:7}
idx2type = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:11, 7:12}


#define a cnn for model
class Cnn(nn.Module):
	def __init__(self) -> None:
		super(Cnn,self).__init__()
		# in_channels , out_channels , kernel_size
		self.conv1 = nn.Conv2d(3, 6, 10)
		self.conv2 = nn.Conv2d(6, 16, 10)
        # an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 45 * 33, 120)  # 5*5 from image dimension
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 8)

	def forward(self, x):
        # Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class DaggerDataset(Dataset):
	def __init__(self,data_batch:list,label_batch:list) -> None:
		super().__init__()

		#transfer the data from (210,160,3) to (3,160,210)
		transf = transforms.ToTensor()
		self.data_batch_t = [transf(data) for data in data_batch]

		#use 0-7 to represent 0,1,2,3,4,5,11,12 sequently
		self.label_batch_t = [type2idx[i] for i in label_batch] 
		assert len(data_batch) == len(label_batch)

	def __getitem__(self, index):
		x = torch.FloatTensor(self.data_batch_t[index])
		y = self.label_batch_t[index]
		return (x,y)

	def __len__(self):
		return len(self.label_batch_t)

class MyDaggerAgent(DaggerAgent):
	def __init__(self):
		super().__init__()
		self.model = Cnn()
		self.optimizer = torch.optim.SGD(self.model.parameters(),lr=0.001,momentum=0.9)
		self.loss_func = nn.CrossEntropyLoss(reduction="sum")
		self.n_epochs = 6 # default epoch: 6
		
		#try to load checkpoint
		self.epoch = 0
		self.model_path = "model.pt"
		try:
			checkpoint = torch.load(self.model_path)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.epoch = checkpoint['epoch']
			self.loss = checkpoint['loss']  #loss will always be 0
		except:
			pass


		

	def update(self, data_batch, label_batch):
		train_data = DaggerDataset(data_batch,label_batch)
		data_loader = DataLoader(train_data,batch_size=4)
		
		for epoch in range(self.n_epochs):
			sum_loss = 0
			train_num = 0
			
			self.model.train()
			for x,y in data_loader:
				self.optimizer.zero_grad()	#initialize the gradient
				pred = self.model(x)	#forward pass
				loss:Tensor = self.loss_func(pred,y)

				sum_loss += loss.detach()
				train_num += len(x)

				loss.backward()	#calculate the gradient
				self.optimizer.step()	#update the parameter
			print(f"Avg loss:{(sum_loss/train_num).item()}")

		torch.save({
			'epoch':self.epoch,
			'model_state_dict':self.model.state_dict(),
			'optimizer_state_dict':self.optimizer.state_dict(),
			'loss':0,
		},self.model_path)


		
	def select_action(self, data_batch):
		transf = transforms.ToTensor()
		data = transf(data_batch)
		prob = list(torch.squeeze(self.model(data.unsqueeze(0))))
		idx =  prob.index(max(prob)) #get the idx of type with max prob
		return idx2type[idx]