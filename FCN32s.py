from FCN import FCN
import torch
import torch.nn as nn
import torch.optim as optim

class FCN32s(FCN):
	def __init__(self, n_classes):
		super().__init__()
		'''self.classifier = nn.Sequential(
			nn.Conv2d(512, 4096, 7, 1, 3),
			nn.Dropout(),
			nn.ReLU(),
			nn.Conv2d(4096, 4096, 1, 1),
			nn.Dropout(),
			nn.ReLU(),
			nn.Conv2d(4096, n_classes, 1, 1))'''
		self.classifier = nn.Sequential(
			nn.ConvTranspose2d(512, 256, 2, 2),
			nn.BatchNorm2d(256),
			nn.ConvTranspose2d(256, 128, 2, 2),
			nn.BatchNorm2d(128),
			nn.ConvTranspose2d(128, 64, 2, 2),
			nn.BatchNorm2d(64),
			nn.ConvTranspose2d(64, 32, 2, 2),
			nn.BatchNorm2d(32),
			nn.ConvTranspose2d(32, n_classes, 2, 2),
			nn.BatchNorm2d(n_classes),
			nn.Conv2d(n_classes, n_classes, 1)
		)
		self.optimizer = optim.SGD([
								{'params': list(self.upto_pool3.parameters())[0::2], 'lr':1e-4},
								{'params': list(self.pool3_to_pool4.parameters())[0::2], 'lr':1e-4},
								{'params': list(self.pool4_to_pool5.parameters())[0::2], 'lr':1e-4},
								{'params': list(self.classifier.parameters())[0::2], 'lr':1e-4},
								{'params': list(self.upto_pool3.parameters())[1::2], 'lr':2e-4},
								{'params': list(self.pool3_to_pool4.parameters())[1::2], 'lr':2e-4},
								{'params': list(self.pool4_to_pool5.parameters())[1::2], 'lr':2e-4},
								{'params': list(self.classifier.parameters())[1::2], 'lr':2e-4}
							], momentum=0.9, weight_decay=5**-4)

	def forward(self, x):
		height, width = x.shape[2:]
		x = self.pool4_to_pool5(self.pool3_to_pool4(self.upto_pool3(x)))
		x = self.classifier(x)
		x = self.pad(x, (height, width))
		#return torch.clamp(x, 0, 20)
		return x

	def save(self, path, epoch):
		torch.save({'upto_pool3': self.upto_pool3.state_dict(),
					'pool3_to_pool4': self.pool3_to_pool4.state_dict(),
					'pool4_to_pool5': self.pool4_to_pool5.state_dict(),
					'classifier': self.classifier.state_dict(),
					'optimizer': self.optimizer.state_dict(),
					'epoch': epoch
					}, path)
	
	def load(self, FCN_32s):
		chkpt = torch.load(FCN_32s)
		self.upto_pool3.load_state_dict(chkpt['upto_pool3'])
		self.pool3_to_pool4.load_state_dict(chkpt['pool3_to_pool4'])
		self.pool4_to_pool5.load_state_dict(chkpt['pool4_to_pool5'])
		self.classifier.load_state_dict(chkpt['classifier'])
		self.optimizer.load_state_dict(chkpt['optimizer'])
		return chkpt['epoch']
