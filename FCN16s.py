from FCN import FCN
import torch
import torch.nn as nn
import torch.optim as optim

class FCN16s(FCN):
	def __init__(self, FCN32s_model, n_classes):
		super().__init__()
		self.initialize(FCN32s_model)
		self.deconv2x_pool5 = nn.Sequential(
			nn.ConvTranspose2d(512, 512, 2, 2),
			nn.BatchNorm2d(512))
		self.predicter_pool4 = nn.Sequential(
			nn.Conv2d(512, 512, 1, 1),
			nn.Dropout(),
			nn.ReLU())

		'''self.classifier = nn.Sequential(
			nn.Conv2d(512, 4096, 7, 1, 3),
			nn.Dropout(),
			nn.ReLU(),
			nn.Conv2d(4096, 4096, 1, 1),
			nn.Dropout(),
			nn.ReLU(),
			nn.Conv2d(4096, n_classes, 1, 1),
			nn.ConvTranspose2d(n_classes, n_classes, 2, 2),
			nn.BatchNorm2d(n_classes),
			nn.ConvTranspose2d(n_classes, n_classes, 2, 2),
			nn.BatchNorm2d(n_classes),
			nn.ConvTranspose2d(n_classes, n_classes, 2, 2),
			nn.BatchNorm2d(n_classes),
			nn.ConvTranspose2d(n_classes, n_classes, 2, 2),
			nn.BatchNorm2d(n_classes),
			nn.Conv2d(n_classes, n_classes, 1)
		)'''
		self.classifier = nn.Sequential(

			nn.ConvTranspose2d(512, 256, 2, 2),
			nn.BatchNorm2d(256),
			nn.ConvTranspose2d(256, 128, 2, 2),
			nn.BatchNorm2d(128),
			nn.ConvTranspose2d(128, 64, 2, 2),
			nn.BatchNorm2d(64),
			nn.ConvTranspose2d(64, n_classes, 2, 2),
			nn.BatchNorm2d(n_classes),
			nn.Conv2d(n_classes, n_classes, 1)
		)
		self.optimizer = optim.SGD([
								{'params': list(self.upto_pool3.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.pool3_to_pool4.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.pool4_to_pool5.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.classifier.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.deconv2x_pool5.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.predicter_pool4.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.upto_pool3.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.pool3_to_pool4.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.pool4_to_pool5.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.classifier.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.deconv2x_pool5.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.predicter_pool4.parameters())[1::2], 'lr':2e-6}
							])
	
	def forward(self, x):
		height, width = x.shape[2:]
		pool4 = self.pool3_to_pool4(self.upto_pool3(x))
		pool5 = self.pool4_to_pool5(pool4)
		predictions = self.predicter_pool4(pool4)
		out = predictions + self.pad(self.deconv2x_pool5(pool5), predictions.shape[2:])
		out = self.classifier(out)
		out = self.pad(out, (height, width))
		return torch.clamp(out, 0, 20)
	
	def initialize(self, FCN32s_model):
		chkpt = torch.load(FCN32s_model)
		self.upto_pool3.load_state_dict(chkpt['upto_pool3'])
		self.pool4_to_pool5.load_state_dict(chkpt['pool4_to_pool5'])
		self.pool3_to_pool4.load_state_dict(chkpt['pool3_to_pool4'])
	
	def save(self, path, epoch):
		torch.save({'upto_pool3': self.upto_pool3.state_dict(),
					'pool3_to_pool4': self.pool3_to_pool4.state_dict(),
					'pool4_to_pool5': self.pool4_to_pool5.state_dict(),
					'deconv2x_pool5': self.deconv2x_pool5.state_dict(),
					'predicter_pool4': self.predicter_pool4.state_dict(),
					'classifier': self.classifier.state_dict(),
					'epoch': epoch,
					'optimizer': self.optimizer.state_dict()
					}, path)

	def load(self, FCN16s_model):
		print(FCN16s_model)
		chkpt = torch.load(FCN16s_model)
		self.upto_pool3.load_state_dict(chkpt['upto_pool3'])
		self.pool4_to_pool5.load_state_dict(chkpt['pool4_to_pool5'])
		self.pool3_to_pool4.load_state_dict(chkpt['pool3_to_pool4'])
		self.deconv2x_pool5.load_state_dict(chkpt['deconv2x_pool5'])
		self.predicter_pool4.load_state_dict(chkpt['predicter_pool4'])
		self.classifier.load_state_dict(chkpt['classifier'])
		self.optimizer.load_state_dict(chkpt['optimizer'])
		return chkpt['epoch']
