from FCN import FCN
import torch
import torch.nn as nn
import torch.optim as optim

class FCN8s(FCN):
	def __init__(self, FCN16s_model, n_classes):
		super().__init__()

		self.deconv2x_pool5 = nn.Sequential(
			nn.ConvTranspose2d(512, 512, 2, 2),
			nn.BatchNorm2d(512))
		self.deconv2x_sum = nn.Sequential(
			nn.ConvTranspose2d(512, 256, 2, 2),
			nn.BatchNorm2d(256))
		self.predicter_pool4 = nn.Sequential(
			nn.Conv2d(512, 512, 1, 1),
			nn.Dropout(),
			nn.ReLU())
		self.predicter_pool3 = nn.Sequential(
			nn.Conv2d(256, 256, 1, 1),
			nn.Dropout(),
			nn.ReLU())
		self.classifier = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 2, 2),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, 2, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, n_classes, 2, 2),
			nn.BatchNorm2d(n_classes),
			nn.ReLU(),
			nn.Conv2d(n_classes, n_classes, 1, 1)
		)
		self.optimizer = optim.SGD([
								{'params': list(self.upto_pool3.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.pool3_to_pool4.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.pool4_to_pool5.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.classifier.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.deconv2x_pool5.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.deconv2x_sum.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.predicter_pool4.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.predicter_pool3.parameters())[0::2], 'lr':1e-6},
								{'params': list(self.upto_pool3.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.pool3_to_pool4.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.pool4_to_pool5.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.classifier.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.deconv2x_pool5.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.deconv2x_sum.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.predicter_pool4.parameters())[1::2], 'lr':2e-6},
								{'params': list(self.predicter_pool3.parameters())[1::2], 'lr':2e-6},
							])
		self.initialize(FCN16s_model)

	def forward(self, x):
		height, width = x.shape[2:]
		pool3 = self.upto_pool3(x)
		pool4 = self.pool3_to_pool4(pool3)
		pool5 = self.pool4_to_pool5(pool4)
		predictions_pool4 = self.predicter_pool4(pool4)
		prediction_pool3 = self.predicter_pool3(pool3)
		out = self.deconv2x_sum(predictions_pool4 + self.pad(self.deconv2x_pool5(pool5), predictions_pool4.shape[2:]))
		out = self.pad(out, prediction_pool3.shape[2:]) + prediction_pool3
		out = self.classifier(out)
		x = self.pad(out, (height, width))
		return x
	
	def initialize(self, FCN16s_model):
		chkpt = torch.load(FCN16s_model)
		self.upto_pool3.load_state_dict(chkpt['upto_pool3'])
		self.pool4_to_pool5.load_state_dict(chkpt['pool4_to_pool5'])
		self.pool3_to_pool4.load_state_dict(chkpt['pool3_to_pool4'])
		self.deconv2x_pool5.load_state_dict(chkpt['deconv2x_pool5'])
		self.predicter_pool4.load_state_dict(chkpt['predicter_pool4'])
	
	def save(self, path, epoch):
		torch.save({'upto_pool3': self.upto_pool3.state_dict(),
					'pool3_to_pool4': self.pool3_to_pool4.state_dict(),
					'pool4_to_pool5': self.pool4_to_pool5.state_dict(),
					'deconv2x_pool5': self.deconv2x_pool5.state_dict(),
					'deconv2x_sum': self.deconv2x_sum.state_dict(),
					'predicter_pool4': self.predicter_pool4.state_dict(),
					'predicter_pool3': self.predicter_pool3.state_dict(),
					'classifier': self.classifier.state_dict(),
					'epoch': epoch,
					'optimizer': self.optimizer.state_dict()
					}, path)

	def load(self, FCN8s_model):
		chkpt = torch.load(FCN8s_model)
		self.upto_pool3.load_state_dict(chkpt['upto_pool3'])
		self.pool4_to_pool5.load_state_dict(chkpt['pool4_to_pool5'])
		self.pool3_to_pool4.load_state_dict(chkpt['pool3_to_pool4'])
		self.deconv2x_pool5.load_state_dict(chkpt['deconv2x_pool5'])
		self.deconv2x_sum.load_state_dict(chkpt['deconv2x_sum'])
		self.predicter_pool4.load_state_dict(chkpt['predicter_pool4'])
		self.predicter_pool3.load_state_dict(chkpt['predicter_pool3'])
		self.classifier.load_state_dict(chkpt['classifier'])
		self.optimizer.load_state_dict(chkpt['optimizer'])
		return chkpt['epoch']
