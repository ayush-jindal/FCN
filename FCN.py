import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class FCN(nn.Module):
	def __init__(self):
		super().__init__()
		self.upto_pool3, self.pool3_to_pool4, self.pool4_to_pool5 = self.VGG16Segmentation()
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = None

	def VGG16Segmentation(self):
		'''
			returns torch.Sequential of VGG16 pretrained
		'''
		vgg_features = models.vgg16(pretrained=True).features
		
		# currently fine tuning VGG also
		for param in vgg_features.parameters():
			param.requires_grad = False
		
		upto_pool3 = vgg_features[:17]
		pool3_to_pool4 = vgg_features[17:24]
		pool4_to_pool5 = vgg_features[24:]
		
		return upto_pool3, pool3_to_pool4, pool4_to_pool5

	def pad(self, x, output_dim):
		hori_pad = output_dim[1] - x.shape[3]
		vert_pad = output_dim[0] - x.shape[2]
		left_pad = hori_pad//2
		right_pad = hori_pad - left_pad
		bottom_pad = vert_pad//2
		top_pad = vert_pad - bottom_pad
		return F.pad(x, (left_pad, right_pad, top_pad, bottom_pad))
