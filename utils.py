import os
import json
import torch
import random
from PIL import Image
import numpy as np

class Iterator():
	
	def __init__(self, lists):
		self.combi = list(zip(*lists))
		random.shuffle(self.combi)
		self.length = len(self.combi)
		
	def __iter__(self):
		self.current = 0
		return self
	
	def __next__(self):
		if self.current < self.length:
			self.current += 1
			return self.combi[self.current-1]
		else:
			raise StopIteration
			self.current = 0

def load_img(path, transforms, max_size):
	img = Image.open(path).convert('RGB')
	if max_size:
		img = img.resize(max_size)
	return transforms(img).unsqueeze(0), path[:-4] + '_segmented_.png'

def loadData(img_path, mode, max_size, transforms, batch_size):
	folder = 'Test' if mode == 'Test' else 'TrainVal'
	file_path = img_path+f'{folder}/ImageSets/Segmentation/{mode}.txt'
	with open(file_path) as file:
		segmentation_img_list = list(map(lambda x: x[:-1]+'.{}', file.readlines()))#[:64]

	img_folder = img_path+f'{folder}/JPEGImages/'
	truth_folder = img_path+f'{folder}/SegmentationObject/'
	palette_exists = os.path.isfile('palette.json')
	imgs = []
	lbls = []
	names = []
	
	img_batch = None
	lbl_batch = None
	name_batch = None
	
	for img_name in segmentation_img_list:
		img = Image.open(img_folder+img_name.format('jpg')).convert('RGB')
		if max_size:
			img = img.resize(max_size)
			if img_batch is not  None:
				if len(img_batch) < batch_size:
					img_batch = torch.cat((img_batch, transforms(img).unsqueeze(0)))
					name_batch.append(img_name.format('png'))
				else:
					imgs.append(img_batch)
					names.append(name_batch)
					img_batch = transforms(img).unsqueeze(0)
					name_batch = [img_name.format('png')]
			else:
				img_batch = transforms(img).unsqueeze(0)
				name_batch = [img_name.format('png')]
		else:
			imgs.append(transforms(img).unsqueeze(0))
			names.append(img_name.format('png'))

		if mode != 'Test':
			lbl = Image.open(truth_folder+img_name.format('png')).convert('P')
			if not palette_exists:
				palette = lbl.getpalette()
				with open('palette.json', 'w') as file:
					json.dump(palette, file)
			if max_size:
				lbl = lbl.resize(max_size)
				if lbl_batch is not  None:
					if len(lbl_batch) < batch_size:
						lbl_batch = torch.cat((lbl_batch, getFeatureMaps(lbl).unsqueeze(0)))
					else:
						lbls.append(lbl_batch)
						lbl_batch = getFeatureMaps(lbl).unsqueeze(0)
				else:
					lbl_batch = getFeatureMaps(lbl).unsqueeze(0)
			else:
				lbls.append(getFeatureMaps(lbl).unsqueeze(0))

	if name_batch is not None:
		imgs.append(img_batch)
		lbls.append(lbl_batch)
		names.append(name_batch)
	del img_batch, lbl_batch, name_batch
	return Iterator([imgs, lbls, names]) if mode != 'Test' else Iterator([imgs, names])

def getFeatureMaps(img):
		#H*W*C
		img = np.array(img)
		img[img > 20] = 0
		#return torch.tensor(np.array(img).flatten('C'), dtype=torch.long)
		return torch.tensor(np.array(img), dtype=torch.long)
	
def visualize(output, name, palette):
	#1*21*h*w
	output = np.array(output.detach().max(dim=1)[0].squeeze(0).cpu()).clip(0, 20).astype(np.uint8)
	img = Image.fromarray(output, mode='P')
	img.putpalette(palette)
	img.save(name)
