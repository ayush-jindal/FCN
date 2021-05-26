import torch
import json
import matplotlib.pyplot as plt
from utils import loadData, visualize
from time import perf_counter

class TrainerModeller():
	def __init__(self, img_folder, save_path, transforms, model, max_size=None):

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = model.to(self.device)
		self.epochs = 350
		self.transforms = transforms
		self.img_path = img_folder
		self.save_path = save_path
		self.palette = None

		self.max_size = max_size
		self.batch_size= 64

	def lossFn(self, output, target):
		# both of the form N*21*h*w
		return self.model.criterion(output, target)

	def train(self, epoch=0):
		traindata = loadData(self.img_path, 'train', self.max_size, self.transforms, self.batch_size)
		valdata = loadData(self.img_path, 'val', self.max_size, self.transforms, self.batch_size)
		train_dataiter = iter(traindata)
		val_dataiter = iter(valdata)
		prev_val_loss = None
		print('Training...')
		for epoch in range(epoch, self.epochs):
			train_loss = 0
			t_start = perf_counter()
			self.model.train()
			for img, gnd, _ in train_dataiter:
				img, gnd = img.to(self.device), gnd.to(self.device)
				pred = self.model(img)
				del img
				loss = self.lossFn(pred, gnd)
				del pred, gnd
				train_loss += float(loss)/len(_)
				self.model.optimizer.zero_grad()
				loss.backward()
				self.model.optimizer.step()
				del loss
			print(f'Epoch {epoch}: train loss: {train_loss}')
			if epoch%3 == 0:
				val_loss = 0
				self.model.eval()
				for img, gnd, _ in val_dataiter:
					img = img.to(self.device)
					gnd = gnd.to(self.device)
					output = self.model(img)
					del img
					loss = self.lossFn(output, gnd)
					del output, gnd
					val_loss += float(loss)/len(_)
					del loss
				print(f'Epoch {epoch}: validation loss: {val_loss}')
				if prev_val_loss is None or val_loss < prev_val_loss or epoch == self.epochs-1:
					self.model.save(self.save_path, epoch)
					print('Model saved...')
					prev_val_loss = val_loss
			print(f'Time elapsed: {perf_counter() - t_start}')

	def resume(self):
		curr_epoch = self.model.load(self.save_path)
		self.train(curr_epoch + 1)

	def test(self):
		self.model.load(self.save_path)
		with open('palette.json', 'r') as file:
			self.palette = json.load(file)
		testdata = loadData(self.img_path, 'Test', self.max_size, self.transforms, self.batch_size)
		test_dataiter = iter(testdata)
		self.model.eval()
		for img, nm in test_dataiter:
			img = img.to(self.device)
			pred = self.model(img)
			visualize(pred, nm, self.palette)

	#def try(self, img):
