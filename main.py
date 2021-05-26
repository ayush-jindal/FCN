from torchvision import transforms
from FCN32s import FCN32s
from FCN16s import FCN16s
from FCN8s import FCN8s
from trainer import TrainerModeller

# when training VGG too (fine tuning), will take a lot of time
# rather use pretrained VGG

if __name__ == '__main__':
	'''model = FCN32s(21)
	transforms = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406), 
													(0.229, 0.224, 0.225))
								])
	#trainer = TrainerModeller('/home/jindal/Desktop/Python/Datasets/VOC2011/', 'SemSeg_FCN32s.pth', transforms, model, (224, 224))
	trainer = TrainerModeller('/home/jindal/Desktop/Python/Datasets/VOC2011/', 'SemSeg_FCN32s.pth', transforms, model)
	del model
	#trainer.train()
	trainer.resume()
	#trainer.test()'''
	'''model = FCN16s('SemSeg_FCN32s.pth', 21)
	transforms = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406), 
													(0.229, 0.224, 0.225))
								])
	#trainer = TrainerModeller('/home/jindal/Desktop/Python/Datasets/VOC2011/', 'SemSeg_FCN16s.pth', transforms, model, (224, 224))
	trainer = TrainerModeller('/home/jindal/Desktop/Python/Datasets/VOC2011/', 'SemSeg_FCN16s.pth', transforms, model)
	del model
	#trainer.train()
	trainer.resume()
	#trainer.test()'''
	
	model = FCN8s('SemSeg_FCN16s.pth', 21)
	transforms = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406), 
													(0.229, 0.224, 0.225))
								])
	trainer = TrainerModeller('/home/jindal/Desktop/Python/Datasets/VOC2011/', 'SemSeg_FCN8s.pth', transforms, model)

	del model
	#trainer.train()
	trainer.resume()
	#trainer.test()
