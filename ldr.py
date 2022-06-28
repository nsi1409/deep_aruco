import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.models as md
import torch.nn as nn
import torch.optim as optim


convert_tensor = transforms.ToTensor()

f = open('aruco_v0.0.json')
data = json.load(f)
f.close()

data_indexed = list(data["_via_img_metadata"].values())
number_tags = 2

def index_load(i):
	label_ten = torch.zeros(9*number_tags)
	path = data_indexed[i]["filename"]
	for j in range(len(data_indexed[i]["regions"])):
		try:
			x = data_indexed[i]["regions"][j]["shape_attributes"]["cx"]
			y = data_indexed[i]["regions"][j]["shape_attributes"]["cy"]
			corner = int(data_indexed[i]["regions"][j]["region_attributes"]["aruco"])
			#print(f'corner {corner}')
			label_ten[(9*(corner//10)) + (2*(corner%10)+1)] = float(x)
			label_ten[(9*(corner//10)) + (2*(corner%10)+2)] = float(y)
			label_ten[9*(corner//10)] = float(1)
		except:
			#print('empty 0 input')
			pass

	img = Image.open(f'imgs/{path}')
	img_ten = convert_tensor(img)
	return img_ten, label_ten


class CustomDataset(torch.utils.data.Dataset):
	def __init__(self):
		return

	def __len__(self):
		return len(data_indexed)

	def __getitem__(self, idx):
		return index_load(idx)


dataset = CustomDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0)


res = md.resnet50(pretrained=True)
model = nn.Sequential(res, nn.Linear(1000, number_tags*9))
#print(res)


inpt, _ = dataset[0]
resize = transforms.Resize(224)
inpt = resize(inpt).unsqueeze(0)
#print(f'inpt: {inpt}')

#print(f'resnet50 test: {model(inpt)}')


def aruco_loss(test, base):
	loss = 0
	loss_ten = torch.tensor([0.0])
	for i in range(base.size(dim=0)):
		base_class = base[i][0].item()
		test_class = test[i][0].item()

		base_class_ten = base[i][0]
		test_class_ten = test[i][0]

		loss += (base_class - test_class) ** 2
		loss_ten += (base_class_ten - test_class_ten) ** 2

		run = 0
		run_ten = torch.tensor([0.0])
		for j in range(base.size(dim=1) - 1):
			#print(f'base class {base_class} , test class {test_class}')
			run += (base[i][j+1].item() - test[i][j+1].item()) ** 2
			run_ten += (base[i][j+1] - test[i][j+1]) ** 2
			#print(f'run loss: {run}, test scalar: {test[i][j]}, base scalar: {base[i][j]}')
		loss += ((base_class * test_class) ** 2) * (run / (base.size(dim=1) - 1))
		loss_ten += ((base_class_ten * test_class_ten) ** 2) * (run_ten / torch.tensor([(base.size(dim=1) - 1)]).float())

	return loss_ten


optimizer = optim.Adam(model.parameters(), lr=0.05)

print("enumerate start")
for epoch in range(1):
	for indx, samples in enumerate(dataloader):
		#print(indx, samples)
		imgs, labels = samples
		imgs = resize(imgs)
		optimizer.zero_grad()
		output = model(imgs)
		print(output.size())
		loss = aruco_loss(output, labels)
		loss.backward()
		optimizer.step()
		#print(f'loss: {loss}')
		#print(model.weight.grad)


