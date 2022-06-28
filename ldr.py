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
resize = transforms.Resize((224, 224))
weight_path = "deep_aruco.trch"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def index_load(i):
	label_ten = torch.zeros(9*number_tags)
	path = data_indexed[i]["filename"]
	img = Image.open(f'imgs/{path}')
	img_ten = convert_tensor(img)

	for j in range(len(data_indexed[i]["regions"])):
		try:
			x = data_indexed[i]["regions"][j]["shape_attributes"]["cx"]
			y = data_indexed[i]["regions"][j]["shape_attributes"]["cy"]
			corner = int(data_indexed[i]["regions"][j]["region_attributes"]["aruco"])
			#print(f'corner {corner}')
			label_ten[(9*(corner//10)) + (2*(corner%10)+1)] = float(x) / img_ten.size(dim=2)
			label_ten[(9*(corner//10)) + (2*(corner%10)+2)] = float(y) / img_ten.size(dim=1)
			label_ten[9*(corner//10)] = float(1)
		except:
			#print('empty 0 input')
			pass

	img_ten = resize(img_ten)
	img_ten = img_ten.to(device)
	label_ten = label_ten.to(device)

	return img_ten, label_ten


class CustomDataset(torch.utils.data.Dataset):
	def __init__(self):
		return

	def __len__(self):
		return len(data_indexed)

	def __getitem__(self, idx):
		item = index_load(idx)
		return item

dataset = CustomDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0)


res = md.resnet50(pretrained=True)
model = nn.Sequential(res, nn.Linear(1000, number_tags*9))
model.load_state_dict(torch.load(weight_path))
model.to(device)
#print(res)


inpt, _ = dataset[0]
inpt = resize(inpt).unsqueeze(0)
#print(f'inpt: {inpt}')

#print(f'resnet50 test: {model(inpt)}')


def aruco_loss(test, base):
	loss_ten = torch.tensor([0.0]).to(device)
	for i in range(base.size(dim=0)):
		for j in range(int(base.size(dim=1)/9)):
			base_class_ten = base[i][9*j]
			test_class_ten = test[i][9*j]

			loss_ten += (base_class_ten - test_class_ten) ** 2

			run_ten = torch.tensor([0.0]).to(device)
			for k in range(8):
				run_ten += (base[i][(9*j)+k+1] - test[i][(9*j)+k+1]) ** 2

			loss_ten += base_class_ten * (run_ten / 8).float()

	return loss_ten


optimizer = optim.Adam(model.parameters(), lr=0.02)

print("train start")
for epoch in range(1024):
	for indx, samples in enumerate(dataloader):
		#print(indx, samples)
		imgs, labels = samples
		#print(imgs.size())
		#imgs = resize(imgs)
		optimizer.zero_grad()
		output = model(imgs)
		loss = aruco_loss(output, labels)
		print(f'epoch: {epoch} | loss: {loss.item()}')
		loss.backward()
		optimizer.step()
		#print(f'loss: {loss}')
		#print(model.weight.grad)


torch.save(model.state_dict(), weight_path)
print('model saved')


