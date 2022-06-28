import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.models as md
import torch.nn as nn

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

vec_i = index_load(6)
print(f'test vec: {vec_i}')

print(f'test indices: {data_indexed[0]["regions"]}')

print(f'dataset length {len(data_indexed)}')

class CustomDataset(torch.utils.data.Dataset):
	def __init__(self):
		return

	def __len__(self):
		return len(data_indexed)

	def __getitem__(self, idx):
		return index_load(idx)


dataset = CustomDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0)

img, label = dataloader

print(f'dataset test indx 0: {dataset[0]}')
print(f'dataloader test indx 0: {img}')


res = md.resnet50(pretrained=True)
model = nn.Sequential(res, nn.Linear(1000, number_tags*9))
print(res)


inpt, _ = dataset[0]
resize = transforms.Resize(224)
inpt = resize(inpt).unsqueeze(0)
print(f'inpt: {inpt}')

print(f'resnet50 test: {model(inpt)}')


