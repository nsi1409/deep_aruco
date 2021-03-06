import torch
import torchvision.models as md
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt


number_tags = 2
weight_path = 'deep_aruco.trch'
test_number = 8

f = open('annotations/aruco_v0.1.json')
data = json.load(f)
f.close()
data_indexed = list(data["_via_img_metadata"].values())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

res = md.resnet50(pretrained=True)
model = nn.Sequential(res, nn.Linear(1000, number_tags*9))
model.load_state_dict(torch.load(weight_path))
model.eval()
model.to(device)

convert_tensor = transforms.ToTensor()
resize = transforms.Resize((224, 224))

def index_load(i):
	label_ten = torch.zeros(9*number_tags)
	#path = data_indexed[i]["filename"]
	path = 'verify/DSCF0030.JPG'
	print(f'label path: {path}')
	#img = Image.open(f'imgs/{path}')
	img = Image.open(f'{path}')	
	img_ten = convert_tensor(img)

	for j in range(len(data_indexed[i]["regions"])):
		try:
			x = data_indexed[i]["regions"][j]["shape_attributes"]["cx"]
			y = data_indexed[i]["regions"][j]["shape_attributes"]["cy"]
			corner = int(data_indexed[i]["regions"][j]["region_attributes"]["aruco"])
			#print(f'corner {corner}')
			label_ten[(9*(corner//10)) + (2*(corner%10)+1)] = float(x) / (img_ten.size(dim=2)-1)
			label_ten[(9*(corner//10)) + (2*(corner%10)+2)] = float(y) / (img_ten.size(dim=1)-1)
			label_ten[9*(corner//10)] = float(1)
		except:
			print('empty 0 input')
			pass

	img_ten = resize(img_ten)
	img_ten = img_ten.to(device)
	label_ten = label_ten.to(device)

	return img_ten, label_ten

while(1):
	test_number = int(input("Enter test number: "))

	img_ten, label_ten = index_load(test_number)

	output = model(img_ten.unsqueeze(0))
	output = output.squeeze()
	print(f'inf: {output}')

	print(f'ground: {label_ten}')

	implot = plt.imshow(img_ten.squeeze().permute(1, 2, 0).to("cpu"))

	conf = 0.7

	output.to("cpu")
	for i in range(int(output.size(dim=0))//9):
		if(output[9*i] > conf):
			for j in range(4):
				x = float(output[(9*i)+(2*j)+1] * 223)
				y = float(output[(9*i)+(2*j)+2] * 223)
				plt.scatter(x=[x], y=[y], c='r', s=7)
				plt.text(x+2, y-2, f'{i}:{j}', c='r', fontsize=9)

	plt.show()


