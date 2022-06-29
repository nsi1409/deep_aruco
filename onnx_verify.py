import onnx
import onnxruntime
import json
import torch
from PIL import Image
from torchvision import transforms

model = onnx.load('deep_aruco.onnx')
onnx.checker.check_model(model)
onnxruntime.get_device()

number_tags = 2
weight_path = 'deep_aruco.trch'
test_number = 7

f = open('annotations/aruco_v0.1.json')
data = json.load(f)
f.close()
data_indexed = list(data["_via_img_metadata"].values())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

convert_tensor = transforms.ToTensor()
resize = transforms.Resize((224, 224))

def index_load(i):
	label_ten = torch.zeros(9*number_tags)
	path = data_indexed[i]["filename"]
	print(f'label path: {path}')
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
			print('empty 0 input')
			pass

	img_ten = resize(img_ten)
	#img_ten = img_ten.to(device)
	#label_ten = label_ten.to(device)
	img_ten = img_ten.unsqueeze(0).numpy()

	return img_ten, label_ten

img_ten, label_ten = index_load(test_number)

ort_session = onnxruntime.InferenceSession('deep_aruco.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
ort_inputs = {ort_session.get_inputs()[0].name: img_ten}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs)



