import torch
import torchvision.models as md
import torch.nn as nn
from torchvision import transforms
from PIL import Image

path = 'DSCF0002.JPG'
number_tags = 2
weight_path = 'deep_aruco.trch'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

res = md.resnet50(pretrained=True)
model = nn.Sequential(res, nn.Linear(1000, number_tags*9))
model.load_state_dict(torch.load(weight_path))
model.eval()
model.to(device)

convert_tensor = transforms.ToTensor()
resize = transforms.Resize((224, 224))
img = Image.open(f'imgs/{path}')
img_ten = convert_tensor(img).to(device)
img_ten = resize(img_ten).unsqueeze(0)

output = model(img_ten)
print(output)


