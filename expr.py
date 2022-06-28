import torch

x = torch.ones(20)
transform = torch.ones(2, 20)
outp = x * transform

print(outp.size(), outp)


