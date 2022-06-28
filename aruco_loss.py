import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

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


x = torch.tensor([[0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
y = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0]])

loss = aruco_loss(x, y)

print(x.size(dim=1))
print(f'loss: {loss}')

model = nn.Linear(9, 9)
x = torch.randn(1, 9)
target = torch.randn(1, 9)
#target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0]])

optimizer = optim.Adam(model.parameters(), lr=0.05)

for i in range(20000):
	optimizer.zero_grad()
	output = model(x)
	loss = aruco_loss(output, target)
	loss.backward()
	optimizer.step()
	#print(f'loss: {loss}')
	#print(model.weight.grad)

print(f'x: {x}')
print(f'target: {target}')
print(f'output: {output}')
print(f'loss: {loss.item()}')


