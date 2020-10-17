import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28*28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
  plt.subplot(2,3, i+1)
  plt.imshow(samples[i][0], cmap='gray')
plt.show()

def MnistModule(input_size, hidden_size, num_classes):
  return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_classes))
model = MnistModule(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_steps = len(train_loader)
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.reshape(-1,28*28).to(device)
    labesl = labels.to(device)

    # Forward Pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward Pass
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()

    if (i+1) % 100 == 0:
      print(f'epoch {epoch+1} Out of {num_epochs}, step {i+1} Out of {num_steps}, loss = {loss.item():.4}')


n_correct = 0
n_samples = 0
for images, labels in test_loader:
  images = images.reshape(-1, 28*28).to(device)
  labels = labels.to(device)
  outputs = model(images)

  _, predictions = torch.max(outputs, 1)
  n_samples += labels.shape[0]
  n_correct += (predictions == labels).sum().item()

acc = 100.0 * n_correct/n_samples
print(f'Accuracy {acc}')