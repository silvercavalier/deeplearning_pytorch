import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1.数据预处理
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))

])
# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# 加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义神经网络

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义了两层卷积，其中第一个参数为输入通道数，第二参数为输出通道，kernel_size为卷积核大小，stride为步长，padding为填充像素的层数
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        # 接下来定义两层全连接层，先将经过卷积层处理的三维张量展平为向量，再输入全连接层
        self.lin1 = nn.Linear(64 * 7 * 7, 128)
        self.lin2 = nn.Linear(128,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # 这里的参数，第一个是处理张量，第二个是池化窗口，每个池化窗口产生一个值，第三个参数为stride，默认等于池化窗口大小，第四个参数padding为填充层，可以让边缘像素更多地参与池化，提取边缘特征，默认为零
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 64*7*7)
        # view()与reshape()作用类似，用于改变张量形状，这里的-1是让pytorch自动推断此处应填入的数
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
# 创建模型实例
model = SimpleCNN()

# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # momentum即带动量的随机梯度下降，0.9表示保留90%的历史速度

# 4. 模型训练
num_epochs = 5
model.train()  # 设置模型为训练模式

for epoch in range(num_epochs):
    total_loss = 0
    for image, label in train_loader:
        outputs = model(image)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        # 为什么会在优化器下执行清空梯度的操作，因为此时parameters已经在创建优化器的时候交给优化器保管了，优化器将梯度写在每个参数的.grad中并批量处理
        loss.backward()
        optimizer.step()

        total_loss += loss.item() # 将1*1的张量转换为数字

    print(f"轮数:{epoch}/{num_epochs}, 损失:{total_loss / len(train_loader)}")

# 模型测评
model.eval()  # 设置模型为评估模式
correct = 0
total = 0

with torch.no_grad():
    for image, label in test_loader:
        outputs = model(image)
        _, predicted = torch.max(outputs, 1) # 沿第一维度，即”类别维度“寻找最大值，并返回最大值和对应索引，在这里我们不关心最大值，只关心最大值的索引
        total += label.size(0) # 衡量label的第一维度，即提取样本个数
        correct += (predicted == label).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


# 可视化
import matplotlib.pyplot as plt

# 切换为评估模式
model.eval()

# 从测试集中取出一个批次
images, labels = next(iter(test_loader))

# 如果模型在 GPU 上，需要把图片也放到相同设备
device = next(model.parameters()).device
images_for_model = images.to(device)

# 进行预测
with torch.no_grad():
    outputs = model(images_for_model)
    predicted = outputs.argmax(dim=1).cpu()

# 转到 CPU，方便画图
images = images.cpu()
labels = labels.cpu()

# 显示前 16 张图片
num_images = 16
plt.figure(figsize=(10, 10))

for i in range(num_images):
    plt.subplot(4, 4, i + 1)

    # MNIST 的形状为 [1, 28, 28]，去掉通道维度
    image = images[i].squeeze()

    # 如果使用了 Normalize((0.5,), (0.5,))，需要反归一化
    image = image * 0.5 + 0.5

    plt.imshow(image, cmap="gray")

    true_label = labels[i].item()
    predicted_label = predicted[i].item()

    # 预测正确显示绿色，错误显示红色
    color = "green" if true_label == predicted_label else "red"

    plt.title(
        f"True: {true_label}\nPred: {predicted_label}",
        color=color
    )
    plt.axis("off")

plt.tight_layout()
plt.show()