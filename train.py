import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch import nn, optim
# optim 是 PyTorch 中的优化器模块，提供了各种优化算法来更新模型的参数
# nn 是 PyTorch 中的神经网络模块，提供了各种预定义的神经网络层和损失函数。

import datasets
import model

device = 'cuda'
myModel = model.ThreeDCNN()
# 实例化自定义的 3D 分类模型
print(torch.cuda.is_available())
# 打印是否有可用的 CUDA 设备
if torch.cuda.is_available():
    myModel = myModel.cuda()
# 如果有可用的 CUDA 设备，则将模型转移到 CUDA 上


# 加载训练集和验证集
print(datasets.train_set[0][0].shape)
# 打印第一个样本的形状
train_loader = datasets.train_loader
val_loader = datasets.val_loader
# 定义训练集和验证集的 DataLoader，用于批量加载数据


loss_fn = nn.BCELoss()
# 定义交叉熵损失函数
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 如果有可用的 CUDA 设备，则将损失函数转移到 CUDA 上

#设置动态学习率
initial_learning_rate = 1e-4
decay_steps = 30
decay_rate = 0.96
staircase = True

#优化器
optimizer = torch.optim.Adam(myModel.parameters(),lr=initial_learning_rate)

# 构建学习率调度器
lr_scheduler = ExponentialLR(optimizer,gamma=decay_rate)


epochs = 20

#定义训练模型
def train(model,train_loader,optimizer,loss_fn,epoch):
    # 将模型设置为训练模式
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播计算模型输出
        outputs = model(inputs)
        # 计算损失
        loss = loss_fn(outputs,labels.float().view(-1,1))
        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        running_loss += loss.item()
        # 对模型输出进行四舍五入，以获取预测结果
        predicted = torch.round(outputs)
        # 计算正确预测的样本数
        correct += (predicted == labels.float().view(-1,1)).sum().item()
        # 统计样本总数
        total += labels.size(0)
    # 计算平均训练损失和准确率
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

#定义验证函数
def validate(model,val_loader,loss_fn):
    # 将模型设置为评估模式
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # 在不计算梯度的情况下，遍历验证数据加载器中的每个批次
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs,labels.float().view(-1,1))
            running_loss += loss.item()
            predicted = torch.round(outputs)
            loss += (predicted == labels.float().view(-1,1)).sum().item()
            total += labels.size(0)
    val_loss = running_loss / len(val_loader)
    val_accuracy = loss / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


#训练模型
for epoch in range(epochs):
    train(myModel, train_loader, optimizer, loss_fn, epoch)
    validate(myModel, val_loader, loss_fn)
    lr_scheduler.step()

print("Done!")

# lr = 1e-4
# decay_steps = 30
# decay_rate = 0.96
#
# # 定义学习率、权重衰减系数
# optim = optim.Adam(myModel.parameters(), lr=lr, weight_decay=decay_rate)
# # 定义优化器，使用 Adam 算法，更新模型的参数
#
# with open('logs.csv', 'a', encoding='utf-8') as f:
#     f.write("\ntrain_loss,test_loss,total_accuracy,total_accuracy_for_AD\n")
# # 打开一个文件，用于记录训练过程中的指标值，并写入表头
#
# train_step = 0
# test_step = 0
# epoch = 20
#
# # 定义迭代轮数
#
# for i in range(epoch):
#     train_loss = 0
#     print("------{}------".format(i))
#     myModel.train()
#     # 进入训练模式，启用 Dropout 和 BatchNormalization 等模型的训练模式
#     for imgs, targets in train_loader:
#         if torch.cuda.is_available():
#             imgs = imgs.cuda()
#             targets = targets.cuda()
#         # 如果有可用的 CUDA 设备，则将数据转移到 CUDA 上
#         # 添加batch_size维度
#         imgs = imgs.unsqueeze(0)
#         output = myModel(imgs)
#         # 前向传播，获取模型的输出
#         optim.zero_grad()
#         # 优化器梯度清零
#         loss = loss_fn(output, targets.float().view(-1,1))
#
#         # 计算损失
#         loss.backward()
#         # 反向传播算梯度
#         optim.step()
#         # 优化器优化模型
#         train_loss += loss.item()
#         train_step += 1
#         # 累加每一批次的训练损失和训练步数
#     print("running_loss: ", train_loss / len(datasets.train_set))
#
#     # 打印训练损失
#     test_loss = 0
#     total_accuracy = 0
#     total_accuracy2 = 0
#     # 初始化验证集总损失、总精度和总 AD 精度
#     with torch.no_grad():
#         myModel.eval()
#         # 进入评估模式，关闭 Dropout 和 BatchNormalization 等模型的训练模式
#         for imgs, targets in val_loader:
#             if torch.cuda.is_available():
#                 imgs = imgs.cuda()
#                 targets = targets.cuda()
#             # 如果有可用的 CUDA 设备，则将数据转移到 CUDA 上
#             output = myModel(imgs)
#             # 前向传播，获取模型的输出
#             loss = loss_fn(output, targets)
#             # 计算损失
#             test_loss += loss.item()
#             # 累加每一个样本的验证损失
#             accuracy = (output.argmax(1) == targets)
#             accuracy = accuracy.sum()
#             # 计算总精度
#             accuracy2 = 0
#             p = 0
#             # 计算 AD 精度
#             for j in output.argmax(1):
#                 if j.item() != 0 and targets[p] != 0:
#                     accuracy2 += 1
#                 if j.item() == 0 and targets[p] == 0:
#                     accuracy2 += 1
#                 p += 1
#             # 判断输出类别是否为 0，以及目标类别是否为 0，如果都是则认为是 AD 类别预测正确
#             total_accuracy += int(accuracy.item())
#             total_accuracy2 += int(accuracy2)
#             # 累加总精度和总 AD 精度
#             test_step += 1
#
#     print("test_loss: ", test_loss / len(datasets.val_set))
#     # 打印验证损失
#     print("total_accuracy: ", total_accuracy / len(datasets.val_set))
#     print("total_accuracy_for_AD: ", total_accuracy2 / len(datasets.val_set))
#     # 打印总精度和总 AD 精度
#     torch.save(myModel, "./model_save/myModel_{}.pth".format(i))
#     # 保存模型参数
#     with open('logs.csv', 'a', encoding='utf-8') as f:
#         data = str(train_loss / len(datasets.train_set)) + "," + str(test_loss / len(datasets.val_set)) + "," + \
#                str(total_accuracy / len(datasets.val_set)) + "," + str(total_accuracy2 / len(datasets.val_set)) + ",\n"
#         f.write(data)
#     # 将训练过程中的指标值写入文件