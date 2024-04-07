import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
import numpy as np

class MyData(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        # # 类别标签字典，键为文件夹名称，值为对应的类别标签
        self.labels = {'CT-0': 0, 'CT-23': 1}
        # # 数据集模式，用于区分训练集和验证集，默认为训练模式
        self.mode = mode
        # 加载数据集
        self.data = self.load_data()

    def load_data(self):
        data = []
        # 遍历类别标签字典。类别标签字典，folder取到键为文件夹名称，label取到值为对应的类别标签
        for folder, label in self.labels.items():
            # 获取当前类别的文件夹路径
            folder_path = os.path.join(self.root_dir, folder)
            for file in os.listdir(folder_path):
                # 获取文件的完整路径
                file_path = os.path.join(folder_path, file)
                # 将文件路径及其对应的标签添加到数据列表中
                data.append((file_path, label))
        return data

    def __len__(self):
        # 返回数据集的大小，即样本数量
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引处的样本数据
        file_path, label = self.data[idx]
        # 使用nibabel库加载NIfTI格式的图像数据，并将其转换为NumPy数组
        img = nib.load(file_path).get_fdata()
        # 对图像数据进行预处理
        img = self.preprocess(img)
        ## 将NumPy数组转换为PyTorch张量，并转换数据类型为float32
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)  # 添加一个维度以匹配模型的输入,
        # 输入张量的形状应该是 (batch_size, 1, depth, height, width)
        return img, label

    def preprocess(self, img):
        img = self.normalize(img) # 对图像进行归一化处理
        img = self.resize_volume(img)
        return img

    def normalize(self, volume):
        """归一化"""
        # 设置归一化的最小值和最大值
        min_val = -1000
        max_val = 400
        # 将超出范围的像素值截断到指定范围内
        volume[volume < min_val] = min_val
        volume[volume > max_val] = max_val
        # 对像素值进行归一化
        volume = (volume - min_val) / (max_val - min_val)
        volume = volume.astype("float32")# 将数据类型转换为float32
        return volume

    def resize_volume(self, img):
        """修改图像大小"""
        # 设置期望的图像大小
        desired_depth = 64
        desired_width = 128
        desired_height = 128
        # 获取当前图像的大小
        current_depth = img.shape[-1]
        current_width = img.shape[0]
        current_height = img.shape[1]
        # 计算当前图像与期望图像大小之间的比例关系
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        # 计算缩放因子
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        # 对图像进行旋转，以调整图像方向
        img = ndimage.rotate(img, 90, reshape=False)
        # 使用ndimage库的zoom函数对图像进行缩放操作
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
        return img

# 定义数据集的根目录
root_dir = "D:/bigdata/data/MosMedData"

# 创建数据集实例
train_dataset = MyData(root_dir, mode='train')
val_dataset = MyData(root_dir, mode='val')

# 定义训练集和验证集的样本数
train_size = int(0.7 * len(train_dataset))
val_size = len(train_dataset) - train_size

# 划分训练集和验证集
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)


if __name__ == "__main__":
    # 测试数据加载器
    print("Number of samples in train and validation are %d and %d." % (len(train_set), len(val_set)))

