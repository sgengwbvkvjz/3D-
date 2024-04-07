from torch import nn
import torch

"""数据应该是 3D 图像数据，通常表示为 (depth, height, width)，
其中 depth 表示图像的深度或切片数量，height 表示图像的高度，width 表示图像的宽度。
模型的输入通道数（channels）为 1，表示灰度图像，因此每个图像是单通道的。
因此，输入张量的形状应该是 (batch_size, 1, depth, height, width)，
其中 batch_size 表示每个批次的样本数量。"""

#构建3D卷积神经网络模型
class Conv3DBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv3DBlock,self).__init__()
        self.conv = nn.Conv3d(in_channels,out_channels,kernel_size=3,padding=1)
        #padding=1 表示在输入的每一侧都填充一层零值，以保持输入和输出的大小相同。
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2)
        #创建了一个 3D 最大池化层，用于下采样操作，
        self.batchnorm = nn.BatchNorm3d(out_channels)
        #创建了一个 3D 批标准化层，用于规范化输入的数据，加速收敛并提高模型的泛化能力。

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.batchnorm(x)
        return x

class ThreeDCNN(nn.Module):
    def __init__(self,in_channels=1,width=128,height=128,depth=64):
        super(ThreeDCNN,self).__init__()
        self.model = nn.Sequential(
            Conv3DBlock(in_channels,64),
            Conv3DBlock(64,64),
            Conv3DBlock(64,128),
            Conv3DBlock(128,256),
            nn.AdaptiveAvgPool3d((1,1,1)),
            ## 创建一个自适应平均池化层，将任意输入大小池化到指定的输出大小
            nn.Flatten(),
            # # 创建一个Flatten层，用于将多维数据扁平化为一维
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            # # 创建一个Dropout层，用于防止过拟合，随机将输入单元的一部分设置为0，以降低过拟合风险
            nn.Linear(512,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model(x)


if __name__ == "__main__":
    model = ThreeDCNN(in_channels=1, width=128, height=128, depth=64)
    print(model)