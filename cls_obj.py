from collections import Counter
import torch
from typing import Union
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.utils.data import WeightedRandomSampler
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from configparser import ConfigParser
import numpy as np
from PIL import Image
import cv2
loss = None  # 可以避免警告，不用这句话也行


def train(model1, device, dataset, optimizer1, epoch1):
    global loss
    model1.train()  # 设置模型为训练模式
    correct = 0
    all_len = 0
    for i, (x, y) in tqdm(enumerate(dataset)):
        x, y = x.to(device), y.to(device)  # 将数据移动到设备上进行计算
        optimizer1.zero_grad()  # 梯度清零
        output = model1(x)  # 模型前向传播
        pred = output.max(1, keepdim=True)[1]  # 获取预测结果
        correct += pred.eq(y.view_as(pred)).sum().item()  # 统计预测正确的数量
        all_len += len(x)  # 统计样本数量
        loss = nn.CrossEntropyLoss()(output, y)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer1.step()  # 更新模型参数
    print(f"第 {epoch1} 次训练的Train准确率：{100. * correct / all_len:.2f}%")  # 打印训练准确率


def vaild(model, device, dataset):
    model.eval()  # 设置模型为评估模式
    global loss
    correct = 0
    test_loss = 0
    all_len = 0
    with torch.no_grad():
        for i, (x, target) in enumerate(dataset):
            x, target = x.to(device), target.to(device)  # 将数据移动到设备上进行计算
            output = model(x)  # 模型前向传播
            loss = nn.CrossEntropyLoss()(output, target)  # 计算损失
            test_loss += loss.item()  # 累计测试损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计预测正确的数量
            all_len += len(x)  # 统计样本数量
    print(f"Test 准确率：{100. * correct / all_len:.2f}%")  # 打印测试准确率
    return 100. * correct / all_len  # 返回测试准确率
#
def cls_predict(img: Union[np.ndarray, Image.Image]):#(img:np.ndarray|Image.Image):
    conf = ConfigParser()
    conf.read('./project.ini',encoding='utf-8')
    cls_dict = {}
    cat_cls = []
    for i,cls in enumerate(conf['PREDICT_CLS']):
        if cls.startswith('cls')==True:
            cls_dict[i] = conf['PREDICT_CLS'][cls]
            cat_cls.append(conf['PREDICT_CLS'][cls])
    predict_transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Resize(256),  # 调整图像大小为256x256
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])])
    re_img = predict_transform(img)
    model = resnet50()
    model.fc = nn.Sequential(nn.Linear(2048,int(conf['PREDICT_CLS']['num_cls'])))
    model.load_state_dict(torch.load(conf['PREDICT_CLS']['root_path']+conf['PREDICT_CLS']['file_cls']))
    model.to('cuda')
    model.eval()

    input_img = re_img.unsqueeze(0)
    input_img = input_img.to('cuda')
    res = model(input_img)
    soft_res =torch.softmax(res,dim=1)
    real_index = int(torch.argmax(soft_res))
    return cls_dict[real_index],soft_res

if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('./project.ini',encoding='utf-8')
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    if int(conf['TRAIN_CLS']['train_switch']) == 1:
        LR = float(conf['TRAIN_CLS']['LR'])
        EPOCH = int(conf['TRAIN_CLS']['EPOCH'])
        BTACH_SIZE = int(conf['TRAIN_CLS']['BTACH_SIZE'])

        train_root = conf['TRAIN_CLS']['data_path']  # 训练数据根目录


        # 数据加载及处理
        train_transform = transforms.Compose([
            transforms.Resize(int(conf['TRAIN_CLS']['RESIZE'])),  # 调整图像大小为256x256
            transforms.RandomResizedCrop(244, scale=(0.6, 1.0), ratio=(0.8, 1.0)),  # 随机裁剪图像为244x244
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),  # 改变图像的亮度
            torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),  # 改变图像的对比度
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])  # 对图像进行标准化
        ])
        # 图像读取转换
        all_data = torchvision.datasets.ImageFolder(
            root=train_root,
            transform=train_transform
        )

        dic = all_data.class_to_idx  # 类别映射表
        print(dic)    #  建议大家打印一下dic,因为在进行读取图片进行分类的时候，不一定按照顺序，为了和真实数据进行比对。
        

        # 计算每个类别的样本数量
        class_counts = Counter(all_data.targets)  # 假设类别信息在 all_data.targets 中

        # 计算每个类别的样本权重
        weights = [1.0 / class_counts[class_idx] for class_idx in all_data.targets]

        # 创建一个权重采样器
        sampler = WeightedRandomSampler(weights, len(all_data), replacement=True)

        # 使用采样器对数据集进行划分
        train_data = torch.utils.data.Subset(all_data, list(sampler))

        # 获取采样器的样本索引
        sampler_indices = list(sampler)

        # 根据采样器的样本索引获取验证集样本索引
        valid_indices = [idx for idx in range(len(all_data)) if idx not in sampler_indices]

        # 创建验证集数据集
        valid_data = torch.utils.data.Subset(all_data, valid_indices)


        # 训练大小，随机
        train_set = torch.utils.data.DataLoader(
            train_data,
            batch_size=BTACH_SIZE,
            shuffle=True
        )

        # 训练大小，随机
        test_set = torch.utils.data.DataLoader(
            valid_data,
            batch_size=BTACH_SIZE,
            shuffle=False
        )
        # 加载预训练的 ResNet-50 模型，并替换掉最后一层全连接层（fc），使其适应当前任务（共12个类别）。
        model_1 = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model_1.fc = nn.Sequential(nn.Linear(2048, 9))

        # 加载已训练好的模型参数, 可选。
        # model_1.load_state_dict(torch.load(r'E:\日常练习\pytorch_Project\best_model_train99.71.pth'))
        # model_1.train()

        # 设置模型为训练模式
        model_1.to(DEVICE)
        # 通过 optim.SGD(model_1.parameters(), lr=LR, momentum=0.9) 定义了 SGD 优化器。这里的 model_1.parameters() 表示优化器需要更新的模型参数，lr=LR 表示学习率为 LR，momentum=0.9 表示使用动量（momentum）参数为0.9。
        optimizer = optim.SGD(model_1.parameters(), lr=LR, momentum=0.9)

        # 设置初始的最高准确率为 90.0，并初始化最优模型。
        max_accuracy = 80.0
        # 最优模型全局变量
        best_model = None

        for epoch in range(1, EPOCH + 1):
            train(model_1, DEVICE, train_set, optimizer, epoch)
            accu = vaild(model_1, DEVICE, test_set)
            # 保存准确率最高的模型
            if accu > max_accuracy:
                max_accuracy = accu
                best_model = model_1.state_dict()  # 或者使用 torch.save() 保存整个模型
        # 打印最高准确率
        print("最高成功率： ", max_accuracy)
        # 保存最优模型
        torch.save(best_model, fr"./train_pth/{max_accuracy:.2f}.pth")


