# 说明
## 大的多图拼接
那个是我们的训练集拼成的一张图纸  
可以叫做训练集掠影  
你也可以发挥一下
## 各个图表
灰线的全连接层为多层感知机，橙色的线为线性层加交叉熵
~~~~ python
import torch
 
model.fc = torch.nn.Sequential(
    torch.nn.Linear(in,in),
    ReLU(),
    torch.nn.Linear(in,in),
    ReLU(),
    torch.nn.Linear(in,class_cnt)
    )
~~~~
![](PLOTDATA/MLP_2_Linear_TrainLoss.png)
训练集Loss Function变化程度
![](PLOTDATA/MLP_2_Linear_ValLoss.png)
验证集Loss Function变化程度
![](PLOTDATA/MLP_2_Linear_TrainAcc.png)
![](PLOTDATA/MLP_2_Linear_ValAcc.png)
训练与验证的准确率
结论：可能产生了过拟合，多层感知机对于这种简单的二分类问题确实会产生这样的结果，可以尝试dropout等方法  
另外还有在验证集6，7个epoch的时候验证的loss function最低，可能在后续的训练采取early stop的策略

![](PLOTDATA/MutiTask_TrainLoss.png)
![](PLOTDATA/MutiTask_ValLoss.png)
两者为训练集损失与验证集损失
![](PLOTDATA/MutiTask_TrainAcc.png)
![](PLOTDATA/MutiTask_ValAcc.png)
两者为训练集损失与验证集准确率