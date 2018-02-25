
# 问题描述
>本数据来源于金风科技研发部载荷计算团队，一共有640个样本点，每个样本点有4个输入特征，对应四个风参，每个样本点有6个浮点数输出值。样本点分布如图5-29，X、Y、Z坐标轴对应风参1、2、3，风参4对应图中球形的大小(适当等比例放大)，图中每一个点对应四个不同大小的球。样本点的随机90%用于训练，随机10%用于验证。图5-30列举了部分样本参数。根据样本点的分布特性，可以将本问题归为回归问题，传统的线性回归方法很难识别该模式，因此本文通过训练神经网络来预测输出值。<br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/data.png" width = "800"><br>
<br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/部分样本参数.png" width = "800"><br>
# 神经网络模型
> 本节计算模型为全连接神经网络，有两层隐藏层，每层有10个节点，如图所示。激活函数为Relu，优化算法为AdamOptimizer，时间步长为0.001，定义损失值为预测值与标准值的均方差。<br>
<br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/部分样本参数.png" width = "800"><br>
<br>

