# 问题描述
>通过风机载荷计算中的一个例子，检验深度学习技术解决此问题的能力。
# 数据集
>本数据来源于金风科技研发部载荷计算团队，一共有640个样本点，每个样本点有4个输入特征，对应四个风参，每个样本点有6个浮点数输出值。样本点分布如图5-29，X、Y、Z坐标轴对应风参1、2、3，风参4对应图中球形的大小(适当等比例放大)，图中每一个点对应四个不同大小的球。样本点的随机90%用于训练，随机10%用于验证。图5-30列举了部分样本参数。根据样本点的分布特性，可以将本问题归为回归问题，传统的线性回归方法很难识别该模式，因此本文通过训练神经网络来预测输出值。<br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/data.png" width = "800"><br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/部分样本参数.png" width = "800"><br>
# 神经网络模型
>计算模型为全连接神经网络，有两层隐藏层，每层有10个节点，如图所示。激活函数为Relu，优化算法为AdamOptimizer，时间步长为0.001，定义损失值为预测值与标准值的均方差。<br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/全连接神经网络.png" width = "800"><br>
# 训练结果
>下图展示了两种误差标准下训练精度和验证精度的变化情况，可以发现，当误差标准为3%时，在第20000步左右，两种准确率都达到了100%，而当误差标准为1%时，在第25000步左右，两种准确率达到了80%左右，之后基本趋于直线，表明神经网络已经基本不在进行学习。<br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/训练过程中准确率变化.png" width = "800"><br>
下图为训练误差的变化情况，可以看到从第25000步开始，此误差趋于稳定，在11.5左右。由预测的结果可以看出本文的计算模型具有一定的可信度和有效性。<br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/训练损失值变化.png" width = "800"><br>
**容忍误差为3%的预测结果<br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/容忍误差为3%时的预测结果.png" width = "800"><br>
**容忍误差为1%的预测结果<br>
<img src="https://github.com/hedongya/GOLDWIND/blob/master/容忍误差为1%时的预测结果.png" width = "800"><br>


