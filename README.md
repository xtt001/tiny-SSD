# tiny-SSD
This is a personal homework for the AI Experiment course in the fall semester of my junior year in SYSU

## 1 quick start

本程序运行的环境配置已在 `requirements.txt` 中列出，通过设置 `train.py` 文件中的超参数 `NUM_EPOCHES` ，可以选择训练的epoch轮数

预训练模型在所需环境下运行测试代码test.py程序即可进行模型测试。


## 2 模型与效果

/models文件夹中存放了一些预训练模型，其中的模型命名规则与模型效果如下：

- **net_X.pkl**是原始未经过优化的模型，X是训练轮数；(通过改变训练轮数并对结果进行分析比较后我发现，在epoch数为30时模型效果达到最优，对测试图片的识别准确率达到了**0.94**，再增加训练轮数后模型效果直线下降，即训练出现了过拟合，为了再提高识别效果的同时避免过拟合，**我使用了两种数据增强的方法对模型效果进行优化**)
- **net_colortransX.pkl**是通过变换图像的饱和度、对比度等进行数据增强后训练出的模型，X是训练轮数；（由于测试图片是黑底白色图标，因此饱和度、对比度变换的数据增强并不能优化模型效果，因此这部分的模型效果并不好）
- **net_greytransX.pkl**是通过将部分数据图像转为灰度图来进行数据增强后训练出的模型，这部分模型中epoch数为50时（net_greytrans50.pkl）模型效果达到了最优，对测试图片的识别准确率达到了**0.99**。

下图分别展示了net_30.pkl的识别效果和net_greytrans50.pkl的识别效果
