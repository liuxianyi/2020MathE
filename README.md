# 2022年研究生数学建模E题
```
├── code
│   ├── config.py 整个代码的配置文件
│   ├── merge.py 数据整合
│   ├── cal_huangmohua.py 问题二，五的数据预处理
│   ├── final_code.py 问题二
│   ├── util.py 问题五
│   ├── question.py 问题三，问题五的解算
│   └── Transformer.py transformer
├── data
│   ├── 基本数据
│   └── 监测点数据
└── train_data
```
## 环境
- pytorch
- keras
- pandas
- numpy
- sklearn
- matplotlib
- seaborn



## 数据准备
```
# 获取所有数据
Config.ismerge = True
python merger.py

# 获取植被覆盖率数据
Config.isAttachment5
python merge
```
### 异常数据处理
为了后续方便地使用数据，先对给定的数据进行处理并以易于操作的矩阵形式进行保存。在处理数据的过程中对其中的异常值和无效值采用滑动平均的方式进行处理。
## 问题二
- LSTM
- Transformer

### 代码


## 问题三
- LSTM
- 支持向量回归机（Support Vector Regression, SVR）
- 装袋回归算法（Bagging Regression，BR）
### 插值
- 牛顿插值
- 拉格朗日插值
- 三次埃尔米特插值
已有数据太少，利用插值，来扩充数据

### 两个角度
- 单元土壤属性时序预测
- 多元土壤属性时序预测（降雨量，植被覆盖率等）

## 使用
查看question.py与Config文件

## 问题四
定义了土壤板结化公式，计算不同放牧强度下的板结化程度

## 问题五
利用问题三构建的模型，预测土壤属性，来反映放牧数量阈值


## 问题六
利用问题四的结论预测不同方案下土地状态变化



