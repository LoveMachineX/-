# main.py
import numpy as np
from data_preparation import load_data_from_chroma, prepare_training_data
from model_training import train_lightgbm

# 1. 加载和准备数据
df = load_data_from_chroma()
X_train, X_test, y_train, y_test = prepare_training_data(df)  # 现在y_train是字符串或数字

# 2. 训练模型
print("\n训练LightGBM模型...")
model, le = train_lightgbm(X_train, y_train, X_test, y_test)  # 返回编码器用于后续预测

# 3. 示例预测
sample_vector = X_test[0].reshape(1, -1)  # 取第一个测试样本
pred_num = model.predict(sample_vector)
pred_label = le.inverse_transform([np.argmax(pred_num)])  # 转换回原始标签
print(f"预测结果: {pred_label[0]}")