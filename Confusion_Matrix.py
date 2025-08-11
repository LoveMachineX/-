from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model_training import train_lightgbm
import numpy as np
from data_preparation import load_data_from_chroma
from data_preparation import prepare_training_data

#加载和准备数据
df = load_data_from_chroma()
X_train, X_test, y_train, y_test = prepare_training_data(df)


price_bins=[0, 10, 30, 40, float('inf')]
price_labels=["低价", "中价", "高价", "奢侈"]

# 1. 训练模型并预测
model, le = train_lightgbm(X_train, y_train, X_test, y_test)
y_pred_prob = model.predict(X_test)
y_pred = le.inverse_transform(np.argmax(y_pred_prob, axis=1))  # 将概率转换为标签

# 2. 生成混淆矩阵和报告
cm = confusion_matrix(y_test, y_pred, labels=price_labels)
print(classification_report(y_test, y_pred, target_names=price_labels))

# 3. 可视化
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=price_labels, 
            yticklabels=price_labels,
            cbar=False)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.title("价格区间分类混淆矩阵", fontsize=14)
plt.tight_layout()
plt.show()