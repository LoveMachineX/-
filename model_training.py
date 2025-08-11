import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def train_lightgbm(X_train, y_train, X_test, y_test):
    """使用LightGBM训练分类模型"""
    # 转换数据格式
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train_encoded)
    
    # 参数设置
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train_encoded)),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    # 训练模型
    model = lgb.train(params,
                     train_data,
                     num_boost_round=100,
                     valid_sets=[train_data])
    
    # 预测和评估
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = accuracy_score(y_test_encoded, y_pred)
    print(f"LightGBM准确率: {acc:.4f}")
    return model, le