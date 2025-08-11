import pandas as pd
import numpy as np
from chromadb import HttpClient
from sklearn.model_selection import train_test_split

def load_data_from_chroma(collection_name="products"):
    """从ChromaDB加载向量数据和标签"""
    client = HttpClient()
    collection = client.get_collection(collection_name)
    
    # 获取所有数据
    items = collection.get(include=["embeddings", "metadatas"])
    
    #将向量列表展平成一维数组
    embeddings = np.array(items["embeddings"])

    # 转换为DataFrame
    data = {
        "supplier": [meta["供应商"] for meta in items["metadatas"]],
        "price": [meta["单价"] for meta in items["metadatas"]],
        "category": [meta.get("类别", "未知") for meta in items["metadatas"]]  # 假设有类别字段
    }
    
    df = pd.DataFrame(data)

    for i in range(embeddings.shape[1]):
        df[f"vec_{i}"] =embeddings[:, i]
    return df


def prepare_training_data(df, test_size=0.2):
    """准备训练和测试数据"""
    # 假设我们要预测价格区间（分类问题）
    price_bins=[0, 10, 30, 40, float('inf')]
    price_labels=["低价", "中价", "高价", "奢侈"]
    

    df["price_range"] = pd.cut(df["price"], 
        bins=price_bins,
        labels=price_labels).astype(str)

    # 分离特征和标签
    vec_columns = [col for col in df.columns if col.startswith("vec_")]
    X = df[vec_columns].values  # 特征矩阵 (n_samples, embedding_dim)
    y = df["price_range"]       # 目标变量
   
    return train_test_split(X, y, test_size=test_size, random_state=42)

# 使用示例
if __name__ == "__main__":
    df = load_data_from_chroma()
    X_train, X_test, y_train, y_test = prepare_training_data(df)
    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")