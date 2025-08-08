import pandas as pd
from chromadb import HttpClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


#初始化中文模型
embed_model = SentenceTransformer("models/bge-base-zh")

def parse_quantity(value):
    """解析数量字符串，提取数字部分"""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    
    # 尝试提取字符串中的数字部分
    import re
    match = re.search(r'(\d+)', str(value))
    return int(match.group(1)) if match else 0

# 1. 读取Excel
df = pd.read_excel("products.xlsx")

# 2. 连接向量数据库
client = HttpClient()
# 先删除已存在的集合（如果存在）
try:
    client.delete_collection("products")
    print("已删除现有的products集合")
except Exception as e:
    print(f"删除集合时出错（可能不存在）: {e}")

# 创建新集合
collection = client.create_collection("products")

# 3. 生成向量并存入数据库
for _, row in df.iterrows():
    # 构建文档文本（商品名称+描述）
    text = f"商品: {row['标题']}, 规格: {row['规格1名字']}, 供应商: {row['云仓']}"
    embedding = embed_model.encode([text])[0].tolist()  # 生成向量
    
    # 构建元数据（包含所有需要检索的字段）
    metadata = {
        "供应商": str(row["云仓"]),
        "条码": str(row["条码"]),
        "单价": float(row["供应最小单位单价"]),
        "规格": str(row["规格1名字"]),
        "起送量": parse_quantity(row["起送量"]),  # 使用解析函数
        "整件规格": str(row["整件规格"]) if pd.notna(row["整件规格"]) else "",
        "整件价格": float(row["整件价格"]) if pd.notna(row["整件价格"]) else 0.0
    }
    
    # 插入数据
    try:
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[f"id_{row.name}"]  # 使用行号作为ID
        )
    except Exception as e:
        print(f"插入第 {row.name} 行时出错:")
        print(f"数据: {metadata}")
        print(f"错误: {str(e)}")
        print("跳过这一行...")
        continue
print("服务器状态:", client.heartbeat())
print("数据插入完成！共插入", len(df), "条记录。")