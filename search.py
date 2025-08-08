# search.py
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("models/bge-base-zh")

def get_collection():
    """获取已存在的集合"""
    client = HttpClient()
    return client.get_collection("products")

def build_where_conditions(price_range=None, supplier=None):
    """构建查询条件"""
    conditions = []
    
    if price_range:
        conditions.append({"单价": {"$gte": price_range[0]}})
        conditions.append({"单价": {"$lte": price_range[1]}})
    
    if supplier:
        conditions.append({"供应商": {"$eq": supplier}})
    
    if len(conditions) == 1:
        return conditions[0]
    elif len(conditions) > 1:
        return {"$and": conditions}
    else:
        return None

def search_products(query, max_results=5, price_range=None, supplier=None):
    """
    搜索商品
    :param query: 搜索查询文本
    :param max_results: 返回的最大结果数
    :param price_range: (min_price, max_price) 元组
    :param supplier: 筛选特定供应商
    :return: 搜索结果列表
    """
    collection = get_collection()
    where_conditions = build_where_conditions(price_range, supplier)
    
    print("正在执行的查询条件:", where_conditions)  # 调试输出
    #使用相同模型生成向量
    query_embedding = embed_model.encode([query])[0].tolist()

    # 执行查询
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_results,
        where=where_conditions
    )
    
    # 格式化结果
    formatted_results = []
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        formatted_results.append({
            "rank": i+1,
            "product": doc,
            "supplier": meta['供应商'],
            "price": meta['单价'],
            "spec": meta['规格'],
            "min_order": meta['起送量']
        })
    
    return formatted_results

def print_results(results):
    """打印搜索结果"""
    for item in results:
        print(f"\n结果 {item['rank']}:")
        print(f"商品: {item['product']}")
        print(f"供应商: {item['supplier']}")
        print(f"价格: {item['price']}")
        print(f"规格: {item['spec']}")
        print(f"起送量: {item['min_order']}")

# 示例使用
if __name__ == "__main__":
    # 示例1: 基本搜索
    print("=== 基本搜索 ===")
    results = search_products("方便面", max_results=5)
    print_results(results)
    

    
