from search import search_products
from export_utils import SearchExporter

def main():
    # 单次查询导出
    print("正在执行单商品导出...")
    drink_results = search_products("饮料", max_results=10)
    SearchExporter.export_to_excel(
        drink_results,
        "饮料商品.xlsx",
        columns=['product', 'supplier', 'price', 'spec']
    )
    
    # 批量导出
    print("\n正在执行批量导出...")
    query_list = [
        ("矿泉水",30),
        ("零食", 30),
        ("酒类", 15),
        ("调味品", 10)
    ]
    SearchExporter.batch_export(
        search_products,
        query_list,
        output_prefix="分类商品",
        price_range=(5, 100)  # 可以传递任意search_products支持的参数
    )

if __name__ == "__main__":
    main()