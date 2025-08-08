import pandas as pd
from typing import List, Dict, Optional

class SearchExporter:
    """
    搜索结果的导出工具类
    功能:将向量搜索结果导出为Excel文件
    """
    
    # 列名映射表
    COLUMN_MAPPING = {
        'rank': '排名',
        'product': '商品名称',
        'supplier': '供应商',
        'price': '单价',
        'spec': '规格',
        'min_order': '起送量',
        'barcode': '条码',
        'case_price': '整件价格',
        'case_spec': '整件规格'
    }
    
    @staticmethod
    def export_to_excel(
        results: List[Dict],
        output_file: str = "搜索结果.xlsx",
        columns: Optional[List[str]] = None
    ) -> None:
        """
        导出搜索结果到Excel文件
        
        参数:
            results: search_products()返回的结果列表
            output_file: 输出文件名(默认"搜索结果.xlsx")
            columns: 指定要导出的列(默认全部列)
        
        返回:
            None (直接生成文件)
        """
        df = pd.DataFrame(results)
        
        # 设置默认列
        default_columns = ['rank', 'product', 'supplier', 'price', 'spec', 'min_order']
        selected_columns = columns or default_columns
        
        # 筛选和重命名列
        df = df[selected_columns].rename(columns=SearchExporter.COLUMN_MAPPING)
        
        # 保存文件
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✓ 成功导出 {len(df)} 条记录到 {output_file}")

    @staticmethod
    def batch_export(
        search_func,  # 搜索函数引用
        queries: List[tuple], 
        output_prefix: str = "搜索结果",
        **search_kwargs
    ) -> None:
        """
        批量查询并导出多个Excel文件
        
        参数:
            search_func: search_products函数引用
            queries: 查询列表，格式[(query_text, max_results), ...]
            output_prefix: 输出文件前缀
            **search_kwargs: 传递给search_func的其他参数
        """
        for query, max_results in queries:
            results = search_func(query, max_results=max_results, **search_kwargs)
            filename = f"{output_prefix}_{query[:20]}.xlsx"  # 限制文件名长度
            SearchExporter.export_to_excel(results, filename)