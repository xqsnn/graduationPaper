# 测试数据库连接和查询 - 用于验证功能
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

# 导入数据库相关模块
from table.TJCAUSP_ORIGIN import TJCAUSP_ORIGIN
from sql.database import get_db_session
import pandas as pd

def test_db_connection():
    print("测试数据库连接和查询...")
    
    # 连接数据库并查询TJCAUSP_ORIGIN表的数据
    with get_db_session() as db:
        # 查询所有数据
        records = db.query(TJCAUSP_ORIGIN).all()
        
        # 将查询结果转换为pandas DataFrame
        data = []
        for record in records:
            data.append({
                'id': record.id,
                'ORDER_NO': record.ORDER_NO,
                'ORDER_MONTH': record.ORDER_MONTH,
                'IN_MAT_MIN_WIDTH': record.IN_MAT_MIN_WIDTH,
                'IN_MAT_MAX_WIDTH': record.IN_MAT_MAX_WIDTH,
                'IN_MAT_MIN_THICK': record.IN_MAT_MIN_THICK,
                'IN_MAT_MAX_THICK': record.IN_MAT_MAX_THICK
            })
        
        df = pd.DataFrame(data)
        print(f"从TJCAUSP_ORIGIN表中查询到 {len(df)} 条记录")
        print("\n前5条记录：")
        if not df.empty:
            print(df.head())
            
            # 显示数据的基本统计信息
            print("\n数据基本统计信息：")
            print(df.describe())
            
            print("\n各订单月份的数量统计：")
            print(df['ORDER_MONTH'].value_counts())
        
        return df

def test_specific_query():
    print("\n" + "="*50)
    print("测试特定查询...")
    
    with get_db_session() as db:
        # 查询入料最大宽度大于1000的数据
        wide_materials = db.query(TJCAUSP_ORIGIN) \
            .filter(TJCAUSP_ORIGIN.IN_MAT_MAX_WIDTH > 1000) \
            .all()
        
        wide_data = []
        for record in wide_materials:
            wide_data.append({
                'ORDER_NO': record.ORDER_NO,
                'ORDER_MONTH': record.ORDER_MONTH,
                'IN_MAT_MIN_WIDTH': record.IN_MAT_MIN_WIDTH,
                'IN_MAT_MAX_WIDTH': record.IN_MAT_MAX_WIDTH,
                'IN_MAT_MIN_THICK': record.IN_MAT_MIN_THICK,
                'IN_MAT_MAX_THICK': record.IN_MAT_MAX_THICK
            })
        
        wide_df = pd.DataFrame(wide_data)
        print(f"入料最大宽度大于1000的记录数：{len(wide_df)}")
        print(wide_df.head())

if __name__ == "__main__":
    df = test_db_connection()
    test_specific_query()
    print("\n测试完成！")