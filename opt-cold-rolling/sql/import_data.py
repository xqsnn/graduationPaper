import pandas as pd
import os
from table.TJCAUSP_ORIGIN import Base, TJCAUSP_ORIGIN
from sql.database import engine, get_db_session

def init_db():
    """
    初始化数据库，创建所有表
    """
    Base.metadata.create_all(bind=engine)
    print("数据库表创建成功！")

def import_excel_to_db(excel_path, db_path="../sql/cold_rolling.db"):
    """
    将Excel文件导入到数据库中
    """
    # 确保数据库表存在
    init_db()
    
    # 读取Excel文件
    try:
        df = pd.read_excel(excel_path)
        
        # 显示列名以供参考
        print("Excel文件的列名:", df.columns.tolist())
        print(f"总共 {len(df)} 行数据")
        
        # 使用数据库会话
        with get_db_session() as db:
            # 遍历DataFrame并插入到数据库
            for index, row in df.iterrows():
                try:
                    origin_record = TJCAUSP_ORIGIN(
                        ORDER_NO=str(row.iloc[0]) if len(row) > 0 and pd.notna(row.iloc[0]) else None,
                        ORDER_MONTH=str(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else None,
                        IN_MAT_MIN_WIDTH=float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else None,
                        IN_MAT_MAX_WIDTH=float(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else None,
                        IN_MAT_MIN_THICK=float(row.iloc[4]) if len(row) > 4 and pd.notna(row.iloc[4]) else None,
                        IN_MAT_MAX_THICK=float(row.iloc[5]) if len(row) > 5 and pd.notna(row.iloc[5]) else None
                    )
                    db.add(origin_record)
                except Exception as e:
                    print(f"处理第{index+1}行时出现错误: {e}")
                    # 如果按位置索引失败，打印该行的数据以供调试
                    print(f"行 {index+1} 的数据: {row.values}")
                    continue
            
            # 提交事务
            db.commit()
            print(f"成功导入 {len(df)} 条记录到数据库")
    except Exception as e:
        print(f"导入Excel文件时出现错误: {e}")

def query_data():
    """
    查询数据库中的数据作为示例
    """
    with get_db_session() as db:
        records = db.query(TJCAUSP_ORIGIN).all()
        print(f"数据库中总共有 {len(records)} 条记录")
        for record in records[:10]:  # 只显示前10条记录
            print(f"订单号: {record.ORDER_NO}, 订单月份: {record.ORDER_MONTH}, "
                  f"入料最小宽度: {record.IN_MAT_MIN_WIDTH}, 入料最大宽度: {record.IN_MAT_MAX_WIDTH}, "
                  f"入料最小厚度: {record.IN_MAT_MIN_THICK}, 入料最大厚度: {record.IN_MAT_MAX_THICK}")
        return records

def query_by_order_number(order_number):
    """
    根据订单号查询特定记录
    """
    with get_db_session() as db:
        record = db.query(TJCAUSP_ORIGIN).filter(TJCAUSP_ORIGIN.ORDER_NO == order_number).first()
        if record:
            print(f"找到订单 {order_number}:")
            print(f"  订单月份: {record.ORDER_MONTH}")
            print(f"  入料最小宽度: {record.IN_MAT_MIN_WIDTH}")
            print(f"  入料最大宽度: {record.IN_MAT_MAX_WIDTH}")
            print(f"  入料最小厚度: {record.IN_MAT_MIN_THICK}")
            print(f"  入料最大厚度: {record.IN_MAT_MAX_THICK}")
            return record
        else:
            print(f"未找到订单号为 {order_number} 的记录")
            return None

if __name__ == "__main__":
    print("开始初始化数据库...")
    init_db()
    
    # 检查Excel文件是否存在
    excel_file_path = "../data/连退作业计划数据.xls"
    if os.path.exists(excel_file_path):
        print(f"找到Excel文件: {excel_file_path}")
        print("开始导入数据...")
        import_excel_to_db(excel_file_path)
    else:
        print(f"未找到Excel文件: {excel_file_path}")
    
    print("\n数据查询示例:")
    query_data()