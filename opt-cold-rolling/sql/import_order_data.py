"""
创建数据库表并导入Excel数据的脚本
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os
from dotenv import load_dotenv

# 添加项目根目录到Python路径，以便导入table模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from table.order import Base, order

def create_database_and_import_data(excel_path, database_url):
    """
    创建数据库表并从Excel文件导入订单数据
    
    Args:
        excel_path: Excel文件路径
        database_url: 数据库连接URL
    """
    # 创建数据库引擎
    engine = create_engine(database_url)
    
    # 创建所有表（如果不存在）
    print("正在创建数据库表...")
    Base.metadata.create_all(engine)
    print("数据库表创建完成!")
    
    # 创建会话
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 读取Excel文件
        print(f"正在读取Excel文件: {excel_path}")
        df = pd.read_excel(excel_path)
        
        # 打印数据形状和前几行作为预览
        print(f"Excel数据形状: {df.shape}")
        print("前5行数据:")
        print(df.head())
        
        # 检查Excel列名是否与数据库表匹配
        required_columns = ['ORDER_NO', 'ORDER_MONTH_PRO', 'ORDER_WT', 'DELIVY_DATE']
        for col in required_columns:
            if col not in df.columns:
                print(f"错误: Excel文件中缺少列 '{col}'")
                return False
                
        # 重命名Excel中的列以匹配数据库表结构
        df = df.rename(columns={
            'ORDER_MONTH_PRO': 'ORDER_MONTH',
            'DELIVY_DATE': 'DELIVERY_DATE'
        })
        
        # 清理数据 - 确保空值被正确处理
        df['ORDER_NO'] = df['ORDER_NO'].fillna('')
        df['ORDER_MONTH'] = df['ORDER_MONTH'].fillna('')
        df['ORDER_WT'] = df['ORDER_WT'].fillna(0.0)
        df['DELIVERY_DATE'] = df['DELIVERY_DATE'].fillna('')
        
        # 检查数据库是否已有数据
        existing_count = session.query(order).count()
        if existing_count > 0:
            print(f"警告: 数据库中已存在 {existing_count} 条记录，将跳过导入以避免重复数据")
            print("如果要重新导入，请先清空表中的数据")
            return True
        
        # 将数据转换为字典列表
        records = df.to_dict('records')
        
        print(f"准备插入 {len(records)} 条记录到order表...")
        
        # 批量插入数据
        for record in records:
            new_order = order(
                ORDER_NO=str(record['ORDER_NO']) if pd.notna(record['ORDER_NO']) else '',
                ORDER_MONTH=str(record['ORDER_MONTH']) if pd.notna(record['ORDER_MONTH']) else '',
                ORDER_WT=float(record['ORDER_WT']) if pd.notna(record['ORDER_WT']) else 0.0,
                DELIVERY_DATE=str(record['DELIVERY_DATE']) if pd.notna(record['DELIVERY_DATE']) else ''
            )
            session.add(new_order)
        
        # 提交事务
        session.commit()
        
        print(f"成功导入 {len(records)} 条记录到order表!")
        
        # 验证导入的数据
        count = session.query(order).count()
        print(f"order表中现在共有 {count} 条记录")
        
        return True
        
    except Exception as e:
        print(f"导入数据时发生错误: {str(e)}")
        session.rollback()  # 回滚事务
        return False
        
    finally:
        session.close()

def query_sample_data(database_url):
    """
    查询并显示一些样本数据以验证导入
    """
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 查询前10条记录
        sample_records = session.query(order).limit(10).all()
        print("\n前10条导入的记录:")
        for record in sample_records:
            print(f"ID: {record.id}, 订单号: {record.ORDER_NO}, 月份: {record.ORDER_MONTH}, "
                  f"重量: {record.ORDER_WT}, 交货日期: {record.DELIVERY_DATE}")
    except Exception as e:
        print(f"查询数据时发生错误: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    # Excel文件路径
    script_path = Path(__file__)
    excel_file_path = script_path.parent.parent / "data" / "合同计划.xls"
    
    # 检查Excel文件是否存在
    if not os.path.exists(excel_file_path):
        print(f"Excel文件不存在: {excel_file_path}")
        exit(1)
    
    # 使用默认数据库URL或从环境变量获取
    # 加载 .env 文件
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")

    # 执行创建表和导入数据
    success = create_database_and_import_data(excel_file_path, database_url)
    
    if success:
        print("数据导入完成!")
        # 显示一些样本数据
        query_sample_data(database_url)
    else:
        print("数据导入失败!")
        exit(1)