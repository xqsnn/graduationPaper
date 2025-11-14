#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查数据库结构和内容
"""
import sqlite3
import os

# 数据库路径
db_path = os.path.join(os.path.dirname(__file__), 'sql', 'cold_rolling.db')
print(f"检查数据库文件: {db_path}")

# 检查文件是否存在
if not os.path.exists(db_path):
    print("数据库文件不存在！")
else:
    print("数据库文件存在")
    
    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 检查所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"数据库中的表: {tables}")
    
    if tables:
        # 检查每个表的结构
        for table_name, in tables:
            print(f"\n表 {table_name} 的结构:")
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col}")
            
            # 检查表中的记录数量
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"  记录数量: {count}")
    
    conn.close()