from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ORM基类
Base = declarative_base()

# 定义ORM模型，匹配TJCAUSP_ORIGIN表结构
class TJCAUSP_ORIGIN(Base):
    __tablename__ = "TJCAUSP_ORIGIN"
    
    id = Column(Integer, primary_key=True, index=True)
    ORDER_NO = Column(String, index=True)  # 订单号
    ORDER_MONTH = Column(String)  # 订单月份
    IN_MAT_MIN_WIDTH = Column(Float)  # 入料最小宽度
    IN_MAT_MAX_WIDTH = Column(Float)  # 入料最大宽度
    IN_MAT_MIN_THICK = Column(Float)  # 入料最小厚度
    IN_MAT_MAX_THICK = Column(Float)  # 入料最大厚度