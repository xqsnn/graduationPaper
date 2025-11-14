from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

# ORM基类
Base = declarative_base()


# 定义ORM模型，匹配TJCAUSP表结构
class TJCAUSP(Base):
    __tablename__ = "TJCAUSP"

    id = Column(Integer, primary_key=True, index=True)
    ORDER_NO = Column(String, index=True)  # 订单号
    ORDER_MONTH = Column(String)  # 订单月份
    IN_MAT_WIDTH = Column(Float)  # 入料宽度
    IN_MAT_THICK = Column(Float)  # 入料厚度