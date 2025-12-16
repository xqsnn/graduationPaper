import pandas as pd
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

# ORM基类
Base = declarative_base()

# 定义ORM模型，匹配order表结构
class order(Base):
    __tablename__ = "order_new"

    id = Column(Integer, primary_key=True, index=True)
    order_no = Column(String, index=True)  # 订单号
    category = Column(String)  # 钢种
    width = Column(Float) # 宽度
    thickness = Column(Float) # 厚度
    delivery_date = Column(String) # 交货日期


    @classmethod
    def to_dataframe(cls, order_data: list):
        """
        将order结果转换为pandas DataFrame

        :param order_data: list的 order 对象
        :return: 包含订单数据的DataFrame
        """
        column_names = order_data[0].__table__.columns.keys()
        data = [{column: getattr(record, column) for column in column_names} for record in order_data]
        df = pd.DataFrame(data)
        return df

    @classmethod
    def df_to_orders(cls, df: pd.DataFrame):
        """
        将DataFrame转换为order对象列表

        :param df: 包含订单数据的DataFrame
        :return: list的 order 对象
        """
        orders = [cls(
            id = item['id'],
            order_no = item['order_no'],
            category = item['category'],
            width = item['width'],
            thickness = item['thickness'],
            delivery_date = item['delivery_date']
        ) for item in df.to_dict(orient='records')]
        return orders