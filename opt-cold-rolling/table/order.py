import pandas as pd
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

# ORM基类
Base = declarative_base()

# 定义ORM模型，匹配order表结构
class order(Base):
    __tablename__ = "order"

    id = Column(Integer, primary_key=True, index=True)
    ORDER_NO = Column(String, index=True)  # 订单号
    ORDER_MONTH = Column(String)  # 订单月份
    ORDER_WT = Column(Float)  # 订单重量
    DELIVERY_DATE = Column(String)  # 交货日期

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
            id=item['id'],
            ORDER_NO=item['ORDER_NO'],
            ORDER_MONTH=item['ORDER_MONTH'],
            ORDER_WT=item['ORDER_WT'],
            DELIVERY_DATE=item['DELIVERY_DATE']
        ) for item in df.to_dict(orient='records')]
        return orders