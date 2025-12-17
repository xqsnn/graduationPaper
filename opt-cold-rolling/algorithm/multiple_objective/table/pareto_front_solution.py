import pandas as pd
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# ORM基类
Base = declarative_base()

class ParetoFrontSolution(Base):
    __tablename__ = "pareto_front_solution"

    id = Column(Integer, primary_key=True, nullable=False)
    task_id = Column(String, nullable=False)
    inventory = Column(Float, nullable=False)
    tardiness = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    start_time = Column(String, nullable=True)
    end_time = Column(String, nullable=True)
    reserved_1 = Column(String, nullable=True)  # 可用于存储其他目标值
    reserved_2 = Column(String, nullable=True)  # 可用于存储额外信息
    reserved_3 = Column(String, nullable=True)  # 可用于存储备注

    @classmethod
    def to_dataframe(cls, table_data: list):
        """
        将 ParetoFrontSolution 对象列表转换为 pandas DataFrame

        :param table_data: list 的 ParetoFrontSolution 对象
        :return: 包含数据的 DataFrame
        """
        if not table_data:
            column_names = [c.name for c in cls.__table__.columns]
            return pd.DataFrame(columns=column_names)

        column_names = [c.name for c in cls.__table__.columns]  # 统一使用类定义，避免依赖实例
        data = [{col: getattr(record, col) for col in column_names} for record in table_data]
        return pd.DataFrame(data)

    @classmethod
    def df_to_orders(cls, df: pd.DataFrame):
        """
        将 DataFrame 转换为 ParetoFrontSolution 对象列表

        :param df: 包含 ParetoFrontSolution 数据的 DataFrame
        :return: list 的 ParetoFrontSolution 对象
        """
        # 确保 DataFrame 列存在，缺失列用 None 填充（尤其 reserved_* 可能为空）
        column_names = [c.name for c in cls.__table__.columns if c.name != 'id']  # 通常 id 由数据库生成，但若 DataFrame 中有则保留

        records = df.to_dict(orient='records')
        res = []
        for item in records:
            # 对 nullable 字段，使用 .get 避免 KeyError
            res.append(cls(
                id=item['id'],
                task_id=item['task_id'],
                inventory=item['inventory'],
                tardiness=item['tardiness'],
                cost=item['cost'],
                start_time=item.get('start_time'),
                end_time=item.get('end_time'),
                reserved_1=item.get('reserved_1'),
                reserved_2=item.get('reserved_2'),
                reserved_3=item.get('reserved_3')
            ))
        return res