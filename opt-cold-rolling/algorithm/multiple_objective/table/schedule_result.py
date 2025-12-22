import pandas as pd
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

# ORM基类
Base = declarative_base()

class ScheduleResult(Base):
    __tablename__ = "schedule_result"

    id = Column(Integer, primary_key=True, nullable=False)
    order_no = Column(String, nullable=False)
    hr_start = Column(String, nullable=False)
    hr_end = Column(String, nullable=False)
    ar_start = Column(String, nullable=False)
    ar_end = Column(String, nullable=False)
    ar_machine = Column(Integer, nullable=False)
    ca_start = Column(String, nullable=False)
    ca_end = Column(String, nullable=False)  # 添加了ca_end字段
    ca_machine = Column(Integer, nullable=False)
    complete_time = Column(String, nullable=False)  # 新增：完成时间
    pareto_front_id = Column(Integer, nullable=False)

    @classmethod
    def to_dataframe(cls, table_data: list):
        """
        将 ScheduleResultDetail 对象列表转换为 pandas DataFrame

        :param table_data: list 的 ScheduleResultDetail 对象
        :return: 包含数据的 DataFrame
        """
        if not table_data:
            # 返回空 DataFrame，结构与表一致
            column_names = [c.name for c in cls.__table__.columns]
            return pd.DataFrame(columns=column_names)

        column_names = [c.name for c in cls.__table__.columns]
        data = [{col: getattr(record, col) for col in column_names} for record in table_data]
        return pd.DataFrame(data)

    @classmethod
    def df_to_orders(cls, df: pd.DataFrame):
        """
        将 DataFrame 转换为 ScheduleResultDetail 对象列表

        :param df: 包含 ScheduleResultDetail 数据的 DataFrame
        :return: list 的 ScheduleResultDetail 对象
        """
        records = df.to_dict(orient='records')
        res = [
            cls(
                id=item['id'],
                order_no=item['order_no'],
                hr_start=item['hr_start'],
                hr_end=item['hr_end'],
                ar_start=item['ar_start'],
                ar_end=item['ar_end'],
                ar_machine=item['ar_machine'],
                ca_start=item['ca_start'],
                ca_end=item['ca_end'],  # 添加了ca_end字段
                ca_machine=item['ca_machine'],
                complete_time=item['complete_time'],
                pareto_front_id=item['pareto_front_id']
            )
            for item in records
        ]
        return res