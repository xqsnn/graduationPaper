"""
数据库查询工具模块，用于从数据库获取调度结果并绘制甘特图
"""
import os
import pandas as pd
from database import get_db_session
from table.schedule_result import ScheduleResult
from plot_charts.plot_gant_charts import plot_schedule_gantt

def get_schedule_result_by_pareto_id(pareto_front_id: int):
    """
    根据pareto_front_id从数据库查询调度结果

    Args:
        pareto_front_id: 帕累托前沿解ID

    Returns:
        pandas.DataFrame: 包含调度结果的DataFrame
    """
    with get_db_session() as db:
        # 查询指定pareto_front_id的调度结果
        results = db.query(ScheduleResult).filter(
            ScheduleResult.pareto_front_id == pareto_front_id
        ).all()

        # 将结果转换为DataFrame
        if results:
            data = []
            for result in results:
                data.append({
                    'id': result.id,
                    'order_no': result.order_no,
                    'hr_start': result.hr_start,
                    'hr_end': result.hr_end,
                    'ar_start': result.ar_start,
                    'ar_end': result.ar_end,
                    'ar_machine': result.ar_machine,
                    'ca_start': result.ca_start,
                    'ca_end': result.ca_end,
                    'ca_machine': result.ca_machine,
                    'complete_time': result.complete_time,
                    'pareto_front_id': result.pareto_front_id
                })
            df = pd.DataFrame(data)
            return df
        else:
            print(f"未找到pareto_front_id为 {pareto_front_id} 的调度结果")
            return pd.DataFrame()


def get_table_data(table_class):
    """
    查询指定表的所有数据

    Args:
        table_class: 表类（如 ParetoFrontSolution, ScheduleResultDetail 等）

    Returns:
        pandas.DataFrame: 包含表中所有数据的DataFrame
    """
    with get_db_session() as db:
        # 查询表中所有记录
        records = db.query(table_class).all()

        if not records:
            # 如果表中没有数据，返回空DataFrame，列名为表的列名
            columns = [column.name for column in table_class.__table__.columns]
            return pd.DataFrame(columns=columns)

        # 获取列名
        columns = [column.name for column in table_class.__table__.columns]

        # 将记录转换为字典列表
        data = []
        for record in records:
            row = {column: getattr(record, column) for column in columns}
            data.append(row)

        # 返回DataFrame
        return pd.DataFrame(data)

