from datetime import datetime

import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from algorithm.order_plan.multi_machines.object.order_multi_feature import OrderMultiFeature
from sql.database import get_db_session
from table.TOCAUSP import TOCAUSP


class DataProcess:
    """
    数据处理类
    """

    orders = list[OrderMultiFeature]


    def get_all_orders(self) -> list[OrderMultiFeature]:
        # 获取所有订单数据
        with get_db_session() as db:
            query = db.query(TOCAUSP)
            TOCAUSP_records = query.all()
            orders = []
            for record in TOCAUSP_records:
                # 将 TOCAUSP 记录转换为 OrderMultiFeature 对象

                plan_start_date = str(record.ORDER_MONTH_PRO) + "01"
                order = OrderMultiFeature(
                    order_no=record.ORDER_NO,
                    delivery_date=datetime.strptime(str(record.DELIVY_DATE), "%Y%m%d"),
                    plan_start_date=datetime.strptime(plan_start_date, "%Y%m%d"),
                    order_wt=float(record.ORDER_WT) if record.ORDER_WT else 0.0,
                    order_width=float(record.ORDER_WIDTH) if record.ORDER_WIDTH else 0.0,
                    order_thick=float(record.ORDER_THICK) if record.ORDER_THICK else 0.0
                )
                orders.append(order)
            return  orders

    def get_all_orders_data_frame(self) -> DataFrame:
        # 获取所有订单数据
        with get_db_session() as db:
            query = db.query(TOCAUSP)
            TOCAUSP_records = query.all()
            orders = []
            for record in TOCAUSP_records:
                # 将 TOCAUSP 记录转换为 OrderMultiFeature 对象
                plan_start_date = str(record.ORDER_MONTH_PRO) + "01"
                order = OrderMultiFeature(
                    order_no=record.ORDER_NO,
                    delivery_date=datetime.strptime(str(record.DELIVY_DATE), "%Y%m%d"),
                    plan_start_date=datetime.strptime(plan_start_date, "%Y%m%d"),
                    order_wt=float(record.ORDER_WT) if record.ORDER_WT else 0.0,
                    order_width=float(record.ORDER_WIDTH) if record.ORDER_WIDTH else 0.0,
                    order_thick=float(record.ORDER_THICK) if record.ORDER_THICK else 0.0
                )
                orders.append(order)
            orders_df = pd.DataFrame([
                {
                    'order_no': order.order_no,
                    'delivery_date': order.delivery_date,
                    'plan_start_date': order.plan_start_date,
                    'order_wt': order.order_wt,
                    'order_width': order.order_width,
                    'order_thick': order.order_thick
                } for order in orders
            ])
            return orders_df

