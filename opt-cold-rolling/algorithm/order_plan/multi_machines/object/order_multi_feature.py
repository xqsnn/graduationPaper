import pandas as pd
from datetime import datetime

from openpyxl.descriptors import DateTime

from algorithm.order_plan.multi_machines.object.Operation import Operation
from algorithm.order_plan.multi_machines.object.static_parameter import StaticParameters


class OrderMultiFeature:
    """
    订单多特征类
    """

    order_no = ""
    delivery_date = None # datetime对象,形如20241225
    plan_start_date = None # datetime对象, 形如20241101
    order_wt = 0.0
    order_width = 0.0
    order_thick = 0.0

    # def __int__(self):
    #     pass

    def __init__(self, order_no: str, delivery_date, plan_start_date, order_wt: float, order_width: float,
                 order_thick: float):
        self.order_no = order_no
        self.delivery_date = delivery_date
        self.plan_start_date = plan_start_date
        self.order_wt = order_wt
        self.order_width = order_width
        self.order_thick = order_thick

    def to_dataframe(self):
        """
        将OrderMultiFeature对象转换为DataFrame
        """
        return pd.DataFrame([self.__dict__])