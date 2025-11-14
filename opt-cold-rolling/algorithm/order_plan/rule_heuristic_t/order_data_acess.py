import pandas as pd

from sql.database import get_db_session
# 导入数据库相关模块
from table.order import order  # 导入新表的ORM
from algorithm.order_plan.order_plan_parameter import process_num, speed, transmission_time


class OrderDataAccess:
    """
    订单数据访问类
    """

    # 合同实体类
    orders = list[order]

    # 合同静态参数，包括处理时间和转运时间
    # 数据格式
    # {
    #     "order_no": {
    #         "process_num": {
    #             "process_time": 12,
    #             "transmission_time": 3
    #         }
    #     }
    # }
    orders_static_parameter = {}

    # 合同静态参数
    order_plan_parameter = {}

    def __init__(self):
        self.order_plan_parameter = {
            "process_num": process_num,
            "speed": speed,
            "transmission_time": transmission_time
        }

    def init_order_plan_parameter(self, _process_num: int, _speed: list[float], _transmission_time: list[float]):
        """
        :param _process_num: 工序号
        :param _speed: 处理速度
        :param _transmission_time: 转运时间
        :return:
        """
        self.order_plan_parameter = {
            "process_num": _process_num,
            "speed": _speed,
            "transmission_time": _transmission_time
        }

    def get_all_orders(self) -> list[order]:
        # 连接数据库并查询order表的数据
        with get_db_session() as db:
            query = db.query(order)
            self.orders = query.all()
            return self.orders

    def get_orders_by_month(self, order_month: str) -> list[order]:
        """
        :param order_month: 合同订货月份
        :return: 过滤后的合同数据
        """
        all_orders = self.get_all_orders()

        # 筛选指定月份的订单数据
        filtered_orders = [order for order in all_orders if order.ORDER_MONTH == order_month]
        if len(filtered_orders) == 0:
            raise ValueError("当前月份没有数据")

        return filtered_orders

    def get_orders_by_month_and_limit(self, order_month: str, limit: int) -> list[order]:
        """
        :param order_month: 合同订货月份
        :param limit: 返回的订单数量限制
        :return: 指定月份的前limit个订单数据
        """
        orders = self.get_orders_by_month(order_month)

        # 返回指定数量的订单数据
        if limit > len(orders):
            raise ValueError("---数据超过当前月份合同总数, 当前月份： %{}，合同数量：%{}", order_month, len(orders))
        return orders[:limit]
        # order_no = ["GG24003300",
        # "G024013563",
        # "G024014114",
        # "GG24003449",
        # "GG24003298",
        # "G024013564",
        # "GG24003326",
        # "GG24003445",
        # "G024013530",
        # "GG24003314"]
        # filtered_orders = [order for order in orders if order.ORDER_NO in order_no]
        #
        # return filtered_orders

    @classmethod
    def get_orders_static_parameter(cls, orders: list[order]) -> dict[str, dict[int, dict[str, float]]]:
        """
        :param orders: 订单数据
        :return: 订单静态参数
        """
        orders_static_parameter = {}

        df_orders = order.to_dataframe(orders)

        # 获取订单号列表
        order_nos = df_orders["ORDER_NO"].tolist()

        for idx, order_no in enumerate(order_nos):
            orders_static_parameter[order_no] = {}

            # 对于每个工序
            for i in range(process_num):
                # 计算处理时间（注意需要转换为list后再索引）
                process_time = (df_orders["ORDER_WT"] / speed[i]).tolist()[idx]
                # 获取传输时间
                trans_time = transmission_time[i]

                # 构建嵌套字典结构
                orders_static_parameter[order_no][i] = {
                    'process_time': process_time,
                    'transmission_time': trans_time
                }
        cls.orders_static_parameter = orders_static_parameter
        return orders_static_parameter