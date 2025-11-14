from datetime import timedelta

import pandas as pd

from algorithm.order_plan.rule_heuristic_t.order_schedule_result import OrderScheduleResult


class strategy:

    strategy_name = "default"

    def schedule(self, orders: pd.DataFrame, static_params: dict[str, dict[int, dict[str, float]]],
                 **kwargs) -> pd.DataFrame:
        # 通用预处理逻辑
        orders = self.preprocess_orders(orders)

        # 提取工序时间参数
        orders = self.extract_process_times(orders, static_params)

        # 转换时间格式
        orders = self.convert_time_formats(orders)

        # 初始化机器时间
        machine_available_time_az, machine_available_time_lt = self.initialize_machine_times(orders)

        # 排序逻辑（子类可重写）
        orders = self.sort_orders(orders)

        # 调度核心逻辑
        return self.schedule_orders(orders, static_params, machine_available_time_az, machine_available_time_lt)

    def preprocess_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        # 将交货日期转换为 datetime 对象
        orders['DELIVERY_DATE'] = pd.to_datetime(orders['DELIVERY_DATE'], format='%Y%m%d')
        # 将开始时间设为订货月份月初
        orders['PLAN_START_DATE'] = orders['ORDER_MONTH'].apply(
            lambda x: pd.to_datetime(str(x)[:4] + '-' + str(x)[4:] + '-01'))
        return orders

    def extract_process_times(self, orders: pd.DataFrame, static_params: dict) -> pd.DataFrame:
        # 提取合同处理时间和传输时间
        for idx, row in orders.iterrows():
            order_no = row['ORDER_NO']

            # 酸轧工序 (process_num = 0) 参数
            az_params = static_params.get(order_no, {}).get(0, {})
            orders.loc[idx, 'AZ_PROCESS_TIME_HOURS'] = az_params.get('process_time', 0)
            orders.loc[idx, 'AZ_TRANSMISSION_TIME_HOURS'] = az_params.get('transmission_time', 0)

            # 连退工序 (process_num = 1) 参数
            lt_params = static_params.get(order_no, {}).get(1, {})
            orders.loc[idx, 'LT_PROCESS_TIME_HOURS'] = lt_params.get('process_time', 0)
            orders.loc[idx, 'LT_TRANSMISSION_TIME_HOURS'] = lt_params.get('transmission_time', 0)
        return orders

    def convert_time_formats(self, orders: pd.DataFrame) -> pd.DataFrame:
        # 将处理时间和传输时间从小时转换为 timedelta 对象，便于日期时间计算
        orders['AZ_PROCESS_TIME_TD'] = orders['AZ_PROCESS_TIME_HOURS'].apply(lambda x: timedelta(hours=x))
        orders['AZ_TRANSMISSION_TIME_TD'] = orders['AZ_TRANSMISSION_TIME_HOURS'].apply(lambda x: timedelta(hours=x))
        orders['LT_PROCESS_TIME_TD'] = orders['LT_PROCESS_TIME_HOURS'].apply(lambda x: timedelta(hours=x))
        return orders

    def initialize_machine_times(self, orders: pd.DataFrame) -> tuple:
        # 初始化机组可用时间，从所有订单中最早的计划开始日期开始
        machine_available_time_az = orders['PLAN_START_DATE'].min()
        machine_available_time_lt = orders['PLAN_START_DATE'].min()
        return machine_available_time_az, machine_available_time_lt

    def sort_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        # 默认排序方法，可在子类中重写
        return orders.sort_values(by='DELIVERY_DATE').reset_index(drop=True)

    def schedule_orders(self, orders: pd.DataFrame, static_params: dict,
                        machine_available_time_az, machine_available_time_lt) -> pd.DataFrame:
        """
        调度核心逻辑
        """
        scheduled_orders_results = []

        for idx, order in orders.iterrows():
            order_no = order['ORDER_NO']

            # 酸轧工序调度
            s_az = max(order['PLAN_START_DATE'], machine_available_time_az)
            e_az = s_az + order['AZ_PROCESS_TIME_TD']
            machine_available_time_az = e_az

            # 连退工序调度
            s_lt = max(e_az + order['AZ_TRANSMISSION_TIME_TD'], machine_available_time_lt)
            e_lt = s_lt + order['LT_PROCESS_TIME_TD']
            machine_available_time_lt = e_lt

            # 结果评估和记录
            delay_hours = max(0, (e_lt - order['DELIVERY_DATE']).total_seconds() / 3600)
            inventory_az_hours = max(0, (s_lt - (e_az + order['AZ_TRANSMISSION_TIME_TD'])).total_seconds() / 3600)
            inventory_az_wt = order['ORDER_WT'] * int(inventory_az_hours > 0)
            inventory_lt_hours = max(0, (order['DELIVERY_DATE'] - e_lt).total_seconds() / 3600)
            inventory_lt_wt = order['ORDER_WT'] * int(inventory_lt_hours > 0)

            result = OrderScheduleResult(order_no, order['DELIVERY_DATE'], s_az, e_az, s_lt, e_lt, delay_hours,
                                         inventory_az_hours, inventory_az_wt, inventory_lt_hours, inventory_lt_wt,
                                         static_params[order_no][0]['process_time'],
                                         static_params[order_no][1]['process_time'])

            scheduled_orders_results.append(result.to_dict())

        return pd.DataFrame(scheduled_orders_results)
