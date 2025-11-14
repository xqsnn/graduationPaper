from datetime import timedelta
import pandas as pd
import numpy as np

from algorithm.order_plan.rule_heuristic_t.order_schedule_result import OrderScheduleResult
from algorithm.order_plan.rule_heuristic_t.strategy.strategy import strategy
from algorithm.order_plan.order_plan_parameter import inventory_limit


class InventoryConstrainedStrategy(strategy):
    """
    库存约束策略，旨在最小化拖期的同时严格满足库存限制要求
    """
    strategy_name = 'INVENTORY_CONSTRAINED'

    def schedule(self, orders: pd.DataFrame, static_params: dict[str, dict[int, dict[str, float]]],
                 **kwargs) -> pd.DataFrame:
        # 通用预处理逻辑
        orders = self.preprocess_orders(orders)

        # 提取工序时间参数
        orders = self.extract_process_times(orders, static_params)

        # 转换时间格式
        orders = self.convert_time_formats(orders)

        # 初始化调度结果列表
        scheduled_orders_results = []

        # 设置默认权重
        w1 = kwargs.get('w1', 1.0)  # 酸轧后库权重
        w2 = kwargs.get('w2', 0.5)  # 连退后库权重
        w3 = kwargs.get('w3', 100.0)  # 拖期惩罚权重
        w4 = kwargs.get('w4', 0.1)  # 提前交货惩罚权重
        
        # 获取库存限制，从参数或默认值中获取
        inv_limit = kwargs.get('inventory_limit', inventory_limit)
        if isinstance(inv_limit, (list, tuple)) and len(inv_limit) >= 2:
            acid_warehouse_limit = inv_limit[0]  # 酸轧后库限制
            anneal_warehouse_limit = inv_limit[1]  # 连退后库限制
        else:
            acid_warehouse_limit = 1000.0
            anneal_warehouse_limit = 3000.0

        # 初始化机器可用时间
        machine_available_time_az = orders['PLAN_START_DATE'].min()
        machine_available_time_lt = orders['PLAN_START_DATE'].min()

        # 使用列表跟踪当前库存情况
        # 每个库存项包含：订单号、重量、开始时间、结束时间
        inventory_az = []  # 酸轧后库存的订单
        inventory_lt = []  # 连退后库存的订单

        # 按照最早交货期优先进行排序
        orders = orders.sort_values(by=['DELIVERY_DATE', 'ORDER_WT'], ascending=[True, False]).reset_index(drop=True)

        # 逐个调度订单
        for idx, order in orders.iterrows():
            order_no = order['ORDER_NO']
            order_wt = order['ORDER_WT']

            # 基础调度尝试
            az_proc_time = order['AZ_PROCESS_TIME_TD']
            lt_proc_time = order['LT_PROCESS_TIME_TD']
            az_trans_time = order['AZ_TRANSMISSION_TIME_TD']
            delivery_date = order['DELIVERY_DATE']

            # 找到一个满足库存约束的酸轧开始时间
            s_az = max(order['PLAN_START_DATE'], machine_available_time_az)
            
            # 循环直到找到满足所有约束的调度时间
            max_delay_hours = 24 * 30  # 最大延迟1个月，避免无限循环
            delay_hours = 0
            
            while delay_hours <= max_delay_hours:
                test_s_az = s_az + timedelta(hours=delay_hours)
                test_e_az = test_s_az + az_proc_time
                test_s_lt = max(test_e_az + az_trans_time, machine_available_time_lt)
                test_e_lt = test_s_lt + lt_proc_time

                # 检查酸轧后库存约束
                acid_inv_ok = self.check_inventory_constraint_at_time(
                    inventory_az, 
                    test_e_az + az_trans_time, 
                    test_s_lt, 
                    acid_warehouse_limit, 
                    order_wt
                )
                
                # 检查连退后库存约束
                anneal_inv_ok = self.check_inventory_constraint_at_time(
                    inventory_lt, 
                    test_e_lt, 
                    delivery_date, 
                    anneal_warehouse_limit, 
                    order_wt
                )
                
                # 检查机器时间约束
                machine_az_ok = test_s_az >= machine_available_time_az
                machine_lt_ok = test_s_lt >= machine_available_time_lt

                # 如果所有约束都满足，就接受这个调度
                if acid_inv_ok and anneal_inv_ok and machine_az_ok and machine_lt_ok:
                    s_az = test_s_az
                    e_az = test_e_az
                    s_lt = test_s_lt
                    e_lt = test_e_lt
                    break

                delay_hours += 1  # 逐步增加延迟时间

            # 如果在最大延迟范围内都没有找到满足约束的调度时间，则使用基础调度并记录
            if delay_hours > max_delay_hours:
                s_az = s_az + timedelta(hours=max_delay_hours)
                e_az = s_az + az_proc_time
                s_lt = max(e_az + az_trans_time, machine_available_time_lt)
                e_lt = s_lt + lt_proc_time

            # 更新酸轧机可用时间
            machine_available_time_az = e_az
            
            # 更新连退机可用时间
            machine_available_time_lt = e_lt

            # 计算拖期和库存
            delay_hours = max(0, (e_lt - delivery_date).total_seconds() / 3600)
            
            # 计算酸轧后库存时间
            inventory_az_time = max(0, (s_lt - (e_az + az_trans_time)).total_seconds() / 3600)
            
            # 计算连退后库存时间
            inventory_lt_time = max(0, (delivery_date - e_lt).total_seconds() / 3600)

            # 计算库存重量
            inventory_az_wt = order_wt if inventory_az_time > 0 else 0
            inventory_lt_wt = order_wt if inventory_lt_time > 0 else 0

            # 添加到库存列表
            if inventory_az_time > 0:
                inventory_az.append({
                    'order_no': order_no,
                    'weight': order_wt,
                    'start_time': e_az + az_trans_time,
                    'end_time': s_lt
                })
                
            if inventory_lt_time > 0:
                inventory_lt.append({
                    'order_no': order_no,
                    'weight': order_wt,
                    'start_time': e_lt,
                    'end_time': delivery_date
                })

            # 从库存中移除已完成的订单（当前时间之后没有库存的订单）
            current_time = max(e_az, e_lt)
            inventory_az = [inv for inv in inventory_az if inv['end_time'] > current_time]
            inventory_lt = [inv for inv in inventory_lt if inv['end_time'] > current_time]

            result = OrderScheduleResult(
                order_no, 
                delivery_date, 
                s_az, 
                e_az, 
                s_lt, 
                e_lt, 
                delay_hours,
                inventory_az_time, 
                inventory_az_wt, 
                inventory_lt_time, 
                inventory_lt_wt,
                static_params[order_no][0]['process_time'],
                static_params[order_no][1]['process_time']
            )

            scheduled_orders_results.append(result.to_dict())

        return pd.DataFrame(scheduled_orders_results)

    def check_inventory_constraint_at_time(self, inventory_list, start_time, end_time, limit, order_weight):
        """
        检查在给定时间段内库存是否超出限制
        """
        # 检查时间点是否有库存重叠
        if not inventory_list:
            return (order_weight <= limit)
        
        # 将新订单加入临时库存列表，检查在时间范围内的最大库存
        temp_inventory = inventory_list + [{
            'start_time': start_time,
            'end_time': end_time,
            'weight': order_weight
        }]
        
        # 获取所有时间点
        time_points = set()
        for inv in temp_inventory:
            time_points.add(inv['start_time'])
            time_points.add(inv['end_time'])
        
        time_points = sorted(list(time_points))
        
        # 检查每个时间段内的库存总量
        for i in range(len(time_points) - 1):
            current_time = time_points[i]
            next_time = time_points[i + 1]
            
            # 计算这个时间段内的库存总重量
            current_inventory = 0
            for inv in temp_inventory:
                if inv['start_time'] <= current_time and inv['end_time'] > current_time:
                    current_inventory += inv['weight']
            
            if current_inventory > limit:
                return False  # 有时间段超限
        
        return True  # 所有时间段都不超限