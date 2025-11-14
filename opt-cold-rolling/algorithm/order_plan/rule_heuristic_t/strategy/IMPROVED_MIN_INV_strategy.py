from datetime import timedelta

import pandas as pd

from algorithm.order_plan.rule_heuristic_t.order_schedule_result import OrderScheduleResult
from algorithm.order_plan.rule_heuristic_t.strategy.strategy import strategy


class IMPROVED_MIN_INV_strategy(strategy):
    """
    改进的MIN_INV策略，更全面地最小化酸轧和连退库存
    """
    strategy_name = 'IMPROVED_MIN_INV'

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

        # 调度核心
        unscheduled = orders.copy()
        scheduled_orders_results = []

        # 设定初始的"最晚可用结束时间"为所有交货日的最大值
        max_delivery = orders['DELIVERY_DATE'].max()
        lt_latest_end = max_delivery
        az_latest_end = max_delivery

        # 权重：可调整
        w1 = kwargs.get('w1', 1.0)  # 酸轧后库权重
        w2 = kwargs.get('w2', 1.0)  # 连退后库权重
        w3 = kwargs.get('w3', 100.0)  # 拖期惩罚权重
        w4 = kwargs.get('w4', 0.1)   # 提前交货惩罚权重（可选）

        while not unscheduled.empty:
            best_candidate = None
            best_score = float('inf')

            for idx, order in unscheduled.iterrows():
                # 处理时间（小时）
                az_proc_h = order['AZ_PROCESS_TIME_HOURS']
                lt_proc_h = order['LT_PROCESS_TIME_HOURS']
                az_trans_h = order['AZ_TRANSMISSION_TIME_HOURS']

                # 尝试从交货期向前倒推 LT 完成时间（不晚于交货日，也不晚于 LT 当前最晚可用结束）
                tentative_e_lt = min(order['DELIVERY_DATE'], lt_latest_end)
                tentative_s_lt = tentative_e_lt - timedelta(hours=lt_proc_h)

                # 预期 AZ 最晚结束 = LT 开始 - 传输（且不能晚于 az_latest_end）
                tentative_e_az = min(tentative_s_lt - timedelta(hours=az_trans_h), az_latest_end)
                tentative_s_az = tentative_e_az - timedelta(hours=az_proc_h)

                # 标记是否需要从前向后推（即倒推到计划开始前，不可行）
                forced_forward = False
                if tentative_s_az < order['PLAN_START_DATE']:
                    # 倒推不可行：把 AZ 开始强制设为计划开始，然后正向推算 AZ/ LT
                    forced_forward = True
                    tentative_s_az = order['PLAN_START_DATE']
                    tentative_e_az = tentative_s_az + timedelta(hours=az_proc_h)
                    tentative_s_lt = tentative_e_az + timedelta(hours=az_trans_h)
                    tentative_e_lt = tentative_s_lt + timedelta(hours=lt_proc_h)
                    
                # 计算酸轧后库库存时间：LT 开始时间 - (AZ 结束时间 + 传输时间)
                I_AZ = max(0.0,
                           (tentative_s_lt - (tentative_e_az + timedelta(hours=az_trans_h))).total_seconds() / 3600.0)
                
                # 计算连退后库库存时间：如果是提前交货，则为提前的时间；如果是延迟交货，则为延迟时间
                # 提前交货时间（库存积压时间）
                early_delivery_hours = max(0.0, (order['DELIVERY_DATE'] - tentative_e_lt).total_seconds() / 3600.0)
                # 延迟交货时间（违约时间）
                delay_hours = max(0.0, (tentative_e_lt - order['DELIVERY_DATE']).total_seconds() / 3600.0)
                
                # 改进评分函数，更全面地考虑库存和交货表现
                # 1. 酸轧后库存时间
                # 2. 连退后库存时间（提前交货）
                # 3. 拖期惩罚（更高的权重）
                # 4. 也可以考虑添加提前交货惩罚（避免过度提前生产）
                score = w1 * I_AZ + w2 * early_delivery_hours + w3 * delay_hours + w4 * early_delivery_hours

                # 可选：轻微偏好不强制前推的方案（避免推早导致连锁影响）
                if forced_forward:
                    score += 1e-6  # tiny penalty

                # 记录最小 score 的候选
                if score < best_score:
                    best_score = score
                    best_candidate = {
                        'idx': idx,
                        'order': order,
                        'S_AZ': tentative_s_az,
                        'E_AZ': tentative_e_az,
                        'S_LT': tentative_s_lt,
                        'E_LT': tentative_e_lt,
                        'I_AZ': I_AZ,
                        'I_LT': early_delivery_hours,  # 实际上是提前交货时间
                        'early_delivery_hours': early_delivery_hours,
                        'delay_hours': delay_hours
                    }

            # 固定最优候选到调度表
            if best_candidate is None:
                # 不应发生，但为安全性加个保护
                break

            b = best_candidate
            order = b['order']
            result = OrderScheduleResult(order['ORDER_NO'], order['DELIVERY_DATE'], b['S_AZ'], b['E_AZ'], b['S_LT'],
                                         b['E_LT'], b['delay_hours'],
                                         b['I_AZ'], order['ORDER_WT'] if b['I_AZ'] > 0 else 0,  # 酸轧后库存重量
                                         b['I_LT'], order['ORDER_WT'] if b['I_LT'] > 0 else 0,  # 连退后库存重量
                                         static_params[order['ORDER_NO']][0]['process_time'],
                                         static_params[order['ORDER_NO']][1]['process_time'])

            scheduled_orders_results.append(result.to_dict())

            # 更新机器的"最晚可用结束时间"：因为我们向前占用了时间段，所以机器可用的最新结束应变为该任务的开始
            lt_latest_end = b['S_LT']
            az_latest_end = b['S_AZ']

            # 从未排集合中移除
            unscheduled = unscheduled.drop(b['idx'])

        result_df = pd.DataFrame(scheduled_orders_results)
        # 如果需要按时间或其他排序返回，可以在这里排序
        return result_df