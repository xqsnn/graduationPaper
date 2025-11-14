from datetime import timedelta

import pandas as pd

from algorithm.order_plan.rule_heuristic_t.order_schedule_result import OrderScheduleResult
from algorithm.order_plan.rule_heuristic_t.strategy.strategy import strategy


class MIN_INV_strategy(strategy):
    """
    MIN_INV策略，最小库存
    """
    strategy_name = 'MIN_INV'

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

        # 设定初始的“最晚可用结束时间”为所有交货日的最大值（也可以用 datetime.max）
        max_delivery = orders['DELIVERY_DATE'].max()
        lt_latest_end = max_delivery
        az_latest_end = max_delivery

        # 权重：可调整
        # w1: 酸轧后库权重，w2: 连退后库权重，w3: 拖期惩罚权重
        # 如果没有传入这些参数，可以在函数签名给默认值，例如 w1=1.0, w2=0.5, w3=100.0
        w1 = kwargs.get('w1', 1.0)
        w2 = kwargs.get('w2', 0.5)
        w3 = kwargs.get('w3', 100.0)

        while not unscheduled.empty:
            best_candidate = None
            best_score = float('inf')

            for idx, order in unscheduled.iterrows():
                # 处理时间（小时）
                az_proc_h = order['AZ_PROCESS_TIME_HOURS']
                lt_proc_h = order['LT_PROCESS_TIME_HOURS']
                az_trans_h = order['AZ_TRANSMISSION_TIME_HOURS']

                # 先尝试从交货期向前倒推 LT 完成时间（不晚于交货日，也不晚于 LT 当前最晚可用结束）
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
                    # LT 开始不能早于 e_az + trans，也不能早于 0；LT 可能需要向后推到 az_latest_end 的限制
                    tentative_s_lt = max(tentative_e_az + timedelta(hours=az_trans_h),
                                         lt_latest_end - timedelta(days=365 * 100))  # keep it flexible; we handle below
                    # 为了简洁，把 tentative_s_lt 至少设为 e_az + trans
                    tentative_s_lt = tentative_e_az + timedelta(hours=az_trans_h)
                    tentative_e_lt = tentative_s_lt + timedelta(hours=lt_proc_h)
                    # 如果这导致 E_LT 超过交货日，则会产生 delay
                # 计算库存与延迟
                I_AZ = max(0.0,
                           (tentative_s_lt - (tentative_e_az + timedelta(hours=az_trans_h))).total_seconds() / 3600.0)
                I_AZ_WT = order['ORDER_WT'] * int(I_AZ > 0)
                           
                # 修正连退后库存计算：当订单提前完成时，库存时间是订单完成到交货期的时间
                I_LT = max(0.0, (order['DELIVERY_DATE'] - tentative_e_lt).total_seconds() / 3600.0)
                I_LT_WT = order['ORDER_WT'] * int(I_LT > 0)

                delay_hours = max(0.0, (tentative_e_lt - order['DELIVERY_DATE']).total_seconds() / 3600.0)

                # 优化评分函数：对延迟施加更高惩罚，同时平衡酸轧后库和连退后库存
                score = w1 * I_AZ + w2 * I_LT + w3 * delay_hours

                # 可选：轻微偏好不强制前推的方案（避免推早导致连锁影响）
                if forced_forward:
                    score += 1e-6  # tiny penalty, 可去掉或调整

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
                        'I_LT': I_LT,
                        'I_AZ_WT': I_AZ_WT,
                        'I_LT_WT': I_LT_WT,
                        'delay': delay_hours
                    }

            # 固定最优候选到调度表
            if best_candidate is None:
                # 不应发生，但为安全性加个保护
                break

            b = best_candidate
            order = b['order']

            result = OrderScheduleResult(order['ORDER_NO'], order['DELIVERY_DATE'], b['S_AZ'], b['E_AZ'], b['S_LT'], b['E_LT'], b['delay'],
                                         b['I_AZ'], b['I_AZ_WT'],b['I_LT'],b['I_LT_WT'],
                                         static_params[order['ORDER_NO']][0]['process_time'],
                                         static_params[order['ORDER_NO']][1]['process_time'])

            scheduled_orders_results.append(result.to_dict())
            # 更新机器的“最晚可用结束时间”：因为我们向前占用了时间段，所以机器可用的最新结束应变为该任务的开始
            lt_latest_end = b['S_LT']
            az_latest_end = b['S_AZ']

            # 从未排集合中移除
            unscheduled = unscheduled.drop(b['idx'])

            # 返回按你固定顺序（这里为向后排的顺序）构造的 DataFrame
        result_df = pd.DataFrame(scheduled_orders_results)
        # 如果需要按时间或其他排序返回，可以在这里排序
        return result_df