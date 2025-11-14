from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
import copy

import pandas as pd
import plotly.graph_objects as go

# 假设你已经定义了 Operation, OrderMultiFeature 和 StaticParameters 类
# 为了代码完整性，我在这里重新定义一下，但请注意，实际使用时你将直接引用你已有的变量。

class Operation(Enum):
    HR = "HR"
    AR = "AR"
    CA = "CA"

@dataclass
class OrderMultiFeature:
    """
    订单多特征类
    """
    order_no: str
    delivery_date: datetime # datetime对象,形如20241225
    plan_start_date: datetime # datetime对象, 形如20241101
    order_wt: float
    order_width: float
    order_thick: float

@dataclass
class StaticParameters:
    """
    静态参数类
    """
    operation_sequence: List[Operation] = field(default_factory=lambda: [Operation.HR, Operation.AR, Operation.CA])
    transmission_time: Dict[Operation, float] = field(default_factory=lambda: {
        Operation.HR: 2.0,
        Operation.AR: 4.0,
        Operation.CA: 1.0
    })
    speed: Dict[Operation, List[float]] = field(default_factory=lambda: {
        Operation.HR: [30],
        Operation.AR: [14, 16],
        Operation.CA: [8.0, 12.0, 10.0]
    })
    stock_limit: Dict[Operation, List[float]] = field(default_factory=lambda: {
        Operation.HR: [0, 500.0],
        Operation.AR: [0, 700.0],
        Operation.CA: [0, 1500.0]
    })
    # 新增搭接费用参数，简化处理，实际中可能更复杂
    changeover_cost_matrix: Dict[Operation, Dict[Tuple[float, float], float]] = field(default_factory=dict)
    delay_penalty_per_hour: float = 6.4953 # 拖期惩罚每小时
    width_change_penalty_factor: float = 1 # 宽度变化惩罚因子，从窄到宽惩罚高

    def __post_init__(self):
        # 预计算搭接费用矩阵 (简化版，仅考虑宽度从宽到窄和从窄到宽两种情况)
        # 实际生产中搭接费用会更复杂，这里仅为示例
        widths = sorted(list(set([o.order_width for o in orders_obj_list]))) # 假设orders_obj_list已传入
        for op in self.operation_sequence:
            self.changeover_cost_matrix[op] = {}
            for w1 in widths:
                for w2 in widths:
                    if w1 == w2:
                        self.changeover_cost_matrix[op][(w1, w2)] = 0.0
                    elif w1 > w2: # 从宽到窄
                        self.changeover_cost_matrix[op][(w1, w2)] = (w1 - w2) * 5.0 # 假设一个较小费用
                    else: # 从窄到宽
                        self.changeover_cost_matrix[op][(w1, w2)] = (w2 - w1) * 20.0 * self.width_change_penalty_factor # 假设一个较高费用

@dataclass
class ScheduleEntry:
    """
    单个合同在某个工序某个机组上的生产计划
    """
    order_no: str
    operation: Operation
    machine_id: int # 机组ID，从0开始
    start_time: datetime
    end_time: datetime
    order_wt: float
    order_width: float

@dataclass
class MachineSchedule:
    """
    单个机组的生产时间线
    """
    machine_id: int
    operation: Operation
    schedule: List[ScheduleEntry] = field(default_factory=list)

@dataclass
class ProductionPlan:
    """
    一个完整的生产方案，包含所有工序和机组的调度
    """
    order_priority: List[str] # 对应染色体，订单号的优先级列表
    schedules: Dict[Operation, Dict[int, MachineSchedule]] = field(default_factory=dict)
    total_tardiness: float = 0.0
    total_cost: float = 0.0
    is_feasible: bool = True
    stock_violations: Dict[Operation, List[Tuple[datetime, float, float]]] = field(default_factory=dict) # (时间, 当前库存, 限制)

class Scheduler:
    def __init__(self, orders: List[OrderMultiFeature], params: StaticParameters):
        self.orders = orders
        self.orders_map = {order.order_no: order for order in orders}
        self.params = params

    def get_production_time(self, order_wt: float, machine_speed: float) -> timedelta:
        """计算生产所需时间"""
        if machine_speed <= 0:  # 避免除以零或负速度
            return timedelta(days=9999)  # 视为无限长时间
        return timedelta(hours=order_wt / machine_speed)



    # def get_production_time(self, order_wt: float, machine_speed: float) -> timedelta:
    #     """计算生产所需时间"""
    #     if machine_speed <= 0: # 避免除以零或负速度
    #         return timedelta(days=9999) # 视为无限长时间
    #     return timedelta(hours=order_wt / machine_speed)
    #
    def find_available_machine_slot_near_due_time(self, operation: Operation, machine_schedules: Dict[int, MachineSchedule],
                                    earliest_start: datetime, latest_end: datetime,
                                    current_order_no: str, current_order_width: float,
                                    consider_stock_at_slot_time: bool = False,
                                    current_plan_for_stock_check: Optional[ProductionPlan] = None) -> Optional[Tuple[int, datetime, datetime, float]]:


        """
        找到在 [earliest_start, latest_end] 内的可行插槽。
        策略：
          1️⃣ 必须在 latest_end 前完成；
          2️⃣ 尽量靠近 latest_end 完成（越晚越好）；
          3️⃣ 同等条件下，速度快（加工时间短）优先；
          4️⃣ 不与已有任务重叠；
          5️⃣ 对空机组，优先以 latest_end 为结束；若放不下，再尝试 earliest_start 开始。
        """
        order_to_schedule = self.orders_map[current_order_no]
        available_slots = []

        for machine_id, speed in enumerate(self.params.speed[operation]):
            production_duration = self.get_production_time(order_to_schedule.order_wt, speed)
            if production_duration > timedelta(days=900):
                continue

            machine_schedule = machine_schedules[machine_id].schedule

            # --- ① 机组完全空闲 ---
            if not machine_schedule:
                # 尝试以 latest_end 为结束时间
                candidate_end = latest_end
                candidate_start = candidate_end - production_duration
                if candidate_start >= earliest_start:
                    available_slots.append((machine_id, candidate_start, candidate_end, speed))
                else:
                    # 放不下时，退一步： earliest_start 开始，看能否在 latest_end 前完成
                    alt_start = earliest_start
                    alt_end = alt_start + production_duration
                    if alt_end <= latest_end:
                        available_slots.append((machine_id, alt_start, alt_end, speed))

            else:
                # --- ② 有排程的机组 ---

                # (1) 第一个任务前的空档
                first_entry = machine_schedule[0]
                gap_start = max(earliest_start, datetime.min)
                gap_finish = min(first_entry.start_time, latest_end)
                candidate_end = gap_finish
                candidate_start = candidate_end - production_duration
                if candidate_start >= gap_start and candidate_end > gap_start:
                    available_slots.append((machine_id, candidate_start, candidate_end, speed))

                # (2) 任务之间的缝隙
                for i in range(len(machine_schedule) - 1):
                    prev_entry = machine_schedule[i]
                    next_entry = machine_schedule[i + 1]
                    gap_start = max(earliest_start, prev_entry.end_time)
                    gap_finish = min(next_entry.start_time, latest_end)
                    candidate_end = gap_finish
                    candidate_start = candidate_end - production_duration
                    if candidate_start >= gap_start and candidate_end <= next_entry.start_time and candidate_end <= latest_end:
                        available_slots.append((machine_id, candidate_start, candidate_end, speed))

                # (3) 最后一个任务之后的空档
                last_entry = machine_schedule[-1]
                gap_start = max(earliest_start, last_entry.end_time)
                gap_finish = latest_end
                candidate_end = gap_finish
                candidate_start = candidate_end - production_duration
                if candidate_start >= gap_start:
                    available_slots.append((machine_id, candidate_start, candidate_end, speed))

        # --- ③ 没有可用插槽 ---
        if not available_slots:
            return None

        # --- ④ 选择最优方案：结束时间最晚优先，其次加工时间短 ---
        available_slots.sort(key=lambda x: (-x[2].timestamp(), (x[2] - x[1]).total_seconds()))
        best_slot = available_slots[0]
        return best_slot

    def find_available_machine_slot_machine_first(self, operation: Operation, machine_schedules: Dict[int, MachineSchedule],
                                    earliest_start: datetime, latest_end: datetime,
                                    current_order_no: str, current_order_width: float,
                                    consider_stock_at_slot_time: bool = False,
                                    current_plan_for_stock_check: Optional[ProductionPlan] = None) -> Optional[
        Tuple[int, datetime, datetime, float]]:
        """
        新策略：
          ✅ 机组空闲就立即安排生产；
          ✅ 不再考虑“靠近 latest_end 完成”；
          ✅ 找到第一个能放下的时间段就安排；
          ✅ 若有多个机组，优先选择能最早开始的（同等条件下速度快）。
        """
        order_to_schedule = self.orders_map[current_order_no]
        available_slots = []

        for machine_id, speed in enumerate(self.params.speed[operation]):
            production_duration = self.get_production_time(order_to_schedule.order_wt, speed)
            if production_duration > timedelta(days=900):
                continue

            machine_schedule = machine_schedules[machine_id].schedule

            # --- ① 完全空闲：从 earliest_start 开始生产 ---
            if not machine_schedule:
                candidate_start = earliest_start
                candidate_end = candidate_start + production_duration
                available_slots.append((machine_id, candidate_start, candidate_end, speed))
                continue

            # --- ② 有排程的情况 ---
            # 检查第一个任务前是否有空档
            first_entry = machine_schedule[0]
            if earliest_start + production_duration <= first_entry.start_time:
                available_slots.append((machine_id, earliest_start, earliest_start + production_duration, speed))
                continue

            # 检查任务之间是否有可用空档
            for i in range(len(machine_schedule) - 1):
                prev_entry = machine_schedule[i]
                next_entry = machine_schedule[i + 1]
                gap_start = max(earliest_start, prev_entry.end_time)
                gap_end = next_entry.start_time
                if gap_end - gap_start >= production_duration:
                    candidate_start = gap_start
                    candidate_end = candidate_start + production_duration
                    available_slots.append((machine_id, candidate_start, candidate_end, speed))
                    break

            # 检查最后一个任务之后的时间
            last_entry = machine_schedule[-1]
            candidate_start = max(earliest_start, last_entry.end_time)
            candidate_end = candidate_start + production_duration
            if candidate_end <= latest_end:
                available_slots.append((machine_id, candidate_start, candidate_end, speed))

        # --- ③ 没有可用插槽 ---
        if not available_slots:
            return None

        # --- ④ 选择最早可开始的方案；若并列，速度快优先 ---
        available_slots.sort(key=lambda x: (x[1].timestamp(), -x[3]))  # start早优先，速度快优先
        return available_slots[0]

    def calculate_stock_levels(self, plan: ProductionPlan) -> Dict[Operation, Dict[datetime, float]]:
        """
        计算每个工序的库存水平随时间的变化。
        考虑同一时间点事件的顺序：先出库，后入库。
        并增加CA成品库的按交货期出库逻辑。
        """
        stock_changes = {op: [] for op in self.params.operation_sequence}  # 存储 (时间, 变化量)

        # 收集所有生产事件的开始和结束时间
        for op_idx, op in enumerate(self.params.operation_sequence):
            for machine_id in plan.schedules.get(op, {}):
                for entry in plan.schedules[op][machine_id].schedule:
                    # 合同生产完成，进入后库 (入库事件)
                    end_time = entry.end_time


                    # 如果不是最后一个工序，考虑下一个工序的消耗
                    if op_idx < len(self.params.operation_sequence) - 1:
                        if op == Operation.HR:
                            # CA成品库的按交货期出库逻辑
                            stock_changes[op].append(
                                (end_time + timedelta(hours=self.params.transmission_time[Operation.AR]), entry.order_wt))
                        else:
                            stock_changes[op].append(
                                (end_time + timedelta(hours=self.params.transmission_time[Operation.CA]), entry.order_wt))


                        next_op = self.params.operation_sequence[op_idx + 1]
                        # 查找该合同在下一个工序的开始时间
                        for next_machine_id in plan.schedules.get(next_op, {}):
                            for next_entry in plan.schedules[next_op][next_machine_id].schedule:
                                if next_entry.order_no == entry.order_no:
                                    # 下一个工序开始生产，库存减少 (出库事件)
                                    stock_changes[op].append((next_entry.start_time, -entry.order_wt))
                                    break
                            else:
                                continue
                            break
                    else:  # 如果是最后一个工序 (CA)
                        order = self.orders_map[entry.order_no]
                        # 入库
                        stock_changes[op].append((end_time+timedelta(hours=self.params.transmission_time[Operation.CA]), entry.order_wt))

                        if order.delivery_date <= end_time:
                            stock_changes[op].append((end_time, -entry.order_wt))
                        else:
                            stock_changes[op].append((order.delivery_date, -entry.order_wt))



        stock_history = {op: {} for op in self.params.operation_sequence}  # 初始化为空字典

        for op in self.params.operation_sequence:
            op_events = sorted(stock_changes[op], key=lambda x: (x[0], x[1]))  # x[1] 为变化量，负值排在前面

            current_stock = 0.0
            # 确保即使没有事件，也能有初始库存0
            if op_events:
                stock_history[op][op_events[0][0] - timedelta(microseconds=1)] = 0.0  # 在第一个事件前稍微加一个0库存点
            else:
                # 如果某个工序没有事件，但为了绘图仍需有个点，则添加一个虚拟点
                # 这在实际生产中不常见，因为所有订单都经过所有工序
                pass  # 暂时不做特殊处理，如果没事件就没曲线

            for t, change in op_events:
                # 处理同一时间点可能发生的多个事件，确保库存变化正确累加
                # 如果前一个时间点与当前时间点相同，则只更新库存值
                # 否则，在新的时间点更新库存

                # 为了绘制阶梯图，我们应该在每个时间点 t，记录 t-epsilon 的库存和 t 的库存
                # 先记录当前时间点前的库存
                if t not in stock_history[op]:
                    # 如果当前时间点t没有记录过，则继承前一个状态
                    # 找到 t 之前的最后一个已知库存点
                    prev_times = [s_t for s_t in stock_history[op] if s_t < t]
                    if prev_times:
                        stock_history[op][t - timedelta(microseconds=1)] = stock_history[op][max(prev_times)]
                    else:
                        stock_history[op][t - timedelta(microseconds=1)] = 0.0  # 否则为0

                current_stock += change
                stock_history[op][t] = current_stock

        return stock_history

    def check_stock_constraints(self, plan: ProductionPlan) -> bool:
        """
        检查生产计划是否满足库存约束。
        如果违规，记录违规信息。
        """
        is_feasible = True
        stock_history_per_op = self.calculate_stock_levels(plan)
        plan.stock_violations = {op: [] for op in self.params.operation_sequence}

        for op in self.params.operation_sequence:
            min_stock, max_stock = self.params.stock_limit[op]
            sorted_times = sorted(stock_history_per_op[op].keys())

            # 确保在每个时间点都检查库存
            for t in sorted_times:
                current_stock = stock_history_per_op[op][t]
                if not (min_stock <= current_stock <= max_stock):
                    # 记录违规信息，包括违规时间、实际库存和限制
                    plan.stock_violations[op].append((t, current_stock, (min_stock, max_stock)))
                    is_feasible = False
                    # 发现一个违规就可以停止对当前工序的检查，因为已经确定不可行
                    # break # 也可以不break，收集所有违规
        return is_feasible

    def generate_production_plan_positive(self, order_priority: List[str]) -> ProductionPlan:
        """
        根据订单优先级生成一个生产方案。
        改为：正向规划（从上游 HR -> AR -> CA 顺排）。
        策略：
          ✅ 每个工序按顺序往后排；
          ✅ 上游结束后考虑转运时间，再启动下游；
          ✅ 机组空闲就生产，不考虑交货期。
        """
        plan = ProductionPlan(order_priority=order_priority)
        plan.schedules = {
            op: {mid: MachineSchedule(mid, op) for mid in range(len(self.params.speed[op]))}
            for op in self.params.operation_sequence
        }

        # 顺向规划：从上游 (HR) 开始到下游 (CA)
        for order_no in order_priority:
            order = self.orders_map[order_no]

            # 存储该订单各工序调度结果（方便下游工序取上游的结束时间）
            order_schedules_in_plan: Dict[Operation, ScheduleEntry] = {}

            for op_idx, current_op in enumerate(self.params.operation_sequence):  # HR -> AR -> CA
                # --- ① 计算最早开始时间 ---
                if op_idx == 0:
                    # 最上游工序 (HR)，从订单可开始日期开始
                    earliest_start_for_current_op = order.plan_start_date
                else:
                    # 下游工序需等上游生产 + 转运完成
                    upstream_op = self.params.operation_sequence[op_idx - 1]
                    upstream_entry = order_schedules_in_plan.get(upstream_op)
                    if upstream_entry is None:
                        print(f"Error: upstream {upstream_op.value} not found for order {order_no}")
                        plan.is_feasible = False
                        return plan
                    # 下游的最早开始时间 = 上游结束 + 转运时间
                    earliest_start_for_current_op = upstream_entry.end_time + timedelta(
                        hours=self.params.transmission_time[current_op]
                    )

                # --- ② 设置一个合理的 latest_end 边界（不再用交货期约束，只防溢出） ---
                latest_end_for_current_op = earliest_start_for_current_op + timedelta(days=365)  # 给个足够大的范围

                # --- ③ 寻找可用机组插槽 ---
                machine_slot = self.find_available_machine_slot_machine_first(
                    current_op,
                    plan.schedules[current_op],
                    earliest_start_for_current_op,
                    latest_end_for_current_op,
                    order_no,
                    order.order_width,
                )

                if machine_slot:
                    machine_id, start_time, end_time, speed = machine_slot
                    entry = ScheduleEntry(
                        order_no=order_no,
                        operation=current_op,
                        machine_id=machine_id,
                        start_time=start_time,
                        end_time=end_time,
                        order_wt=order.order_wt,
                        order_width=order.order_width,
                    )
                    plan.schedules[current_op][machine_id].schedule.append(entry)
                    plan.schedules[current_op][machine_id].schedule.sort(key=lambda x: x.start_time)
                    order_schedules_in_plan[current_op] = entry
                else:
                    # 没地方排，方案不可行
                    plan.is_feasible = False
                    print(
                        f"Order {order_no} at {current_op.value}: No slot found after {earliest_start_for_current_op}")
                    return plan

        # --- ④ 可选：检查库存约束 ---
        plan.is_feasible = self.check_stock_constraints(plan)
        if not plan.is_feasible:
            return plan

        # --- ⑤ 可选：计算拖期和成本 ---
        plan.total_tardiness = self.calculate_tardiness(plan)
        plan.total_cost = self.calculate_production_cost(plan)

        return plan

    def generate_production_plan(self, order_priority: List[str]) -> ProductionPlan:
        """
        根据订单优先级生成一个生产方案。
        按需调度：从最下游产线 (CA) 开始逆向规划。
        """
        plan = ProductionPlan(order_priority=order_priority)
        plan.schedules = {op: {mid: MachineSchedule(mid, op) for mid in range(len(self.params.speed[op]))}
                          for op in self.params.operation_sequence}

        # 逆向规划：从最下游产线 (CA) 开始
        for order_no in order_priority:
            order = self.orders_map[order_no]

            # 存储该订单在所有工序的调度结果，方便后续查找下游工序的开始时间
            order_schedules_in_plan: Dict[Operation, ScheduleEntry] = {}

            # 从最下游工序 (CA) 向上游工序 (AR, HR) 规划
            for op_idx in range(len(self.params.operation_sequence) - 1, -1, -1): # CA -> AR -> HR
                current_op = self.params.operation_sequence[op_idx]

                # 计算当前工序的最晚完成时间 (latest_end)
                if current_op == Operation.CA:
                    # CA 的最晚完成时间是订单交期减去其出库转运时间 (如果是成品出库，则需要)
                    # 这里定义为：CA 工序本身的生产必须在 order.delivery_date 之前完成
                    # 如果转运时间是从 CA 完成后开始计算到客户手中的，那么 CA 的生产结束时间就是交货期
                    # 但为了给转运留时间，通常是 delivery_date - transmission_time
                    latest_end_for_current_op = order.delivery_date - timedelta(hours=self.params.transmission_time[current_op])
                    # print(f"CA 最晚结束时间：{latest_end_for_current_op}")
                else:
                    # 上游工序的最晚完成时间是下游工序的开始时间减去向下游的转运时间
                    downstream_op = self.params.operation_sequence[op_idx + 1]
                    downstream_entry = order_schedules_in_plan.get(downstream_op)
                    # print(f"下游工序：{downstream_op}，开始时间{downstream_entry.start_time}")

                    if downstream_entry is None:
                        # 这不应该发生，因为我们是逆向规划，下游工序应该已经调度完成
                        # 如果出现，说明逻辑有误，或者上游无法满足下游需求
                        # 暂时先标记为不可行
                        print(f"Error: Downstream schedule for {order_no} in {downstream_op.value} not found during {current_op.value} planning.")
                        plan.is_feasible = False
                        return plan

                    # 上游工序必须在下游工序开始前，将物料转运到位
                    # 所以上游工序的结束时间 <= 下游工序的开始时间 - (上游到下游的转运时间)
                    latest_end_for_current_op = downstream_entry.start_time - timedelta(hours=self.params.transmission_time[downstream_op])
                    # print(f"{current_op.value} 最晚开始时间：{latest_end_for_current_op}")

                # 计算当前工序的最早开始时间 (earliest_start)
                # 理论上可以从订单计划开始日期开始，但也要考虑机组的可用性
                earliest_start_for_current_op = order.plan_start_date # 这是订单最早可以开始生产的时间

                # 确保最早开始时间不晚于最晚结束时间，否则无解
                if earliest_start_for_current_op >= latest_end_for_current_op:
                    plan.is_feasible = False
                    print(f"Order {order_no} at {current_op.value}: earliest_start ({earliest_start_for_current_op}) >= latest_end ({latest_end_for_current_op})")
                    return plan

                # 在机组中找到合适的插槽
                # 传入 `current_plan_for_stock_check` 是为了在 find_available_machine_slot 中可以进行实时库存检查
                machine_slot = self.find_available_machine_slot_near_due_time(
                    current_op,
                    plan.schedules[current_op], # 传入当前工序的机组调度状态
                    earliest_start_for_current_op,
                    latest_end_for_current_op,
                    order_no,
                    order.order_width,
                    # consider_stock_at_slot_time=True, # 暂时不开启实时库存检查，等调度逻辑稳定再考虑
                    # current_plan_for_stock_check=plan
                )

                if machine_slot:
                    machine_id, start_time, end_time, speed = machine_slot
                    entry = ScheduleEntry(
                        order_no=order_no,
                        operation=current_op,
                        machine_id=machine_id,
                        start_time=start_time,
                        end_time=end_time,
                        order_wt=order.order_wt,
                        order_width=order.order_width
                    )
                    plan.schedules[current_op][machine_id].schedule.append(entry)
                    plan.schedules[current_op][machine_id].schedule.sort(key=lambda x: x.start_time) # 保持机组时间线有序
                    order_schedules_in_plan[current_op] = entry # 记录当前订单在当前工序的调度结果

                else:
                    # 无法找到可行的插槽，方案不可行
                    plan.is_feasible = False
                    print(f"Order {order_no} at {current_op.value}: No slot found between {earliest_start_for_current_op} and {latest_end_for_current_op}")
                    return plan # 直接返回，标记为不可行

        # 2. 检查库存约束 (在所有调度完成后统一检查)
        plan.is_feasible = self.check_stock_constraints(plan)
        if not plan.is_feasible:
            # print("库存约束违规。")
            return plan

        # 3. 计算拖期和成本
        plan.total_tardiness = self.calculate_tardiness(plan)
        plan.total_cost = self.calculate_production_cost(plan)

        return plan

    # 计算总拖期
    def calculate_tardiness(self, plan: ProductionPlan) -> float:
        """
        计算总拖期。
        拖期的计算方式：交货期 - （最后一道工序的加工结束时间 + 转运时间）
        """
        total_tardiness_hours = 0.0
        for order_no in plan.order_priority:
            order = self.orders_map[order_no]
            ca_schedule = None
            for machine_id in plan.schedules[Operation.CA]:
                for entry in plan.schedules[Operation.CA][machine_id].schedule:
                    if entry.order_no == order_no:
                        ca_schedule = entry
                        break
                if ca_schedule:
                    break

            if ca_schedule:
                finish_time = ca_schedule.end_time + timedelta(hours=self.params.transmission_time[Operation.CA])
                if finish_time > order.delivery_date:
                    tardiness_td = finish_time - order.delivery_date
                    total_tardiness_hours += tardiness_td.total_seconds() / 3600.0
            else:
                # 如果 CA 阶段没有排程，则认为拖期非常大
                total_tardiness_hours += 999999.0  # 假设一个非常大的值

        return total_tardiness_hours

    # 计算生产成本
    def calculate_production_cost(self, plan: ProductionPlan) -> float:
        """
        计算生产成本，主要考虑搭接费用。
        """
        total_cost = 0.0
        for op in self.params.operation_sequence:
            for machine_id in plan.schedules[op]:
                machine_schedule = plan.schedules[op][machine_id].schedule
                for i in range(1, len(machine_schedule)):
                    prev_entry = machine_schedule[i - 1]
                    current_entry = machine_schedule[i]

                    # 只有紧邻生产才计算搭接费用
                    # 这里定义紧邻生产为：前一个合同结束和当前合同开始之间没有其他合同插入
                    # 实际可能需要更严格的定义，例如时间间隔很小
                    # 为了简化，我们假设只要在同一个机组上连续排产就计算搭接

                    prev_width = prev_entry.order_width
                    current_width = current_entry.order_width

                    # 查找预计算的搭接费用
                    cost_key = (prev_width, current_width)
                    if cost_key in self.params.changeover_cost_matrix[op]:
                        total_cost += self.params.changeover_cost_matrix[op][cost_key]
                    else:
                        # 兜底逻辑，如果没找到精确的宽度搭接费用，根据规则估算
                        if prev_width == current_width:
                            total_cost += 0.0
                        elif prev_width > current_width:  # 从宽到窄
                            total_cost += (prev_width - current_width) * 5.0
                        else:  # 从窄到宽
                            total_cost += (current_width - prev_width) * 20.0 * self.params.width_change_penalty_factor

        return total_cost


class GeneticAlgorithm:
    def __init__(self, orders: List[OrderMultiFeature], params: StaticParameters,
                 population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.orders = orders
        self.params = params
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scheduler = Scheduler(orders, params)
        self.order_nos = [order.order_no for order in orders]

    def initialize_population(self) -> List[List[str]]:
        """
        初始化种群，生成多种优先级染色体。
        """
        population = []

        # 1. 随机优先级
        for _ in range(self.population_size // 3):
            random_priority = self.order_nos[:]
            random.shuffle(random_priority)
            population.append(random_priority)

        # 2. 交期早的优先级高 (EDD - Earliest Due Date)
        edd_priority = sorted(self.orders, key=lambda x: x.delivery_date)
        population.append([o.order_no for o in edd_priority])

        # 3. 交期晚的优先级高 (LDD - Latest Due Date)
        ldd_priority = sorted(self.orders, key=lambda x: x.delivery_date, reverse=True)
        population.append([o.order_no for o in ldd_priority])

        # 4. 计划开始日期早的优先级高 (SPT - Shortest Processing Time is also an option, but not in our current data)
        # 这里用 plan_start_date 替代类似概念
        spt_priority = sorted(self.orders, key=lambda x: x.plan_start_date)
        population.append([o.order_no for o in spt_priority])

        # 5. 宽度递增排序 (假设有助于减少搭接成本，宽到窄费用低)
        width_inc_priority = sorted(self.orders, key=lambda x: x.order_width)
        population.append([o.order_no for o in width_inc_priority])

        # 6. 宽度递减排序
        width_dec_priority = sorted(self.orders, key=lambda x: x.order_width, reverse=True)
        population.append([o.order_no for o in width_dec_priority])

        # 填充剩余部分，确保种群大小
        while len(population) < self.population_size:
            random_priority = self.order_nos[:]
            random.shuffle(random_priority)
            population.append(random_priority)

        return population

    def calculate_fitness(self, plan: ProductionPlan) -> float:
        """
        计算适应度。目标是最小化拖期和生产成本。
        适应度函数通常是最大化问题，所以我们将成本和拖期取负数或者倒数。
        对于不可行方案，给予非常低的适应度。
        """
        if not plan.is_feasible:
            # 对于不可行方案，给予一个非常低的惩罚值
            return -1e9

        # 惩罚函数，将最小化问题转化为最大化问题
        # 拖期惩罚系数，成本惩罚系数
        tardiness_penalty = plan.total_tardiness * self.params.delay_penalty_per_hour
        total_penalty = tardiness_penalty + plan.total_cost

        # 适应度 = 1 / (1 + total_penalty) 或者 (max_penalty - total_penalty)
        # 这里我们使用一个简单的负值，目标是让其最大化（即 total_penalty 最小化）
        return -total_penalty

    def select(self, population_plans: List[ProductionPlan]) -> List[List[str]]:
        """
        选择操作：轮盘赌选择 (Roulette Wheel Selection) 或者锦标赛选择 (Tournament Selection)。
        这里使用锦标赛选择，更稳定。
        """
        selected_chromosomes = []
        # 过滤掉不合法的染色体
        feasible_plans = [p for p in population_plans if p.is_feasible]

        if not feasible_plans: # 如果所有方案都不可行，则随机选择
            for _ in range(self.population_size):
                selected_chromosomes.append(random.choice(population_plans).order_priority)
            return selected_chromosomes

        # 进行锦标赛选择
        tournament_size = 5 # 每次锦标赛选5个个体
        for _ in range(self.population_size):
            # 从可行方案中选择
            tournament_candidates = random.sample(feasible_plans, min(tournament_size, len(feasible_plans)))
            winner = max(tournament_candidates, key=lambda p: self.calculate_fitness(p))
            selected_chromosomes.append(winner.order_priority)

        return selected_chromosomes

    def crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """
        交叉操作：顺序交叉 (Order Crossover - OX) 适用于排列问题。
        """
        if random.random() < self.crossover_rate:
            size = len(parent1)
            p1, p2 = [0] * size, [0] * size

            # 随机选择两个交叉点
            cx_point1, cx_point2 = sorted(random.sample(range(size), 2))

            # 复制中间段
            p1[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
            p2[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]

            # 填充剩余部分
            # 对于 P1，从 P1 的 cx_point2 开始，按 P2 的顺序填充未被复制的元素
            fill_p1_idx = cx_point2
            for item in parent1:
                if item not in p1[cx_point1:cx_point2]:
                    if fill_p1_idx >= size:
                        fill_p1_idx = (fill_p1_idx % size) # 循环填充
                    p1[fill_p1_idx] = item
                    fill_p1_idx += 1

            # 对于 P2，从 P2 的 cx_point2 开始，按 P1 的顺序填充未被复制的元素
            fill_p2_idx = cx_point2
            for item in parent2:
                if item not in p2[cx_point1:cx_point2]:
                    if fill_p2_idx >= size:
                        fill_p2_idx = (fill_p2_idx % size) # 循环填充
                    p2[fill_p2_idx] = item
                    fill_p2_idx += 1
            return p1, p2
        else:
            return parent1, parent2

    def mutate(self, chromosome: List[str]) -> List[str]:
        """
        变异操作：交换变异 (Swap Mutation)，随机交换两个元素。
        """
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def run(self) -> ProductionPlan:
        """
        运行遗传算法。
        """
        population = self.initialize_population()
        best_plan: Optional[ProductionPlan] = None
        best_fitness = -float('inf')

        for generation in range(self.generations):
            if (generation+1) % 20 == 0:
                print(f"迭代次数： {generation+1}/{self.generations}, 当前最优个体适应度：{best_fitness}, 总成本: {best_plan.total_cost}, 拖期:{best_plan.total_tardiness}, ")

            # 评估种群中的每个个体
            population_plans = []

            for chromosome in population:
                plan = self.scheduler.generate_production_plan(chromosome)

                population_plans.append(plan)

                fitness = self.calculate_fitness(plan)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_plan = copy.deepcopy(plan) # 深拷贝最优方案

            # 选择
            selected_chromosomes = self.select(population_plans)

            # 交叉和变异，生成新的种群
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = random.choice(selected_chromosomes)
                parent2 = random.choice(selected_chromosomes)

                child1, child2 = self.crossover(parent1, parent2)

                new_population.append(self.mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child2))

            population = new_population

        # 确保返回的 best_plan 是一个完整的、经过调度计算的方案
        if best_plan is None and population_plans:
            # 如果在循环中没有更新 best_plan (例如所有方案都不可行)，则取最后一代中最好的一个
            best_plan = max(population_plans, key=self.calculate_fitness)

        return best_plan

def plot_gantt_chart(plan: ProductionPlan, orders_map: Dict[str, OrderMultiFeature]):
    """
    使用 Plotly 绘制生产甘特图。
    """
    if not plan or not plan.is_feasible:
        print("无法绘制甘特图：生产计划不可行或为空。")
        return

    gantt_data = []
    colors = {}
    unique_orders = list(orders_map.keys())
    # 为每个订单生成一个唯一的颜色
    for i, order_no in enumerate(unique_orders):
        r = int((i * 30 % 255 + 50) % 255)
        g = int((i * 50 % 255 + 100) % 255)
        b = int((i * 70 % 255 + 150) % 255)
        colors[order_no] = f'rgb({r},{g},{b})'

    for op_idx, op in enumerate(plan.schedules):
        for machine_id, machine_schedule in plan.schedules[op].items():
            for entry in machine_schedule.schedule:
                gantt_data.append(
                    dict(
                        Task=f"{op.value} - 机组{entry.machine_id+1}",
                        Start=entry.start_time.isoformat(),
                        Finish=entry.end_time.isoformat(),
                        Resource=entry.order_no, # 使用订单号作为资源，方便染色
                        Description=f"订单: {entry.order_no}<br>重量: {entry.order_wt}t<br>宽度: {entry.order_width}mm",
                        Operation=op.value,
                        OrderNo=entry.order_no,
                        Machine=entry.machine_id+1,
                    )
                )

    # 对数据进行排序，以便在甘特图中显示顺序
    # 优先按工序排序，然后按机组，最后按开始时间
    # gantt_data.sort(key=lambda x: (
    #     [op.value for op in static_params.operation_sequence].index(x['Operation']),
    #     x['Machine'],
    #     x['Start']
    # ))

    # 构建自定义颜色映射
    color_map = {order_no: colors[order_no] for order_no in unique_orders}

    # 使用 plotly express 的 timeline
    import plotly.express as px
    import pandas as pd
    df = pd.DataFrame(gantt_data)
    # 确保时间列是 datetime 类型
    df["Start"] = pd.to_datetime(df["Start"], format="ISO8601")
    df["Finish"] = pd.to_datetime(df["Finish"], format="ISO8601")
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="OrderNo",
        hover_data=["Description"],
        color_discrete_map=colors,
        category_orders={"Task": df["Task"].unique().tolist()}
    )

    fig.update_layout(
        title="生产计划甘特图",
        xaxis_title="时间",
        yaxis_title="工序 - 机组",
        height=800,
        margin=dict(l=150, r=50, t=80, b=80),
    )

    fig.show()


import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict

def plot_stock_changes(plan: ProductionPlan, static_params: StaticParameters, orders_map: Dict[str, OrderMultiFeature]):
    """
    绘制每个工序的库存水平随时间变化的图表。
    """
    if not plan or not plan.is_feasible:
        print("无法绘制库存变化图：生产计划不可行或为空。")
        return

    scheduler_instance = Scheduler(orders_obj_list, static_params)
    stock_history_per_op_full = scheduler_instance.calculate_stock_levels(plan)

    fig = go.Figure()

    colors = {
        Operation.HR: 'blue',
        Operation.AR: 'green',
        Operation.CA: 'red'
    }

    all_event_times = set()  # 包含所有库存变化事件的时间点
    all_finish_times = []  # 包含所有订单的最终完成时间 (CA)
    all_delivery_dates = []  # 包含所有订单的交货期

    for op in static_params.operation_sequence:
        all_event_times.update(stock_history_per_op_full[op].keys())

    for order_no, order in orders_map.items():
        all_delivery_dates.append(order.delivery_date)

        ca_schedule = None
        for machine_id in plan.schedules[Operation.CA]:
            for entry in plan.schedules[Operation.CA][machine_id].schedule:
                if entry.order_no == order_no:
                    ca_schedule = entry
                    break
            if ca_schedule:
                break
        if ca_schedule:
            all_finish_times.append(
                ca_schedule.end_time + timedelta(hours=static_params.transmission_time[Operation.CA]))

    # 确定图表的实际时间范围
    actual_min_time = datetime.now()  # 默认当前时间
    if all_event_times:
        actual_min_time = min(all_event_times)

    actual_max_time = max(all_delivery_dates)  # 默认当前时间
    if all_finish_times:
        actual_max_time = max(actual_max_time, max(all_finish_times))
    if all_delivery_dates:
        actual_max_time = max(actual_max_time, max(all_delivery_dates))

    # 确保有足够的显示范围，尤其是在没有事件时，或者只有一个事件时
    if actual_max_time <= actual_min_time:
        actual_max_time = actual_min_time + timedelta(days=1)  # 至少一天

    # 设定图表的显示范围，稍微提前和延后一点
    plot_min_time = actual_min_time - timedelta(hours=1)  # 提前12小时
    plot_max_time = actual_max_time + timedelta(hours=4)  # 延后24小时，给库存出库留时间

    # 确保 plot_min_time 不会过于早导致绘图问题，可以限制在一个合理范围内
    # 例如，不早于所有订单的 plan_start_date 的最早值
    earliest_plan_start = min(o.plan_start_date for o in orders_map.values())
    if plot_min_time < earliest_plan_start - timedelta(hours=24):  # 留一点缓冲
        plot_min_time = earliest_plan_start - timedelta(hours=24)

    if plot_min_time > actual_min_time:  # 确保不会超过实际事件的最小时间
        plot_min_time = actual_min_time - timedelta(hours=1)

    # 为每个工序绘制库存曲线
    for op in static_params.operation_sequence:
        stock_data_raw = sorted(stock_history_per_op_full[op].items())

        times = []
        stocks = []

        # 在图表的实际起始点 (plot_min_time) 处添加初始库存
        times.append(plot_min_time)
        stocks.append(0.0)  # 假设初始库存为0

        # 跟踪当前库存水平
        current_stock_level = 0.0
        # 从 plot_min_time 到第一个事件发生前的库存
        first_event_time = stock_data_raw[0][0] if stock_data_raw else plot_max_time  # 如果没有事件，则用 max_time
        if plot_min_time < first_event_time:
            # 找到在 plot_min_time 之后，第一个实际事件之前，最早的事件点
            # 应该找到所有事件中，第一个大于 plot_min_time 的事件，其之前的库存是0
            # 或者更准确地说，找到在 plot_min_time 时的理论库存。
            # 简单起见，从 plot_min_time 到第一个事件前，库存为0
            current_stock_level = 0.0  # 假设在所有事件发生前库存为0

            # 遍历 stock_data_raw，填充 times 和 stocks
            for i in range(len(stock_data_raw)):
                t_event, stock_val_at_event = stock_data_raw[i]

                # 如果这个事件在 plot_min_time 之后
                if t_event >= plot_min_time:
                    # 如果上一个记录的时间点和当前事件时间之间有空白，先填充一个阶梯
                    if times[-1] < t_event:
                        times.append(t_event - timedelta(microseconds=1))  # 在事件发生前一刻
                        stocks.append(current_stock_level)  # 保持之前的库存

                    times.append(t_event)  # 事件发生时
                    stocks.append(stock_val_at_event)  # 更新库存
                    current_stock_level = stock_val_at_event

            # 在图表的实际结束点 (plot_max_time) 处，保持最后一个库存值
            if plot_max_time > times[-1]:
                times.append(plot_max_time)
                stocks.append(current_stock_level)
        else:  # 如果 plot_min_time 在第一个事件之后，这通常不应该发生
            # 这种情况比较复杂，需要从 plot_min_time 处的实际库存开始
            # 简单起见，如果 plot_min_time 已经包含在事件中，则直接从该点开始绘制

            # Find the stock level at plot_min_time
            current_stock_level_at_min = 0.0
            for t_event, stock_val_at_event in stock_data_raw:
                if t_event < plot_min_time:
                    current_stock_level_at_min = stock_val_at_event
                else:
                    break  # 已经超过 plot_min_time

            times.append(plot_min_time)
            stocks.append(current_stock_level_at_min)
            current_stock_level = current_stock_level_at_min

            for i in range(len(stock_data_raw)):
                t_event, stock_val_at_event = stock_data_raw[i]
                if t_event >= plot_min_time:
                    if times[-1] < t_event:
                        times.append(t_event - timedelta(microseconds=1))
                        stocks.append(current_stock_level)

                    times.append(t_event)
                    stocks.append(stock_val_at_event)
                    current_stock_level = stock_val_at_event

            if plot_max_time > times[-1]:
                times.append(plot_max_time)
                stocks.append(current_stock_level)

        fig.add_trace(go.Scatter(
            x=times,
            y=stocks,
            mode='lines',
            name=f'{op.value} 库存',
            line=dict(color=colors[op], shape='hv'),
        ))

        min_stock, max_stock = static_params.stock_limit[op]
        fig.add_hline(y=min_stock, line_dash="dash", line_color=colors[op], annotation_text=f"{op.value} Min",
                      annotation_position="bottom right", opacity=0.5)
        fig.add_hline(y=max_stock, line_dash="dash", line_color=colors[op], annotation_text=f"{op.value} Max",
                      annotation_position="top right", opacity=0.5)

        violations = plan.stock_violations.get(op, [])
        if violations:
            violation_times = [v[0] for v in violations]
            violation_stocks = [v[1] for v in violations]
            fig.add_trace(go.Scatter(
                x=violation_times,
                y=violation_stocks,
                mode='markers',
                marker=dict(color='black', size=10, symbol='x'),
                name=f'{op.value} 违规'
            ))

    fig.update_layout(
        title="工序库存变化图",
        xaxis_title="时间",
        yaxis_title="库存量 (吨)",
        hovermode="x unified",
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
    )
    fig.show()


def plot_tardiness_chart(plan: ProductionPlan, orders_map: Dict[str, OrderMultiFeature], static_params: StaticParameters):
    """
    绘制每个订单的拖期（或提前）情况图表。
    """
    if not plan or not plan.is_feasible:
        print("无法绘制拖期图：生产计划不可行或为空。")
        return

    order_tardiness_data = []
    # 需要重新计算每个订单的拖期，因为 plan.total_tardiness 是总拖期
    scheduler_instance = Scheduler(orders_obj_list, static_params) # 需要一个 Scheduler 实例

    for order_no in plan.order_priority:
        order = orders_map[order_no]
        ca_schedule = None
        for machine_id in plan.schedules[Operation.CA]:
            for entry in plan.schedules[Operation.CA][machine_id].schedule:
                if entry.order_no == order_no:
                    ca_schedule = entry
                    break
            if ca_schedule:
                break

        if ca_schedule:
            finish_time = ca_schedule.end_time + timedelta(hours=static_params.transmission_time[Operation.CA])
            tardiness_td = finish_time - order.delivery_date
            tardiness_hours = tardiness_td.total_seconds() / 3600.0
            order_tardiness_data.append({
                'OrderNo': order_no,
                'DeliveryDate': order.delivery_date,
                'FinishDate': finish_time,
                'Tardiness': tardiness_hours
            })
        else:
            # 如果 CA 阶段没有排程，则认为拖期非常大
            order_tardiness_data.append({
                'OrderNo': order_no,
                'DeliveryDate': order.delivery_date,
                'FinishDate': None, # 无法确定
                'Tardiness': 99999.0 # 假设一个非常大的值
            })

    # 根据订单号排序，或者按照优先级排序
    order_tardiness_data.sort(key=lambda x: plan.order_priority.index(x['OrderNo']))

    order_labels = [d['OrderNo'] for d in order_tardiness_data]
    tardiness_values = [d['Tardiness'] for d in order_tardiness_data]
    colors = ['red' if t > 0 else 'green' for t in tardiness_values] # 拖期为红色，提前为绿色

    fig = go.Figure(data=[go.Bar(
        x=order_labels,
        y=tardiness_values,
        marker_color=colors,
        text=[f'{val:.2f}h' for val in tardiness_values], # 显示具体数值
        textposition='outside'
    )])

    fig.add_shape(
        type="line", x0=-0.5, x1=len(order_labels)-0.5, y0=0, y1=0,
        line=dict(color="grey", width=2, dash="dash"),
        name="准时线"
    )

    fig.update_layout(
        title="订单拖期/提前完成情况",
        xaxis_title="订单号",
        yaxis_title="拖期时间 (小时) (正:拖期, 负:提前)",
        hovermode="x unified",
        yaxis_tickformat=".2f",
        showlegend=False
    )
    fig.show()

def plot_ca_contract_widths(plan: ProductionPlan, orders_map: Dict[str, OrderMultiFeature]):
    """
    绘制 CA 工序每个机组上生产的合同的宽度柱状图。
    每个机组单独一个图，显示其上每个合同的宽度。
    """
    if not plan or not plan.is_feasible:
        print("无法绘制合同宽度图：生产计划不可行或为空。")
        return

    ca_schedules = plan.schedules.get(Operation.CA, {})

    if not ca_schedules:
        print("CA 工序没有调度信息。")
        return

    # 为每个机组创建一个子图
    fig = go.Figure()
    machine_plots = {} # 用于存储每个机组的数据

    for machine_id in sorted(ca_schedules.keys()): # 确保按机组ID排序
        machine_schedule = ca_schedules[machine_id]
        sorted_schedule = sorted(machine_schedule.schedule, key=lambda x: x.start_time) # 确保按时间排序

        if not sorted_schedule:
            continue # 如果机组上没有任务，则跳过

        order_nos = [entry.order_no for entry in sorted_schedule]
        order_widths = [entry.order_width for entry in sorted_schedule]
        start_times = [entry.start_time.strftime('%Y-%m-%d %H:%M') for entry in sorted_schedule]

        # 准备数据用于 DataFrame
        machine_data = {
            'OrderNo': order_nos,
            'Width': order_widths,
            'StartTime': start_times,
            'Machine': f"CA - 机组{machine_id + 1}"
        }
        machine_plots[machine_id] = pd.DataFrame(machine_data)

    if not machine_plots:
        print("CA 工序所有机组都没有找到有效的生产任务。")
        return

    # 生成一个颜色映射，确保每个合同有唯一的颜色
    unique_orders = list(orders_map.keys())
    colors = {}
    for i, order_no in enumerate(unique_orders):
        colors[order_no] = f'rgb({147},{151},{255})'

    # 为每个机组添加一个柱状图
    for machine_id, df_machine in machine_plots.items():
        fig.add_trace(go.Bar(
            x=df_machine['OrderNo'],
            y=df_machine['Width'],
            name=df_machine['Machine'].iloc[0], # 使用机组名称作为图例
            marker_color=[colors.get(order_no, 'grey') for order_no in df_machine['OrderNo']], # 根据订单号上色
            customdata=df_machine[['StartTime']],
            hovertemplate="<b>订单: %{x}</b><br>" +
                          "机组: %{full_text}<br>" +
                          "生产宽度: %{y}mm<br>" +
                          "开始时间: %{customdata[0]}<extra></extra>",
            text=[f"{w:.0f}" for w in df_machine['Width']], # 显示宽度值
            textposition='outside',
            textfont=dict(size=10, color='black'),
        ))

    fig.update_layout(
        title="CA 工序各机组合同生产宽度",
        xaxis_title="订单号",
        yaxis_title="合同宽度 (mm)",
        barmode='group', # 将不同机组的柱子分组显示，或者 stacked
        hovermode="x unified",
        showlegend=True,
        height=600,
        legend_title_text='CA 机组',
        xaxis_tickangle=-45, # 倾斜x轴标签，如果订单多的话
    )
    fig.show()

def generate_random_orders(num_orders: int = 5) -> List[OrderMultiFeature]:
    """
    随机生成订单列表

    Args:
        num_orders: 要生成的订单数量，默认为5

    Returns:
        List[OrderMultiFeature]: 随机生成的订单列表
    """
    orders = []

    random.seed(42)

    # 定义一些基础参数范围
    start_date = datetime(2024, 11, 1)
    end_date = datetime(2024, 11, 15)
    date_range = 10
    widths = [792, 802, 830, 821, 854, 910, 914, 926, 952, 984, 1035]
    thicknesses = [8.0, 9.0, 10.0, 11.0, 12.0]
    weights = [300.0, 350.0, 400.0, 180.0, 260]

    for i in range(num_orders):
        order_no = f"order_{i + 1:03d}"  # 生成订单号，如 O001, O002...

        # 随机生成计划开始日期（基于基准日期）
        plan_start = start_date + timedelta(days=random.randint(0, date_range))

        # 随机生成交货日期（在计划开始日期之后的几天内）
        delivery_date = plan_start + timedelta(days=random.randint(date_range, date_range + 5))

        # 随机选择其他属性
        order_wt = random.choice(weights)
        order_width = random.choice(widths)
        order_thick = random.choice(thicknesses)

        order = OrderMultiFeature(
            order_no=order_no,
            delivery_date=delivery_date,
            plan_start_date=plan_start,
            order_wt=order_wt,
            order_width=order_width,
            order_thick=order_thick
        )

        orders.append(order)

    return orders
# 假设的 orders_obj_list 和 static_params
# 在实际运行时，你会从外部获取这些数据
orders_obj_list = [
    OrderMultiFeature(order_no="O001",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 19),
                      order_wt=240.0, order_width=1212.0, order_thick=10.0),
    OrderMultiFeature(order_no="O002",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 19),
                          order_wt=250.0, order_width=1201.0, order_thick=10.0),
    OrderMultiFeature(order_no="O003",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 19),
                              order_wt=240.0, order_width=1211.0, order_thick=10.0),
    OrderMultiFeature(order_no="O004",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 19),
                                  order_wt=270.0, order_width=1203.0, order_thick=10.0),
    OrderMultiFeature(order_no="O005",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 21, 21),
                          order_wt=240.0, order_width=1200.0, order_thick=10.0),
    OrderMultiFeature(order_no="O006",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 23),
                          order_wt=240.0, order_width=1204.0, order_thick=10.0),
    OrderMultiFeature(order_no="O007",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 23),
                          order_wt=250.0, order_width=1205.0, order_thick=10.0),
    OrderMultiFeature(order_no="O008",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 22),
                          order_wt=240.0, order_width=1206.0, order_thick=10.0),
    OrderMultiFeature(order_no="O009",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 19),
                          order_wt=260.0, order_width=1207.0, order_thick=10.0),
    OrderMultiFeature(order_no="O0010",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 20),
                          order_wt=240.0, order_width=1208.0, order_thick=10.0),
    OrderMultiFeature(order_no="O0011",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 21),
                          order_wt=240.0, order_width=1209.0, order_thick=10.0),
OrderMultiFeature(order_no="O0012",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 21),
                          order_wt=230.0, order_width=1208.0, order_thick=10.0),
OrderMultiFeature(order_no="O0013",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 21),
                          order_wt=240.0, order_width=1211.0, order_thick=10.0),
OrderMultiFeature(order_no="O0014",  plan_start_date=datetime(2024, 11, 15), delivery_date=datetime(2024, 11, 21),
                          order_wt=240.0, order_width=1212.0, order_thick=10.0),


]

# orders_obj_list = generate_random_orders(10)
static_params = StaticParameters() # 初始化时会自动计算搭接费用矩阵


# 将 orders_obj_list 转换为以 order_no 为键的字典，方便查找
orders_map = {order.order_no: order for order in orders_obj_list}

print("开始运行遗传算法...")
ga = GeneticAlgorithm(orders_obj_list, static_params,
                      population_size=100, generations=200, # 调整种群大小和代数以获得更好的结果
                      mutation_rate=0.15, crossover_rate=0.8)

best_overall_plan = ga.run()

if best_overall_plan and best_overall_plan.is_feasible:
    print("\n找到的最优生产计划：")
    print(f"订单优先级: {best_overall_plan.order_priority}")
    print(f"总拖期 (小时): {best_overall_plan.total_tardiness:.2f}")
    print(f"总生产成本: {best_overall_plan.total_cost:.2f}")
    print("详细调度计划：")
    # 绘制甘特图
    plot_gantt_chart(best_overall_plan, orders_map)
    plot_stock_changes(best_overall_plan, static_params, orders_map)
    plot_tardiness_chart(best_overall_plan, orders_map, static_params)
    plot_ca_contract_widths(best_overall_plan, orders_map)
else:
    print("\n未能找到可行的生产计划。")
    if best_overall_plan:
        print("最近一个不可行方案的库存违规信息:")
        for op, violations in best_overall_plan.stock_violations.items():
            if violations:
                print(f"  工序 {op.value} 违规:")
                for t, current, limits in violations:
                    print(f"    时间: {t.strftime('%Y-%m-%d %H:%M')}, 库存: {current:.2f}, 限制: {limits}")