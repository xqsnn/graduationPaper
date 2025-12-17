"""
NSGA-II求解器 - 用于混合流水车间调度问题

问题描述:
- 三道工序: 热轧(1台) -> 酸轧(2台) -> 连退(3台)
- 约束条件:
  * 热轧: 钢种相同才能相邻生产
  * 酸轧: 相邻材料宽度递减
  * 连退: 厚度集中(软约束)
- 优化目标:
  * 最小化交期拖期
  * 最小化库存水平
  * 最小化工艺违规惩罚
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import copy
import random
from collections import defaultdict


class Operation(Enum):
    """工序枚举"""
    HR = "hot_rolling"  # 热轧
    AR = "acid_rolling"  # 酸轧
    CA = "continuous_annealing"  # 连退


@dataclass
class Material:
    """材料类"""
    id: int
    order_no: str
    category: str  # 钢种
    width: float  # 宽度
    thickness: float  # 厚度
    weight: float  # 重量
    delivery_date: datetime  # 交货日期

    def __repr__(self):
        return f"Material({self.id}, cat={self.category}, w={self.width:.1f}, t={self.thickness:.2f})"


@dataclass
class ScheduleResult:
    """单个材料的调度结果"""
    material_id: int
    hr_start: float  # 热轧开始时间
    hr_end: float  # 热轧结束时间
    ar_machine: int  # 酸轧机器编号
    ar_start: float  # 酸轧开始时间
    ar_end: float  # 酸轧结束时间
    ca_machine: int  # 连退机器编号
    ca_start: float  # 连退开始时间
    ca_end: float  # 连退结束时间
    complete_time: float = 0.0  # 完成时间（连退结束时间 + 转运时间）

    def to_dict(self):
        """将调度结果转换为字典格式"""
        return {
            'material_id': self.material_id,
            'hr_start': self.hr_start,
            'hr_end': self.hr_end,
            'ar_machine': self.ar_machine,
            'ar_start': self.ar_start,
            'ar_end': self.ar_end,
            'ca_machine': self.ca_machine,
            'ca_start': self.ca_start,
            'ca_end': self.ca_end,
            'complete_time': self.complete_time
        }


@dataclass
class Solution:
    """
    解的编码方案

    编码结构:
    - hr_sequence: 热轧工序的材料加工顺序 (列表索引)
    - ar_assignment: 酸轧工序的机器分配 (0或1表示机器编号)
    - ar_sequences: 每台酸轧机上的材料加工顺序
    - ca_assignment: 连退工序的机器分配 (0,1或2表示机器编号)
    - ca_sequences: 每台连退机上的材料加工顺序
    """
    hr_sequence: List[int]  # 热轧顺序
    ar_assignment: List[int]  # 酸轧机器分配
    ar_sequences: List[List[int]]  # 每台酸轧机的顺序
    ca_assignment: List[int]  # 连退机器分配
    ca_sequences: List[List[int]]  # 每台连退机的顺序

    # 目标函数值
    max_tardiness: float = 0.0  # f1 最大拖期
    avg_inventory: float = 0.0  # f2 平均库存
    process_instability: float = 0.0  # f3 工艺不稳定性

    # Pareto相关
    rank: int = 0  # 非支配排序等级
    crowding_distance: float = 0.0  # 拥挤度

    # 详细调度结果
    schedule_results: List[ScheduleResult] = None


class StaticParameters:
    """静态参数配置"""

    start_time = datetime.strptime("20260125", "%Y%m%d")

    # 产线顺序
    operation_sequence = [Operation.HR, Operation.AR, Operation.CA]

    # 产线之间的运输时间(小时)
    transmission_time = {
        Operation.HR: 2.0,
        Operation.AR: 4.0,
        Operation.CA: 1.0
    }

    # 不同机组的处理速度(吨/小时)
    speed = {
        Operation.HR: [30.0],
        Operation.AR: [14.0, 16.0],
        Operation.CA: [8.0, 12.0, 10.0]
    }

    # 机器数量
    num_machines = {
        Operation.HR: 1,
        Operation.AR: 2,
        Operation.CA: 3
    }


class NSGA2Solver:
    """NSGA-II求解器"""

    def __init__(self, materials: List[Material], params: StaticParameters):
        """
        初始化求解器

        Args:
            materials: 材料列表
            params: 静态参数
        """
        self.materials = materials
        self.params = params
        self.n_materials = len(materials)

        # 按钢种分组 - 用于热轧约束
        self.category_groups = self._group_by_category()

    def _group_by_category(self) -> Dict[str, List[int]]:
        """按钢种对材料进行分组"""
        groups = defaultdict(list)
        for i, mat in enumerate(self.materials):
            groups[mat.category].append(i)
        return dict(groups)

    def create_initial_population(self, pop_size: int) -> List[Solution]:
        """
        创建初始种群

        策略:
        - 热轧: 钢种分组内随机排序,组间随机排序
        - 酸轧: 随机分配 + 按宽度降序排序
        - 连退: 随机分配 + 按厚度聚类排序

        Args:
            pop_size: 种群大小

        Returns:
            初始种群
        """
        population = []

        for _ in range(pop_size):
            # 1. 热轧顺序 - 按钢种分组
            hr_sequence = self._create_hr_sequence()

            # 2. 酸轧分配和顺序
            ar_assignment, ar_sequences = self._create_ar_assignment()

            # 3. 连退分配和顺序
            ca_assignment, ca_sequences = self._create_ca_assignment()

            solution = Solution(
                hr_sequence=hr_sequence,
                ar_assignment=ar_assignment,
                ar_sequences=ar_sequences,
                ca_assignment=ca_assignment,
                ca_sequences=ca_sequences
            )

            population.append(solution)

        return population

    def _create_hr_sequence(self) -> List[int]:
        """
        创建热轧顺序 - 满足钢种约束

        Returns:
            热轧顺序(材料索引列表)
        """
        # 随机打乱钢种组的顺序
        categories = list(self.category_groups.keys())
        random.shuffle(categories)

        hr_sequence = []
        for cat in categories:
            # 组内材料随机排序
            group = self.category_groups[cat].copy()
            random.shuffle(group)
            hr_sequence.extend(group)

        return hr_sequence

    def _create_ar_assignment(self) -> Tuple[List[int], List[List[int]]]:
        """
        创建酸轧机器分配和顺序 - 考虑宽度递减约束

        Returns:
            (机器分配列表, 每台机器的顺序列表)
        """
        n_machines = self.params.num_machines[Operation.AR]

        # 按宽度降序排序材料
        sorted_indices = sorted(range(self.n_materials),
                               key=lambda i: self.materials[i].width,
                               reverse=True)

        # 轮流分配到不同机器
        assignment = [0] * self.n_materials
        sequences = [[] for _ in range(n_machines)]

        for idx, mat_idx in enumerate(sorted_indices):
            machine = idx % n_machines
            assignment[mat_idx] = machine
            sequences[machine].append(mat_idx)

        # 为每台机器的序列添加随机扰动
        for seq in sequences:
            if len(seq) > 2 and random.random() < 0.3:
                i, j = random.sample(range(len(seq)), 2)
                seq[i], seq[j] = seq[j], seq[i]

        return assignment, sequences

    def _create_ca_assignment(self) -> Tuple[List[int], List[List[int]]]:
        """
        创建连退机器分配和顺序 - 考虑厚度集中约束

        Returns:
            (机器分配列表, 每台机器的顺序列表)
        """
        n_machines = self.params.num_machines[Operation.CA]

        # 按厚度排序
        sorted_indices = sorted(range(self.n_materials),
                               key=lambda i: self.materials[i].thickness)

        # 分段分配 - 使相似厚度的材料在同一台机器
        assignment = [0] * self.n_materials
        sequences = [[] for _ in range(n_machines)]

        chunk_size = len(sorted_indices) // n_machines + 1
        for i, mat_idx in enumerate(sorted_indices):
            machine = min(i // chunk_size, n_machines - 1)
            assignment[mat_idx] = machine
            sequences[machine].append(mat_idx)

        # 组内按交货期排序
        for seq in sequences:
            seq.sort(key=lambda i: self.materials[i].delivery_date)

        return assignment, sequences

    def evaluate_solution(self, solution: Solution, use_continuous_inventory: bool = True) -> Solution:
        """
        评估解的目标函数值

        通过仿真计算:
        1. 每个材料的实际完工时间
        2. 总拖期时间
        3. 库存水平
        4. 约束违规惩罚

        Args:
            solution: 待评估的解
            use_continuous_inventory: 是否使用改进的连续库存计算方法

        Returns:
            评估后的解(更新目标函数值)
        """
        # 仿真调度过程
        schedule_results = self._simulate_schedule(solution)
        solution.schedule_results = schedule_results

        # 计算目标函数
        max_tardiness = 0.0

        for result in schedule_results:
            mat = self.materials[result.material_id]
            # 使用complete_time作为实际完成时间（连退结束时间+转运时间）
            completion_days = result.complete_time / 24.0
            delivery_days = (mat.delivery_date - StaticParameters.start_time).days
            tardiness = max(0, completion_days - delivery_days)
            max_tardiness = max(max_tardiness, tardiness)

        # 计算库存
        if use_continuous_inventory:
            # 使用改进的连续库存计算方法
            inventory = self._calculate_continuous_inventory(schedule_results)
        else:
            # 使用原有方法
            inventory = self._calculate_inventory(schedule_results)

        # 计算约束违规惩罚
        hr_switch = self._hr_switch_count(solution)
        ar_jump = self._ar_width_jump(solution)
        ca_switch = self._ca_thickness_switch(solution)
        process_instability = (
                5.0 * hr_switch +
                1.0 * ar_jump +
                10.0 * ca_switch
        )
        # constraint_penalty = self._calculate_constraint_penalty(solution)

        solution.max_tardiness = max_tardiness
        solution.avg_inventory = inventory
        solution.process_instability = process_instability

        return solution

    def _hr_switch_count(self, solution: Solution) -> int:
        """
        计算HR切换次数

        Args:
            solution: 待评估的解

        Returns:
            切换次数
        """
        count = 0
        seq = solution.hr_sequence
        for i in range(len(seq) - 1):
            if self.materials[seq[i]].category != self.materials[seq[i + 1]].category:
                count += 1
        return count

    def _ar_width_jump(self, solution: Solution) -> float:
        """
        计算AR宽度跳变

        Args:
            solution: 待评估的解

        Returns:
            宽度跳变距离
        """
        jump = 0.0
        for seq in solution.ar_sequences:
            for i in range(len(seq) - 1):
                w1 = self.materials[seq[i]].width
                w2 = self.materials[seq[i + 1]].width
                if w2 > w1:
                    jump += (w2 - w1)
        return jump

    def _ca_thickness_switch(self, solution: Solution, delta: float = 0.15) -> int:
        """
        计算CA厚度切换次数

        Args:
            solution: 待评估的解
            delta: 厚度阈值

        Returns:
            切换次数
        """
        switches = 0
        for seq in solution.ca_sequences:
            for i in range(len(seq) - 1):
                t1 = self.materials[seq[i]].thickness
                t2 = self.materials[seq[i + 1]].thickness
                if abs(t2 - t1) > delta:
                    switches += 1
        return switches

    def _simulate_schedule(self, solution: Solution) -> List[ScheduleResult]:
        """
        仿真调度过程,计算每个材料在各工序的开始和结束时间

        Args:
            solution: 调度方案

        Returns:
            每个材料的调度结果
        """
        results = [None] * self.n_materials

        # 1. 热轧工序 - 单机,按顺序处理
        hr_speed = self.params.speed[Operation.HR][0]
        current_time = 0.0  # 从0开始，表示从起始时间开始调度

        for mat_idx in solution.hr_sequence:
            mat = self.materials[mat_idx]
            start_time = current_time
            processing_time = mat.weight / hr_speed
            end_time = start_time + processing_time

            results[mat_idx] = ScheduleResult(
                material_id=mat_idx,
                hr_start=start_time,
                hr_end=end_time,
                ar_machine=-1,
                ar_start=0,
                ar_end=0,
                ca_machine=-1,
                ca_start=0,
                ca_end=0,
                complete_time=0.0
            )

            current_time = end_time

        # 2. 酸轧工序 - 多台机器并行
        ar_speeds = self.params.speed[Operation.AR]
        ar_machine_time = [0.0] * len(ar_speeds)  # 每台机器的当前时间，相对起始时间

        for machine_id, sequence in enumerate(solution.ar_sequences):
            for mat_idx in sequence:
                mat = self.materials[mat_idx]
                result = results[mat_idx]

                # 开始时间 = max(热轧完成时间+运输时间, 机器可用时间)
                earliest_start = result.hr_end + self.params.transmission_time[Operation.HR]
                start_time = max(earliest_start, ar_machine_time[machine_id])

                processing_time = mat.weight / ar_speeds[machine_id]
                end_time = start_time + processing_time

                result.ar_machine = machine_id
                result.ar_start = start_time
                result.ar_end = end_time

                ar_machine_time[machine_id] = end_time

        # 3. 连退工序 - 多台机器并行
        ca_speeds = self.params.speed[Operation.CA]
        ca_machine_time = [0.0] * len(ca_speeds)  # 每台机器的当前时间，相对起始时间

        for machine_id, sequence in enumerate(solution.ca_sequences):
            for mat_idx in sequence:
                mat = self.materials[mat_idx]
                result = results[mat_idx]

                # 开始时间 = max(酸轧完成时间+运输时间, 机器可用时间)
                earliest_start = result.ar_end + self.params.transmission_time[Operation.AR]
                start_time = max(earliest_start, ca_machine_time[machine_id])

                processing_time = mat.weight / ca_speeds[machine_id]
                end_time = start_time + processing_time

                result.ca_machine = machine_id
                result.ca_start = start_time
                result.ca_end = end_time

                # 计算完成时间 = 连退结束时间 + Operation.CA的转运时间
                result.complete_time = end_time + self.params.transmission_time[Operation.CA]

                ca_machine_time[machine_id] = end_time

        return results

    def _calculate_inventory(self, schedule_results: List[ScheduleResult]) -> float:
        """
        计算库存水平(简化版)

        采用最大在制品数量作为库存指标

        Args:
            schedule_results: 调度结果

        Returns:
            最大库存水平
        """
        # 收集所有时间事件
        events = []
        for result in schedule_results:
            # 热轧完成 -> 进入热轧后仓库
            events.append((result.hr_end, 1, 'hr'))
            # 酸轧开始 -> 离开热轧后仓库
            events.append((result.ar_start, -1, 'hr'))

            # 酸轧完成 -> 进入酸轧后仓库
            events.append((result.ar_end, 1, 'ar'))
            # 连退开始 -> 离开酸轧后仓库
            events.append((result.ca_start, -1, 'ar'))

        events.sort()

        # 计算最大库存
        max_inventory = 0
        current_inventory = 0

        for time, delta, warehouse in events:
            current_inventory += delta
            max_inventory = max(max_inventory, current_inventory)

        return float(max_inventory)

    def _calculate_continuous_inventory(self, schedule_results: List[ScheduleResult]) -> float:
        """
        计算连续生产的库存水平(改进版)

        基于事件驱动的精确库存计算方法
        按重量统计，根据入库/出库事件实时更新库存
        返回加权平均库存（考虑事件间隔时间）

        Args:
            schedule_results: 调度结果

        Returns:
            平均库存水平(吨)
        """
        if not schedule_results:
            return 0.0

        # 收集所有库存事件 (时间, 重量变化, 仓库类型)
        events = []

        for result in schedule_results:
            mat_weight = self.materials[result.material_id].weight

            # 热轧完成后入库事件 (热轧后-酸轧前缓冲区)
            events.append((result.hr_end, mat_weight, 'hr_to_ar'))
            # 酸轧开始前出库事件 (从热轧后-酸轧前缓冲区)
            events.append((result.ar_start, -mat_weight, 'hr_to_ar'))

            # 酸轧完成后入库事件 (酸轧后-连退前缓冲区)
            events.append((result.ar_end, mat_weight, 'ar_to_ca'))
            # 连退开始前出库事件 (从酸轧后-连退前缓冲区)
            events.append((result.ca_start, -mat_weight, 'ar_to_ca'))

        # 按时间排序事件
        events.sort(key=lambda x: x[0])

        # 遍历事件并计算加权平均库存
        if not events:
            return 0.0

        # 计算每个时间段的库存并累加面积
        total_inventory_time = 0.0  # 库存×时间的累积值
        hr_to_ar_inventory = 0.0  # 热轧后-酸轧前库存
        ar_to_ca_inventory = 0.0  # 酸轧后-连退前库存
        prev_time = 0.0  # 上一个事件时间

        for event_time, weight_change, warehouse_type in events:
            # 计算从prev_time到event_time时间段的库存面积
            if event_time > prev_time:
                time_interval = event_time - prev_time
                current_total_inventory = hr_to_ar_inventory + ar_to_ca_inventory
                total_inventory_time += current_total_inventory * time_interval

            # 更新库存（执行事件）
            if warehouse_type == 'hr_to_ar':
                hr_to_ar_inventory += weight_change
            elif warehouse_type == 'ar_to_ca':
                ar_to_ca_inventory += weight_change

            prev_time = event_time

        # 计算总的调度时间长度
        max_time = max(result.ca_end for result in schedule_results)
        total_duration = max_time - 0.0

        if total_duration <= 0:
            return 0.0

        # 计算加权平均库存
        avg_inventory = total_inventory_time / total_duration
        return avg_inventory

    def _calculate_inventory_peak_and_average(self, schedule_results: List[ScheduleResult]) -> Tuple[float, float]:
        """
        计算库存峰值和平均值（完整版）

        Args:
            schedule_results: 调度结果

        Returns:
            (峰值库存, 平均库存)
        """
        if not schedule_results:
            return 0.0, 0.0

        # 收集所有库存事件 (时间, 重量变化, 仓库类型)
        events = []

        for result in schedule_results:
            mat_weight = self.materials[result.material_id].weight

            # 热轧完成后入库事件 (热轧后-酸轧前缓冲区)
            events.append((result.hr_end, mat_weight, 'hr_to_ar'))
            # 酸轧开始前出库事件 (从热轧后-酸轧前缓冲区)
            events.append((result.ar_start, -mat_weight, 'hr_to_ar'))

            # 酸轧完成后入库事件 (酸轧后-连退前缓冲区)
            events.append((result.ar_end, mat_weight, 'ar_to_ca'))
            # 连退开始前出库事件 (从酸轧后-连退前缓冲区)
            events.append((result.ca_start, -mat_weight, 'ar_to_ca'))

        # 按时间排序事件
        events.sort(key=lambda x: x[0])

        # 遍历事件并计算库存变化
        hr_to_ar_inventory = 0.0  # 热轧后-酸轧前库存
        ar_to_ca_inventory = 0.0  # 酸轧后-连退前库存
        max_inventory = 0.0  # 库存峰值
        total_inventory_time = 0.0  # 库存×时间的累积值
        prev_time = 0.0  # 上一个事件时间

        for event_time, weight_change, warehouse_type in events:
            # 计算从prev_time到event_time时间段的库存面积（在事件发生前）
            if event_time > prev_time:
                time_interval = event_time - prev_time
                current_total_inventory = hr_to_ar_inventory + ar_to_ca_inventory
                total_inventory_time += current_total_inventory * time_interval
                max_inventory = max(max_inventory, current_total_inventory)

            # 更新库存（执行事件）
            if warehouse_type == 'hr_to_ar':
                hr_to_ar_inventory += weight_change
            elif warehouse_type == 'ar_to_ca':
                ar_to_ca_inventory += weight_change

            prev_time = event_time

        # 计算总的调度时间长度
        max_time = max(result.ca_end for result in schedule_results)
        total_duration = max_time - 0.0

        if total_duration <= 0:
            return 0.0, 0.0

        # 计算加权平均库存
        avg_inventory = total_inventory_time / total_duration if total_duration > 0 else 0.0

        return max_inventory, avg_inventory

    def _calculate_constraint_penalty(self, solution: Solution) -> float:
        """
        计算约束违规惩罚

        Args:
            solution: 调度方案

        Returns:
            惩罚值
        """
        penalty = 0.0

        # 1. 热轧钢种约束检查(硬约束,不应该违反)
        for i in range(len(solution.hr_sequence) - 1):
            curr_idx = solution.hr_sequence[i]
            next_idx = solution.hr_sequence[i + 1]
            if self.materials[curr_idx].category != self.materials[next_idx].category:
                penalty += 1000.0  # 严重惩罚

        # 2. 酸轧宽度递减约束检查
        for sequence in solution.ar_sequences:
            for i in range(len(sequence) - 1):
                curr_width = self.materials[sequence[i]].width
                next_width = self.materials[sequence[i + 1]].width
                if curr_width < next_width:
                    penalty += 100.0 * (next_width - curr_width)

        # 3. 连退厚度集中约束检查(软约束)
        for sequence in solution.ca_sequences:
            if len(sequence) > 1:
                thicknesses = [self.materials[i].thickness for i in sequence]
                # 计算标准差作为离散程度
                std_dev = np.std(thicknesses)
                penalty += 10.0 * std_dev

        return penalty

    def fast_non_dominated_sort(self, population: List[Solution]) -> List[List[Solution]]:
        """
        快速非支配排序

        Args:
            population: 种群

        Returns:
            分层后的种群(每层是一个列表)
        """
        # 每个解被多少个解支配
        domination_count = [0] * len(population)
        # 每个解支配哪些解
        dominated_solutions = [[] for _ in range(len(population))]
        # Pareto前沿
        fronts = [[]]

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # 比较i和j
                if self._dominates(population[i], population[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j], population[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

            # 如果i不被任何解支配,加入第一层
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])

        # 生成后续层
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for sol_i in fronts[current_front]:
                i = population.index(sol_i)
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = current_front + 1
                        next_front.append(population[j])

            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]  # 移除最后的空列表

    def _dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """
        判断sol1是否支配sol2

        支配条件: sol1在所有目标上不差于sol2,且至少在一个目标上更好

        目标: 最小化拖期、最小化库存、最小化惩罚

        Args:
            sol1, sol2: 待比较的解

        Returns:
            True如果sol1支配sol2
        """
        better_in_any = False

        # 检查三个目标
        # objectives1 = [sol1.max_tardiness, sol1.avg_inventory, sol1.process_instability]
        # objectives2 = [sol2.max_tardiness, sol2.avg_inventory, sol2.process_instability]

        objectives1 = [
            sol1.max_tardiness,
            sol1.avg_inventory,
            sol1.process_instability
        ]

        objectives2 = [
            sol2.max_tardiness,
            sol2.avg_inventory,
            sol2.process_instability
        ]

        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 > obj2:  # sol1在某个目标上更差
                return False
            if obj1 < obj2:  # sol1在某个目标上更好
                better_in_any = True

        return better_in_any

    def calculate_crowding_distance(self, front: List[Solution]):
        """
        计算拥挤度距离

        Args:
            front: 同一层的解集合
        """
        n = len(front)
        if n <= 2:
            for sol in front:
                sol.crowding_distance = float('inf')
            return

        # 初始化
        for sol in front:
            sol.crowding_distance = 0.0

        # 对每个目标计算拥挤度
        objectives = ['max_tardiness', 'avg_inventory', 'process_instability']

        for obj in objectives:
            # 按该目标排序
            front.sort(key=lambda x: getattr(x, obj))

            # 边界解设为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # 目标值范围
            obj_min = getattr(front[0], obj)
            obj_max = getattr(front[-1], obj)
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # 计算中间解的拥挤度
            for i in range(1, n - 1):
                if front[i].crowding_distance != float('inf'):
                    distance = (getattr(front[i + 1], obj) - getattr(front[i - 1], obj)) / obj_range
                    front[i].crowding_distance += distance

    def selection(self, population: List[Solution], n_select: int) -> List[Solution]:
        """
        选择操作 - 锦标赛选择

        Args:
            population: 种群
            n_select: 选择数量

        Returns:
            选择的个体
        """
        selected = []
        for _ in range(n_select):
            # 随机选择两个个体
            candidates = random.sample(population, 2)
            # 比较rank,rank小的更好
            if candidates[0].rank < candidates[1].rank:
                selected.append(copy.deepcopy(candidates[0]))
            elif candidates[0].rank > candidates[1].rank:
                selected.append(copy.deepcopy(candidates[1]))
            else:
                # 同一rank,比较拥挤度,拥挤度大的更好
                if candidates[0].crowding_distance > candidates[1].crowding_distance:
                    selected.append(copy.deepcopy(candidates[0]))
                else:
                    selected.append(copy.deepcopy(candidates[1]))

        return selected

    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """
        交叉操作

        对不同部分采用不同的交叉策略:
        - 热轧顺序: 顺序交叉(OX)
        - 酸轧/连退分配: 单点交叉
        - 酸轧/连退顺序: 顺序交叉

        Args:
            parent1, parent2: 父代

        Returns:
            两个子代
        """
        child1 = Solution(
            hr_sequence=[],
            ar_assignment=[],
            ar_sequences=[],
            ca_assignment=[],
            ca_sequences=[]
        )
        child2 = Solution(
            hr_sequence=[],
            ar_assignment=[],
            ar_sequences=[],
            ca_assignment=[],
            ca_sequences=[]
        )

        # 1. 热轧顺序 - 顺序交叉(保持钢种分组)
        child1.hr_sequence = self._crossover_sequence_with_groups(
            parent1.hr_sequence, parent2.hr_sequence
        )
        child2.hr_sequence = self._crossover_sequence_with_groups(
            parent2.hr_sequence, parent1.hr_sequence
        )

        # 2. 酸轧分配 - 单点交叉
        point = random.randint(0, self.n_materials - 1)
        child1.ar_assignment = parent1.ar_assignment[:point] + parent2.ar_assignment[point:]
        child2.ar_assignment = parent2.ar_assignment[:point] + parent1.ar_assignment[point:]

        # 根据分配重建序列
        child1.ar_sequences = self._rebuild_sequences(child1.ar_assignment,
                                                       self.params.num_machines[Operation.AR])
        child2.ar_sequences = self._rebuild_sequences(child2.ar_assignment,
                                                       self.params.num_machines[Operation.AR])

        # 3. 连退分配 - 单点交叉
        point = random.randint(0, self.n_materials - 1)
        child1.ca_assignment = parent1.ca_assignment[:point] + parent2.ca_assignment[point:]
        child2.ca_assignment = parent2.ca_assignment[:point] + parent1.ca_assignment[point:]

        # 根据分配重建序列
        child1.ca_sequences = self._rebuild_sequences(child1.ca_assignment,
                                                       self.params.num_machines[Operation.CA])
        child2.ca_sequences = self._rebuild_sequences(child2.ca_assignment,
                                                       self.params.num_machines[Operation.CA])

        return child1, child2

    def _crossover_sequence_with_groups(self, seq1: List[int], seq2: List[int]) -> List[int]:
        """
        带钢种分组约束的顺序交叉

        策略: 保持钢种组的顺序,但在组内进行交叉

        Args:
            seq1, seq2: 两个序列

        Returns:
            交叉后的序列
        """
        # 提取seq1中的钢种组顺序
        seen_categories = []
        for idx in seq1:
            cat = self.materials[idx].category
            if cat not in seen_categories:
                seen_categories.append(cat)

        # 按组构建新序列
        child_seq = []
        for cat in seen_categories:
            # 找到该组在两个序列中的材料
            group_in_seq1 = [i for i in seq1 if self.materials[i].category == cat]
            group_in_seq2 = [i for i in seq2 if self.materials[i].category == cat]

            # 组内顺序交叉
            if len(group_in_seq1) > 1 and random.random() < 0.5:
                # 使用seq2的组内顺序
                child_seq.extend(group_in_seq2)
            else:
                # 使用seq1的组内顺序
                child_seq.extend(group_in_seq1)

        return child_seq

    def _rebuild_sequences(self, assignment: List[int], n_machines: int) -> List[List[int]]:
        """
        根据机器分配重建每台机器的加工顺序

        Args:
            assignment: 机器分配列表
            n_machines: 机器数量

        Returns:
            每台机器的顺序列表
        """
        sequences = [[] for _ in range(n_machines)]
        for mat_idx, machine in enumerate(assignment):
            sequences[machine].append(mat_idx)
        return sequences

    def mutation(self, solution: Solution, mutation_rate: float = 0.1):
        """
        变异操作

        Args:
            solution: 待变异的解
            mutation_rate: 变异概率
        """
        # 1. 热轧顺序变异 - 组内交换
        if random.random() < mutation_rate:
            self._mutate_hr_sequence(solution)

        # 2. 酸轧分配变异
        if random.random() < mutation_rate:
            self._mutate_assignment(solution.ar_assignment,
                                   solution.ar_sequences,
                                   self.params.num_machines[Operation.AR])

        # 3. 酸轧顺序变异
        if random.random() < mutation_rate:
            self._mutate_sequences(solution.ar_sequences)

        # 4. 连退分配变异
        if random.random() < mutation_rate:
            self._mutate_assignment(solution.ca_assignment,
                                   solution.ca_sequences,
                                   self.params.num_machines[Operation.CA])

        # 5. 连退顺序变异
        if random.random() < mutation_rate:
            self._mutate_sequences(solution.ca_sequences)

    def _mutate_hr_sequence(self, solution: Solution):
        """热轧顺序变异 - 在钢种组内交换"""
        # 随机选择一个钢种组
        if not self.category_groups:
            return

        category = random.choice(list(self.category_groups.keys()))
        group_indices = self.category_groups[category]

        if len(group_indices) < 2:
            return

        # 找到该组在hr_sequence中的位置
        positions = [solution.hr_sequence.index(idx) for idx in group_indices]

        # 随机交换组内两个位置
        if len(positions) >= 2:
            i, j = random.sample(positions, 2)
            solution.hr_sequence[i], solution.hr_sequence[j] = \
                solution.hr_sequence[j], solution.hr_sequence[i]

    def _mutate_assignment(self, assignment: List[int], sequences: List[List[int]], n_machines: int):
        """机器分配变异"""
        if self.n_materials == 0:
            return

        # 随机选择一个材料,改变其机器分配
        mat_idx = random.randint(0, self.n_materials - 1)
        old_machine = assignment[mat_idx]
        new_machine = random.randint(0, n_machines - 1)

        if old_machine != new_machine:
            assignment[mat_idx] = new_machine
            # 更新序列
            sequences[old_machine].remove(mat_idx)
            sequences[new_machine].append(mat_idx)

    def _mutate_sequences(self, sequences: List[List[int]]):
        """顺序变异 - 随机交换"""
        for seq in sequences:
            if len(seq) >= 2 and random.random() < 0.5:
                i, j = random.sample(range(len(seq)), 2)
                seq[i], seq[j] = seq[j], seq[i]

    def solve(self, pop_size: int = 100, n_generations: int = 200,
              mutation_rate: float = 0.1, verbose: bool = True) -> List[Solution]:
        """
        主求解流程

        Args:
            pop_size: 种群大小
            n_generations: 迭代代数
            mutation_rate: 变异概率
            verbose: 是否打印进度

        Returns:
            Pareto最优解集
        """
        # 1. 初始化种群
        if verbose:
            print(f"初始化种群 (大小: {pop_size})...")
        population = self.create_initial_population(pop_size)

        # 2. 评估初始种群
        if verbose:
            print("评估初始种群...")
        for sol in population:
            self.evaluate_solution(sol, use_continuous_inventory=True)

        # 3. 迭代进化
        for gen in range(n_generations):
            # 非支配排序
            fronts = self.fast_non_dominated_sort(population)

            # 计算拥挤度
            for front in fronts:
                self.calculate_crowding_distance(front)

            # 打印当前代信息
            if verbose and (gen % 10 == 0 or gen == n_generations - 1):
                best_tardiness = min(sol.max_tardiness for sol in fronts[0])
                best_inventory = min(sol.avg_inventory for sol in fronts[0])
                avg_instability = np.mean([sol.process_instability for sol in fronts[0]])

                print(
                    f"代数 {gen + 1}/{n_generations}: "
                    f"Pareto={len(fronts[0])}, "
                    f"Max拖期={best_tardiness:.2f}天, "
                    f"库存={best_inventory:.1f}吨, "
                    f"工艺不稳={avg_instability:.1f}"
                )

                # best_tardiness = min(sol.max_tardiness for sol in fronts[0])
                # best_inventory = min(sol.avg_inventory for sol in fronts[0])
                # avg_penalty = np.mean([sol.process_instability for sol in fronts[0]])
                # print(f"代数 {gen+1}/{n_generations}: "
                #       f"Pareto前沿大小={len(fronts[0])}, "
                #       f"最优拖期={best_tardiness:.2f}天, "
                #       f"最优库存={best_inventory:.1f}, "
                #       f"平均惩罚={avg_penalty:.2f}")

            # 选择
            offspring_size = pop_size
            parents = self.selection(population, offspring_size)

            # 交叉和变异生成子代
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                self.mutation(child1, mutation_rate)
                self.mutation(child2, mutation_rate)
                offspring.extend([child1, child2])

            # 评估子代
            for sol in offspring:
                self.evaluate_solution(sol, use_continuous_inventory=True)

            # 合并父代和子代
            combined = population + offspring

            # 环境选择 - 选择下一代
            fronts = self.fast_non_dominated_sort(combined)
            next_population = []

            for front in fronts:
                if len(next_population) + len(front) <= pop_size:
                    next_population.extend(front)
                else:
                    # 最后一层需要根据拥挤度选择
                    self.calculate_crowding_distance(front)
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    remaining = pop_size - len(next_population)
                    next_population.extend(front[:remaining])
                    break

            population = next_population

        # 返回最终的Pareto前沿
        fronts = self.fast_non_dominated_sort(population)
        if verbose:
            print(f"\n求解完成! Pareto最优解数量: {len(fronts[0])}")

        return fronts[0]


def load_materials_from_db() -> List[Material]:
    """
    从数据库加载材料数据

    Returns:
        材料列表
    """
    try:
        from database import get_db_session
        from table.order_new import order_new

        with get_db_session() as db:
            records = db.query(order_new).all()
        # records = records[:1] + records[10:11]
        materials = []
        for record in records:
            # 解析交货日期
            try:
                delivery_date = datetime.strptime(record.delivery_date, '%Y%m%d')
            except:
                # 如果解析失败,使用30天后作为默认值
                delivery_date = datetime.now() + timedelta(days=30)

            mat = Material(
                id=record.id,
                order_no=record.order_no,
                category=record.category,
                width=float(record.width),
                thickness=float(record.thickness),
                weight=float(record.weight),
                delivery_date=delivery_date
            )
            materials.append(mat)

        return materials
    except Exception as e:
        print(f"从数据库加载数据失败: {e}")
        print("将使用示例数据...")
        raise e



import uuid

def save_results_to_db(pareto_solutions, materials, clear_existing: bool = False):
    """
    保存帕累托前沿结果到数据库

    Args:
        pareto_solutions: 帕累托前沿解列表
        materials: 材料列表（用于获取订单号等信息）
        clear_existing: 是否在插入前清空现有数据，默认为False
    """
    try:
        from database_util import save_pareto_front_results

        # 生成任务ID
        task_id = f"nsga2_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # 设置结束时间
        for sol in pareto_solutions:
            if hasattr(sol, '__dict__') and 'start_time' in sol.__dict__:
                sol.end_time = datetime.now().isoformat()

        save_pareto_front_results(task_id, pareto_solutions, materials, clear_existing=clear_existing)
        print(f"结果已成功保存到数据库，任务ID: {task_id}")
    except ImportError:
        print("警告: 无法导入数据库工具，结果将不会保存到数据库")
    except Exception as e:
        print(f"保存结果到数据库时出错: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("NSGA-II求解器 - 混合流水车间调度问题")
    print("=" * 80)

    # 加载数据
    print("\n正在加载数据...")
    materials = load_materials_from_db()
    print(f"成功加载 {len(materials)} 个材料订单")
    print(f"钢种分布: {pd.Series([m.category for m in materials]).value_counts().to_dict()}")

    # 创建求解器
    params = StaticParameters()
    solver = NSGA2Solver(materials, params)

    # 求解
    print("\n开始求解...")
    print("-" * 80)

    pareto_front = solver.solve(
        pop_size=100,
        n_generations=100,
        mutation_rate=0.15,
        verbose=True
    )

    # 输出结果
    print("\n" + "=" * 80)
    print("求解结果")
    print("=" * 80)

    print(f"\nPareto最优解集大小: {len(pareto_front)}")
    print("\nTop 5 解决方案:")
    print("-" * 80)

    # 按拖期排序
    pareto_front_sorted = sorted(pareto_front, key=lambda x: x.max_tardiness)

    for i, sol in enumerate(pareto_front_sorted[:1]):
        print(f"\n方案 {i+1}:")
        print(f"  总拖期: {sol.max_tardiness:.2f} 天")
        print(f"  库存水平: {sol.avg_inventory:.1f} 吨")
        print(f"  工艺不稳: {sol.process_instability:.2f}")

        # 统计每台机器的负载
        ar_loads = [len(seq) for seq in sol.ar_sequences]
        ca_loads = [len(seq) for seq in sol.ca_sequences]
        print(f"  酸轧机负载: {ar_loads}")
        print(f"  连退机负载: {ca_loads}")

    # 保存结果到数据库
    print("\n正在保存结果到数据库...")
    save_results_to_db(pareto_front, materials, clear_existing=True)

    print("\n" + "=" * 80)
    print("求解完成!")
    print("=" * 80)
    # from visualization import Visualizer
    # visualizer = Visualizer(materials)
    # visualizer.plot_pareto_front(pareto_front, save_path="pareto_front.png")
    # visualizer.plot_pareto_2d(pareto_front, save_path="pareto_front_2d.png")
    # for i, sol in enumerate(pareto_front[:1]):
    #     visualizer.plot_inventory_curve(sol, save_path=f"inventory_curve_{i+1}.png")
    #     visualizer.plot_gantt_chart(sol, save_path=f"gantt_chart_{i+1}.png")



