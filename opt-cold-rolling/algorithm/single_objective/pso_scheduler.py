"""
粒子群优化算法 (PSO) - 用于混合流水车间调度问题
"""
import numpy as np
import random
from typing import List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


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

    # 详细调度结果
    schedule_results: List[ScheduleResult] = None


class StaticParameters:
    """静态参数配置"""
    start_time = datetime.strptime("20260118", "%Y%m%d")

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


class Particle:
    """粒子类"""
    def __init__(self, solution: Solution):
        self.position = solution  # 粒子位置（解）
        self.velocity = self._initialize_velocity()  # 粒子速度
        self.personal_best = solution  # 个体最优解
        self.personal_best_fitness = float('inf')  # 个体最优适应度
    
    def _initialize_velocity(self):
        """初始化粒子速度"""
        # 速度用一个字典表示对解各个部分的扰动程度
        velocity = {
            'hr_perturbation': random.uniform(-0.5, 0.5),
            'ar_assignment_perturbation': random.uniform(-0.5, 0.5),
            'ca_assignment_perturbation': random.uniform(-0.5, 0.5)
        }
        return velocity
    
    def update_position(self):
        """更新粒子位置"""
        # 这里需要根据速度更新解的编码
        # 实现对调度顺序和分配的扰动
        self._perturb_solution()
    
    def _perturb_solution(self):
        """根据速度扰动解"""
        # 简单的扰动策略：对热轧顺序进行交换
        if random.random() < abs(self.velocity['hr_perturbation']):
            # 随机交换热轧顺序中的两个元素
            if len(self.position.hr_sequence) > 1:
                i, j = random.sample(range(len(self.position.hr_sequence)), 2)
                self.position.hr_sequence[i], self.position.hr_sequence[j] = \
                    self.position.hr_sequence[j], self.position.hr_sequence[i]
        
        # 对酸轧分配进行扰动
        if random.random() < abs(self.velocity['ar_assignment_perturbation']):
            if len(self.position.ar_assignment) > 0:
                idx = random.randint(0, len(self.position.ar_assignment) - 1)
                n_machines = StaticParameters.num_machines[Operation.AR]
                self.position.ar_assignment[idx] = random.randint(0, n_machines - 1)
        
        # 对连退分配进行扰动
        if random.random() < abs(self.velocity['ca_assignment_perturbation']):
            if len(self.position.ca_assignment) > 0:
                idx = random.randint(0, len(self.position.ca_assignment) - 1)
                n_machines = StaticParameters.num_machines[Operation.CA]
                self.position.ca_assignment[idx] = random.randint(0, n_machines - 1)


class PSOScheduler:
    """基于PSO的调度求解器"""
    
    def __init__(self, materials: List[Material], params: StaticParameters):
        self.materials = materials
        self.params = params
        self.n_materials = len(materials)
        
        # 按钢种分组 - 用于热轧约束
        self.category_groups = self._group_by_category()
    
    def _group_by_category(self) -> dict:
        """按钢种对材料进行分组"""
        groups = {}
        for i, mat in enumerate(self.materials):
            if mat.category not in groups:
                groups[mat.category] = []
            groups[mat.category].append(i)
        return groups

    def create_initial_solution(self) -> Solution:
        """创建初始解"""
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

        return solution

    def _create_hr_sequence(self) -> List[int]:
        """创建热轧顺序 - 满足钢种约束"""
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
        """创建酸轧机器分配和顺序 - 考虑宽度递减约束"""
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
        """创建连退机器分配和顺序 - 考虑厚度集中约束"""
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

    def evaluate_solution(self, solution: Solution) -> float:
        """评估解的适应度（多目标加权和）"""
        # 仿真调度过程
        schedule_results = self._simulate_schedule(solution)
        solution.schedule_results = schedule_results

        # 计算目标函数
        max_tardiness = 0.0
        for result in schedule_results:
            mat = self.materials[result.material_id]
            completion_days = result.ca_end / 24.0
            delivery_days = (mat.delivery_date - StaticParameters.start_time).days
            tardiness = max(0, completion_days - delivery_days)
            max_tardiness = max(max_tardiness, tardiness)

        # 计算库存
        avg_inventory = self._calculate_continuous_inventory(schedule_results)

        # 计算工艺不稳定性
        hr_switch = self._hr_switch_count(solution)
        ar_jump = self._ar_width_jump(solution)
        ca_switch = self._ca_thickness_switch(solution)
        process_instability = (
                5.0 * hr_switch +
                1.0 * ar_jump +
                10.0 * ca_switch
        )

        solution.max_tardiness = max_tardiness
        solution.avg_inventory = avg_inventory
        solution.process_instability = process_instability

        # 多目标加权和作为适应度函数
        # 可以根据实际需要调整权重
        fitness = 1.0 * max_tardiness + 0.01 * avg_inventory + 0.001 * process_instability
        return fitness

    def _simulate_schedule(self, solution: Solution) -> List[ScheduleResult]:
        """仿真调度过程,计算每个材料在各工序的开始和结束时间"""
        results = [None] * self.n_materials

        # 1. 热轧工序 - 单机,按顺序处理
        hr_speed = self.params.speed[Operation.HR][0]
        current_time = 0.0

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
                ca_end=0
            )

            current_time = end_time

        # 2. 酸轧工序 - 多台机器并行
        ar_speeds = self.params.speed[Operation.AR]
        ar_machine_time = [0.0] * len(ar_speeds)  # 每台机器的当前时间

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
        ca_machine_time = [0.0] * len(ca_speeds)

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

                ca_machine_time[machine_id] = end_time

        return results

    def _calculate_continuous_inventory(self, schedule_results: List[ScheduleResult]) -> float:
        """计算连续生产的库存水平(改进版)"""
        # 获取调度结束时间
        max_time = max(result.ca_end for result in schedule_results)

        # 采样间隔(2小时)
        sample_interval = 2.0

        # 存储各时间段的库存
        inventory_samples = []

        # 按时间间隔采样库存
        current_time = 0.0
        while current_time <= max_time:
            # 计算当前时间点的在制品库存
            hr_to_ar_inventory = 0.0  # 热轧后等待酸轧的库存
            ar_to_ca_inventory = 0.0  # 酸轧后等待连退的库存

            for result in schedule_results:
                mat_weight = self.materials[result.material_id].weight

                # 检查是否在热轧后等待酸轧区域
                if result.hr_end <= current_time < result.ar_start:
                    hr_to_ar_inventory += mat_weight

                # 检查是否在酸轧后等待连退区域
                if result.ar_end <= current_time < result.ca_start:
                    ar_to_ca_inventory += mat_weight

            total_inventory = hr_to_ar_inventory + ar_to_ca_inventory
            inventory_samples.append(total_inventory)

            current_time += sample_interval

        # 返回平均库存水平
        if len(inventory_samples) > 0:
            average_inventory = sum(inventory_samples) / len(inventory_samples)
            return average_inventory
        else:
            return 0.0

    def _hr_switch_count(self, solution: Solution) -> int:
        """热轧钢种切换次数"""
        count = 0
        for i in range(len(solution.hr_sequence) - 1):
            curr_idx = solution.hr_sequence[i]
            next_idx = solution.hr_sequence[i + 1]
            if self.materials[curr_idx].category != self.materials[next_idx].category:
                count += 1
        return count

    def _ar_width_jump(self, solution: Solution) -> float:
        """酸轧宽度跳跃总和"""
        total_jump = 0.0
        for sequence in solution.ar_sequences:
            for i in range(len(sequence) - 1):
                curr_width = self.materials[sequence[i]].width
                next_width = self.materials[sequence[i + 1]].width
                if curr_width < next_width:  # 违反宽度递减约束
                    total_jump += (next_width - curr_width)
        return total_jump

    def _ca_thickness_switch(self, solution: Solution) -> float:
        """连退厚度切换程度"""
        total_switch = 0.0
        for sequence in solution.ca_sequences:
            if len(sequence) > 1:
                thicknesses = [self.materials[i].thickness for i in sequence]
                # 计算相邻厚度差异
                for i in range(len(thicknesses) - 1):
                    total_switch += abs(thicknesses[i] - thicknesses[i + 1])
        return total_switch

    def solve(self, n_particles: int = 30, n_iterations: int = 100, 
              w: float = 0.7, c1: float = 1.5, c2: float = 1.5, verbose: bool = True) -> Solution:
        """
        PSO主求解流程
        
        Args:
            n_particles: 粒子数量
            n_iterations: 迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            verbose: 是否打印进度
        
        Returns:
            最优解
        """
        # 初始化粒子群
        particles = []
        global_best = None
        global_best_fitness = float('inf')
        
        if verbose:
            print(f"初始化粒子群 (大小: {n_particles})...")
        
        for _ in range(n_particles):
            solution = self.create_initial_solution()
            particle = Particle(solution)
            fitness = self.evaluate_solution(particle.position)
            
            # 更新个体最优
            particle.personal_best_fitness = fitness
            particle.personal_best = solution
            
            # 更新全局最优
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best = solution
            
            particles.append(particle)
        
        if verbose:
            print(f"初始最优适应度: {global_best_fitness:.2f}")
        
        # 迭代优化
        for iter in range(n_iterations):
            for particle in particles:
                # 计算当前适应度
                fitness = self.evaluate_solution(particle.position)
                
                # 更新个体最优
                if fitness < particle.personal_best_fitness:
                    particle.personal_best_fitness = fitness
                    particle.personal_best = particle.position
            
                # 更新全局最优
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particle.position
            
            # 更新每个粒子的速度和位置
            for particle in particles:
                # 更新速度
                r1, r2 = random.random(), random.random()
                
                # 对速度进行更新（简化处理）
                for key in particle.velocity:
                    particle.velocity[key] = (
                        w * particle.velocity[key] +
                        c1 * r1 * (particle.personal_best.hr_sequence[0] - particle.position.hr_sequence[0]) * 0.01 +
                        c2 * r2 * (global_best.hr_sequence[0] - particle.position.hr_sequence[0]) * 0.01
                    )
                    # 限制速度范围
                    particle.velocity[key] = max(-1.0, min(1.0, particle.velocity[key]))
                
                # 更新位置
                particle.update_position()
            
            if verbose and (iter + 1) % 10 == 0:
                print(f"迭代 {iter + 1}/{n_iterations}, 最优适应度: {global_best_fitness:.2f}")
        
        # 返回最优解
        self.evaluate_solution(global_best)  # 确保最优解的目标函数值已计算
        return global_best


if __name__ == "__main__":
    # 这里可以添加测试代码
    pass