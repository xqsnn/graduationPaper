#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试调度仿真是否正确
"""
import sys
import os

# 添加项目路径
sys.path.append('/mnt/c/Users/lxq/Desktop/GraduationPaper/opt-cold-rolling')

from algorithm.multiple_objective.nsga2_solver import NSGA2Solver, Material, Solution, StaticParameters
from datetime import datetime, timedelta
import random


def create_test_data():
    """创建测试数据"""
    # 设置随机种子
    random.seed(42)
    
    # 创建一些测试材料
    materials = []
    categories = ['A', 'B', 'C']
    
    for i in range(5):  # 少量材料进行快速测试
        mat = Material(
            id=i,
            order_no=f"ORD-{i:03d}",
            category=random.choice(categories),
            width=random.uniform(1000, 1500),
            thickness=random.uniform(1.0, 2.0),
            weight=random.uniform(25, 35),  # 使用相对均匀的重量
            delivery_date=datetime.now() + timedelta(days=random.randint(10, 30))
        )
        materials.append(mat)
    
    return materials


def test_schedule_simulation():
    """测试调度仿真是否正确"""
    print("=" * 80)
    print("测试调度仿真是否正确 - 检查热轧工序时间分配")
    print("=" * 80)
    
    # 创建测试数据
    materials = create_test_data()
    print(f"创建了 {len(materials)} 个测试材料")
    
    # 打印材料信息
    for i, mat in enumerate(materials):
        print(f"  材料 {i}: 重量={mat.weight:.1f}吨, 钢种={mat.category}")
    
    # 创建求解器
    params = StaticParameters()
    solver = NSGA2Solver(materials, params)
    
    # 创建一个测试解，热轧顺序为 [0,1,2,3,4]
    solution = Solution(
        hr_sequence=[0, 1, 2, 3, 4],  # 按ID顺序
        ar_assignment=[i % 2 for i in range(len(materials))],  # 酸轧机分配
        ar_sequences=[[], []],
        ca_assignment=[i % 3 for i in range(len(materials))],  # 连退机分配
        ca_sequences=[[], [], []]
    )
    
    # 重建序列
    for i, machine in enumerate(solution.ar_assignment):
        solution.ar_sequences[machine].append(i)
    for i, machine in enumerate(solution.ca_assignment):
        solution.ca_sequences[machine].append(i)
    
    print(f"\n热轧工序顺序: {solution.hr_sequence}")
    print(f"酸轧分配: {solution.ar_assignment}")
    print(f"连退分配: {solution.ca_assignment}")
    
    # 执行调度仿真
    schedule_results = solver._simulate_schedule(solution)
    solution.schedule_results = schedule_results
    
    print("\n调度仿真结果:")
    print("材料ID | 热轧开始 | 热轧结束 | 酸轧机 | 酸轧开始 | 酸轧结束 | 连退机 | 连退开始 | 连退结束")
    print("-" * 85)
    
    # 按热轧顺序排序显示
    for mat_idx in solution.hr_sequence:
        result = schedule_results[mat_idx]
        print(f"  {mat_idx}    | {result.hr_start:6.2f}  | {result.hr_end:6.2f}  | "
              f" AR-{result.ar_machine+1}  | {result.ar_start:6.2f}  | {result.ar_end:6.2f}  | "
              f" CA-{result.ca_machine+1}  | {result.ca_start:6.2f}  | {result.ca_end:6.2f}")
    
    # 检查热轧工序是否正确串行执行
    print("\n热轧工序检查:")
    prev_end = 0.0
    hr_correct = True
    for i, mat_idx in enumerate(solution.hr_sequence):
        result = schedule_results[mat_idx]
        print(f"材料 {mat_idx}: 开始={result.hr_start:.2f}, 结束={result.hr_end:.2f}, "
              f"与前一个结束时间比较: prev_end={prev_end:.2f}")
        
        if abs(result.hr_start - prev_end) > 0.001:  # 允许小的浮点误差
            print(f"  错误: 材料 {mat_idx} 的开始时间({result.hr_start:.2f}) != 前一个结束时间({prev_end:.2f})")
            hr_correct = False
        else:
            print(f"  正确: 材料 {mat_idx} 在前一个材料结束后立即开始")
        
        prev_end = result.hr_end
    
    if hr_correct:
        print("\n[Y] 热轧工序仿真正确：严格按照顺序执行，无时间重叠")
    else:
        print("\n[N] 热轧工序仿真有问题：存在时间重叠或间隙")

    print("\n" + "=" * 80)
    print("调度仿真测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_schedule_simulation()