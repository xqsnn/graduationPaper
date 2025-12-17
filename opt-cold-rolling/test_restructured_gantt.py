#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试重构后的甘特图功能
"""
import sys
import os

# 添加项目路径
sys.path.append('/mnt/c/Users/lxq/Desktop/GraduationPaper/opt-cold-rolling')

from algorithm.multiple_objective.nsga2_solver import NSGA2Solver, Material, Solution, StaticParameters
from algorithm.multiple_objective.visualization import Visualizer
from datetime import datetime, timedelta
import random


def create_test_data():
    """创建测试数据"""
    # 设置随机种子
    random.seed(42)
    
    # 创建一些测试材料
    materials = []
    categories = ['A', 'B', 'C']
    
    for i in range(8):  # 少量材料进行快速测试
        mat = Material(
            id=i,
            order_no=f"ORD-{i:03d}",
            category=random.choice(categories),
            width=random.uniform(1000, 1500),
            thickness=random.uniform(1.0, 2.0),
            weight=random.uniform(20, 40),
            delivery_date=datetime.now() + timedelta(days=random.randint(10, 30))
        )
        materials.append(mat)
    
    return materials


def test_restructured_gantt():
    """测试重构后的甘特图"""
    print("=" * 60)
    print("测试重构后的甘特图功能")
    print("=" * 60)
    
    # 创建测试数据
    materials = create_test_data()
    print(f"创建了 {len(materials)} 个测试材料")
    
    # 创建求解器
    params = StaticParameters()
    solver = NSGA2Solver(materials, params)
    
    # 创建一个测试解
    solution = Solution(
        hr_sequence=list(range(len(materials))),
        ar_assignment=[i % 2 for i in range(len(materials))],
        ar_sequences=[[], []],
        ca_assignment=[i % 3 for i in range(len(materials))],
        ca_sequences=[[], [], []]
    )
    
    # 重建序列
    for i, machine in enumerate(solution.ar_assignment):
        solution.ar_sequences[machine].append(i)
    for i, machine in enumerate(solution.ca_assignment):
        solution.ca_sequences[machine].append(i)
    
    # 评估解以生成调度结果
    solution = solver.evaluate_solution(solution, use_continuous_inventory=True)
    
    print(f"解已评估，调度结果数: {len(solution.schedule_results) if solution.schedule_results else 0}")
    
    # 创建可视化器并测试重构的甘特图
    visualizer = Visualizer(materials)
    print("\n正在生成重构后的甘特图...")
    
    # 保存甘特图
    try:
        visualizer.plot_gantt_chart(solution, save_path="test_restructured_gantt.png")
        print("甘特图已保存为 test_restructured_gantt.png")
    except Exception as e:
        print(f"生成甘特图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("甘特图重构测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_restructured_gantt()