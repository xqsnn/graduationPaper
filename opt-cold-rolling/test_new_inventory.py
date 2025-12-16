#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新库存计算方法
"""
import sys
import os

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
    
    for i in range(10):
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


def test_new_inventory_calculation():
    """测试新库存计算方法"""
    print("=" * 60)
    print("测试新的连续库存计算方法")
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
        ar_assignment=[i % 2 for i in range(len(materials))],  # 酸轧机分配
        ar_sequences=[[], []],  # 初始化
        ca_assignment=[i % 3 for i in range(len(materials))],  # 连退机分配
        ca_sequences=[[], [], []]  # 初始化
    )
    
    # 重建序列
    for i, machine in enumerate(solution.ar_assignment):
        solution.ar_sequences[machine].append(i)
    for i, machine in enumerate(solution.ca_assignment):
        solution.ca_sequences[machine].append(i)
    
    # 运行调度仿真
    schedule_results = solver._simulate_schedule(solution)
    solution.schedule_results = schedule_results

    # 使用旧方法计算库存
    old_inventory = solver._calculate_inventory(schedule_results)
    print(f"旧方法计算的库存(单位: 件): {old_inventory}")
    
    # 使用新方法计算库存
    new_inventory = solver._calculate_continuous_inventory(schedule_results)
    print(f"新方法计算的库存(单位: 吨): {new_inventory:.2f}")
    
    print("\n调度结果概览:")
    for i, result in enumerate(schedule_results[:3]):  # 仅显示前3个
        mat = materials[result.material_id]
        print(f"  材料 {i}: 重量={mat.weight:.1f}吨, "
              f"热轧({result.hr_start:.1f}-{result.hr_end:.1f}), "
              f"酸轧({result.ar_start:.1f}-{result.ar_end:.1f}), "
              f"连退({result.ca_start:.1f}-{result.ca_end:.1f})")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


def compare_evaluation_methods():
    """比较新旧评估方法"""
    print("\n" + "=" * 60)
    print("比较新旧评估方法")
    print("=" * 60)
    
    # 创建测试数据
    materials = create_test_data()
    
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
    
    # 使用旧方法评估
    old_solution = solver.evaluate_solution(solution, use_continuous_inventory=False)
    print(f"使用旧库存方法 - 拖期: {old_solution.max_tardiness:.2f}, 库存: {old_solution.max_inventory:.2f}")
    
    # 使用新方法评估
    new_solution = solver.evaluate_solution(solution, use_continuous_inventory=True)
    print(f"使用新库存方法 - 拖期: {new_solution.max_tardiness:.2f}, 库存: {new_solution.max_inventory:.2f}")
    
    print("\n" + "=" * 60)
    print("比较完成")
    print("=" * 60)


if __name__ == "__main__":
    test_new_inventory_calculation()
    compare_evaluation_methods()