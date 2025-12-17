#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新的库存计算方法
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


def test_new_inventory():
    """测试新库存计算方法"""
    print("=" * 60)
    print("测试新的事件驱动库存计算方法")
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

    # 使用新方法计算库存
    new_inventory = solver._calculate_continuous_inventory(schedule_results)
    print(f"新方法计算的加权平均库存: {new_inventory:.2f} 吨")
    
    # 计算库存峰值和平均值
    peak_inventory, avg_inventory = solver._calculate_inventory_peak_and_average(schedule_results)
    print(f"库存峰值: {peak_inventory:.2f} 吨")
    print(f"加权平均库存: {avg_inventory:.2f} 吨 (应该与上面相同)")
    
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


if __name__ == "__main__":
    test_new_inventory()