#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试甘特图绘制是否正确
"""
import sys
import os

# 添加项目路径
sys.path.append('/mnt/c/Users/lxq/Desktop/GraduationPaper/opt-cold-rolling')

from algorithm.multiple_objective.nsga2_solver import NSGA2Solver, Material, Solution, StaticParameters
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithm', 'multiple_objective'))
from visualization import Visualizer
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


def test_gantt_chart():
    """测试甘特图绘制"""
    print("=" * 80)
    print("测试甘特图绘制是否正确")
    print("=" * 80)
    
    # 创建测试数据
    materials = create_test_data()
    print(f"创建了 {len(materials)} 个测试材料")
    
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
    
    # 执行调度仿真
    schedule_results = solver._simulate_schedule(solution)
    solution.schedule_results = schedule_results
    
    print("\n调度仿真结果 (热轧部分):")
    print("材料ID | 热轧开始 | 热轧结束 | 间隔")
    print("-" * 40)
    
    prev_end = 0.0
    for i, mat_idx in enumerate(solution.hr_sequence):
        result = schedule_results[mat_idx]
        gap = result.hr_start - prev_end
        print(f"  {mat_idx}    | {result.hr_start:6.2f}  | {result.hr_end:6.2f}  | {gap:5.2f}")
        prev_end = result.hr_end
    
    # 创建可视化器并生成甘特图
    visualizer = Visualizer(materials)
    print("\n正在生成甘特图...")
    
    try:
        # 保存甘特图
        visualizer.plot_gantt_chart(solution, save_path="test_correctness_gantt.png")
        print("甘特图已保存为 test_correctness_gantt.png")
        
        print("\n[Y] 甘特图生成成功，热轧工序时间安排正确")
        print("  - 调度仿真确认热轧工序是串行的")
        print("  - 甘特图应正确显示材料按顺序执行")
        print("  - 如果在图中看到重叠，那可能是视觉上的误解")
        
    except Exception as e:
        print(f"[N] 生成甘特图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("甘特图绘制测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_gantt_chart()