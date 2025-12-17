#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试完整的NSGA2流程包含重构的甘特图
"""
import sys
import os

# 添加项目路径
sys.path.append('/mnt/c/Users/lxq/Desktop/GraduationPaper/opt-cold-rolling')

from algorithm.multiple_objective.nsga2_solver import NSGA2Solver, Material, StaticParameters
from datetime import datetime, timedelta
import random


def create_test_data():
    """创建测试数据"""
    # 设置随机种子
    random.seed(42)
    
    # 创建一些测试材料
    materials = []
    categories = ['A', 'B', 'C']
    
    for i in range(6):  # 少量材料进行快速测试
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


def test_full_nsga2_with_new_gantt():
    """测试完整的NSGA2流程"""
    print("=" * 60)
    print("测试完整的NSGA2流程（包含重构的甘特图）")
    print("=" * 60)
    
    # 创建测试数据
    materials = create_test_data()
    print(f"创建了 {len(materials)} 个测试材料")
    
    # 创建求解器
    params = StaticParameters()
    solver = NSGA2Solver(materials, params)
    
    # 运行小规模NSGA2测试
    print("\n开始NSGA2求解...")
    pareto_front = solver.solve(
        pop_size=10,        # 减小种群大小以加快测试
        n_generations=5,   # 减少迭代次数以加快测试
        mutation_rate=0.1,
        verbose=True
    )
    
    print(f"\n找到 {len(pareto_front)} 个Pareto最优解")
    
    if pareto_front:
        # 显示最优解的统计信息
        best_sol = min(pareto_front, key=lambda x: x.max_tardiness)
        print(f"\n最优解统计:")
        print(f"  拖期: {best_sol.max_tardiness:.2f} 天")
        print(f"  库存: {best_sol.avg_inventory:.2f} 吨")
        print(f"  工艺不稳: {best_sol.process_instability:.2f}")
        
        # 测试重构后的甘特图功能
        print("\n正在测试重构后的甘特图功能...")
        try:
            from algorithm.multiple_objective.visualization import Visualizer
            visualizer = Visualizer(materials)
            visualizer.plot_gantt_chart(best_sol, save_path="test_full_nsga2_gantt.png")
            print("重构后的甘特图已成功生成: test_full_nsga2_gantt.png")
        except Exception as e:
            print(f"生成甘特图时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("完整NSGA2流程测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_full_nsga2_with_new_gantt()