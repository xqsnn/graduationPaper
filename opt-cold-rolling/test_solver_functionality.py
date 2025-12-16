#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试NSGA2求解器功能
"""
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
    
    for i in range(15):  # 使用15个材料进行测试
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


def test_solver_functionality():
    """测试求解器功能"""
    print("=" * 60)
    print("测试NSGA2求解器功能")
    print("=" * 60)
    
    # 创建测试数据
    materials = create_test_data()
    print(f"创建了 {len(materials)} 个测试材料")
    
    # 统计信息
    print(f"钢种分布: A={sum(1 for m in materials if m.category == 'A')}, "
          f"B={sum(1 for m in materials if m.category == 'B')}, "
          f"C={sum(1 for m in materials if m.category == 'C')}")
    
    # 创建求解器
    params = StaticParameters()
    solver = NSGA2Solver(materials, params)
    
    # 测试种群初始化
    print("\n测试种群初始化...")
    pop_size = 10
    population = solver.create_initial_population(pop_size)
    print(f"成功创建 {len(population)} 个初始解")
    
    # 测试解的评估
    print("\n测试解的评估...")
    for i, sol in enumerate(population[:3]):  # 只测试前3个
        sol = solver.evaluate_solution(sol, use_continuous_inventory=True)
        print(f"  解 {i}: 拖期={sol.max_tardiness:.2f}, 库存={sol.avg_inventory:.2f}, 惩罚={sol.process_instability:.2f}")
    
    # 简单测试算法运行
    print("\n测试完整算法流程 (小规模)...")
    try:
        pareto_front = solver.solve(
            pop_size=20,
            n_generations=10,  # 减少代数以加快测试
            mutation_rate=0.1,
            verbose=True
        )
        print(f"成功找到 {len(pareto_front)} 个Pareto最优解")
        
        if len(pareto_front) > 0:
            best_sol = min(pareto_front, key=lambda x: x.max_tardiness)
            print(f"最优解 - 拖期: {best_sol.max_tardiness:.2f}, "
                  f"库存: {best_sol.avg_inventory:.2f}, "
                  f"惩罚: {best_sol.process_instability:.2f}")
        else:
            print("未找到任何Pareto最优解")
            
    except Exception as e:
        print(f"算法运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("功能测试完成")
    print("=" * 60)


def test_new_vs_old_inventory():
    """测试新旧库存计算方法对比"""
    print("\n" + "=" * 60)
    print("新旧库存计算方法对比测试")
    print("=" * 60)
    
    materials = create_test_data()
    params = StaticParameters()
    solver = NSGA2Solver(materials, params)
    
    # 创建一个测试解
    solution = solver.create_initial_population(1)[0]
    
    # 使用旧方法评估（注意：这里需要临时修改参数来使用旧方法）
    sol_old = solver.evaluate_solution(solution, use_continuous_inventory=False)
    print(f"旧方法: 拖期={sol_old.max_tardiness:.2f}, 库存={sol_old.avg_inventory:.2f}")

    # 使用新方法评估
    sol_new = solver.evaluate_solution(solution, use_continuous_inventory=True)
    print(f"新方法: 拖期={sol_new.max_tardiness:.2f}, 库存={sol_new.avg_inventory:.2f}")

    print(f"差异: 拖期={sol_new.max_tardiness - sol_old.max_tardiness:.2f}, "
          f"库存={sol_new.avg_inventory - sol_old.avg_inventory:.2f}")
    
    print("\n" + "=" * 60)
    print("对比测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_solver_functionality()
    test_new_vs_old_inventory()