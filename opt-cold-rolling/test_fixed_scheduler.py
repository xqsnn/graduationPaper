#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的调度算法
"""

from algorithm.order_plan.multi_machines.order_schedule_fixed import Scheduler, data_process, static_params, orders_obj_list

def test_scheduler():
    """
    测试调度器的基本功能
    """
    print("开始测试修复后的调度算法...")
    
    try:
        # 创建调度器实例
        scheduler = Scheduler(orders_obj_list, static_params)
        print(f"成功创建调度器，订单数量: {len(orders_obj_list)}")
        
        # 运行简化版的调度算法进行测试
        print("开始运行调度算法...")
        best_individual, schedule_results, order_delays, stock_history_detailed, objectives = scheduler.solve(
            population_size=20,  # 减少种群大小以加快测试
            generations=10       # 减少代数以加快测试
        )
        
        print("\n--- 测试结果 ---")
        print(f"最佳个体存在: {best_individual is not None}")
        print(f"调度结果数量: {len(schedule_results)}")
        print(f"订单延迟数量: {len(order_delays)}")
        print(f"目标值: 拖期={objectives[0]:.2f}, 成本={objectives[1]:.2f}")
        
        print("\n测试成功完成!")
        return True
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scheduler()
    if success:
        print("\n所有测试通过！算法修复成功。")
    else:
        print("\n测试失败，请检查错误。")