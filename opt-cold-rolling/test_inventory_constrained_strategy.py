"""
测试库存约束策略
"""
from algorithm.order_plan.rule_heuristic_t.solver import HeuristicSolver
from algorithm.order_plan.rule_heuristic_t.order_data_acess import OrderDataAccess
from table.order import order


def test_inventory_constrained_strategy():
    print("开始测试库存约束策略...")
    
    solver = HeuristicSolver()

    # 设置使用新策略
    solver.set_solve_strategy('INVENTORY_CONSTRAINED')

    order_data_access = OrderDataAccess()

    # 获取订单数据
    orders = order_data_access.get_orders_by_month_and_limit('202411', 30)

    dataframe = order.to_dataframe(orders)

    static_params = order_data_access.get_orders_static_parameter(orders)

    # 解决调度问题
    solve_result = solver.solve(dataframe, static_params)

    print("\n调度结果:")
    print(solve_result.head(10))
    
    print(f"\n总拖期 (小时): {solve_result['L_i_hours'].sum():.2f}")
    print(f"总酸轧后库库存时间 (小时): {solve_result['I_AZ_hours'].sum():.2f}")
    print(f"总连退后库库存时间 (小时): {solve_result['I_LT_hours'].sum():.2f}")
    print(f"平均拖期 (小时): {solve_result['L_i_hours'].mean():.2f}")

    # 可视化结果
    try:
        solver.plot_schedule_gantt(solve_result)
        solver.plot_inventory_curve(solve_result, static_params)
    except Exception as e:
        print(f"绘图时出现错误: {e}")

    print("\n库存约束策略测试完成!")


if __name__ == '__main__':
    test_inventory_constrained_strategy()