"""
比较不同策略的性能
"""
from algorithm.order_plan.rule_heuristic_t.solver import HeuristicSolver
from algorithm.order_plan.rule_heuristic_t.order_data_acess import OrderDataAccess
from table.order import order


def compare_strategies():
    print("比较不同策略的性能...")
    
    order_data_access = OrderDataAccess()

    # 获取订单数据
    orders = order_data_access.get_orders_by_month_and_limit('202411', 10)

    dataframe = order.to_dataframe(orders)

    static_params = order_data_access.get_orders_static_parameter(orders)
    
    # 定义要测试的策略
    strategies = ['EDD', 'FCFS', 'MIN_INV', 'IMPROVED_MIN_INV', 'INVENTORY_CONSTRAINED']
    
    results = {}
    
    for strategy_name in strategies:
        print(f"\n正在测试策略: {strategy_name}")
        
        solver = HeuristicSolver()
        solver.set_solve_strategy(strategy_name)
        
        try:
            solve_result = solver.solve(dataframe, static_params)
            
            total_delay = solve_result['L_i_hours'].sum()
            total_az_inventory = solve_result['I_AZ_hours'].sum()
            total_lt_inventory = solve_result['I_LT_hours'].sum()
            
            results[strategy_name] = {
                'total_delay': total_delay,
                'total_az_inventory': total_az_inventory,
                'total_lt_inventory': total_lt_inventory,
                'avg_delay': solve_result['L_i_hours'].mean()
            }
            
            print(f"  总拖期: {total_delay:.2f}")
            print(f"  总酸轧后库存时间: {total_az_inventory:.2f}")
            print(f"  总连退后库存时间: {total_lt_inventory:.2f}")
            print(f"  平均拖期: {solve_result['L_i_hours'].mean():.2f}")
        except Exception as e:
            print(f"  策略 {strategy_name} 执行出错: {e}")
    
    print("\n策略比较结果:")
    print("-" * 80)
    print(f"{'策略':<20} {'总拖期':<12} {'酸轧库存':<12} {'连退库存':<12} {'平均拖期':<12}")
    print("-" * 80)
    
    for strategy_name, metrics in results.items():
        print(f"{strategy_name:<20} {metrics['total_delay']:<12.2f} {metrics['total_az_inventory']:<12.2f} {metrics['total_lt_inventory']:<12.2f} {metrics['avg_delay']:<12.2f}")


if __name__ == '__main__':
    compare_strategies()