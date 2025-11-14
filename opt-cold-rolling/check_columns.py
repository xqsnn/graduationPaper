"""
检查调度结果的列名
"""
from algorithm.order_plan.rule_heuristic_t.solver import HeuristicSolver
from algorithm.order_plan.rule_heuristic_t.order_data_acess import OrderDataAccess
from table.order import order


def check_schedule_result_columns():
    print("检查调度结果的列名...")
    
    solver = HeuristicSolver()
    
    # 设置使用一个简单的策略
    solver.set_solve_strategy('EDD')

    order_data_access = OrderDataAccess()

    # 获取订单数据
    orders = order_data_access.get_orders_by_month_and_limit('202411', 5)

    dataframe = order.to_dataframe(orders)

    static_params = order_data_access.get_orders_static_parameter(orders)

    # 解决调度问题
    solve_result = solver.solve(dataframe, static_params)
    
    print("调度结果的列名:")
    print(solve_result.columns.tolist())
    
    print("\n调度结果前几行:")
    print(solve_result.head())


if __name__ == '__main__':
    check_schedule_result_columns()