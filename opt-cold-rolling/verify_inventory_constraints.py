"""
验证库存约束策略是否真正满足库存限制
"""
from algorithm.order_plan.rule_heuristic_t.solver import HeuristicSolver
from algorithm.order_plan.rule_heuristic_t.order_data_acess import OrderDataAccess
from table.order import order
import pandas as pd
from datetime import timedelta


def verify_inventory_constraints(schedule_result, original_orders_df, static_params, inventory_limits):
    """
    验证调度结果是否满足库存限制
    """
    print("验证库存约束...")
    
    # 按酸轧结束时间排序
    sorted_result = schedule_result.sort_values(by='E_AZ').reset_index(drop=True)
    
    # 创建时间序列来跟踪库存变化
    acid_inventory_timeline = []  # 酸轧后库
    anneal_inventory_timeline = []  # 连退后库
    
    acid_warehouse_limit, anneal_warehouse_limit = inventory_limits
    
    for _, row in sorted_result.iterrows():
        # 从原始订单数据中获取订单重量
        order_no = row['ORDER_NO']
        order_weight = original_orders_df[original_orders_df['ORDER_NO'] == order_no]['ORDER_WT'].iloc[0]
        
        # 计算酸轧后库存时间（从酸轧结束+传输时间到连退开始）
        # 传输时间在static_params中定义
        az_transmission_time = static_params[order_no][0]['transmission_time']
        
        # 添加酸轧后库存事件（酸轧结束到连退开始之间的库存）
        if row['I_AZ_hours'] > 0:
            inventory_start = row['E_AZ'] + timedelta(hours=az_transmission_time)
            inventory_end = row['S_LT']
            acid_inventory_timeline.append({
                'start': inventory_start,
                'end': inventory_end,
                'weight': order_weight
            })
        
        # 添加连退后库存事件（连退结束到交货日期之间的库存）
        if row['I_LT_hours'] > 0:
            inventory_start = row['E_LT']
            inventory_end = row['DELIVERY_DATE']
            anneal_inventory_timeline.append({
                'start': inventory_start,
                'end': inventory_end,
                'weight': order_weight
            })
    
    # 检查酸轧后库是否超限
    max_acid_inventory = check_inventory_timeline(acid_inventory_timeline, acid_warehouse_limit, "酸轧后库")
    
    # 检查连退后库是否超限
    max_anneal_inventory = check_inventory_timeline(anneal_inventory_timeline, anneal_warehouse_limit, "连退后库")
    
    print(f"酸轧后库最大库存量: {max_acid_inventory:.2f}, 限制: {acid_warehouse_limit}")
    print(f"连退后库最大库存量: {max_anneal_inventory:.2f}, 限制: {anneal_warehouse_limit}")
    
    acid_within_limit = max_acid_inventory <= acid_warehouse_limit
    anneal_within_limit = max_anneal_inventory <= anneal_warehouse_limit
    
    print(f"酸轧后库是否在限制内: {acid_within_limit}")
    print(f"连退后库是否在限制内: {anneal_within_limit}")
    
    return acid_within_limit and anneal_within_limit


def check_inventory_timeline(inventory_timeline, limit, warehouse_name):
    """
    检查库存时间线是否在限制内
    """
    if not inventory_timeline:
        return 0
    
    # 获取所有时间点
    time_points = set()
    for inv in inventory_timeline:
        time_points.add(inv['start'])
        time_points.add(inv['end'])
    
    time_points = sorted(list(time_points))
    
    max_inventory = 0
    
    # 对每个时间段计算库存
    for i in range(len(time_points) - 1):
        current_time = time_points[i]
        next_time = time_points[i + 1]
        
        # 计算这个时间段内的库存总重量
        current_inventory = 0
        for inv in inventory_timeline:
            if inv['start'] <= current_time and inv['end'] > current_time:
                current_inventory += inv['weight']
        
        if current_inventory > max_inventory:
            max_inventory = current_inventory
    
    return max_inventory


def test_inventory_constrained_strategy_with_verification():
    print("测试库存约束策略并验证结果...")
    
    solver = HeuristicSolver()
    
    # 设置使用新策略
    solver.set_solve_strategy('INVENTORY_CONSTRAINED')

    order_data_access = OrderDataAccess()

    # 获取订单数据
    orders = order_data_access.get_orders_by_month_and_limit('202411', 15)

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

    # 验证库存约束
    inventory_limits = [1000.0, 3000.0]  # 与配置文件中的值一致
    is_valid = verify_inventory_constraints(solve_result, dataframe, static_params, inventory_limits)

    print(f"\n库存约束是否满足: {is_valid}")
    
    if is_valid:
        print("新策略成功满足了库存限制要求！")
    else:
        print("新策略未能满足库存限制要求。")
    
    return is_valid


if __name__ == '__main__':
    test_inventory_constrained_strategy_with_verification()