from typing import Dict

import pandas as pd

from algorithm.order_plan.rule_heuristic_t.order_data_acess import OrderDataAccess
from algorithm.order_plan.rule_heuristic_t.plot_visualization import PlotVisualization
from algorithm.order_plan.rule_heuristic_t.strategy.strategy_factory import StrategyFactory
from algorithm.order_plan.rule_heuristic_t.strategy.strategy import strategy
from table.order import order


# 求解器

class HeuristicSolver:
    """
    启发式求解器主类，提供统一的接口用于调度问题的求解和可视化
    """


    def __init__(self):
        self.solve_strategy = strategy()
        self.plot_visualization = PlotVisualization()
        pass

    def set_solve_strategy(self, strategy_name: str):
        """
        设置调度策略

        Args:
            strategy_name: 调度策略名称
        """
        strategy_factory = StrategyFactory()
        self.solve_strategy = strategy_factory.create_strategy(strategy_name)

    def solve(self, orders: pd.DataFrame, static_params: Dict[str, Dict[int, Dict[str, float]]], **kwargs) -> pd.DataFrame:
        """
        调度问题求解

        Args:
            orders: 订单数据
            static_params: 静态参数
            **kwargs: 调度策略的额外参数

        Returns:
            pd.DataFrame: 调度结果
        """

        if self.solve_strategy is None:
            raise ValueError("调度策略未设置")
        if self.solve_strategy.strategy_name not in StrategyFactory.get_strategy_list():
            raise ValueError(f"调度策略 {self.solve_strategy.strategy_name} 不存在")

        print(f"当前使用调度策略: {self.solve_strategy.strategy_name}")


        schedule_result = self.solve_strategy.schedule(orders, static_params, **kwargs)

        print(f"总拖期 (小时): {schedule_result['L_i_hours'].sum():.2f}")
        print(f"总酸轧后库库存时间 (小时): {schedule_result['I_AZ_hours'].sum():.2f}")
        print(f"总连退后库库存时间 (小时): {schedule_result['I_LT_hours'].sum():.2f}")

        return schedule_result

    def plot_schedule_gantt(self, schedule_result: pd.DataFrame):
        """
        生成可视化甘特图
        Args:
            schedule_result: 调度结果
        """
        if schedule_result is None:
            raise ValueError("调度结果为空")
        fig = self.plot_visualization.plot_schedule_gantt(schedule_result)
        fig.show()

    def plot_delivery_performance(self, schedule_result: pd.DataFrame):
        """
        生成可视化交货情况图
        Args:
            schedule_result: 调度结果
        """
        if schedule_result is None:
            raise ValueError("调度结果为空")
        fig = self.plot_visualization.plot_delivery_performance(schedule_result)
        fig.show()

    def plot_inventory_performance(self, schedule_result: pd.DataFrame):
        """
        生成可视化库存情况图
        Args:
            schedule_result: 调度结果
        """
        if schedule_result is None:
            raise ValueError("调度结果为空")
        fig = self.plot_visualization.plot_inventory_performance(schedule_result)
        fig.show()
    def plot_inventory_curve(self, scheduled_df: pd.DataFrame, static_params: dict, title: str = "各工序库存随时间变化图"):
        """
        生成库存曲线
        Args:
            scheduled_df: 调度结果
            static_params: 静态参数
            title: 图表标题
        """
        if scheduled_df is None:
            raise ValueError("调度结果为空")
        fig = self.plot_visualization.plot_inventory_curve(scheduled_df, static_params, title)
        fig.show()

if __name__ == '__main__':
    solver = HeuristicSolver()

    solver.set_solve_strategy('MIN_INV')

    order_data_access = OrderDataAccess()

    orders = order_data_access.get_orders_by_month_and_limit('202411', 30)

    dataframe = order.to_dataframe(orders)

    static_params = order_data_access.get_orders_static_parameter(orders)

    solve_result = solver.solve(order.to_dataframe(orders), static_params)

    solver.plot_inventory_curve(solve_result, static_params)

    solver.plot_schedule_gantt(solve_result)
    # solver.plot_delivery_performance(solve_result)
    # solver.plot_inventory_performance(solve_result)
