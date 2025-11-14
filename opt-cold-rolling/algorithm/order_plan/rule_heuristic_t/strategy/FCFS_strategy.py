import pandas as pd

from algorithm.order_plan.rule_heuristic_t.strategy.strategy import strategy


class FCFS_strategy(strategy):
    """
    先来先服务 (First Come First Served)，按订单ID排序
    """

    strategy_name = 'FCFS'
    def sort_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        return orders.sort_values(by='id').reset_index(drop=True)