from datetime import timedelta

import pandas as pd

from algorithm.order_plan.rule_heuristic_t.strategy.strategy import strategy


class EDD_strategy(strategy):
    """
    Earliest Due Date策略，最早交货期优先生产
    """
    strategy_name = "EDD"
    def sort_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        # 默认排序方法，可在子类中重写
        return orders.sort_values(by='DELIVERY_DATE').reset_index(drop=True)
