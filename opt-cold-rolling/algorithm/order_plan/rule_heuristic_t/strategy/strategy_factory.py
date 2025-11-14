from typing import Dict, Type

from algorithm.order_plan.rule_heuristic_t.strategy.EDD_strategy import EDD_strategy
from algorithm.order_plan.rule_heuristic_t.strategy.FCFS_strategy import FCFS_strategy
from algorithm.order_plan.rule_heuristic_t.strategy.MIN_INV_strategy import MIN_INV_strategy
from algorithm.order_plan.rule_heuristic_t.strategy.IMPROVED_MIN_INV_strategy import IMPROVED_MIN_INV_strategy
from algorithm.order_plan.rule_heuristic_t.strategy.inventory_constrained_strategy import InventoryConstrainedStrategy
from algorithm.order_plan.rule_heuristic_t.strategy.strategy import strategy


class StrategyFactory:
    """
    策略工厂类，用于创建和管理不同的调度策略
    """

    _strategies: Dict[str, Type[strategy]] = {
        'EDD': EDD_strategy,
        'FCFS': FCFS_strategy,
        'MIN_INV': MIN_INV_strategy,
        'IMPROVED_MIN_INV': IMPROVED_MIN_INV_strategy,
        'INVENTORY_CONSTRAINED': InventoryConstrainedStrategy
    }

    @staticmethod
    def get_strategy_list() -> list[str]:
        """
        获取所有可用的策略名称

        Returns:
            list: 可用策略名称列表
        """
        return list(StrategyFactory._strategies.keys())

    @staticmethod
    def create_strategy(strategy_name: str) -> strategy:
        """
        根据名称创建策略实例

        Args:
            strategy_name: 策略名称

        Returns:
            strategy: 策略实例
        """
        if strategy_name not in StrategyFactory._strategies:
            raise ValueError(f"未知的策略: {strategy_name}，当前的可用策略为: {StrategyFactory.get_strategy_list()}")

        strategy_class = StrategyFactory._strategies[strategy_name]
        return strategy_class()