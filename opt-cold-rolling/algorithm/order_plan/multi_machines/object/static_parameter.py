from dataclasses import dataclass

from algorithm.order_plan.multi_machines.object.Operation import Operation


@dataclass
class StaticParameters:
    """
    静态参数类
    """
    # 产线顺序：HR -> AZ -> CA
    operation_sequence = [Operation.HR, Operation.AZ, Operation.CA]

    # 产线之间的运输时间，小时
    transmission_time = {
        Operation.HR: 2.0,
        Operation.AZ: 4.0,
        Operation.CA: 1.0 # 最后一站出库
    }

    # 产线不同机组的处理速度，吨/小时
    speed = {
        Operation.HR: [30],
        Operation.AZ: [14, 16],
        Operation.CA: [8.0, 12.0, 10.0]
    }

    # 库存限制，单位吨 [最小值, 最大值]
    stock_limit = {
        Operation.HR: [0.0, 100.0],
        Operation.AZ: [0.0, 100.0],
        Operation.CA: [0.0, 100.0]
    }

    # 其他配置
    TIME_UNIT_HOURS = 1 # 模拟时间步长，1小时
    MAX_SIMULATION_HOURS = 24 * 30 # 最长模拟时间，例如30天
