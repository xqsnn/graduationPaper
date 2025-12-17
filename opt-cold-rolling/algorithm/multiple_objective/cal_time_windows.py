import sys
import os
from dataclasses import dataclass
from typing import List, Dict
import heapq

# 获取当前文件所在目录并添加到模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from table.order_new import order_new
import database

# --------- 枚举工序 ---------
class Operation:
    HR = "HR"
    AR = "AR"
    CA = "CA"


# --------- 数据结构 ---------
@dataclass
class Job:
    id: int
    weight: float


# --------- 工艺参数 ---------
operation_sequence = [Operation.HR, Operation.AR, Operation.CA]

transmission_time = {
    Operation.HR: 2.0,
    Operation.AR: 4.0,
    Operation.CA: 1.0,
}

speed = {
    Operation.HR: [30],
    Operation.AR: [14, 16],
    Operation.CA: [8.0, 12.0, 10.0],
}


# --------- 调度仿真 ---------
def simulate(jobs: List[Job]) -> float:
    # 每道工序的机器"可用时间"（小顶堆）
    machine_available: Dict[str, List[float]] = {}

    for op in operation_sequence:
        machine_available[op] = [0.0 for _ in speed[op]]
        heapq.heapify(machine_available[op])

    # 每个 job 当前完成时间
    completion_time = {job.id: 0.0 for job in jobs}

    for op in operation_sequence:
        new_completion = {}

        for job in jobs:
            # 取最早可用的机器
            machine_ready = heapq.heappop(machine_available[op])

            start_time = max(
                completion_time[job.id],
                machine_ready
            )

            # 加工时间（选该机器的速度）
            machine_index = len(speed[op]) - len(machine_available[op]) - 1
            process_speed = speed[op][machine_index]
            process_time = job.weight / process_speed

            finish_time = start_time + process_time

            # 更新机器时间
            heapq.heappush(machine_available[op], finish_time)

            # 加上传输时间
            new_completion[job.id] = finish_time + transmission_time[op]

        completion_time = new_completion

    # 返回最晚完工时间
    return max(completion_time.values())


def cal_time_windows():
    with database.get_db_session() as db:
        records = db.query(order_new).all()

        # 将数据库中的订单转换为调度系统所需的Job对象
        jobs = [Job(id=record.id, weight=float(record.weight)) for record in records]

        total_num = len(jobs)
        total_weight = sum(job.weight for job in jobs)

        if jobs:
            makespan = simulate(jobs)
            print(f"总计订单数：{total_num}，总重量：{total_weight} 吨")
            print(f"预计全部生产完成时间：{makespan:.2f} 小时")
            print(f"约合：{makespan/24:.2f} 天")
        else:
            print("没有找到订单数据")


if __name__ == "__main__":
    cal_time_windows()