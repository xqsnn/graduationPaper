from algorithm.order_plan.multi_machines.object.static_parameter import StaticParameters
from algorithm.order_plan.multi_machines.data_process import DataProcess
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from deap import base, creator, tools, algorithms
from algorithm.order_plan.multi_machines.object.order_multi_feature import OrderMultiFeature

data_process = DataProcess()
orders_obj_list = data_process.get_all_orders() # 获取订单对象列表
static_params = StaticParameters()

# ------------------------------------------------------------------------------------------------------
# 成本计算函数
# ------------------------------------------------------------------------------------------------------

def calculate_adjacency_cost(order1: OrderMultiFeature, order2: OrderMultiFeature) -> float:
    """
    计算两个相邻生产的订单之间的成本。
    主要考虑宽度递减、厚度集中原则，以及宽度反跳的惩罚。
    成本值越大表示越不优，越小表示越优。
    """
    cost = 0.0

    # 1. 宽度递减原则
    width_diff = order1.order_width - order2.order_width
    if width_diff < 0:
        # 宽度增加 (反跳)
        width_jump_ratio = abs(width_diff) / max(order1.order_width, order2.order_width, 1)  # 防止除零
        if width_jump_ratio < 0.1:  # 小于10%的反跳
            cost += 30.0
        elif width_jump_ratio < 0.3:  # 小于30%的反跳
            cost += 80.0
        else:  # 大于30%的反跳，相当于重开一个批次，成本更高
            cost += 200.0
    else:
        # 宽度递减或保持，鼓励
        cost -= 15.0  # 增加奖励幅度

    # 2. 厚度集中原则
    thick_diff = abs(order1.order_thick - order2.order_thick)
    if thick_diff > 5.0:  # 厚度差异较大，惩罚
        cost += thick_diff * 3.0
    else:  # 厚度差异小，鼓励
        cost -= 8.0  # 增加奖励

    # 3. 宽度变化的平滑度
    # 根据订单宽度进行加权，大宽订单间的跳变影响更大
    width_weight = (order1.order_width + order2.order_width) / 2.0
    cost += abs(width_diff) * 0.05 * width_weight

    # 4. 如果是相同订单号，则成本为0 (尽管在实际生产中可能不会相邻)
    if order1.order_no == order2.order_no:
        return 0.0

    return max(0.0, cost)  # 确保成本非负

# 预计算所有订单对之间的成本，以便快速查询
order_adjacency_costs = {}
for i, order1 in enumerate(orders_obj_list):
    for j, order2 in enumerate(orders_obj_list):
        if i != j:
            order_adjacency_costs[(order1.order_no, order2.order_no)] = calculate_adjacency_cost(order1, order2)
        else:
            order_adjacency_costs[(order1.order_no, order2.order_no)] = 0.0 # 同一个订单相邻生产，成本为0

# ------------------------------------------------------------------------------------------------------
# 智能算法求解 (NSGA-II)
# ------------------------------------------------------------------------------------------------------

# DEAP 库的设置
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0)) # 最小化拖期和最小化成本
creator.create("Individual", list, fitness=creator.FitnessMulti)

class Scheduler:
    def __init__(self, orders: list[OrderMultiFeature], static_params: StaticParameters):
        if not orders:
            raise ValueError("订单列表不能为空")
        if not static_params:
            raise ValueError("静态参数不能为空")
            
        self.orders = orders
        self.static_params = static_params
        self.num_orders = len(orders)
        self.operations = static_params.operation_sequence
        self.num_operations = len(self.operations)

        # 订单到索引的映射，方便查找
        self.order_to_idx = {order.order_no: i for i, order in enumerate(orders)}
        self.idx_to_order = {i: order for i, order in enumerate(orders)}

        # 各产线各机组的处理速度
        self.machine_speeds = static_params.speed
        # 产线间传输时间
        self.transfer_times = static_params.transmission_time
        # 库存限制
        self.stock_limits = static_params.stock_limit

        # 模拟时间步长
        self.time_unit_hours = static_params.TIME_UNIT_HOURS
        self.max_simulation_hours = static_params.MAX_SIMULATION_HOURS

    def evaluate(self, individual):
        """
        评估函数，计算拖期总和和生产成本总和。
        individual 代表一个调度方案，通常是订单在各产线上的生产顺序和机组分配。
        这里我们简化individual为订单的全局生产顺序列表，机组分配在模拟时随机或按规则选择。
        更复杂的individual可以编码每个订单在每个工序的机组选择。
        为了简化，我们先假设：
        Individual: 一个表示订单生产顺序的列表 (permutation)。
                    例如: [order_idx_3, order_idx_1, order_idx_2, ...]
                    机组的选择会在模拟过程中基于可用性进行。
        """

        # 解码 individual (订单生产顺序)
        scheduled_order_nos = [self.idx_to_order[i].order_no for i in individual]

        # 存储调度结果
        schedule_results = {} # {order_no: {op_enum: {'start_time': dt, 'end_time': dt, 'machine_idx': int}}}

        # 存储每个机组的可用时间
        machine_availability = {op: [datetime.min for _ in range(len(self.machine_speeds[op]))]
                                for op in self.operations}

        # 存储每个产线的库存，初始化为最小值 (假设初始库存为0，模型中C_0j = C_j^initial)
        # 实际模拟中，可以考虑一个初始库存值
        current_stocks = {op: self.stock_limits[op][0] for op in self.operations if op != self.operations[-1]} # 最后一站没有库存概念

        # 存储库存历史
        stock_history = {op: [(datetime.min, self.stock_limits[op][0])] for op in self.operations if op != self.operations[-1]}

        current_time = datetime.min # 模拟开始时间，可以设置为当前时间

        # 假设所有订单都在0时刻准备就绪
        order_ready_times = {order.order_no: datetime.min for order in self.orders}

        # 约束违反惩罚
        constraint_violation_penalty = 0.0

        # 遍历每个订单按顺序调度
        for order_no in scheduled_order_nos:
            order = self.orders[self.order_to_idx[order_no]]

            # 调度该订单在所有工序上
            for op_idx, operation in enumerate(self.operations):

                # 计算订单在该工序上的加工时间 (需要选择机组)
                # P_i,j = w_i / r_j,k
                # 这里我们选择最空闲的机组

                # 获取该工序所有机组的可用时间
                available_machines = machine_availability[operation]

                # 找到最早可用的机组及其索引
                earliest_available_time = datetime.max
                best_machine_idx = -1
                for k, avail_time in enumerate(available_machines):
                    if avail_time < earliest_available_time:
                        earliest_available_time = avail_time
                        best_machine_idx = k

                # 订单在该工序上的最早开始时间
                start_time_candidates = [order_ready_times[order_no], earliest_available_time]

                # 如果不是第一个工序，需要考虑前一道工序的完成时间和传输时间
                if op_idx > 0:
                    prev_op = self.operations[op_idx - 1]
                    if order_no in schedule_results and prev_op in schedule_results[order_no]:
                        prev_op_end_time = schedule_results[order_no][prev_op]['end_time']
                        transfer_time = self.transfer_times[prev_op] # 上一工序到当前工序的传输时间
                        start_time_candidates.append(prev_op_end_time + timedelta(hours=transfer_time))
                    else:
                        # 如果前一工序未完成，说明调度出错，给予极大惩罚
                        return float('inf'), float('inf')

                actual_start_time = max(start_time_candidates)

                # 计算加工时间
                processing_rate = self.machine_speeds[operation][best_machine_idx]
                if processing_rate <= 0:
                    # 处理速度为0或负数，给予惩罚
                    constraint_violation_penalty += 1000.0
                    continue
                    
                processing_time_hours = order.order_wt / processing_rate

                actual_end_time = actual_start_time + timedelta(hours=processing_time_hours)

                # 检查是否超出时间限制
                if (actual_end_time - datetime.min).total_seconds() / 3600 > self.max_simulation_hours:
                    constraint_violation_penalty += 1000.0

                # 更新机组可用时间
                machine_availability[operation][best_machine_idx] = actual_end_time

                # 记录调度结果
                if order_no not in schedule_results:
                    schedule_results[order_no] = {}
                schedule_results[order_no][operation] = {
                    'start_time': actual_start_time,
                    'end_time': actual_end_time,
                    'machine_idx': best_machine_idx
                }

                # 更新订单在该工序结束后可以进入下一工序的时间
                order_ready_times[order_no] = actual_end_time

        # ----------------------------------------------------------------------------------
        # 1. 计算总拖期 (L_i)
        total_delay = 0.0
        order_delays = {}
        for order in self.orders:
            # 最后一个工序的完成时间
            final_op = self.operations[-1]
            if order.order_no in schedule_results and final_op in schedule_results[order.order_no]:
                completion_time = schedule_results[order.order_no][final_op]['end_time']
                # 确保order.delivery_date是datetime对象
                if not isinstance(order.delivery_date, datetime):
                    # 如果不是datetime对象，则尝试转换
                    raise TypeError(f"order.delivery_date must be datetime object, got {type(order.delivery_date)}")
                delay = (completion_time - order.delivery_date).total_seconds() / 3600.0 # 小时
                order_delays[order.order_no] = delay
                total_delay += max(0, delay)
            else:
                # 未能完成调度的订单，给予极大惩罚
                total_delay += self.max_simulation_hours * 2 # 确保它不被选中

        # ----------------------------------------------------------------------------------
        # 2. 计算总生产成本 (y_i,i',j,k * cost_i,i')
        total_production_cost = 0.0
        # 遍历所有工序，检查相邻生产的订单
        for operation in self.operations:
            # 获取在该工序上所有机组的生产顺序
            machine_sequences = {k: [] for k in range(len(self.machine_speeds[operation]))}

            # 收集每个机组的订单及其开始/结束时间
            for order_no, op_schedule in schedule_results.items():
                if operation in op_schedule:
                    machine_idx = op_schedule[operation]['machine_idx']
                    machine_sequences[machine_idx].append({
                        'order_no': order_no,
                        'start_time': op_schedule[operation]['start_time'],
                        'end_time': op_schedule[operation]['end_time']
                    })

            # 对每个机组的生产序列按开始时间排序
            for machine_idx in machine_sequences:
                machine_sequences[machine_idx].sort(key=lambda x: x['start_time'])

                # 计算相邻订单的成本
                for i in range(len(machine_sequences[machine_idx]) - 1):
                    order1_no = machine_sequences[machine_idx][i]['order_no']
                    order2_no = machine_sequences[machine_idx][i+1]['order_no']

                    order1 = self.orders[self.order_to_idx[order1_no]]
                    order2 = self.orders[self.order_to_idx[order2_no]]

                    # 只有当两个订单在同一个机组上连续生产时才计算相邻成本
                    # 这里简化为只要是生产序列上相邻就计算
                    total_production_cost += order_adjacency_costs[(order1_no, order2_no)]

        # ----------------------------------------------------------------------------------
        # 3. 库存约束检查
        # 改进库存变化的模拟逻辑
        # 找到最早的开始时间和最晚的结束时间作为模拟的起止点
        all_start_times = []
        all_end_times = []
        for order_no, op_schedules in schedule_results.items():
            for op, times in op_schedules.items():
                all_start_times.append(times['start_time'])
                all_end_times.append(times['end_time'])

        if not all_start_times or not all_end_times:
            # 没有成功调度任何订单
            return float('inf'), float('inf')

        simulation_start_time = min(all_start_times)
        simulation_end_time = max(all_end_times) + timedelta(hours=24) # 额外24小时用于观察库存平稳

        # 初始化每个工序的库存 (假设初始库存为最小值)
        current_stocks = {op: self.stock_limits[op][0] for op in self.operations if op != self.operations[-1]}
        stock_history_detailed = {op: [(simulation_start_time, current_stocks[op])
                                  for op in self.operations if op != self.operations[-1]]}

        # 定义事件点：订单完成当前工序(产生库存)、订单开始下一工序(消耗库存)
        events = []
        for order_no, op_schedules in schedule_results.items():
            order_obj = self.orders[self.order_to_idx[order_no]]
            for op_idx, op in enumerate(self.operations):
                if op in op_schedules:
                    # 订单完成当前工序时，为该工序库存增加产出
                    if op != self.operations[-1]: # 不是最后一道工序时，会产生库存
                        end = op_schedules[op]['end_time']
                        events.append({
                            'time': end, 
                            'type': 'produce', 
                            'operation': op, 
                            'order': order_obj, 
                            'amount': order_obj.order_wt
                        })

                    # 订单开始下一道工序时，消耗库存
                    if op_idx < len(self.operations) - 1:  # 不是最后一道工序
                        next_op = self.operations[op_idx + 1]
                        next_op_start = schedule_results[order_no][next_op]['start_time']
                        events.append({
                            'time': next_op_start, 
                            'type': 'consume', 
                            'operation': op,  # 消耗当前工序的库存到下一道工序
                            'order': order_obj, 
                            'amount': order_obj.order_wt
                        })

        # 将事件按时间排序
        events.sort(key=lambda x: x['time'])

        # 模拟库存变化
        stock_penalty = 0.0
        time_ptr = simulation_start_time

        for event in events:
            # 在事件发生前，更新当前时间段内的库存历史
            for op in current_stocks:
                if time_ptr < event['time']: # 避免重复添加相同时间点的记录
                    stock_history_detailed[op].append((event['time'], current_stocks[op]))

            # 执行事件
            op_event = event['operation']
            amount = event['amount']

            if event['type'] == 'produce':
                # 当订单完成一道工序时，为对应的库存增加产出
                if op_event in current_stocks:
                    current_stocks[op_event] += amount
            elif event['type'] == 'consume':
                # 当订单进入下一道工序时，消耗前一道工序的库存
                if op_event in current_stocks:
                    current_stocks[op_event] -= amount
                    # 检查是否库存不足
                    if current_stocks[op_event] < 0:
                        # 库存不足，给予惩罚
                        stock_penalty += abs(current_stocks[op_event]) * 50.0  # 库存不足的惩罚

            # 检查库存限制 - 对所有工序进行检查
            for op in current_stocks:
                min_stock, max_stock = self.stock_limits[op]
                if not (min_stock <= current_stocks[op] <= max_stock):
                    # 库存超出限制，给予惩罚
                    excess = max(abs(current_stocks[op] - min_stock), abs(current_stocks[op] - max_stock))
                    stock_penalty += excess * 10.0  # 按超出量给予惩罚

            time_ptr = event['time']

        # 最后的库存快照
        for op in current_stocks:
             stock_history_detailed[op].append((simulation_end_time, current_stocks[op]))

        # 总库存惩罚加到总成本中
        total_production_cost += stock_penalty + constraint_violation_penalty

        return total_delay, total_production_cost, schedule_results, order_delays, stock_history_detailed

    def init_individual(self):
        """
        初始化一个个体，即订单的随机生产顺序。
        """
        individual = list(range(self.num_orders))
        random.shuffle(individual)
        return creator.Individual(individual)

    def solve(self, population_size=100, generations=50, cxpb=0.9, mutpb=0.1):
        """
        使用NSGA-II算法求解多目标优化问题。
        """
        try:
            # 参数验证
            if population_size <= 0:
                raise ValueError("种群大小必须大于0")
            if generations <= 0:
                raise ValueError("迭代次数必须大于0")
            if not (0 <= cxpb <= 1):
                raise ValueError("交叉概率必须在0到1之间")
            if not (0 <= mutpb <= 1):
                raise ValueError("变异概率必须在0到1之间")

            toolbox = base.Toolbox()
            toolbox.register("individual", self.init_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", self.evaluate_wrapper) # 包裹评估函数以只返回目标值
            toolbox.register("mate", tools.cxOrdered) # 顺序交叉
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05) # 随机打乱一部分索引
            toolbox.register("select", tools.selNSGA2)

            pop = toolbox.population(n=population_size)

            # 第一次评估
            try:
                fits = toolbox.map(toolbox.evaluate, pop)
                for ind, fit in zip(pop, fits):
                    ind.fitness.values = fit
            except Exception as e:
                print(f"初始评估出错: {e}")
                raise

            # 迭代优化
            for gen in range(generations):
                print(f"Generation {gen}/{generations}")
                try:
                    # 选择
                    offspring = toolbox.select(pop, len(pop))
                    offspring = list(map(toolbox.clone, offspring))

                    # 交叉
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < cxpb:
                            toolbox.mate(child1, child2)
                            del child1.fitness.values
                            del child2.fitness.values

                    # 变异
                    for mutant in offspring:
                        if random.random() < mutpb:
                            toolbox.mutate(mutant)
                            del mutant.fitness.values

                    # 评估新个体
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    if invalid_ind:  # 只有当有无效个体时才评估
                        fits = toolbox.map(toolbox.evaluate, invalid_ind)
                        for ind, fit in zip(invalid_ind, fits):
                            ind.fitness.values = fit

                    # 合并父代和子代，选择新的种群
                    pop = toolbox.select(pop + offspring, population_size)
                except Exception as e:
                    print(f"在第 {gen} 代发生错误: {e}")
                    # 在发生错误时继续执行，而不是终止
                    continue

            # 返回最优解 (帕累托前沿)
            # 找出帕累托前沿的个体，并返回其详细调度结果
            final_front = tools.selNSGA2(pop, len(pop))

            if not final_front:
                print("未找到有效的帕累托前沿解")
                return None, {}, {}, {}, (float('inf'), float('inf'))

            # 找到最佳个体 (例如，拖期和成本都最小的)
            # 通常NSGA-II会返回一个帕累托前沿，你需要从中选择一个你认为"最好"的解
            # 这里为了演示，我们选择拖期最小的那个解中的一个 (如果多个，随便取一个)
            try:
                best_individual = min(final_front, key=lambda ind: ind.fitness.values[0] if hasattr(ind.fitness, 'values') and ind.fitness.values[0] is not None else float('inf'))
            except (IndexError, TypeError):
                print("无法确定最优个体，使用第一个个体")
                best_individual = final_front[0]

            # 再次评估最佳个体，获取详细结果
            total_delay, total_production_cost, schedule_results, order_delays, stock_history_detailed = self.evaluate(best_individual)

            return best_individual, schedule_results, order_delays, stock_history_detailed, (total_delay, total_production_cost)
        except Exception as e:
            print(f"求解过程发生错误: {e}")
            # 返回一个默认结果而不是崩溃
            return None, {}, {}, {}, (float('inf'), float('inf'))

    def evaluate_wrapper(self, individual):
        """
        评估函数的包装器，只返回目标值，供DEAP使用。
        """
        try:
            total_delay, total_production_cost, _, _, _ = self.evaluate(individual)
            # 确保返回的值是有效的数字
            if not isinstance(total_delay, (int, float)) or not isinstance(total_production_cost, (int, float)):
                return float('inf'), float('inf')
            return total_delay, total_production_cost
        except Exception as e:
            print(f"评估函数包装器发生错误: {e}")
            # 发生错误时返回极大值
            return float('inf'), float('inf')


# ------------------------------------------------------------------------------------------------------
# 可视化函数
# ------------------------------------------------------------------------------------------------------

def plot_gantt_chart(schedule_results, orders_obj_list, static_params):
    """
    绘制甘特图
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    operations = static_params.operation_sequence
    num_operations = len(operations)
    orders_map = {order.order_no: order for order in orders_obj_list}

    # 为每个订单和工序分配一个颜色
    colors = plt.cm.get_cmap("tab20", len(schedule_results))
    order_color_map = {order_no: colors(i) for i, order_no in enumerate(schedule_results.keys())}

    # 绘制任务条
    y_labels = []
    y_pos = 0

    # 将所有调度事件扁平化，按开始时间排序
    all_tasks = []
    for order_no, op_schedules in schedule_results.items():
        for op, details in op_schedules.items():
            all_tasks.append({
                'order_no': order_no,
                'op': op,
                'start': details['start_time'],
                'end': details['end_time'],
                'machine_idx': details['machine_idx']
            })

    # 按照工序和机组进行分组排序，以便更好地展示
    task_lanes = {} # { (op, machine_idx): [] }
    for task in all_tasks:
        key = (task['op'], task['machine_idx'])
        if key not in task_lanes:
            task_lanes[key] = []
        task_lanes[key].append(task)

    # 按照工序顺序和机组索引来绘制
    sorted_lane_keys = sorted(task_lanes.keys(), key=lambda x: (static_params.operation_sequence.index(x[0]), x[1]))

    for key in sorted_lane_keys:
        op, machine_idx = key
        tasks_in_lane = sorted(task_lanes[key], key=lambda x: x['start'])

        for task in tasks_in_lane:
            start_dt = task['start']
            end_dt = task['end']
            duration = (end_dt - start_dt).total_seconds() / 3600.0 # 小时

            ax.barh(y_pos, duration, left=start_dt, height=0.6,
                    color=order_color_map[task['order_no']], edgecolor='black', alpha=0.8)

            # 添加文本标签
            text_x = start_dt + timedelta(hours=duration / 2)
            ax.text(text_x, y_pos, f"{task['order_no']}", va='center', ha='center', color='white', fontsize=8)

        y_labels.append(f"{op.value} (Machine {machine_idx + 1})")
        y_pos += 1

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("时间")
    ax.set_ylabel("工序与机组")
    ax.set_title("合同生产甘特图")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 格式化x轴为日期时间
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate() # 自动调整日期标签

    # 添加图例
    handles = []
    labels = []
    for order_no, color in order_color_map.items():
        handles.append(plt.Rectangle((0,0),1,1, color=color))
        labels.append(order_no)
    ax.legend(handles, labels, title="订单", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.show()

def plot_stock_change(stock_history_detailed, static_params):
    """
    绘制库存变化图
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    for op, history in stock_history_detailed.items():
        times = [entry[0] for entry in history]
        stocks = [entry[1] for entry in history]
        ax.step(times, stocks, where='post', label=f'库存: {op.value}') # 'post' 表示阶梯图

        # 绘制库存限制
        min_stock, max_stock = static_params.stock_limits[op]
        ax.axhline(y=min_stock, color='gray', linestyle='--', alpha=0.6, label=f'{op.value} 最低库存')
        ax.axhline(y=max_stock, color='red', linestyle='--', alpha=0.6, label=f'{op.value} 最高库存')

    ax.set_xlabel("时间")
    ax.set_ylabel("库存量 (吨)")
    ax.set_title("库存变化图")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()

    plt.tight_layout()

    plt.show()


def plot_delay_change(order_delays):
    """
    绘制拖期变化图 (有正有负)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    order_nos = list(order_delays.keys())
    delays = list(order_delays.values())

    # 将正拖期和负拖期分开绘制，方便区分
    positive_delays = [d if d > 0 else 0 for d in delays]
    negative_delays = [d if d < 0 else 0 for d in delays]

    # 条形图
    ax.bar(order_nos, positive_delays, color='red', label='拖期 (小时)', alpha=0.7)
    ax.bar(order_nos, negative_delays, color='green', label='提前 (小时)', alpha=0.7)

    ax.set_xlabel("订单号")
    ax.set_ylabel("时间 (小时)")
    ax.set_title("订单拖期/提前变化图")
    ax.axhline(0, color='black', linewidth=0.8) # 零线
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    plt.tight_layout()

    plt.show()