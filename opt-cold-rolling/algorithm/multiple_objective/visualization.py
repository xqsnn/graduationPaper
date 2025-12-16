"""
结果可视化模块

包含:
1. Pareto前沿可视化
2. 甘特图(调度方案)
3. 库存曲线图
4. 结果导出
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np
from typing import List
from nsga2_solver import Solution, Material, ScheduleResult, Operation, StaticParameters
import os


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class Visualizer:
    """结果可视化器"""

    def __init__(self, materials: List[Material]):
        """
        初始化可视化器

        Args:
            materials: 材料列表
        """
        self.materials = materials

    def plot_pareto_front(self, pareto_solutions: List[Solution], save_path: str = None):
        """
        绘制Pareto前沿(3D散点图)

        Args:
            pareto_solutions: Pareto最优解集
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 提取目标值
        tardiness = [sol.max_tardiness for sol in pareto_solutions]
        inventory = [sol.avg_inventory for sol in pareto_solutions]
        penalty = [sol.process_instability for sol in pareto_solutions]

        # 绘制散点
        scatter = ax.scatter(tardiness, inventory, penalty,
                            c=tardiness, cmap='viridis',
                            s=100, alpha=0.6, edgecolors='black')

        ax.set_xlabel('总拖期 (天)', fontsize=12, labelpad=10)
        ax.set_ylabel('平均库存 (吨)', fontsize=12, labelpad=10)
        ax.set_zlabel('工艺不稳', fontsize=12, labelpad=10)
        ax.set_title('Pareto前沿 - 三目标优化结果', fontsize=14, pad=20)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('总拖期 (天)', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pareto前沿图已保存至: {save_path}")

        plt.show()

    def plot_pareto_2d(self, pareto_solutions: List[Solution], save_path: str = None):
        """
        绘制2D Pareto前沿(拖期 vs 库存)

        Args:
            pareto_solutions: Pareto最优解集
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        tardiness = [sol.max_tardiness for sol in pareto_solutions]
        inventory = [sol.avg_inventory for sol in pareto_solutions]
        penalty = [sol.process_instability for sol in pareto_solutions]

        # 使用惩罚值作为颜色
        scatter = ax.scatter(tardiness, inventory, c=penalty,
                            cmap='RdYlGn_r', s=100, alpha=0.7,
                            edgecolors='black', linewidths=1.5)

        ax.set_xlabel('总拖期 (天)', fontsize=12)
        ax.set_ylabel('平均库存 (吨)', fontsize=12)
        ax.set_title('Pareto前沿 - 拖期 vs 库存', fontsize=14)
        ax.grid(True, alpha=0.3)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('约束惩罚', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D Pareto前沿图已保存至: {save_path}")

        plt.show()

    def plot_gantt_chart(self, solution: Solution, save_path: str = None):
        """
        绘制甘特图 - 显示调度方案

        Args:
            solution: 调度方案
            save_path: 保存路径
        """
        if not solution.schedule_results:
            print("错误: 解未包含调度结果")
            return

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('生产调度甘特图', fontsize=16, y=0.995)

        # 为不同钢种分配颜色
        categories = list(set(mat.category for mat in self.materials))
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        category_color_map = dict(zip(categories, colors))

        # 使用StaticParameters.start_time作为起始时间
        start_time = StaticParameters.start_time

        # 1. 热轧甘特图
        ax = axes[0]
        ax.set_title('热轧工序 (1台机器)', fontsize=12, pad=10)
        ax.set_xlabel('时间 (真实时间)', fontsize=10)
        ax.set_ylabel('机器', fontsize=10)

        for result in solution.schedule_results:
            mat = self.materials[result.material_id]
            color = category_color_map[mat.category]

            # 将相对时间转换为真实时间
            hr_start_time = start_time + pd.Timedelta(hours=result.hr_start)
            hr_duration = result.hr_end - result.hr_start

            ax.barh(0, hr_duration,
                   left=hr_start_time, height=0.6,
                   color=color, edgecolor='black', linewidth=0.5)

            # 添加材料ID标签
            mid_time = hr_start_time + pd.Timedelta(hours=hr_duration/2)
            ax.text(mid_time, 0, f'{mat.id}',
                   ha='center', va='center', fontsize=8)

        ax.set_yticks([0])
        ax.set_yticklabels(['HR-1'])
        ax.grid(True, axis='x', alpha=0.3)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d\n%H:%M'))

        # 2. 酸轧甘特图
        ax = axes[1]
        ax.set_title('酸轧工序 (2台机器)', fontsize=12, pad=10)
        ax.set_xlabel('时间 (真实时间)', fontsize=10)
        ax.set_ylabel('机器', fontsize=10)

        for result in solution.schedule_results:
            mat = self.materials[result.material_id]
            color = category_color_map[mat.category]
            y_pos = result.ar_machine

            # 将相对时间转换为真实时间
            ar_start_time = start_time + pd.Timedelta(hours=result.ar_start)
            ar_duration = result.ar_end - result.ar_start

            ax.barh(y_pos, ar_duration,
                   left=ar_start_time, height=0.6,
                   color=color, edgecolor='black', linewidth=0.5)

            mid_time = ar_start_time + pd.Timedelta(hours=ar_duration/2)
            ax.text(mid_time, y_pos, f'{mat.id}',
                   ha='center', va='center', fontsize=8)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['AR-1', 'AR-2'])
        ax.grid(True, axis='x', alpha=0.3)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d\n%H:%M'))

        # 3. 连退甘特图
        ax = axes[2]
        ax.set_title('连退工序 (3台机器)', fontsize=12, pad=10)
        ax.set_xlabel('时间 (真实时间)', fontsize=10)
        ax.set_ylabel('机器', fontsize=10)

        for result in solution.schedule_results:
            mat = self.materials[result.material_id]
            color = category_color_map[mat.category]
            y_pos = result.ca_machine

            # 将相对时间转换为真实时间
            ca_start_time = start_time + pd.Timedelta(hours=result.ca_start)
            ca_duration = result.ca_end - result.ca_start

            ax.barh(y_pos, ca_duration,
                   left=ca_start_time, height=0.6,
                   color=color, edgecolor='black', linewidth=0.5)

            mid_time = ca_start_time + pd.Timedelta(hours=ca_duration/2)
            ax.text(mid_time, y_pos, f'{mat.id}',
                   ha='center', va='center', fontsize=8)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['CA-1', 'CA-2', 'CA-3'])
        ax.grid(True, axis='x', alpha=0.3)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d\n%H:%M'))

        # 添加图例
        legend_patches = [mpatches.Patch(color=category_color_map[cat], label=f'钢种 {cat}')
                         for cat in categories]
        fig.legend(handles=legend_patches, loc='upper right',
                  bbox_to_anchor=(0.98, 0.98), fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"甘特图已保存至: {save_path}")

        plt.show()

    def plot_inventory_curve(self, solution: Solution, save_path: str = None):
        """
        绘制库存曲线

        Args:
            solution: 调度方案
            save_path: 保存路径
        """
        if not solution.schedule_results:
            print("错误: 解未包含调度结果")
            return

        # 使用StaticParameters.start_time作为起始时间
        start_time = StaticParameters.start_time

        # 构建时间事件列表
        events_hr = []  # 热轧后库存
        events_ar = []  # 酸轧后库存

        for result in solution.schedule_results:
            # 热轧后库存变化
            events_hr.append((start_time + pd.Timedelta(hours=result.hr_end), 1))  # 进入库存
            events_hr.append((start_time + pd.Timedelta(hours=result.ar_start), -1))  # 离开库存

            # 酸轧后库存变化
            events_ar.append((start_time + pd.Timedelta(hours=result.ar_end), 1))
            events_ar.append((start_time + pd.Timedelta(hours=result.ca_start), -1))

        # 排序事件
        events_hr.sort()
        events_ar.sort()

        # 计算库存曲线
        def compute_inventory_curve(events):
            times = [start_time]
            inventory = [0]
            current_inv = 0

            for time, delta in events:
                times.append(time)
                inventory.append(current_inv)
                current_inv += delta
                times.append(time)
                inventory.append(current_inv)

            return times, inventory

        times_hr, inventory_hr = compute_inventory_curve(events_hr)
        times_ar, inventory_ar = compute_inventory_curve(events_ar)

        # 绘图
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('库存变化曲线', fontsize=14)

        # 热轧后库存
        ax = axes[0]
        ax.plot(times_hr, inventory_hr, linewidth=2, color='steelblue')
        ax.fill_between(times_hr, inventory_hr, alpha=0.3, color='steelblue')
        ax.set_ylabel('库存数量 (件)', fontsize=11)
        ax.set_title('热轧后仓库库存', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d\n%H:%M'))

        # 酸轧后库存
        ax = axes[1]
        ax.plot(times_ar, inventory_ar, linewidth=2, color='coral')
        ax.fill_between(times_ar, inventory_ar, alpha=0.3, color='coral')
        ax.set_xlabel('时间 (真实时间)', fontsize=11)
        ax.set_ylabel('库存数量 (件)', fontsize=11)
        ax.set_title('酸轧后仓库库存', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d\n%H:%M'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"库存曲线图已保存至: {save_path}")

        plt.show()

    def export_results_to_excel(self, pareto_solutions: List[Solution],
                                output_path: str = "results.xlsx"):
        """
        导出结果到Excel文件

        Args:
            pareto_solutions: Pareto最优解集
            output_path: 输出文件路径
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. Pareto解摘要
            summary_data = []
            for i, sol in enumerate(pareto_solutions):
                summary_data.append({
                    '方案编号': i + 1,
                    '总拖期(天)': round(sol.max_tardiness, 2),
                    '平均库存(吨)': round(sol.avg_inventory, 1),
                    '工艺不稳': round(sol.process_instability, 2),
                    'Pareto等级': sol.rank,
                    '拥挤度': round(sol.crowding_distance, 4)
                })

            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Pareto解摘要', index=False)

            # 2. 详细调度结果(选择最优解)
            best_sol = min(pareto_solutions, key=lambda x: x.max_tardiness)

            if best_sol.schedule_results:
                schedule_data = []
                for result in best_sol.schedule_results:
                    mat = self.materials[result.material_id]
                    schedule_data.append({
                        '材料ID': mat.id,
                        '订单号': mat.order_no,
                        '钢种': mat.category,
                        '宽度': mat.width,
                        '厚度': mat.thickness,
                        '重量': mat.weight,
                        '交货日期': mat.delivery_date.strftime('%Y-%m-%d'),
                        '热轧开始': round(result.hr_start, 2),
                        '热轧结束': round(result.hr_end, 2),
                        '酸轧机器': f'AR-{result.ar_machine + 1}',
                        '酸轧开始': round(result.ar_start, 2),
                        '酸轧结束': round(result.ar_end, 2),
                        '连退机器': f'CA-{result.ca_machine + 1}',
                        '连退开始': round(result.ca_start, 2),
                        '连退结束': round(result.ca_end, 2),
                        '完工时间(天)': round(result.ca_end / 24, 2)
                    })

                df_schedule = pd.DataFrame(schedule_data)
                df_schedule.to_excel(writer, sheet_name='最优方案详细调度', index=False)

            # 3. 机器负载统计
            load_data = []
            for i, sol in enumerate(pareto_solutions):
                ar_loads = [len(seq) for seq in sol.ar_sequences]
                ca_loads = [len(seq) for seq in sol.ca_sequences]

                load_data.append({
                    '方案编号': i + 1,
                    'AR-1负载': ar_loads[0],
                    'AR-2负载': ar_loads[1],
                    'CA-1负载': ca_loads[0],
                    'CA-2负载': ca_loads[1],
                    'CA-3负载': ca_loads[2]
                })

            df_load = pd.DataFrame(load_data)
            df_load.to_excel(writer, sheet_name='机器负载统计', index=False)

        print(f"\n结果已导出至Excel文件: {output_path}")


def plot_convergence_curve(history: List[float], save_path: str = None):
    """
    绘制收敛曲线

    Args:
        history: 每代的最优目标值历史
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(history) + 1), history,
           linewidth=2, marker='o', markersize=4,
           color='steelblue', label='最优解')

    ax.set_xlabel('迭代代数', fontsize=12)
    ax.set_ylabel('最优总拖期 (天)', fontsize=12)
    ax.set_title('算法收敛曲线', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"收敛曲线图已保存至: {save_path}")

    plt.show()


def analyze_solution_quality(pareto_solutions: List[Solution], materials: List[Material]):
    """
    分析解的质量

    Args:
        pareto_solutions: Pareto最优解集
        materials: 材料列表
    """
    print("\n" + "=" * 80)
    print("解的质量分析")
    print("=" * 80)

    # 统计信息
    tardiness_values = [sol.max_tardiness for sol in pareto_solutions]
    inventory_values = [sol.avg_inventory for sol in pareto_solutions]
    penalty_values = [sol.process_instability for sol in pareto_solutions]

    print(f"\nPareto最优解数量: {len(pareto_solutions)}")
    print("\n目标函数统计:")
    print("-" * 80)
    print(f"总拖期 (天):")
    print(f"  最小值: {min(tardiness_values):.2f}")
    print(f"  最大值: {max(tardiness_values):.2f}")
    print(f"  平均值: {np.mean(tardiness_values):.2f}")
    print(f"  标准差: {np.std(tardiness_values):.2f}")

    print(f"\n平均库存 (吨):")
    print(f"  最小值: {min(inventory_values):.1f}")
    print(f"  最大值: {max(inventory_values):.1f}")
    print(f"  平均值: {np.mean(inventory_values):.1f}")

    print(f"\n约束惩罚:")
    print(f"  最小值: {min(penalty_values):.2f}")
    print(f"  最大值: {max(penalty_values):.2f}")
    print(f"  平均值: {np.mean(penalty_values):.2f}")

    # 找到极端解
    best_tardiness = min(pareto_solutions, key=lambda x: x.max_tardiness)
    best_inventory = min(pareto_solutions, key=lambda x: x.avg_inventory)
    best_penalty = min(pareto_solutions, key=lambda x: x.process_instability)

    print("\n极端Pareto解:")
    print("-" * 80)
    print(f"最优拖期解: 拖期={best_tardiness.max_tardiness:.2f}天, "
          f"库存={best_tardiness.avg_inventory:.1f}, "
          f"工艺不稳={best_tardiness.process_instability:.2f}")

    print(f"最优库存解: 拖期={best_inventory.max_tardiness:.2f}天, "
          f"库存={best_inventory.avg_inventory:.1f}, "
          f"工艺不稳={best_inventory.process_instability:.2f}")

    print(f"最优约束解: 拖期={best_penalty.max_tardiness:.2f}天, "
          f"库存={best_penalty.avg_inventory:.1f}, "
          f"工艺不稳={best_penalty.process_instability:.2f}")

    # 分析拖期情况
    if best_tardiness.schedule_results:
        tardy_materials = 0
        for result in best_tardiness.schedule_results:
            mat = materials[result.material_id]
            completion_days = result.ca_end / 24
            delivery_days = (mat.delivery_date - pd.Timestamp.now()).days
            if completion_days > delivery_days:
                tardy_materials += 1

        print(f"\n拖期材料数量: {tardy_materials}/{len(materials)} "
              f"({tardy_materials/len(materials)*100:.1f}%)")

    print("=" * 80)


if __name__ == "__main__":
    # 示例: 如何使用可视化模块
    print("可视化模块 - 请在主程序中调用相关函数")
