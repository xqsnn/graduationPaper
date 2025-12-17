"""
结果可视化模块

包含:
1. Pareto前沿可视化
2. 甘特图(调度方案)
3. 库存曲线图
4. 结果导出
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List
from nsga2_solver import Solution, Material, ScheduleResult, Operation, StaticParameters
import os


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
        # 提取目标值
        tardiness = [sol.max_tardiness for sol in pareto_solutions]
        inventory = [sol.avg_inventory for sol in pareto_solutions]
        penalty = [sol.process_instability for sol in pareto_solutions]

        # 创建3D散点图
        fig = go.Figure(data=[go.Scatter3d(
            x=tardiness,
            y=inventory,
            z=penalty,
            mode='markers',
            marker=dict(
                size=6,
                color=tardiness,  # 根据总拖期设置颜色
                colorscale='Viridis',
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            text=[f'拖期: {t:.2f}<br>库存: {i:.2f}<br>工艺不稳: {p:.2f}'
                  for t, i, p in zip(tardiness, inventory, penalty)],
            hovertemplate='<b>%{text}</b><extra></extra>'
        )])

        fig.update_layout(
            title='Pareto前沿 - 三目标优化结果',
            scene=dict(
                xaxis_title='总拖期 (天)',
                yaxis_title='平均库存 (吨)',
                zaxis_title='工艺不稳'
            ),
            width=800,
            height=600
        )

        if save_path:
            fig.write_image(save_path)
            print(f"Pareto前沿图已保存至: {save_path}")

        fig.show()

    def plot_pareto_2d(self, pareto_solutions: List[Solution], save_path: str = None):
        """
        绘制2D Pareto前沿(拖期 vs 库存)

        Args:
            pareto_solutions: Pareto最优解集
            save_path: 保存路径
        """
        tardiness = [sol.max_tardiness for sol in pareto_solutions]
        inventory = [sol.avg_inventory for sol in pareto_solutions]
        penalty = [sol.process_instability for sol in pareto_solutions]

        # 创建散点图
        fig = go.Figure(data=go.Scatter(
            x=tardiness,
            y=inventory,
            mode='markers',
            marker=dict(
                size=12,
                color=penalty,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="约束惩罚"),
                opacity=0.7,
                line=dict(width=1.5, color='black')
            ),
            text=[f'拖期: {t:.2f}<br>库存: {i:.2f}<br>工艺不稳: {p:.2f}'
                  for t, i, p in zip(tardiness, inventory, penalty)],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

        fig.update_layout(
            title='Pareto前沿 - 拖期 vs 库存',
            xaxis_title='总拖期 (天)',
            yaxis_title='平均库存 (吨)',
            width=800,
            height=600
        )

        # 添加网格
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        if save_path:
            fig.write_image(save_path)
            print(f"2D Pareto前沿图已保存至: {save_path}")

        fig.show()

    def plot_gantt_chart(self, solution: Solution, save_path: str = None):
        """
        绘制甘特图 - 显示调度方案（所有工序在一个图中）

        Args:
            solution: 调度方案
            save_path: 保存路径
        """
        if not solution.schedule_results:
            print("错误: 解未包含调度结果")
            return

        # 为不同钢种分配颜色
        categories = list(set(mat.category for mat in self.materials))
        # 生成颜色映射（使用plotly的颜色序列）
        colors = px.colors.qualitative.Set3[:len(categories)]
        category_color_map = dict(zip(categories, colors))

        # 使用StaticParameters.start_time作为起始时间
        start_time = StaticParameters.start_time

        # 定义每台机器的y轴位置
        machine_positions = {
            'HR-1': 0,
            'AR-1': 1,
            'AR-2': 2,
            'CA-1': 3,
            'CA-2': 4,
            'CA-3': 5
        }

        # 准备甘特图数据
        df_data = []
        for result in solution.schedule_results:
            mat = self.materials[result.material_id]

            # 热轧工序
            hr_start_time = start_time + pd.Timedelta(hours=result.hr_start)
            hr_end_time = start_time + pd.Timedelta(hours=result.hr_end)
            df_data.append({
                'Task': f'HR-1',
                'Start': hr_start_time,
                'Finish': hr_end_time,
                'Material': f"order_{mat.id}",
                'Category': mat.category,
                'Operation': '热轧'
            })

            # 酸轧工序
            ar_machine_key = f'AR-{result.ar_machine + 1}'
            if ar_machine_key not in machine_positions:
                print(f"警告: 无效的酸轧机器编号 {ar_machine_key}")
                continue

            ar_start_time = start_time + pd.Timedelta(hours=result.ar_start)
            ar_end_time = start_time + pd.Timedelta(hours=result.ar_end)
            df_data.append({
                'Task': ar_machine_key,
                'Start': ar_start_time,
                'Finish': ar_end_time,
                'Material': f"order_{mat.id}",
                'Category': mat.category,
                'Operation': '酸轧'
            })

            # 连退工序
            ca_machine_key = f'CA-{result.ca_machine + 1}'
            if ca_machine_key not in machine_positions:
                print(f"警告: 无效的连退机器编号 {ca_machine_key}")
                continue

            ca_start_time = start_time + pd.Timedelta(hours=result.ca_start)
            ca_end_time = start_time + pd.Timedelta(hours=result.ca_end)
            df_data.append({
                'Task': ca_machine_key,
                'Start': ca_start_time,
                'Finish': ca_end_time,
                'Material': f"order_{mat.id}",
                'Category': mat.category,
                'Operation': '连退'
            })

        df = pd.DataFrame(df_data)

        # 使用plotly express的timeline函数更适合处理时间范围
        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Category",
            title="生产调度甘特图 - 所有机床",
            labels={"Task": "机器", "Material": "材料"},
            color_discrete_map=category_color_map
        )

        # 设置文本和悬停模板
        # 假设你想显示：订单、钢种、工序类型、开始/结束时间（字符串格式）
        df['Start_str'] = df['Start'].dt.strftime('%Y-%m-%d %H:%M')
        df['Finish_str'] = df['Finish'].dt.strftime('%Y-%m-%d %H:%M')

        # 定义要传入 customdata 的列（顺序很重要！）
        custom_cols = ['Material', 'Category', 'Operation', 'Start_str', 'Finish_str']

        fig.update_traces(
            text=None,  # 不在色块上显示任何文字
            customdata=df[custom_cols].values,
            hovertemplate=(
                    "<b>订单:</b> %{customdata[0]}<br>" +
                    "<b>钢种:</b> %{customdata[1]}<br>" +
                    "<b>工序:</b> %{customdata[2]}<br>" +
                    "<b>开始:</b> %{customdata[3]}<br>" +
                    "<b>结束:</b> %{customdata[4]}<extra></extra>"
            )
        )

        fig.update_layout(
            title='生产调度甘特图 - 所有机床',
            xaxis_title='时间',
            yaxis_title='机器'
        )

        # 设置x轴格式
        fig.update_xaxes(
            type='date',
            tickformat='%m-%d\n%H:%M',  # 格式化为月-日\n时:分
            title_text='时间'
        )

        fig.update_yaxes(
            title_text='机器',
            categoryorder='array',
            categoryarray=['CA-3', 'CA-2', 'CA-1', 'AR-2', 'AR-1', 'HR-1']  # HR-1 在最后 → 显示在最上方
        )

        fig.update_layout(
            title='生产调度甘特图 - 所有机床',
            xaxis_title='时间',
            yaxis_title='机器',
            barmode='overlay',
            bargap=0.2,
            width=1200,
            height=600
        )

        # 设置x轴格式
        fig.update_xaxes(type='date')

        # 添加垂直线表示机器分隔

        # 添加图例
        fig.update_layout(legend_title_text='钢种')

        if save_path:
            fig.write_image(save_path)
            print(f"甘特图已保存至: {save_path}")

        fig.show()

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

        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('热轧后仓库库存', '酸轧后仓库库存'),
            vertical_spacing=0.1
        )

        # 热轧后库存曲线
        fig.add_trace(
            go.Scatter(
                x=times_hr,
                y=inventory_hr,
                mode='lines',
                line=dict(color='steelblue', width=2),
                fill='tonexty',  # 填充下方区域
                fillcolor='rgba(70, 130, 180, 0.3)',  # steelblue的透明版本
                name='热轧后库存',
                hovertemplate='<b>时间:</b> %{x}<br><b>库存数量:</b> %{y}<extra></extra>'
            ),
            row=1, col=1
        )

        # 酸轧后库存曲线
        fig.add_trace(
            go.Scatter(
                x=times_ar,
                y=inventory_ar,
                mode='lines',
                line=dict(color='coral', width=2),
                fill='tonexty',  # 填充下方区域
                fillcolor='rgba(255, 127, 80, 0.3)',  # coral的透明版本
                name='酸轧后库存',
                hovertemplate='<b>时间:</b> %{x}<br><b>库存数量:</b> %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        # 更新布局
        fig.update_layout(
            title_text='库存变化曲线',
            height=600,
            showlegend=False
        )

        # 设置y轴标题
        fig.update_yaxes(title_text='库存数量 (件)', row=1, col=1)
        fig.update_yaxes(title_text='库存数量 (件)', row=2, col=1)

        # 设置x轴标题
        fig.update_xaxes(title_text='时间 (真实时间)', row=2, col=1)

        # 设置网格
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)

        if save_path:
            fig.write_image(save_path)
            print(f"库存曲线图已保存至: {save_path}")

        fig.show()

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
    generations = list(range(1, len(history) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=generations,
        y=history,
        mode='lines+markers',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6, symbol='circle'),
        name='最优解',
        hovertemplate='<b>代数:</b> %{x}<br><b>最优总拖期:</b> %{y}<extra></extra>'
    ))

    fig.update_layout(
        title='算法收敛曲线',
        xaxis_title='迭代代数',
        yaxis_title='最优总拖期 (天)',
        width=800,
        height=600,
        showlegend=True
    )

    # 添加网格
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    if save_path:
        fig.write_image(save_path)
        print(f"收敛曲线图已保存至: {save_path}")

    fig.show()


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
