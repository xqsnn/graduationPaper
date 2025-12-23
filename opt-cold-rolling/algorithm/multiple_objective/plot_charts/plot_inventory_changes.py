import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


def plot_inventory_changes(schedule_results_df, order_data, staticParameters, operation) -> go.Figure:
    """
    绘制三个仓库的库存变化曲线图

    Args:
        schedule_results_df: pandas DataFrame，包含调度结果
        order_data: pandas DataFrame，包含订单信息（重量、交付日期等）
        staticParameters: 静态参数配置

    Returns:
        包含三个仓库库存变化曲线的Plotly图表对象
    """

    # 将DataFrame转换为字典，方便通过订单号查找材料信息
    order_info_map = {}
    for idx, row in order_data.iterrows():
        order_info_map[row['order_no']] = row

    # 收集所有库存事件 (时间, 重量变化, 仓库类型)
    events = []
    unit_ton = 50.0

    # 转换时间字符串为浮点数小时
    def time_str_to_float(time_str, base_date=None):
        """将时间字符串转换为从基准日期开始的小时数"""
        import re

        if pd.isna(time_str):  # 处理空值
            return 0.0
        elif isinstance(time_str, str):
            # 如果是datetime格式的字符串，如 "2026-01-31 01:00:00"
            # 尝试不同的日期时间格式
            time_str = time_str.strip()

            # 如果包含空格，用T替换（ISO格式）
            if ' ' in time_str:
                time_str = time_str.replace(' ', 'T')

            try:
                dt = datetime.fromisoformat(time_str)
            except ValueError:
                # 如果不是ISO格式，尝试其他常见格式
                formats_to_try = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d %H:%M',
                    '%Y-%m-%d',
                    '%Y/%m/%d %H:%M:%S',
                    '%Y/%m/%d %H:%M',
                    '%Y/%m/%d'
                ]

                dt = None
                for fmt in formats_to_try:
                    try:
                        dt = datetime.strptime(time_str.replace('T', ' '), fmt)
                        break
                    except ValueError:
                        continue

                if dt is None:
                    raise ValueError(f"无法解析时间格式: {time_str}")
        elif isinstance(time_str, pd.Timestamp):
            # 如果是pandas时间戳格式
            dt = time_str.to_pydatetime()
        else:
            # 假设time_str已经是datetime对象
            dt = time_str

        # 计算相对于StaticParameters.start_time的小时数
        time_diff = dt - staticParameters.start_time
        hours_diff = time_diff.total_seconds() / 3600.0

        return hours_diff

    for idx, result in schedule_results_df.iterrows():
        # 通过订单号获取材料信息
        if result['order_no'] not in order_info_map:
            continue  # 跳过找不到材料信息的订单

        order_info = order_info_map[result['order_no']]

        # 确保重量存在
        if pd.isna(order_info.get('weight')):
            raise ValueError(f"材料 {result['order_no']} 缺少重量信息")

        total_weight = order_info['weight']

        # 将时间字符串转换为小时数
        hr_start = time_str_to_float(result['hr_start'])
        hr_end = time_str_to_float(result['hr_end'])
        ar_start = time_str_to_float(result['ar_start'])
        ar_end = time_str_to_float(result['ar_end'])
        ca_start = time_str_to_float(result['ca_start'])
        ca_end = time_str_to_float(result['ca_end'])
        complete_time = time_str_to_float(result['complete_time'])

        # 时间转换：从静态参数start_time开始计算偏移
        start_time_offset = 0  # 在这个上下文中，我们直接使用时间差值

        num_units = int(math.ceil(total_weight / unit_ton))

        # HR → AR 缓冲区事件
        hr_duration = hr_end - hr_start
        hr_unit_time = hr_duration / num_units
        for k in range(num_units):
            t = hr_start + (k + 1) * hr_unit_time
            w = unit_ton if k < num_units - 1 else total_weight - unit_ton * (num_units - 1)
            events.append((t, w, 'hr_to_ar'))

        ar_duration = ar_end - ar_start
        ar_unit_time = ar_duration / num_units
        for k in range(num_units):
            t = ar_start + (k + 1) * ar_unit_time - staticParameters.transmission_time[operation.HR]
            w = unit_ton if k < num_units - 1 else total_weight - unit_ton * (num_units - 1)
            events.append((t, -w, 'hr_to_ar'))

        # AR → CA 缓冲区事件
        for k in range(num_units):
            t = ar_start + (k + 1) * ar_unit_time
            w = unit_ton if k < num_units - 1 else total_weight - unit_ton * (num_units - 1)
            events.append((t, w, 'ar_to_ca'))

        ca_duration = ca_end - ca_start
        ca_unit_time = ca_duration / num_units
        for k in range(num_units):
            t = ca_start + (k + 1) * ca_unit_time - staticParameters.transmission_time[operation.AR]
            w = unit_ton if k < num_units - 1 else total_weight - unit_ton * (num_units - 1)
            events.append((t, -w, 'ar_to_ca'))

        # CA → 最终库存事件
        for k in range(num_units):
            t = ca_start + (k + 1) * ca_unit_time
            w = unit_ton if k < num_units - 1 else total_weight - unit_ton * (num_units - 1)
            events.append((t, w, 'ca_to_final'))

        # 获取交付日期并转换为小时数
        delivery_date_str = order_info.get('delivery_date')
        if isinstance(delivery_date_str, str):
            # 假设delivery_date格式是YYYY-MM-DD HH:MM:SS或类似格式
            delivery_datetime = datetime.fromisoformat(delivery_date_str.replace(' ', 'T'))
            delivery_time = (delivery_datetime - staticParameters.start_time).total_seconds() / 3600.0
        elif pd.isna(delivery_date_str):
            raise ValueError("无法解析交付日期")
        else:
            delivery_time = (delivery_date_str - staticParameters.start_time).total_seconds() / 3600.0

        events.append((max(complete_time, delivery_time) - staticParameters.transmission_time[operation.CA], -total_weight, 'ca_to_final'))

    # 按时间排序事件
    events.sort(key=lambda x: x[0])

    if not events:
        raise ValueError("没有库存事件数据可绘制")

    # 计算各仓库库存变化序列
    time_points = [events[0][0]]  # 相对小时数
    hr_to_ar_inventory = [0.0]
    ar_to_ca_inventory = [0.0]
    ca_to_final_inventory = [0.0]

    current_hr = 0.0
    current_ar = 0.0
    current_ca = 0.0

    for event_time, delta, warehouse in events:
        # 更新库存
        if warehouse == 'hr_to_ar':
            current_hr += delta
        elif warehouse == 'ar_to_ca':
            current_ar += delta
        elif warehouse == 'ca_to_final':
            current_ca += delta

        # 记录时间点和库存值
        time_points.append(event_time)
        hr_to_ar_inventory.append(current_hr)
        ar_to_ca_inventory.append(current_ar)
        ca_to_final_inventory.append(current_ca)

    # 将相对小时数转换为真实时间戳
    real_time_points = [
        staticParameters.start_time + timedelta(hours=time_point)
        for time_point in time_points
    ]

    # 创建子图
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('HR→AR缓冲区库存变化', 'AR→CA缓冲区库存变化', 'CA→最终库存变化'),
        shared_xaxes=True,
        vertical_spacing=0.1,
        y_title='库存水平 (吨)'
    )

    # 添加HR→AR库存曲线
    fig.add_trace(
        go.Scatter(
            x=real_time_points,
            y=hr_to_ar_inventory,
            mode='lines',
            name='HR→AR库存',
            line=dict(color='blue'),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ),
        row=1, col=1
    )

    # 添加AR→CA库存曲线
    fig.add_trace(
        go.Scatter(
            x=real_time_points,
            y=ar_to_ca_inventory,
            mode='lines',
            name='AR→CA库存',
            line=dict(color='green'),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ),
        row=2, col=1
    )

    # 添加CA→最终库存曲线
    fig.add_trace(
        go.Scatter(
            x=real_time_points,
            y=ca_to_final_inventory,
            mode='lines',
            name='CA→最终库存',
            line=dict(color='orange'),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.1)'
        ),
        row=3, col=1
    )

    # 更新布局
    fig.update_layout(
        title='各仓库库存水平变化曲线',
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # 更新x轴标签
    fig.update_xaxes(title_text='时间', row=3, col=1)

    # 添加零基准线
    for i in range(1, 4):
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, row=i, col=1)

    fig.show()
    return fig