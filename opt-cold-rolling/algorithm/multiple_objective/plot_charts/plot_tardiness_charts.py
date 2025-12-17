"""
绘制拖期情况柱状图的模块
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime


def plot_tardiness_chart(df: pd.DataFrame, order_delivery_df: pd.DataFrame, title: str = "Tardiness Chart"):
    """
    绘制拖期情况柱状图

    Args:
        df: 调度结果DataFrame，包含ca_end等时间列
        order_delivery_df: 订单交货期DataFrame，包含order_no和delivery_date列
        title: 图表标题
    """
    if df.empty or order_delivery_df.empty:
        print("无法绘制拖期图，因为输入数据为空")
        return

    # 将时间字符串转换为datetime类型
    df = df.copy()
    for col in ["hr_start", "hr_end", "ar_start", "ar_end", "ca_start", "ca_end", "complete_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    order_delivery_df = order_delivery_df.copy()
    order_delivery_df['delivery_date'] = pd.to_datetime(order_delivery_df['delivery_date'])

    # 将调度结果与交货期信息合并
    merged_df = df.merge(order_delivery_df[['order_no', 'delivery_date']], on='order_no', how='inner')

    # 计算每个订单的拖期/提前情况（以complete_time为准，即最终完成时间）
    results = []
    for _, row in merged_df.iterrows():
        order_no = row['order_no']
        # 优先使用complete_time（最终完成时间），如果不存在则使用ca_end（连退结束时间）
        if 'complete_time' in row and pd.notna(row['complete_time']):
            completion_time = row['complete_time']
        elif 'ca_end' in row and pd.notna(row['ca_end']):
            completion_time = row['ca_end']
        else:
            completion_time = None
        delivery_date = row['delivery_date']

        if completion_time is not None:
            # 计算天数差异
            diff_days = (completion_time - delivery_date).days
            results.append({
                'order_no': order_no,
                'delivery_date': delivery_date,
                'completion_date': completion_time,
                'diff_days': diff_days
            })

    if not results:
        print("没有足够的数据来绘制拖期图")
        return

    result_df = pd.DataFrame(results)

    # 按订单号排序
    result_df = result_df.sort_values('order_no')

    # 创建柱状图，处理拖期（正数）和提前（负数）
    fig = go.Figure()

    # 分离拖期和提前
    result_df['tardiness'] = result_df['diff_days'].apply(lambda x: x if x > 0 else 0)
    result_df['earliness'] = result_df['diff_days'].apply(lambda x: x if x < 0 else 0)

    # 拖期柱（正数，向上）
    fig.add_trace(go.Bar(
        x=result_df['order_no'],
        y=result_df['tardiness'],
        name='拖期 (天)',
        marker_color='red',
        text=result_df['tardiness'],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>拖期: %{y}天<br><extra></extra>'
    ))

    # 提前柱（负数，向下）
    fig.add_trace(go.Bar(
        x=result_df['order_no'],
        y=result_df['earliness'],
        name='提前 (天)',
        marker_color='green',
        text=result_df['earliness'].abs(),  # 显示绝对值
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>提前: %{y}天<br><extra></extra>'
    ))

    # 更新布局
    fig.update_layout(
        title=title,
        xaxis_title="订单号",
        yaxis_title="天数",
        barmode='relative',  # 使得正负值分别向上向下绘制
        showlegend=True,
        xaxis_tickangle=-45,
        height=600
    )

    # 添加零基准线
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

    fig.show()

    return fig


def plot_multiple_tardiness_charts(df_list, order_delivery_df_list, titles=None, main_title="Multiple Tardiness Charts Comparison"):
    """
    绘制多个拖期情况柱状图（并排对比）

    Args:
        df_list: 调度结果DataFrame列表
        order_delivery_df_list: 订单交货期DataFrame列表
        titles: 各图表标题列表
        main_title: 主标题
    """
    n_plots = len(df_list)
    if n_plots != len(order_delivery_df_list):
        print("错误：调度结果列表和订单交货期列表长度不匹配")
        return

    if titles is None:
        titles = [f"Tardiness Chart #{i+1}" for i in range(n_plots)]

    # 计算子图布局
    n_cols = min(2, n_plots)  # 最多2列
    n_rows = (n_plots + n_cols - 1) // n_cols

    if n_plots == 1:
        return plot_tardiness_chart(df_list[0], order_delivery_df_list[0], titles[0])

    # 创建子图
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        shared_xaxes=True
    )

    for i in range(n_plots):
        df = df_list[i]
        order_delivery_df = order_delivery_df_list[i]

        if df.empty or order_delivery_df.empty:
            continue

        # 将时间字符串转换为datetime类型
        df = df.copy()
        for col in ["hr_start", "hr_end", "ar_start", "ar_end", "ca_start", "ca_end", "complete_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        order_delivery_df = order_delivery_df.copy()
        order_delivery_df['delivery_date'] = pd.to_datetime(order_delivery_df['delivery_date'])

        # 将调度结果与交货期信息合并
        merged_df = df.merge(order_delivery_df[['order_no', 'delivery_date']], on='order_no', how='inner')

        # 计算每个订单的拖期/提前情况（以complete_time为准，即最终完成时间）
        results = []
        for _, row in merged_df.iterrows():
            order_no = row['order_no']
            # 优先使用complete_time（最终完成时间），如果不存在则使用ca_end（连退结束时间）
            if 'complete_time' in row and pd.notna(row['complete_time']):
                completion_time = row['complete_time']
            elif 'ca_end' in row and pd.notna(row['ca_end']):
                completion_time = row['ca_end']
            else:
                completion_time = None
            delivery_date = row['delivery_date']

            if completion_time is not None:
                # 计算天数差异
                diff_days = (completion_time - delivery_date).days
                results.append({
                    'order_no': order_no,
                    'delivery_date': delivery_date,
                    'completion_date': completion_time,
                    'diff_days': diff_days
                })

        if not results:
            continue

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('order_no')

        # 分离拖期和提前
        result_df['tardiness'] = result_df['diff_days'].apply(lambda x: x if x > 0 else 0)
        result_df['earliness'] = result_df['diff_days'].apply(lambda x: x if x < 0 else 0)

        # 计算子图位置
        row_idx = (i // n_cols) + 1
        col_idx = (i % n_cols) + 1

        # 添加拖期柱
        fig.add_trace(go.Bar(
            x=result_df['order_no'],
            y=result_df['tardiness'],
            name=f'S{i+1}-拖期',
            marker_color='red',
            showlegend=(i==0),  # 只在第一个子图显示图例
            text=result_df['tardiness'],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>拖期: %{y}天<br><extra></extra>'
        ), row=row_idx, col=col_idx)

        # 添加提前柱
        fig.add_trace(go.Bar(
            x=result_df['order_no'],
            y=result_df['earliness'],
            name=f'S{i+1}-提前',
            marker_color='green',
            showlegend=(i==0),  # 只在第一个子图显示图例
            text=result_df['earliness'].abs(),  # 显示绝对值
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>提前: %{y}天<br><extra></extra>'
        ), row=row_idx, col=col_idx)

    # 更新布局
    fig.update_layout(
        title=main_title,
        barmode='relative',
        showlegend=True,
        height=300*n_rows
    )

    # 为每个子图添加零基准线
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3,
                         row=i, col=j)

    fig.show()

    return fig


if __name__ == "__main__":
    print("拖期情况绘图模块已加载")
    print("使用方法: plot_tardiness_chart(df, order_delivery_df, title)")