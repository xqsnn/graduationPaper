"""
可视化模块，负责生成甘特图等图表
"""
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# 设置默认模板
pio.templates.default = "plotly_white"

class PlotVisualization:
    """
    可视化类
    """

    @staticmethod
    def plot_schedule_gantt(scheduled_df: pd.DataFrame, title: str = "生产调度甘特图") -> go.Figure:
        """
        使用 Plotly 绘制调度结果的甘特图，支持多个工序动态处理。

        Args:
            scheduled_df (pd.DataFrame): heuristic_scheduler 返回的调度结果DataFrame。
            title (str): 甘特图的标题。

        Returns:
            go.Figure: 生成的图表
        """
        gantt_data = []

        # 动态识别工序
        process_columns = [col for col in scheduled_df.columns if col.startswith('S_')]
        process_names = [col.replace('S_', '') for col in process_columns]

        for idx, row in scheduled_df.iterrows():
            order_no = row['ORDER_NO']

            # 动态处理所有工序
            for process in process_names:
                start_col = f'S_{process}'
                end_col = f'E_{process}'
                time_col = f'PROCESS_TIME_{process}'

                if start_col in row and end_col in row and pd.notna(row[start_col]) and pd.notna(row[end_col]):
                    gantt_data.append(dict(
                        Task=order_no,
                        Start=row[start_col],
                        Finish=row[end_col],
                        Resource=process,  # 动态工序名称
                        ORDER_NO=order_no,
                        Process_Time_Hours=row.get(time_col, 0),
                        Delivery_Date=row['DELIVERY_DATE']
                    ))

        gantt_df = pd.DataFrame(gantt_data)

        # 确保 'Start' 和 'Finish' 列是 datetime 类型
        gantt_df['Start'] = pd.to_datetime(gantt_df['Start'])
        gantt_df['Finish'] = pd.to_datetime(gantt_df['Finish'])

        fig = px.timeline(gantt_df,
                          x_start="Start",
                          x_end="Finish",
                          y="Resource",
                          color="ORDER_NO",
                          hover_name="ORDER_NO",
                          hover_data={
                              "Start": "|%Y-%m-%d %H:%M",
                              "Finish": "|%Y-%m-%d %H:%M",
                              "Resource": True,
                              "ORDER_NO": True,
                              "Process_Time_Hours": True,
                              "Delivery_Date": True
                          },
                          title=title)

        # 动态设置Y轴顺序（可以根据需要调整）
        fig.update_yaxes(categoryorder="array", categoryarray=process_names[::-1])  # 倒序显示

        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="工序",
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
            height=len(gantt_df['Resource'].unique()) * 100 + 200,
            title_x=0.5,
            xaxis=dict(
                tickformat="%Y-%m-%d\n%H:%M",
            )
        )
        return fig

    @staticmethod
    def plot_delivery_performance(scheduled_df: pd.DataFrame, title: str = "订单交货情况 (拖期/提前)") -> go.Figure:
        """
        绘制订单交货情况的柱状图。
        向上红色表示拖期，向下绿色表示提前交货。

        Args:
            scheduled_df (pd.DataFrame): 调度结果DataFrame
            title (str): 图表的标题

        Returns:
            go.Figure: 生成的图表
        """
        plot_df = scheduled_df.copy()

        # 计算实际的交货时间差：负值表示提前，正值表示拖期
        plot_df['DELIVERY_DIFF_HOURS'] = (plot_df['E_LT'] - plot_df['DELIVERY_DATE']).dt.total_seconds() / 3600

        # 根据交货情况分配颜色
        plot_df['COLOR'] = plot_df['DELIVERY_DIFF_HOURS'].apply(lambda x: 'red' if x > 0 else 'green')

        fig = px.bar(plot_df,
                     x='ORDER_NO',
                     y='DELIVERY_DIFF_HOURS',
                     color='COLOR',  # 使用自定义颜色
                     color_discrete_map={'red': 'red', 'green': 'green'},  # 明确颜色映射
                     title=title,
                     labels={
                         'ORDER_NO': '订单号',
                         'DELIVERY_DIFF_HOURS': '交货时间差 (小时, 正为拖期, 负为提前)'
                     },
                     hover_data={
                         'DELIVERY_DATE': '|%Y-%m-%d %H:%M',
                         'E_LT': '|%Y-%m-%d %H:%M',
                         'COLOR': False  # 不在hover信息中显示颜色分类
                     })

        fig.update_layout(
            xaxis_title="订单号",
            yaxis_title="时间差 (小时)",
            title_x=0.5,
            showlegend=False  # 不显示颜色图例，因为颜色很直观
        )

        # 添加一条零线，更清晰地表示拖期和提前的界限
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

        return fig

    @staticmethod
    def plot_inventory_performance(scheduled_df: pd.DataFrame, title: str = "订单工序后库库存时间") -> go.Figure:
        """
        绘制订单各工序后库库存时间的柱状图，支持动态多个工序。
        X轴为订单号，Y轴为库存时间，颜色区分不同工序后库。

        Args:
            scheduled_df (pd.DataFrame): 调度结果DataFrame
            title (str): 图表的标题

        Returns:
            go.Figure: 生成的图表
        """
        # 动态识别库存列
        inventory_columns = [col for col in scheduled_df.columns if col.startswith('I_') and col.endswith('_hours')]

        # 将库存数据进行重塑，支持任意数量的工序库存
        inventory_data = []
        for idx, row in scheduled_df.iterrows():
            for inv_col in inventory_columns:
                # 从列名提取工序名称，例如 'I_AZ_hours' -> '酸轧后库库存'
                process_name = inv_col.replace('I_', '').replace('_hours', '')
                # 可以通过映射表自定义显示名称
                process_display_map = {
                    'AZ': '酸轧后库库存',
                    'LT': '连退后库库存'
                }
                display_name = process_display_map.get(process_name, f'{process_name}后库库存')

                inventory_data.append({
                    'ORDER_NO': row['ORDER_NO'],
                    '库存类型': display_name,
                    '库存时间 (小时)': row[inv_col]
                })

        inventory_df = pd.DataFrame(inventory_data)

        # 默认颜色映射，可扩展
        default_color_map = {
            '酸轧后库库存': 'skyblue',
            '连退后库库存': 'lightcoral'
        }

        # 只为实际存在的库存类型设置颜色
        existing_types = inventory_df['库存类型'].unique()
        color_discrete_map = {k: v for k, v in default_color_map.items() if k in existing_types}

        fig = px.bar(inventory_df,
                     x='ORDER_NO',
                     y='库存时间 (小时)',
                     color='库存类型',
                     barmode='group',
                     title=title,
                     labels={
                         'ORDER_NO': '订单号',
                         '库存时间 (小时)': '库存时间 (小时)'
                     },
                     color_discrete_map=color_discrete_map
                     )

        fig.update_layout(
            xaxis_title="订单号",
            yaxis_title="库存时间 (小时)",
            title_x=0.5,
            legend_title="库存类型"
        )

        return fig

    @staticmethod
    def plot_inventory_curve(scheduled_df: pd.DataFrame, static_params: dict, title: str = "各工序库存随时间变化图"):
        """
        绘制时间-库存量折线图，不同颜色代表不同工序。

        Args:
            scheduled_df (pd.DataFrame): 调度结果
            static_params (dict): 静态参数（包含process_time和transmission_time）
            title (str): 图表标题

        Returns:
            go.Figure: Plotly折线图
        """
        # 将字符串时间转为 datetime
        for col in ["S_AZ", "E_AZ", "S_LT", "E_LT", "DELIVERY_DATE"]:
            if pd.api.types.is_string_dtype(scheduled_df[col]):
                scheduled_df[col] = pd.to_datetime(scheduled_df[col])

        inventory_events = []  # 用于存放库存事件（时间点、工序、库存变化）

        for _, row in scheduled_df.iterrows():
            order_no = row["ORDER_NO"]
            az_end = row["E_AZ"]
            lt_start = row["S_LT"]
            lt_end = row["E_LT"]
            delivery_date = row["DELIVERY_DATE"]

            # 获取转运时间
            trans_az_to_lt = static_params.get(order_no, {}).get(0, {}).get("transmission_time", 0)
            trans_lt_to_deliver = static_params.get(order_no, {}).get(1, {}).get("transmission_time", 0)

            # 酸轧库存事件
            az_inv_start_time = az_end + timedelta(hours=trans_az_to_lt)
            if az_inv_start_time < lt_start:
                inventory_events.append({
                    "时间": az_inv_start_time,
                    "工序": "酸轧后库",
                    "变化量": row["I_AZ_wt"]
                })
                inventory_events.append({
                    "时间": lt_start,
                    "工序": "酸轧后库",
                    "变化量": -row["I_AZ_wt"]
                })

            # 连退库存事件
            lt_inv_start_time = lt_end + timedelta(hours=trans_lt_to_deliver)
            if lt_inv_start_time < delivery_date:
                inventory_events.append({
                    "时间": lt_inv_start_time,
                    "工序": "连退后库",
                    "变化量": row["I_LT_wt"]
                })
                inventory_events.append({
                    "时间": delivery_date,
                    "工序": "连退后库",
                    "变化量": -row["I_LT_wt"]
                })

        # 将事件整理成 DataFrame
        inv_df = pd.DataFrame(inventory_events)

        # 汇总成时间序列库存曲线
        time_series = []
        for process_name, group in inv_df.groupby("工序"):
            g = group.sort_values("时间").copy()
            g["库存量"] = g["变化量"].cumsum()
            time_series.append(g[["时间", "工序", "库存量"]])

        inv_curve_df = pd.concat(time_series, ignore_index=True)

        fig = px.line(
            inv_curve_df,
            x="时间",
            y="库存量",
            color="工序",
            title=title,
            labels={
                "时间": "时间",
                "库存量": "库存量（吨）",
                "工序": "库存类型"
            },
            line_shape="hv",
            # markers=True
        )

        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="库存量（吨）",
            title_x=0.5,
            legend_title="工序",
            hovermode="x unified"
        )

        fig.update_xaxes(
            tickformat="%Y-%m-%d %H:%M",
            ticklabelmode="instant"
        )
        fig.update_traces(
            hovertemplate="时间: %{x|%Y-%m-%d %H:%M}<br>库存量: %{y}<extra></extra>"
        )

        return fig

# 我需要画一个库存的折线图，横坐标为时间，纵坐标为库存量，颜色区分不同工序。
# 仿照这种画图
# @staticmethod
#     def plot_inventory_performance(scheduled_df: pd.DataFrame, title: str = "订单工序后库库存时间") -> go.Figure:
#         """
#         绘制订单各工序后库库存时间的柱状图，支持动态多个工序。
#         X轴为订单号，Y轴为库存时间，颜色区分不同工序后库。
#
#         Args:
#             scheduled_df (pd.DataFrame): 调度结果DataFrame
#             title (str): 图表的标题
#
#         Returns:
#             go.Figure: 生成的图表
#         """
#         # 动态识别库存列
#         inventory_columns = [col for col in scheduled_df.columns if col.startswith('I_') and col.endswith('_hours')]
#
#         # 将库存数据进行重塑，支持任意数量的工序库存
#         inventory_data = []
#         for idx, row in scheduled_df.iterrows():
#             for inv_col in inventory_columns:
#                 # 从列名提取工序名称，例如 'I_AZ_hours' -> '酸轧后库库存'
#                 process_name = inv_col.replace('I_', '').replace('_hours', '')
#                 # 可以通过映射表自定义显示名称
#                 process_display_map = {
#                     'AZ': '酸轧后库库存',
#                     'LT': '连退后库库存'
#                 }
#                 display_name = process_display_map.get(process_name, f'{process_name}后库库存')
#
#                 inventory_data.append({
#                     'ORDER_NO': row['ORDER_NO'],
#                     '库存类型': display_name,
#                     '库存时间 (小时)': row[inv_col]
#                 })
#
#         inventory_df = pd.DataFrame(inventory_data)
#
#         # 默认颜色映射，可扩展
#         default_color_map = {
#             '酸轧后库库存': 'skyblue',
#             '连退后库库存': 'lightcoral'
#         }
#
#         # 只为实际存在的库存类型设置颜色
#         existing_types = inventory_df['库存类型'].unique()
#         color_discrete_map = {k: v for k, v in default_color_map.items() if k in existing_types}
#
#         fig = px.bar(inventory_df,
#                      x='ORDER_NO',
#                      y='库存时间 (小时)',
#                      color='库存类型',
#                      barmode='group',
#                      title=title,
#                      labels={
#                          'ORDER_NO': '订单号',
#                          '库存时间 (小时)': '库存时间 (小时)'
#                      },
#                      color_discrete_map=color_discrete_map
#                      )
#
#         fig.update_layout(
#             xaxis_title="订单号",
#             yaxis_title="库存时间 (小时)",
#             title_x=0.5,
#             legend_title="库存类型"
#         )
#
#         return fig

# 调度结果在scheduled_df: pd.DataFrame中，dataframe的一列是这样的：
# class OrderScheduleResult:
#     """
#     订单调度结果
#     """
#     order_no: str
#     delivery_date: str
#     s_az: str
#     e_az: str
#     s_lt: str
#     e_lt: str
#     l_i_hours: float
#     i_az_hours: float
#     i_az_wt: float
#     i_lt_hours: float
#     i_lt_wt: float
#     process_time_az: float
#     process_time_lt: float
#     def __init__(self, order_no: str, delivery_date: str, s_az: str, e_az: str, s_lt: str, e_lt: str, l_i_hours: float,
#                  i_az_hours: float, i_az_wt: float, i_lt_hours: float, i_lt_wt: float,
#                  process_time_az: float, process_time_lt: float):
#         self.order_no = order_no
#         self.delivery_date = delivery_date
#         self.s_az = s_az
#         self.e_az = e_az
#         self.s_lt = s_lt
#         self.e_lt = e_lt
#         self.l_i_hours = l_i_hours
#         self.i_az_hours = i_az_hours
#         self.i_az_wt = i_az_wt
#         self.i_lt_hours = i_lt_hours
#         self.i_lt_wt = i_lt_wt
#         self.process_time_az = process_time_az
#         self.process_time_lt = process_time_lt
#
#
#     def to_dict(self):
#         return {
#             'ORDER_NO': self.order_no,
#             'DELIVERY_DATE': self.delivery_date,
#             'S_AZ': self.s_az,
#             'E_AZ': self.e_az,
#             'S_LT': self.s_lt,
#             'E_LT': self.e_lt,
#             'L_i_hours': self.l_i_hours,
#             'I_AZ_hours': self.i_az_hours,
#             'I_AZ_wt': self.i_az_wt,
#             'I_LT_hours': self.i_lt_hours,
#             'I_LT_wt': self.i_lt_wt,
#             'PROCESS_TIME_AZ': self.process_time_az,
#             'PROCESS_TIME_LT': self.process_time_lt
#         }
#
#     def to_dict_chinese(self):
#         return {
#             '订单编号': self.order_no,
#             '交货日期': self.delivery_date,
#             '酸轧开始时间': self.s_az,
#             '酸轧结束时间': self.e_az,
#             '连退开始时间': self.s_lt,
#             '连退结束时间': self.e_lt,
#             '总拖期 (小时)': self.l_i_hours,
#             '总酸轧后库库存时间 (小时)': self.i_az_hours,
#             '总酸轧后库库存时间 (重量)': self.i_az_wt,
#             '总连退后库库存时间 (小时)': self.i_lt_hours,
#             '总连退后库库存时间 (重量)': self.i_lt_wt,
#             '酸轧工序处理时间 (小时)': self.process_time_az,
#             '连退工序处理时间 (小时)': self.process_time_lt
#         }
#
# 还有一个static_params，# 合同静态参数，包括处理时间和转运时间
# # 数据格式
# # {
# #     "order_no": {
# #         "process_num": {
# #             "process_time": 12,
# #             "transmission_time": 3
# #         }
# #     }
# # }
# 其数据类型为：dict[str, dict[int, dict[str, float]]]
# 其中process_num为工序号，酸轧为0，连退为1，一些固定参数在这：# 工序数量
# process_num = 2


# 现在需要从最小的时间开始到最晚的生产时间结束，绘制一个折线图，横坐标为时间，纵坐标为库存量，库存量为重量，颜色区分不同工序。
# 酸轧的库存表示为前一个酸轧工序生产结束时间加上运输时间，这个时间点开始如果连退还未开始生产，则进入库存，否则认为进入下一道工序开始开始生产，不加库存。
# 连退的后库一样，连退的库存表示为连退工序生产结束时间加上运输时间，这个时间点开始如果未到交期，则进入库存，否则认为已道交货期，不加库存。

