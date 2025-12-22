

import pandas as pd
import plotly.express as px

def plot_schedule_gantt(
    df: pd.DataFrame,
    title="Schedule Gantt (HR → AR → CA)",
    save_path: str = None,
    height: int = None,
):
    """
    用 Plotly 绘制调度甘特图（HR → AR → CA → Complete）
    - 色块内不放文字
    - 信息通过 hover 展示

    必需列:
      order_no, hr_start, hr_end,
      ar_start, ar_end, ar_machine,
      ca_start, ca_end, ca_machine
    可选列:
      complete_time
    """
    if df is None or df.empty:
        print("无法绘制甘特图，因为数据为空")
        return

    df = df.copy()

    # --- 解析日期时间 ---
    datetime_cols = ["hr_start", "hr_end", "ar_start", "ar_end", "ca_start", "ca_end", "complete_time"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- 统一把三段工序摊平为“任务表” ---
    tasks = []

    # HR（单线）
    if {"hr_start", "hr_end"}.issubset(df.columns):
        sub = df[df["hr_start"].notna() & df["hr_end"].notna()]
        if not sub.empty:
            t = sub[["order_no", "hr_start", "hr_end"]].copy()
            t["stage"] = "HR"
            t["lane"] = "HR"
            t["machine"] = '0'
            t = t.rename(columns={"hr_start": "start", "hr_end": "end"})
            tasks.append(t)

    # AR（多机）
    if {"ar_start", "ar_end", "ar_machine"}.issubset(df.columns):
        sub = df[df["ar_start"].notna() & df["ar_end"].notna() & df["ar_machine"].notna()]
        if not sub.empty:
            t = sub[["order_no", "ar_start", "ar_end", "ar_machine"]].copy()
            t["stage"] = "AR"
            t["machine"] = t["ar_machine"].astype(str)
            t["lane"] = "AR-M" + t["machine"]
            t = t.rename(columns={"ar_start": "start", "ar_end": "end"})
            tasks.append(t.drop(columns=["ar_machine"]))

    # CA（多机）
    if {"ca_start", "ca_end", "ca_machine"}.issubset(df.columns):
        sub = df[df["ca_start"].notna() & df["ca_end"].notna() & df["ca_machine"].notna()]
        if not sub.empty:
            t = sub[["order_no", "ca_start", "ca_end", "ca_machine"]].copy()
            t["stage"] = "CA"
            t["machine"] = t["ca_machine"].astype(str)
            t["lane"] = "CA-M" + t["machine"]
            t = t.rename(columns={"ca_start": "start", "ca_end": "end"})
            tasks.append(t.drop(columns=["ca_machine"]))


    if not tasks:
        print("没有可绘制的数据（所有 start/end 都为空或缺列）")
        return

    tasks = pd.concat(tasks, ignore_index=True)
    tasks["order_no"] = tasks["order_no"].astype(str)

    # 计算时长（可选，用于 hover）
    tasks["duration_min"] = (tasks["end"] - tasks["start"]).dt.total_seconds() / 60.0

    # --- lane 顺序：上→下：HR, AR..., CA..., Complete ---
    lane_order = ["HR"]
    if "lane" in tasks.columns:
        ar_lanes = sorted([x for x in tasks["lane"].unique() if x.startswith("AR-M")])
        ca_lanes = sorted([x for x in tasks["lane"].unique() if x.startswith("CA-M")])
        # 如果有Complete列，也添加它
        complete_lanes = ["Complete"] if "Complete" in tasks["lane"].values else []
        lane_order += ar_lanes + ca_lanes + complete_lanes
        # 只保留存在的
        lane_order = [x for x in lane_order if x in set(tasks["lane"].unique())]

    # --- 颜色：按 order_no 固定 ---
    orders = tasks["order_no"].unique().tolist()
    # 用 Plotly 内置 qualitative 色板（数量不够会循环）
    palette = px.colors.qualitative.Plotly
    color_map = {o: palette[i % len(palette)] for i, o in enumerate(orders)}

    tasks = tasks.copy()
    tasks["start_str"] = tasks["start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    tasks["end_str"] = tasks["end"].dt.strftime("%Y-%m-%d %H:%M:%S")
    tasks["duration_hours"] = tasks["duration_min"] / 60.0
    tasks["machine"] = tasks["machine"]

    fig = px.timeline(
        tasks,
        x_start="start",
        x_end="end",
        y="lane",
        color="order_no",
        color_discrete_map=color_map,
        category_orders={"lane": lane_order},

        # ✅ 显式指定 customdata 的列顺序（关键）
        custom_data=["order_no", "stage", "machine", "start_str", "end_str", "duration_hours"],

        # hover_data 你可以留着（控制默认 hover 表格里出现哪些字段）
        hover_data={
            "lane": False,
            "order_no": False,  # 因为 hovertemplate 里会显示
            "stage": False,
            "machine": False,
            "start": False,
            "end": False,
            "duration_hours": False,
        },
    )

    # Plotly 时间线默认 y 轴从下往上；我们希望从上到下显示 lane_order
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=lane_order,
        autorange="reversed",
        title="生产线",
    )

    # 不显示色块内文字（Plotly timeline 默认也不放文字；这里明确禁用）
    fig.update_traces(text=None)

    # hover 更清晰一点（可按你喜好调整）
    fig.update_traces(
        hovertemplate=(
            "<b>订单</b>: %{customdata[0]}<br>"
            "<b>工序</b>: %{customdata[1]}<br>"
            "<b>设备</b>: %{customdata[2]}<br>"
            "<b>开始</b>: %{customdata[3]|%Y-%m-%d %H:%M:%S}<br>"
            "<b>结束</b>: %{customdata[4]|%Y-%m-%d %H:%M:%S}<br>"
            "<b>时长(小时)</b>: %{customdata[5]:.1f}<br>"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        legend_title_text="订单",
        bargap=0.25,
        hoverlabel=dict(namelength=-1),
        height=height if height is not None else max(450, 120 + 40 * len(lane_order)),
    )

    # 保存
    if save_path:
        if save_path.lower().endswith(".html"):
            fig.write_html(save_path)
            print(f"甘特图已保存到: {save_path}")
        else:
            # 需要: pip install -U kaleido
            fig.write_image(save_path, scale=2)
            print(f"甘特图已保存到: {save_path}")

    fig.show()
    return fig



def plot_multiple_gantt_charts(df_list, titles, save_paths=None):
    """
    绘制多个甘特图（用于比较不同的帕累托解）

    Args:
        df_list: DataFrame列表
        titles: 标题列表
        save_paths: 保存路径列表（可选）
    """
    if save_paths is None:
        save_paths = [None] * len(df_list)

    for i, (df, title, save_path) in enumerate(zip(df_list, titles, save_paths)):
        print(f"正在绘制第 {i+1}/{len(df_list)} 个甘特图: {title}")
        plot_schedule_gantt(df, title, save_path)
