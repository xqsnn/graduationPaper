"""
数据库工具模块，用于将NSGA2求解结果保存到数据库
"""
import os
from datetime import datetime, timedelta
from database import get_db_session
from table.pareto_front_solution import ParetoFrontSolution
from table.schedule_result import ScheduleResultDetail
from nsga2_solver import StaticParameters

def save_pareto_front_results(task_id: str, pareto_solutions: list, materials: list, clear_existing: bool = False):
    """
    保存帕累托前沿结果到数据库

    Args:
        task_id: 任务ID
        pareto_solutions: 帕累托前沿解列表
        materials: 材料列表（用于获取订单号等信息）
        clear_existing: 是否在插入前清空现有数据，默认为False
    """
    with get_db_session() as db:
        # 如果需要清空现有数据
        if clear_existing:
            # 根据task_id获取现有的pareto_front_solution记录ID以清空对应的详细记录
            existing_pareto_records = db.query(ParetoFrontSolution.id).filter(
                ParetoFrontSolution.task_id.like(f'{task_id.split("_")[0]}_%')
            ).all()

            existing_pareto_ids = [r.id for r in existing_pareto_records]

            if existing_pareto_ids:
                # 清空schedule_result表中对应pareto_front_id的数据
                db.query(ScheduleResultDetail).filter(
                    ScheduleResultDetail.pareto_front_id.in_(existing_pareto_ids)
                ).delete(synchronize_session=False)

            # 清空pareto_front_solution表中相关数据
            db.query(ParetoFrontSolution).filter(
                ParetoFrontSolution.task_id.like(f'{task_id.split("_")[0]}_%')
            ).delete(synchronize_session=False)

            db.commit()
            print("已清空现有数据")

        # 插入帕累托前沿解到pareto_front_solution表
        pareto_records = []
        for sol in pareto_solutions:
            record = ParetoFrontSolution(
                task_id=task_id,
                inventory=sol.avg_inventory,
                tardiness=sol.max_tardiness,
                cost=sol.process_instability,  # 这里暂时将工艺不稳定性作为成本
                start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 修正时间格式
                end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 修正时间格式
                reserved_1=str(sol.rank),  # 保留等级信息
                reserved_2=str(sol.crowding_distance),  # 保留拥挤度信息
                reserved_3=f"Solutions with {len(sol.schedule_results or [])} items"  # 保留更多信息
            )
            db.add(record)
            pareto_records.append(record)

        # 提交帕累托前沿记录以获取ID
        db.commit()

        # 为每个帕累托解保存详细调度结果到schedule_result表
        for i, sol in enumerate(pareto_solutions):
            pareto_id = pareto_records[i].id  # 获取刚插入记录的ID

            # 获取开始时间基准（使用项目静态参数）

            start_time = StaticParameters.start_time

            # 遍历调度结果，保存每个订单的详细时间安排
            if sol.schedule_results:
                for schedule_res in sol.schedule_results:
                    material = materials[schedule_res.material_id]

                    # 转换时间戳为实际日期时间字符串
                    hr_start_dt = start_time + timedelta(hours=schedule_res.hr_start)
                    hr_end_dt = start_time + timedelta(hours=schedule_res.hr_end)
                    ar_start_dt = start_time + timedelta(hours=schedule_res.ar_start)
                    ar_end_dt = start_time + timedelta(hours=schedule_res.ar_end)
                    ca_start_dt = start_time + timedelta(hours=schedule_res.ca_start)
                    ca_end_dt = start_time + timedelta(hours=schedule_res.ca_end)

                    # 计算完成时间
                    complete_time_dt = start_time + timedelta(hours=schedule_res.complete_time)

                    detail_record = ScheduleResultDetail(
                        order_no=material.order_no,
                        hr_start=hr_start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        hr_end=hr_end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        ar_start=ar_start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        ar_end=ar_end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        ar_machine=schedule_res.ar_machine,
                        ca_start=ca_start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        ca_end=ca_end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        ca_machine=schedule_res.ca_machine,
                        complete_time=complete_time_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        pareto_front_id=pareto_id
                    )
                    db.add(detail_record)

        # 提交详细调度结果
        db.commit()
        print(f"已保存 {len(pareto_solutions)} 个帕累托解及其 {sum(len(sol.schedule_results or []) for sol in pareto_solutions)} 条详细调度记录到数据库")