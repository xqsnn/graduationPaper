
from plot_charts.plot_gant_charts import plot_schedule_gantt
from db_query_util import get_schedule_result_by_pareto_id, get_table_data
from plot_charts.plot_tardiness_charts import plot_tardiness_chart
from plot_charts.plot_inventory_changes import plot_inventory_changes
from nsga2_solver import StaticParameters,Operation
from table.order_new import order_new


if __name__ == "__main__":
    result = get_schedule_result_by_pareto_id(150)
    order_new_data = get_table_data(order_new)

    plot_schedule_gantt(result)
    plot_tardiness_chart(result, order_new_data)
    plot_inventory_changes(result, order_new_data, StaticParameters,Operation)
