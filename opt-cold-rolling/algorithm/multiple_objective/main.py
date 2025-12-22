
from plot_charts.plot_gant_charts import plot_schedule_gantt
from db_query_util import get_schedule_result_by_pareto_id, get_table_data
from plot_charts.plot_tardiness_charts import plot_tardiness_chart

from table.order_new import order_new


if __name__ == "__main__":
    result = get_schedule_result_by_pareto_id(1)
    order_new_data = get_table_data(order_new)

    plot_schedule_gantt(result)
    plot_tardiness_chart(result, order_new_data)
