import pandas as pd


class OrderScheduleResult:
    """
    订单调度结果
    """
    order_no: str
    delivery_date: str
    s_az: str
    e_az: str
    s_lt: str
    e_lt: str
    l_i_hours: float
    i_az_hours: float
    i_az_wt: float
    i_lt_hours: float
    i_lt_wt: float
    process_time_az: float
    process_time_lt: float
    def __init__(self, order_no: str, delivery_date: str, s_az: str, e_az: str, s_lt: str, e_lt: str, l_i_hours: float,
                 i_az_hours: float, i_az_wt: float, i_lt_hours: float, i_lt_wt: float,
                 process_time_az: float, process_time_lt: float):
        self.order_no = order_no
        self.delivery_date = delivery_date
        self.s_az = s_az
        self.e_az = e_az
        self.s_lt = s_lt
        self.e_lt = e_lt
        self.l_i_hours = l_i_hours
        self.i_az_hours = i_az_hours
        self.i_az_wt = i_az_wt
        self.i_lt_hours = i_lt_hours
        self.i_lt_wt = i_lt_wt
        self.process_time_az = process_time_az
        self.process_time_lt = process_time_lt


    def to_dict(self):
        return {
            'ORDER_NO': self.order_no,
            'DELIVERY_DATE': self.delivery_date,
            'S_AZ': self.s_az,
            'E_AZ': self.e_az,
            'S_LT': self.s_lt,
            'E_LT': self.e_lt,
            'L_i_hours': self.l_i_hours,
            'I_AZ_hours': self.i_az_hours,
            'I_AZ_wt': self.i_az_wt,
            'I_LT_hours': self.i_lt_hours,
            'I_LT_wt': self.i_lt_wt,
            'PROCESS_TIME_AZ': self.process_time_az,
            'PROCESS_TIME_LT': self.process_time_lt
        }

    def to_dict_chinese(self):
        return {
            '订单编号': self.order_no,
            '交货日期': self.delivery_date,
            '酸轧开始时间': self.s_az,
            '酸轧结束时间': self.e_az,
            '连退开始时间': self.s_lt,
            '连退结束时间': self.e_lt,
            '总拖期 (小时)': self.l_i_hours,
            '总酸轧后库库存时间 (小时)': self.i_az_hours,
            '总酸轧后库库存时间 (重量)': self.i_az_wt,
            '总连退后库库存时间 (小时)': self.i_lt_hours,
            '总连退后库库存时间 (重量)': self.i_lt_wt,
            '酸轧工序处理时间 (小时)': self.process_time_az,
            '连退工序处理时间 (小时)': self.process_time_lt
        }
