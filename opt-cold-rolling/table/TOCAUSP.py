import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from algorithm.order_plan.multi_machines.object.order_multi_feature import OrderMultiFeature

# ORM基类
Base = declarative_base()


# 定义ORM模型，匹配TOCAUSP表结构
class TOCAUSP(Base):
    __tablename__ = "TOCAUSP"

    NO = Column(Integer, primary_key=True)
    ORDER_NO = Column(String(255))
    PROD_CODE_TYPE = Column(String(255))
    PLAN_COMMENT = Column(String(255))
    PROD_CPIN = Column(String(255))
    DELIVY_DATE = Column(Integer)
    ORDER_MONTH_PRO = Column(Integer)
    ORDER_MONTH = Column(Integer)
    SG_SIGN = Column(String(255))
    ST_NO = Column(String(255))
    SURF_QUALITY_GRADE = Column(String(255))
    ORDER_THICK = Column(String(255))
    ORDER_WIDTH = Column(Integer)
    LACK_WT = Column(Float)
    ORDER_WT = Column(Integer)
    SURF_STRUC_DESC = Column(String(255))
    SURF_STRUC_CODE = Column(String(255))
    ORDER_TYPE_CODE = Column(String(255))
    APN = Column(String(255))
    ORDER_RECV_TIME = Column(Integer)
    FIN_CUST_CNAME = Column(String(255))
    APN_DESC = Column(String(255))
    CONTRACT_NO = Column(String(255))
    ORDER_ZD = Column(String(255))
    ORDER_ND = Column(String(255))
    OAK_TEMP_MIN = Column(Integer)
    OAK_TEMP_MAX = Column(Integer)
    OAK_TEMP_AIM = Column(Integer)


    def to_dataframe(self):
        """
        将ORM对象转换为DataFrame
        """
        return pd.DataFrame([self.__dict__])
