from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

import os

# 确保sql目录存在
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# 数据库配置
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DB_PATH = os.path.join(PROJECT_ROOT, "sql", "cold_rolling.db")

DB_PATH = r"C:\Users\lxq\Desktop\GraduationPaper\opt-cold-rolling\sql\cold_rolling.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# 创建数据库引擎
engine = create_engine(DATABASE_URL, echo=False)

# 创建会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    获取数据库会话的依赖项
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    """
    数据库会话上下文管理器
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
