from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from cfg.config import *
import logging
from functools import wraps
import time
from contextvars import ContextVar

# 定义 Base
Base = declarative_base()
# 创建数据库引擎
engine = create_engine(
    MYSQL_URL_REMTOTE,
    pool_size=30,       # 连接池大小
    max_overflow=50,    # 允许的溢出连接
    pool_timeout=60,     # 连接超时时间
    pool_recycle=9600    # 连接回收时间
)

# 创建全局 SessionFactory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# 线程安全的 Scoped Session
session_factory = scoped_session(SessionLocal)
# 创建数据库表（如果不存在）
Base.metadata.create_all(engine)
session_logger = logging.getLogger("session_logger")
if not session_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    session_logger.addHandler(handler)
    session_logger.setLevel(logging.INFO)

# 记录当前线程的 session 层级（栈深度）
_session_depth = ContextVar("_session_depth", default=0)

def use_session(commit=True, log=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "UNKNOWN"

            depth = _session_depth.get()
            _session_depth.set(depth + 1)  # 进入一层

            session_created = False
            if 'session' in kwargs and kwargs['session'] is not None:
                session = kwargs['session']
            else:
                session = session_factory()
                kwargs['session'] = session
                session_created = True

            try:
                result = func(*args, **kwargs)
                if commit and session_created:
                    session.commit()
                    status = "COMMIT"
                elif not commit:
                    status = "READ-ONLY"
                else:
                    status = "NO COMMIT (external session)"
                return result
            except Exception as e:
                if session_created:
                    session.rollback()
                status = f"ROLLBACK ({type(e).__name__}: {str(e)})"
                raise
            finally:
                # 退出一层
                _session_depth.set(_session_depth.get() - 1)

                # 只有是内部创建的 session 且是最外层，才 remove
                if session_created and _session_depth.get() == 0:
                    session_factory.remove()

                if log:
                    elapsed = round(time.time() - start_time, 3)
                    session_logger.info(f"[{func.__name__}] {status} in {elapsed}s")
        return wrapper
    return decorator
