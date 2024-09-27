import os

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

CPU_COUNT: int = 8
PARALLEL_CHUNK_SIZE: int = 5
DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
