import os
import sys
import logging

login_format = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'

log_dir = "logs"
logging_path = os.path.join(log_dir,"runnig_logs.log")
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig ( 
    level = logging.INFO, 
    format = login_format, 

    handlers = [
                 logging.FileHandler(logging_path),
                 logging.StreamHandler(sys.stdout)
    ]
)


logger = logging.getLogger("chest_cancer_logger")