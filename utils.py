import logging
import os

# 创建文件夹
def create_log_dir(dest_directory,filename):
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    # filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    # 创建txt文件
    file = open(filepath, 'w')

    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return None