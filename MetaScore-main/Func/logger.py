import logging
import os

class Logger:
    def __init__(self, log_dir, log_file='train.log'):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(levelname)s | %(message)s')

        # file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
