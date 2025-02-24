import logging
from pathlib import Path 
import datetime as dt 



def create_log_path(module_name: str) -> str:
    """
    Create a log file path based on current date and the provided module_name
    
    Parameters:
    - module_name (str): The name of the log file

    Return: 
    - str: The complete path of log file
    """

    current_date = dt.date.today()
    
    # create a logs folder in root directory 
    root_path = Path(__file__).parent.parent
    
    # create path for logs folder
    log_dir_path = root_path/ 'logs'
    log_dir_path.mkdir(exist_ok=True)

    # create folder for specific module
    module_log_path = log_dir_path / module_name
    module_log_path.mkdir(exist_ok=True)

    # covert the date to str
    current_date_str = current_date.strftime("%d-%m-%Y")

    log_file_name = module_log_path /( current_date_str + '.log')

    return log_file_name


class CustomLogger:
    def __init__(self, logger_name, log_filename):
        """
        Initializes a custom logger with the specified name and log file.

        Parameters:
        - logger_name (str): Name of the logger.
        - log_filename (str): Path to the log file.
        """

        self.__logger = logging.getLogger(name=logger_name)
        self.__log_path = log_filename

        # make the file handler

        file_handler = logging.FileHandler(filename = self.__log_path, 
                                                    mode='a')

        # add file handler to the logger
        self.__logger.addHandler(hdlr= file_handler)
        # formatter for logs
        log_format = "%(asctime)s - %(levelname)s : %(message)s"
        time_format = '%d-%m-%Y %H:%M:%S'
        formatter = logging.Formatter(fmt=log_format, 
                                            datefmt=time_format)

        file_handler.setFormatter(fmt = formatter)


    def get_log_path(self):
        """
        Return the path of the log file

        Returns:
        - str: Log file path
        """

        return self.__log_path

    def get_logger(self):
        """
        Return the logger object

        Returns:
        - logging.Logger: Logger object
        """

        return self.__logger

    def set_log_level(self, level=logging.DEBUG):
        """
        Sets the log level for logger.

        Parameters:
        - level(int): Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        logger = self.get_logger()
        logger.setLevel(level=level)

    def save_logs(self, msg, log_level='info'):

        """
        Saves logs to the specified log file with given message and log level.

        Parameters:
        - msg (str): Log message.
        - log_level: (str): Log Level ('debug', 'info', 'warning', 'error', 'exception', 'critcal'). 
        """

        # get logger
        logger = self.get_logger()

        if log_level == 'debug':
            logger.debug(msg=msg)
        elif log_level == 'info':
            logger.info(msg=msg)
        elif log_level == 'warning':
            logger.warning(msg=msg)
        elif log_level == 'error':
            logger.error(msg=msg)
        elif log_level == 'exception':
            logger.exception(msg=msg)
        elif log_level == 'critical':
            logger.critical(msg=msg)


if __name__ == '__main__':

    logger = CustomLogger(logger_name = 'my_logger',
                            log_filename= create_log_path('test'))
    logger.set_log_level()
    
    logger.save_logs('save me code is breaking',log_level='critical')
