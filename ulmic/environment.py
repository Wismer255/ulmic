import datetime
import os
import logging




class UlmicEnvironment:

    @staticmethod
    def get_daily_logger():
        date_today = str(datetime.datetime.now().date())
        daily_log = os.path.join(os.environ['ULMIC_LOG'], date_today)
        logging.basicConfig(filename=daily_log + '.log', level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        return logger

    @staticmethod
    def get_daily_log_dir():
        date_today = str(datetime.datetime.now().date())
        daily_log_dir = os.path.join(os.environ['ULMIC_LOG'], date_today)
        return daily_log_dir

    @staticmethod
    def get_log_dir():
        return UlmicEnvironment.get_path_from_environment_variable('ULMIC_LOG')

    @staticmethod
    def get_home_dir():
        return UlmicEnvironment.get_path_from_environment_variable('ULMIC_HOME')

    @staticmethod
    def get_data_dir():
        return UlmicEnvironment.get_path_from_environment_variable('ULMIC_DATA')

    @staticmethod
    def get_test_dir():
        return UlmicEnvironment.get_path_from_environment_variable('ULMIC_DATA')

    @staticmethod
    def get_path_from_environment_variable(path):
        try:
            path = os.environ[path]
        except:
            path = os.getcwd()
        return path

    @staticmethod
    def get_threads():
        return 1


if not os.environ['ULMIC_LOG']:
    os.environ['ULMIC_LOG'] = os.getcwd()

date_today = str(datetime.datetime.now().date())
daily_log = os.path.join(os.environ['ULMIC_LOG'],date_today)

logging.basicConfig(filename=daily_log+'.log',level=logging.DEBUG)
logger = logging.getLogger(__name__)

if not os.environ['ULMIC_HOME']:
    logger.info('Environment variable ULMIC_HOME not found. Setting to %s' %os.getcwd())
    os.environ['ULMIC_HOME'] = os.getcwd()

if not os.environ['ULMIC_DATA']:
    logger.info('Environment variable ULMIC_DATA not found. Setting to %s' %os.getcwd())
    os.environ['ULMIC_DATA'] = os.getcwd()

if not os.environ['ULMIC_TEST']:
    logger.info('Environment variable ULMIC_TEST not found. Setting to %s' %os.getcwd())
    os.environ['ULMIC_TEST'] = os.getcwd()

