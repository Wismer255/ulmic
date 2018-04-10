import os
from ulmic.environment import UlmicEnvironment as ue

class Logs(object):

    def __init__(self, name, show_header=True):
        self.name = name
        self.directory = ue.get_daily_log_dir()
        self.log_file = os.path.join(self.directory,name+'.log')
        self.create_dirs()
        if show_header:
            header = '='*20 + ' {} '.format(self.name) + '='*20
            self.log(header)


    def log(self, text):
        with open(self.log_file,'a') as f:
            f.write(text)
            f.write('\n')


    def create_dirs(self):
        try:
            os.makedirs(self.directory)
        except:
            pass



