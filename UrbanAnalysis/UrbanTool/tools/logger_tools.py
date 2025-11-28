from datetime import datetime
import os
from tabnanny import verbose

class Log:

    verbose = True

    def __init__(self, project_folder, resume = False, add = ''):
        self.path = f'{project_folder}/log{add}.txt'
        if os.path.exists(self.path):
            resume = True
        if resume == False:
            self.creates_file()
        else:
            self.resume_file()


    def creates_file(self):
        f = open(self.path, 'w')
        f.write('LOG FILE')
        f.write('\n')
        f.close()


    def resume_file(self):
        self.write_log('Resuming script ...')


    def write_log(self, message):
        time = Log.get_time()
        log = f'{time} - {message}'
        if Log.verbose:
            print(log)
        f = open(self.path, 'a')
        f.write(log)
        f.write('\n')
        f.close()
    
    
    def write_error(self, error):
        time = Log.get_time()
        log = f'{time} - Error:\n\n{repr(error)}'
        if Log.verbose:
            print(log)
        f = open(self.path, 'a')
        f.write(log)
        f.write('\n\n')
        f.close()
    

    def write_blank(self):
        if Log.verbose:
            print('')
        f = open(self.path, 'a')
        f.write('')
        f.close()


    def get_time():
        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y %H:%M:%S")
        return current_time
    


