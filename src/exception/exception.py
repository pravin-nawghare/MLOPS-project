import sys

class CustomeException(Exception):
    
    def __init__(self, error_messsage, error_detail:sys): # sys module is handling every error detail. It knows
        self.error_message = error_messsage       # everything hapening inside the file
        _,_,exc_tb = error_detail.exc_info() # exc_info  gives 3 things but 3rd one is important
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occcured in the python script [{0}] line number [{1}] and error message [{2}]".format(
                                                            self.file_name, self.lineno, str(self.error_message))

if __name__ == '__main__':
    try:
        a= 1/0
    except Exception as e:
        raise CustomeException(e,sys)