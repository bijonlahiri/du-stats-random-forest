import os, sys

class DUStatsException(Exception):

    def __init__(self, error_message: str, system_trace: sys):
        self.error_message = error_message
        self.system_trace = system_trace
        _, _, exc_tb = system_trace.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename
    
    def __str__(self):
        return f"\nError: {self.error_message}\nFilename: {self.filename}\nLineno: {self.lineno}"