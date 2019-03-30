import os
import sys
import logging

def setup_logger(log_file_path):
  """
  adapted from https://github.com/lvpengyuan/ssd.tf/blob/fctd-box/src/utils.py
  Setup a logger that simultaneously output to a file and stdout
  ARGS
    log_file_path: string, path to the logging file
  """
  # logging settings
  log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.DEBUG)
  # file handler
  log_file_handler = logging.FileHandler(log_file_path)
  log_file_handler.setFormatter(log_formatter)
  root_logger.addHandler(log_file_handler)
  # stream handler (stdout)
  log_stream_handler = logging.StreamHandler(sys.stdout)
  log_stream_handler.setFormatter(log_formatter)
  root_logger.addHandler(log_stream_handler)

  logging.info('Logging file is %s' % log_file_path)