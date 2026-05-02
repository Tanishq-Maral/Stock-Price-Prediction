import os
from pathlib import Path

# Suppress TensorFlow and oneDNN warnings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging (0=all, 1=info, 2=warning, 3=error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (disable GPU warnings if not available)

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus


def build_mysql_url():
	url = os.getenv("MYSQL_URL")
	if url:
		return url
	user = os.getenv("MYSQL_USER", "root")
	pwd = os.getenv("MYSQL_PASS", "root")
	host = os.getenv("MYSQL_HOST", "localhost")
	port = os.getenv("MYSQL_PORT", "3306")
	db = os.getenv("MYSQL_DB", "stock_prediction")
	return f"mysql+pymysql://{user}:{quote_plus(pwd)}@{host}:{port}/{db}"