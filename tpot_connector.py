# tpot_connector.py

import pandas as pd
import joblib

# These can be globals, file-backed, or Streamlit session_state
_tpot_model = None
_X_train = None
_y_train = None

def set_latest_model_and_data(model, X_train, y_train):
    global _tpot_model, _X_train, _y_train
    _tpot_model = model
    _X_train = X_train
    _y_train = y_train

def get_latest_model_and_data():
    return _tpot_model, _X_train, _y_train
