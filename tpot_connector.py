
# tpot_connector.py
_tpot_cache = {}

def update_tpot_cache(key, value):
    _tpot_cache[key] = value


# This module stores and shares the latest TPOT model and data for use in SHAP and Q&A panels.

latest_tpot_model = None
latest_X_train = None
latest_y_train = None
latest_X_test = None
latest_y_test = None
