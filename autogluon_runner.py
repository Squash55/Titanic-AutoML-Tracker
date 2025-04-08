# autogluon_runner.py

import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from tpot_connector import _tpot_cache


def train_autogluon_model(X_train, y_train, label_column='target', output_dir='autogluon_models/'):
    # Combine X and y into one DataFrame
    train_data = pd.concat([X_train.copy(), y_train.copy()], axis=1)
    train_data.columns = list(X_train.columns) + [label_column]

    predictor = TabularPredictor(label=label_column, path=output_dir)
    predictor.fit(train_data)

    # Store model in cache for SHAP comparison and prediction
    _tpot_cache['all_models'] = _tpot_cache.get('all_models', {})
    _tpot_cache['all_models']['AutoGluon'] = predictor

    return predictor


def get_autogluon_predictor():
    if os.path.exists('autogluon_models/'): 
        return TabularPredictor.load('autogluon_models/')
    return None
