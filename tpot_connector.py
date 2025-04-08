# tpot_connector.py

# Shared in-memory cache for model and data
_tpot_cache = {
    "model": None,
    "X_train": None,
    "y_train": None
}

def set_latest_model_and_data(model, X_train, y_train):
    """
    Store the latest trained model and data into shared memory.
    """
    _tpot_cache["model"] = model
    _tpot_cache["X_train"] = X_train
    _tpot_cache["y_train"] = y_train

def get_latest_model_and_data():
    """
    Retrieve the latest trained model and data from shared memory.
    """
    return (
        _tpot_cache.get("model", None),
        _tpot_cache.get("X_train", None),
        _tpot_cache.get("y_train", None)
    )
