import time

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    start_time = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end_time = time.time()

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    result = {
        "模型": model_name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "训练与预测耗时(s)": end_time - start_time
    }

    return result, y_pred