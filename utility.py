import numpy as np
from sklearn.ensemble import RandomForestRegressor


def rand_search_reg(X_train, y_train, X_test, y_test,
                            params, runs=16, reg=RandomForestRegressor(random_state=2, n_jobs=-1)):

    from sklearn.model_selection import RandomizedSearchCV

    rand_reg = RandomizedSearchCV(reg, params, 
                n_iter=runs, cv=10, random_state=2, n_jobs=-1,
                scoring="neg_mean_squared_error")

    # Fit rand_reg on X_train and y_train
    rand_reg.fit(X_train, y_train)

    # Extract the best estimator
    best_model = rand_reg.best_estimator_

    # Extract best params
    best_params = rand_reg.best_params_
    print(f"Best Params: {best_params}")

    # Compute the best score
    best_score = np.sqrt(-rand_reg.best_score_)
    print(f"Training score: {np.round(best_score, 3)}")

    # Predict test set labels
    y_pred = best_model.predict(X_test)

    # Compute rmse test
    from sklearn.metrics import mean_squared_error
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test set score: {rmse_test}")

    