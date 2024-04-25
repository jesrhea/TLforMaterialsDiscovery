import pandas as pd
import logging , os
logging.captureWarnings(True)
from joblib import load, dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

def train_test_data(X,y):
    # training data size : test data size = 0.8 : 0.2
    # fixed seed using the random_state parameter, so it always has the same split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0)

    # normalise
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # save original X
    X_train_ori = X_train
    X_test_ori = X_test
    # transform data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return(X_train, y_train, X_test, y_test)

def rf_randomsearch(X_train, y_train):
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 20, 30],
        'bootstrap': [True, False],
       # 'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=0,n_estimators=128),
        param_distributions=param_grid, 
        n_iter=100, 
        cv=5, 
        scoring='accuracy'
    )
    
    # Fit the RandomizedSearchCV object to the training data
    random_search.fit(X_train, y_train)

    # re-train a model using best hyperparameters
    rf_rs = RandomForestRegressor(**random_search.best_params_, random_state=0)
    print('Best paramters: ', random_search.best_params_)

    return(rf_rs)

def rf_gridsearch(X_train, y_train):
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 20, 30]
    }

    # use 5-folds cross validation during grid searching
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=0,n_estimators=128),
        param_grid,
        cv=5
    )
    grid_search.fit(X_train, y_train)

    # re-train a model using best hyperparameters
    rf_gs = RandomForestRegressor(**grid_search.best_params_, random_state=0)
    print('Best paramters: ', grid_search.best_params_)
    
    
    return(rf_gs)

def concate_df(df1,*df2):
    result = [df1,*df2]
    return(pd.concat(result, axis=1))

    
def train_test_model(model, X_train, y_train, X_test, y_test, job_name):
    """
    Function that trains a model, and tests it.
    Inputs: sklearn model, train_data, test_data
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Calculate RMSE on training
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    model_train_mse = mean_squared_error(y_train, y_pred_train)
    model_test_mse = mean_squared_error(y_test, y_pred_test)
    model_train_rmse = model_train_mse ** 0.5
    model_test_rmse = model_test_mse ** 0.5
    model_train_mae = mean_absolute_error(y_train, y_pred_train)
    model_test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"RMSE on train set: {model_train_rmse:.4f}, and test set: {model_test_rmse:.4f}.\n")
    print(f"MAE on train set: {model_train_mae:.4f}, and test set: {model_test_mae:.4f}.\n")
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred_test))
    
    save_model(model, job_name)
    
    return(y_pred_train, y_pred_test)


def save_model(model, jn):
    dump(model, os.path.join('report_model_bank',jn+'.joblib'))

def xgb_gridsearch(X_train, y_train):
    # Define the hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.7, 1]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        XGBRegressor(random_state=0), 
        param_grid, 
        cv=5, 
        scoring='accuracy'
    )

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best set of hyperparameters and the corresponding score
    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    
    xgb_opt_model =  XGBRegressor(**grid_search.best_params_, random_state=0)
        
    return(xgb_opt_model)

def xgb_randomsearch(X_train, y_train):
    
    # Define the hyperparameter distributions
    param_dist = {
        'max_depth': stats.randint(3, 10),
        'learning_rate': stats.uniform(0.01, 0.1),
        'subsample': stats.uniform(0.5, 0.5),
        'n_estimators':stats.randint(50, 200)
    }

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        XGBRegressor(random_state=0),
        param_distributions=param_dist, 
        n_iter=10, 
        cv=5, 
        scoring='accuracy'
    )

    # Fit the RandomizedSearchCV object to the training data
    random_search.fit(X_train, y_train)

    # Print the best set of hyperparameters and the corresponding score
    print("Best set of hyperparameters: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)
    
    xgb_opt_model = XGBRegressor(**random_search.best_params_, random_state=0)
    
    return(xgb_opt_model)

if __name__ == '__main__':
    
    data = pd.read_csv('all_features_rm_volume.csv', index_col=0)
    k_data = data.iloc[:,-4:-1].to_numpy()
    data_top = pd.read_csv('X-features_top-he_rm-vol.csv', index_col=0)
    lcd_only = concate_df(data_top.iloc[:,:3],data_top.iloc[:,10:])
    
    byinspection_splits = [
        (0,.1),
        (.1,.25),
        (.25,70)
    ]
    
    a = pd.concat([lcd_only, data.iloc[:,-4:]], axis=1)
    
    X_0_01 = a.loc[a['kmean'].between(byinspection_splits[0][0], byinspection_splits[0][1])].iloc[:,:-4]
    y_0_01 = a.loc[a['kmean'].between(byinspection_splits[0][0], byinspection_splits[0][1])].iloc[:,-4:-1].to_numpy()

    X_01_025 = a.loc[a['kmean'].between(byinspection_splits[1][0], byinspection_splits[1][1])].iloc[:,:-4]
    y_01_025 = a.loc[a['kmean'].between(byinspection_splits[1][0], byinspection_splits[1][1])].iloc[:,-4:-1].to_numpy()

    X_025_70 = a.loc[a['kmean'].between(byinspection_splits[2][0], byinspection_splits[2][1])].iloc[:,:-4]
    y_025_70 = a.loc[a['kmean'].between(byinspection_splits[2][0], byinspection_splits[2][1])].iloc[:,-4:-1].to_numpy()

    log_kdata = pd.read_csv('log_kdata.csv').to_numpy()
    
    X_train, y_train, X_test, y_test = train_test_data(data_top, k_data)
        
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    
    #model = XGBRegressor(n_estimators=128, random_state=0)
    
    lcd_only_noopt = train_test_model(model, X_train, y_train, X_test, y_test, 'rf-n10_topdata_kdata_alldata')