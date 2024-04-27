from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error


def evaluate_model_freq(pipe, X_test, y_test):
    """
    """
    proba = pipe.predict_proba(X_test)
    matrix = confusion_matrix(y_test, pipe.predict(X_test))
    report = classification_report(y_test, pipe.predict(X_test))
    return proba, matrix, report 

def evaluate_model_cost(pipe, X_test, y_test):
    """
    """
    mse = mean_squared_error(y_test, pipe.predict(X_test))
    return mse