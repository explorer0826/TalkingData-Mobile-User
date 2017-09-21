# -*- coding: utf-8 -*-


from sklearn.model_selection import GridSearchCV


def get_best_model(model, X_train, y_train, params, cv=5):
    """
        交叉验证获取最优模型
        默认5折交叉验证
    """
    clf = GridSearchCV(model, params, cv=cv)
    clf.fit(X_train, y_train)
    return clf.best_estimator_
