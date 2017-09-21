# -*- coding: utf-8 -*-


import pandas as pd
import os
from pd_tools import split_train_test, get_part_data
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
from ml_tools import get_best_model
from sklearn.metrics import log_loss
from sklearn.feature_selection import VarianceThreshold

# 数据集变量声明
dataset_path = './dataset'
gender_age_filename = 'gender_age.csv'
phone_brand_device_model_filename = 'phone_brand_device_model.csv'
events_filename = 'events.csv'
app_events_filename = 'app_events.csv'
app_labels_filename = 'app_labels.csv'
label_categories_filename = 'label_categories.csv'

train_gender_age_filename = 'gender_age_train.csv'
test_gender_age_filename = 'gender_age_test.csv'

is_first_run = False


def run_main():
    """
        主函数
    """
    if is_first_run:
        # 1. 分割数据集
        print('分割数据集')
        all_gender_age = pd.read_csv(os.path.join(dataset_path, gender_age_filename))
        df_train, df_test = split_train_test(all_gender_age)
        # 查看训练集测试集基本信息
        print('训练集中各类的数据个数：', df_train.groupby('group').size())
        print('测试集中各类的数据个数：', df_test.groupby('group').size())

        # 保存分割的数据集
        df_train.to_csv(os.path.join(dataset_path, train_gender_age_filename),
                        index=False)
        df_test.to_csv(os.path.join(dataset_path, test_gender_age_filename),
                       index=False)

    # 2. 加载数据
    print('加载数据')
    # 加载数据
    gender_age_train = pd.read_csv(os.path.join(dataset_path, train_gender_age_filename),
                                   index_col='device_id')
    gender_age_test = pd.read_csv(os.path.join(dataset_path, test_gender_age_filename),
                                  index_col='device_id')

    # 选取部分数据用于实验
    percent = 0.1
    gender_age_train = get_part_data(gender_age_train, percent=percent)
    gender_age_test = get_part_data(gender_age_test, percent=percent)

    phone_brand_device_model = pd.read_csv(os.path.join(dataset_path, phone_brand_device_model_filename))
    # 去掉重复数据
    phone_brand_device_model = phone_brand_device_model.drop_duplicates('device_id').set_index('device_id')

    events = pd.read_csv(os.path.join(dataset_path, events_filename),
                         usecols=['device_id', 'event_id'], index_col='event_id')
    app_events = pd.read_csv(os.path.join(dataset_path, app_events_filename),
                             usecols=['event_id', 'app_id'])
    # app_labels = pd.read_csv(os.path.join(dataset_path, app_labels_filename))

    # 3. 特征工程
    # 3.1 手机品牌特征
    # 使用LabelEncoder将类别转换为数字
    brand_label_encoder = LabelEncoder()
    brand_label_encoder.fit(phone_brand_device_model['phone_brand'].values)
    phone_brand_device_model['brand_label_code'] = \
        brand_label_encoder.transform(phone_brand_device_model['phone_brand'].values)
    gender_age_train['brand_label_code'] = phone_brand_device_model['brand_label_code']
    gender_age_test['brand_label_code'] = phone_brand_device_model['brand_label_code']

    # 使用OneHotEncoder将数字转换为OneHot码
    brand_onehot_encoder = OneHotEncoder()
    brand_onehot_encoder.fit(phone_brand_device_model['brand_label_code'].values.reshape(-1, 1))
    tr_brand_feat = brand_onehot_encoder.transform(gender_age_train['brand_label_code'].values.reshape(-1, 1))
    te_brand_feat = brand_onehot_encoder.transform(gender_age_test['brand_label_code'].values.reshape(-1, 1))

    print('[手机品牌]特征维度：', tr_brand_feat.shape[1])

    # 3.2 手机型号特征
    # 合并手机品牌与型号字符串
    phone_brand_device_model['brand_model'] = \
        phone_brand_device_model['phone_brand'].str.cat(phone_brand_device_model['device_model'])

    # 使用LabelEncoder将类别转换为数字
    model_label_encoder = LabelEncoder()
    model_label_encoder.fit(phone_brand_device_model['brand_model'].values)
    phone_brand_device_model['brand_model_label_code'] = \
        model_label_encoder.transform(phone_brand_device_model['brand_model'].values)
    gender_age_train['brand_model_label_code'] = phone_brand_device_model['brand_model_label_code']
    gender_age_test['brand_model_label_code'] = phone_brand_device_model['brand_model_label_code']

    # 使用OneHotEncoder将数字转换为OneHot码
    model_onehot_encoder = OneHotEncoder()
    model_onehot_encoder.fit(phone_brand_device_model['brand_model_label_code'].values.reshape(-1, 1))
    tr_model_feat = model_onehot_encoder.transform(gender_age_train['brand_model_label_code'].values.reshape(-1, 1))
    te_model_feat = model_onehot_encoder.transform(gender_age_test['brand_model_label_code'].values.reshape(-1, 1))

    print('[手机型号]特征维度：', tr_model_feat.shape[1])

    # 3.3 安装app特征
    device_app = app_events.merge(events, how='left', left_on='event_id', right_index=True)
    # 运行app的总次数
    n_run_s = device_app['app_id'].groupby(device_app['device_id']).size()

    # 运行app的个数
    n_app_s = device_app['app_id'].groupby(device_app['device_id']).nunique()

    gender_age_train['n_run'] = n_run_s
    gender_age_train['n_app'] = n_app_s

    # 填充缺失数据
    gender_age_train['n_run'].fillna(0, inplace=True)
    gender_age_train['n_app'].fillna(0, inplace=True)

    gender_age_test['n_run'] = n_run_s
    gender_age_test['n_app'] = n_app_s

    # 填充缺失数据
    gender_age_test['n_run'].fillna(0, inplace=True)
    gender_age_test['n_app'].fillna(0, inplace=True)

    tr_run_feat = gender_age_train['n_run'].values.reshape(-1, 1)
    tr_app_feat = gender_age_train['n_app'].values.reshape(-1, 1)

    te_run_feat = gender_age_test['n_run'].values.reshape(-1, 1)
    te_app_feat = gender_age_test['n_app'].values.reshape(-1, 1)

    # 3.4 合并所有特征
    tr_feat = np.hstack((tr_brand_feat.toarray(), tr_model_feat.toarray(), tr_run_feat, tr_app_feat))
    te_feat = np.hstack((te_brand_feat.toarray(), te_model_feat.toarray(), te_run_feat, te_app_feat))
    print('特征提取结束')
    print('每个样本特征维度：', tr_feat.shape[1])

    # 3.5 特征范围归一化
    scaler = StandardScaler()
    tr_feat_scaled = scaler.fit_transform(tr_feat)
    te_feat_scaled = scaler.transform(te_feat)

    # 3.6 特征选择
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    tr_feat_scaled_sel = sel.fit_transform(tr_feat_scaled)
    te_feat_scaled_sel = sel.transform(te_feat_scaled)

    # 3.7 PCA降维操作
    pca = PCA(n_components=0.95)  # 保留95%共享率的特征向量
    tr_feat_scaled_sel_pca = pca.fit_transform(tr_feat_scaled_sel)
    te_feat_scaled_sel_pca = pca.transform(te_feat_scaled_sel)
    print('特征处理结束')
    print('处理后每个样本特征维度：', tr_feat_scaled_sel_pca.shape[1])

    # 4 为数据添加标签
    group_label_encoder = LabelEncoder()
    group_label_encoder.fit(gender_age_train['group'].values)
    y_train = group_label_encoder.transform(gender_age_train['group'].values)
    y_test = group_label_encoder.transform(gender_age_test['group'].values)

    # 5. 训练模型
    # 5.1 逻辑回归模型
    print('训练逻辑回归模型...')
    lr_param_grid = [
        {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}
    ]
    lr_model = LogisticRegression()
    best_lr_model = get_best_model(lr_model,
                                   tr_feat_scaled_sel_pca, y_train,
                                   lr_param_grid, cv=3)
    y_pred_lr = best_lr_model.predict_proba(te_feat_scaled_sel_pca)

    # 5.2 SVM
    print('训练SVM模型...')
    svm_param_grid = [
        {'C': [1e-2, 1e-1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # 设置probability=True用于输出预测概率
    svm_model = svm.SVC(probability=True)
    best_svm_model = get_best_model(svm_model,tr_feat_scaled_sel_pca, y_train,
                                    svm_param_grid, cv=3)
    y_pred_svm = best_svm_model.predict_proba(te_feat_scaled_sel_pca)

    # 6. 查看结果
    print('逻辑回归模型 logloss:', log_loss(y_test, y_pred_lr))
    print('SVM logloss:', log_loss(y_test, y_pred_svm))


if __name__ == '__main__':
    run_main()
