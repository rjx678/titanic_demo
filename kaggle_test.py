import pandas as pd #数据分析
import numpy as np #科学计算
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,train_test_split
import matplotlib.pyplot as plt
import matplotlib

data_train = pd.read_csv(r"C:\Users\daigang\Desktop\train.csv")
myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\AdobeHeitiStd-Regular.otf')
# 使用 RandomForestRegressor 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # print(age_df)
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # print(known_age)
    # print(unknown_age)
    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000,n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
    #print(predictedAges)

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
# print(data_train)
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# print(df)

import sklearn.preprocessing as preprocessing
#对训练集数据进行标准化处理
scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1),age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1),fare_scale_param)
#print(df.head())#只列出前五项

from sklearn import linear_model

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

#print(clf)


data_test = pd.read_csv(r"C:\Users\daigang\Desktop\test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)
#print(df_test.head())

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

# result.to_csv(r"C:\Users\daigang\Desktop\logistic_regression_predictions.csv", index=False)

# print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))


#
# #简单看看打分情况
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# X = all_data.values[:,1:]
# y = all_data.values[:,0]
#
# # print(cross_val_score(clf, X, y, cv=5))
#
# # 分割数据，按照 训练数据:cv数据 = 7:3的比例
# split_train, split_cv = train_test_split(df, test_size=0.3, random_state=42)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# # 生成模型
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])
#
# # 对cross validation数据进行预测
#
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(cv_df.as_matrix()[:,1:])
#
# origin_data_train = pd.read_csv(r"C:\Users\daigang\Desktop\train.csv")
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
# #print(bad_cases.head(10))


from sklearn.model_selection import learning_curve
# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title,fontproperties = myfont)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数",fontproperties = myfont)
        plt.ylabel(u"得分",fontproperties = myfont)
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best",prop=myfont)

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


# plot_learning_curve(clf, u"学习曲线", X, y)
if __name__=="__main__":
    from sklearn.ensemble import BaggingClassifier
    from sklearn.svm import SVC
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到BaggingRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    bagging_clf = BaggingClassifier(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
    bagging_clf.fit(X, y)
    print("baggingclassifier is",cross_val_score(bagging_clf, X, y, cv=5))

    # model = SVC(kernel='linear',C=1,probability=True)
    # model.fit(X,y)
    # print(cross_val_score(model,X,y,cv=5))

    bagging_clf = SVC(kernel='poly', C=0.01, gamma=0.6, probability=True)
    bagging_clf.fit(X, y)
    print("0.6 is",cross_val_score(bagging_clf, X, y, cv=5))
    sum = 0
    for i in range(5):
        sum += cross_val_score(bagging_clf, X, y, cv=5)[i]
    print(sum)





    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    # model = SVC(probability=True)
    # # param_grid = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10,100]},
    #               # {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10,100], 'gamma': [0.5]},
    # param_grid = [{'kernel': ['poly'], 'C': [0.01],
    #                'gamma': [0.1,0.2,0.3,0.4,0.5], 'degree': [3]}
    #               ]
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10,
    #                     scoring='accuracy')
    # # 针对每个参数对进行了10次交叉验证。scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
    # grid.fit(X, y)
    #
    # print('网格搜索-度量记录：', grid.cv_results_)  # 包含每次训练的相关信息
    # print('网格搜索-最佳度量值:', grid.best_score_)  # 获取最佳度量值
    # print('网格搜索-最佳参数：', grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
    # print('网格搜索-最佳模型：', grid.best_estimator_)  # 获取最佳度量时的分类器模型




    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
    predictions = bagging_clf.predict(test)
    # print(predictions,type(predictions))
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
    # result.to_csv(r"C:\Users\daigang\Desktop\svm_classification_predictions.csv", index=False)
