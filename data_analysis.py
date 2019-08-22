import matplotlib.pyplot as plt
import pandas as pd #数据分析
import matplotlib
data_train = pd.read_csv(r"C:\Users\daigang\Desktop\train.csv")
# print(data_train.head())
# print(data_train.info())
# print(data_train.describe())


myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\AdobeHeitiStd-Regular.otf')
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
# data_train.Survived.value_counts().plot(kind='bar')# 柱状图
# plt.title(u"获救情况 (1为获救)",fontproperties = myfont) # 标题
# plt.ylabel(u"人数",fontproperties = myfont)
#
# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"人数",fontproperties = myfont)
# plt.title(u"乘客等级分布",fontproperties = myfont)
#
# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"年龄",fontproperties = myfont)                         # 设定纵坐标名称
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"按年龄看获救分布 (1为获救)",fontproperties = myfont)
#
#
# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')#密度图
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄",fontproperties = myfont)# plots an axis lable
# plt.ylabel(u"密度",fontproperties = myfont)
# plt.title(u"各等级的乘客年龄分布",fontproperties = myfont)
# plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best',prop=myfont) # sets our legend for our graph.
#
#
# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数",fontproperties = myfont)
# plt.ylabel(u"人数",fontproperties = myfont)
# plt.show()


#看看各乘客等级的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
# df.plot(kind='bar', stacked=True)#对比柱状图
# plt.title(u"各乘客等级的获救情况",fontproperties = myfont)
# plt.xlabel(u"乘客等级",fontproperties = myfont)
# plt.ylabel(u"人数",fontproperties = myfont)
# plt.legend(loc='best',prop=myfont)
# plt.show()

# 看看各性别的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
# m = data_train.Sex[data_train.Sex=='male'].value_counts()
# print("男性总数",m)
#
# f = data_train.Sex[data_train.Sex=='female'].value_counts()
# print("女性总数",f)
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# print("男性获救情况 ",Survived_m)
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# print("女性获救情况",Survived_f)
# df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
# df.plot(kind='bar', stacked=True)
# plt.title(u"按性别看获救情况",fontproperties = myfont)
# plt.xlabel(u"获救情况",fontproperties = myfont)
# plt.ylabel(u"人数",fontproperties = myfont)
# plt.legend(loc='best',prop=myfont)
# plt.show()

#然后我们再来看看各种舱级别情况下各性别的获救情况
# fig=plt.figure()
# fig.set(alpha=0.65) # 设置图像透明度，无所谓
# plt.title(u"根据舱等级和性别的获救情况",fontproperties = myfont)
#
# ax1=fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().sort_index().plot(kind='bar', label="female highclass", color='#FA2479')
# ax1.set_xticks([0,1])
# ax1.set_xticklabels([u"未获救", u"获救"], rotation=0,fontproperties = myfont)
# ax1.legend([u"女性/高级舱"], loc='best',prop=myfont)
#
# ax2=fig.add_subplot(142, sharey=ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().sort_index().plot(kind='bar', label='female, low class', color='pink')
# ax2.set_xticklabels([u"未获救", u"获救"], rotation=0,fontproperties = myfont)
# plt.legend([u"女性/低级舱"], loc='best',prop=myfont)
#
# ax3=fig.add_subplot(143, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().sort_index().plot(kind='bar', label='male, high class',color='lightblue')
# ax3.set_xticklabels([u"未获救", u"获救"], rotation=0,fontproperties = myfont)
# plt.legend([u"男性/高级舱"], loc='best',prop=myfont)
#
# ax4=fig.add_subplot(144, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().sort_index().plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels([u"未获救", u"获救"], rotation=0,fontproperties = myfont)
# plt.legend([u"男性/低级舱"], loc='best',prop=myfont)
#
# plt.show()

# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'未获救':Survived_0,u'获救':Survived_1})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各登船港口乘客的获救情况",fontproperties=myfont)
# plt.xlabel(u"登船港口",fontproperties=myfont)
# plt.ylabel(u"人数",fontproperties=myfont)
# plt.legend(loc='best',prop=myfont)
# plt.show()

# gg = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(gg.count()['PassengerId'])
# print(df)
#
# gp = data_train.groupby(['Parch','Survived'])
# df = pd.DataFrame(gp.count()['PassengerId'])
# print(df)

# print(data_train.Cabin.value_counts())

# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df=pd.DataFrame({u'have':Survived_cabin, u'no':Survived_nocabin}).transpose()
# df.plot(kind='bar', stacked=True)
# plt.title(u"按Cabin有无看获救情况",fontproperties=myfont)
# plt.xlabel(u"Cabin有无",fontproperties=myfont)
# plt.ylabel(u"人数",fontproperties=myfont)
# plt.legend(loc='best',prop=myfont)
# plt.show()