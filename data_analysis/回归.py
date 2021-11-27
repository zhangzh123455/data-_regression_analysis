import pandas as pd
import os
import pymysql
import pyecharts.charts as pyc
import pyecharts.options as opts
import pyecharts.globals as glbs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing as ppcs
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import LinearSVR
from sklearn import model_selection

# dff = pd.read_csv('data.csv')
# data = dff.sort_values(by="shopnum", ascending=False)
# data.to_csv('data.csv')

# milk_sql = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='zzh223236', database='milktea', charset='utf8')
#
# df = pd.read_sql("select city,AVG(avgprice) as AVGprice,count(shopid) as shopnum from milktea_shop where isMain=1 and record=1 group by city ;", con=milk_sql)
# milk_sql.close()
# sns.set(style='whitegrid', context='notebook')   #style控制默认样式,context控制着默认的画幅大小
# sns.pairplot(df, size=7)
# plt.savefig('x.png')
#
# sns.heatmap(corr, cmap='GnBu_r', square=True, annot=True)
# plt.savefig('xx.png')

df = pd.read_csv('data.csv', usecols=[1])
citys = df['city']
citys = citys.values.tolist()

def plot_corr_matrix(df):
    cm = df.corr()
    value = [[i, j, round(cm[x][y], 4)] for i, x in enumerate(cm.index) for j, y in enumerate(cm.columns)]
    heatmap = pyc.HeatMap(
        init_opts=opts.InitOpts(theme=glbs.ThemeType.DARK, width="360px", height="360px", bg_color='#1a1c1d')
    ).add_xaxis(list(cm.index)
                ).add_yaxis('相关系数', list(cm.columns), value
                            ).set_series_opts(
        label_opts=opts.LabelOpts(is_show=False)
    ).set_global_opts(
        legend_opts=opts.LegendOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
        visualmap_opts=opts.VisualMapOpts(
            min_=-1, max_=1, precision=2,
            range_color=["#0000FF", "#FFFFFF", "#B50A24"],
            orient='horizontal', pos_top='1%', pos_left='center',
            item_width='12px', item_height='200px'
        )
    )
    heatmap.render('heatmap.html')


def plot_predict(model, features, labels, title=''):
    '''绘制拟合曲线以及残差'''
    pred = model.predict(features)
    residual = - (labels - pred)
    res_up = residual.map(lambda x: x if x >= 0 else None)
    res_dw = residual.map(lambda x: x if x < 0 else None)
    init_options = opts.InitOpts(theme=glbs.ThemeType.WALDEN, bg_color='#1a1c1d', width='700px', height='360px')
    line = pyc.Line(init_opts=init_options
        ).add_xaxis(citys
        ).add_yaxis('truth', [round(l, 2) for l in labels], symbol_size=6
        ).add_yaxis(
            'prediction', [round(p, 2) for p in pred],
            itemstyle_opts=opts.ItemStyleOpts(color='#A35300')
        ).set_series_opts(label_opts=opts.LabelOpts(is_show=False)
        ).set_global_opts(title_opts=opts.TitleOpts(title=title, subtitle='R²='+str(model.score(features, labels))),
                          toolbox_opts=opts.ToolboxOpts(feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(type_=['stack', 'tiled']), data_zoom=None, brush=None)),
                          datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=40))
    bar = pyc.Bar(init_opts=init_options
        ).add_xaxis(citys
        ).add_yaxis(
            'residual+', [round(r, 2) for r in res_up], bar_width='30%',
            itemstyle_opts=opts.ItemStyleOpts(color='#DA6964')
        ).add_yaxis(
            'residual-', [round(r, 2) for r in res_dw], bar_width='30%',
            itemstyle_opts=opts.ItemStyleOpts(color='#6F9F71')
        ).set_series_opts(label_opts=opts.LabelOpts(is_show=False)
        )
    line.overlap(bar)
    try:
        os.remove(f'{title}.html')
    except:
        pass
    line.render(f'{title}.html')


def print_coefficients(model):
    text = f'y = {model.intercept_}'
    for i, c in enumerate(model.coef_):
        text += f' + {c} * X{i+1}'
    print('\n'+'-'*60+'\n' + text + '\n'+'-'*60+'\n')


if __name__ == '__main__':
    df1 = pd.read_csv('data.csv', usecols=[1, 2, 3, 4, 5, 6, 7])
    print(df1)
    corr = df1.corr()
    selected = ['GDP', 'population', 'latitude', 'longitude']
    y_value1 = 'shopnum'
    y_value2 = 'AVGprice'
    features = df1[selected]
    labels1 = df1[y_value1]
    labels2 = df1[y_value2]
    plot_corr_matrix(df1)

    # 线性回归
    # lr = linear_model.LinearRegression()
    # lr.fit(features, labels2)
    # plot_predict(lr, features, labels2, 'Linear')
    # print_coefficients(lr)
    # print(lr.score(features, labels2))

    # 岭回归
    # ridge = linear_model.Ridge(alpha=1)
    # ridge.fit(features, labels1)
    # plot_predict(ridge, features, labels1, 'Ridge')
    # print_coefficients(ridge)
    # print(ridge.score(features, labels1))




    poly_feat = ppcs.PolynomialFeatures(degree=2, include_bias=False).fit_transform(features)
    # print(poly_feat)
    # 这里有四个变量，交互项 c(4, 2) = 6，加上平方项 4 个、一次项 4 个、变成 14 个特征
    # 因为要得出回归方程，确定一下交互项的排序（文档没写...）
    # df = pd.DataFrame({'x1': [3], 'x2': [5], 'x3': [7], 'x4': [11]})
    # print(df)
    # print(ppcs.PolynomialFeatures(degree=2, include_bias=False).fit_transform(df))
    # 应用多项式线性回归
    poly_lr = linear_model.LinearRegression()
    poly_lr.fit(poly_feat, labels1)
    plot_predict(poly_lr, poly_feat, labels1, y_value1)
    print_coefficients(poly_lr)
    print(poly_lr.score(poly_feat, labels1))
    # print(poly_lr.coef_)
    # print(poly_lr.intercept_)
    # for i in poly_feat:
    #     result = 0
    #     for j in range(14):
    #         result += i[j] * poly_lr.coef_[j]
    #     result += poly_lr.intercept_
    #     print(result)

    poly_lr = linear_model.LinearRegression()
    poly_lr.fit(poly_feat, labels2)
    plot_predict(poly_lr, poly_feat, labels2, y_value2)
    print_coefficients(poly_lr)
    print(poly_lr.score(poly_feat, labels2))

    scores = []
    for train, test in model_selection.KFold(5).split(poly_feat):
        plr = linear_model.LinearRegression()
        plr.fit(poly_feat[train], labels1[train])
        s = metrics.r2_score(labels1[test], plr.predict(poly_feat[test]))
        scores.append(s)
    plt.plot(scores)
    plt.show()
    # 明显的低偏差高方差，这里相当于用 5% 的【非随机采样】来预测整体（中国一共六百多个城市）
    # 而且可用的特征不多，能不能泛化看天意...
    model = {
        'label': y_value1,
        'columns': list(features.columns),
        'poly-degree': 2,
        'intercept': float(poly_lr.intercept_),
        'coefs': [float(c) for c in poly_lr.coef_]
    }