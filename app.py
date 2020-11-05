from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from bokeh.plotting import figure, show
from bokeh.resources import CDN
from bokeh.embed import file_html, components
from bokeh.models import ColumnDataSource
from bokeh.palettes import Dark2_5 as palette
from bokeh.transform import dodge

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)
df = pd.read_excel('rawData_select.xlsx')
df_2019 = pd.read_excel('data2019_select.xlsx')

state_group = df.groupby('State', as_index=False)
count = state_group['Award Amount'].agg('count')
count = count.rename(columns={'Award Amount':'Tot'})

@app.route('/')
def index():

    def findTotal(df):
        yr_group = df.groupby('Award Year',as_index=False)
        award_tot = yr_group['Award Amount'].agg('sum')
        award_tot = award_tot.rename(columns={'Award Amount':'Annual Sum'})
        return award_tot
    def extractData(key, df):
        keySum = key + ' Sum'
        ind = df['Abstract'].str.contains(key, case=False, na=False)
        filter_df = df[ind]
        filter_tot = pd.DataFrame(filter_df.groupby('Award Year',as_index=False)['Award Amount'].sum())
        filter_tot = filter_tot.rename(columns={'Award Amount':keySum})
        return filter_tot
    def dataAnalysis():
        result = []
        result_2019 = []

        input = request.args.get('input')
        if input:
            keys = input.split(',')
        else:
            keys = ['machine learning']

        for key in keys:
            filter_tot = extractData(key, df)
            award_tot = findTotal(df)
            result_df = award_tot.merge(filter_tot,how='outer',on='Award Year')
            result_df = result_df.fillna(0)
            result_df['Award Proportion'] = result_df[key + ' Sum']/result_df['Annual Sum']*100
            result.append(result_df)


            filter_tot_2019 = extractData(key, df_2019)
            award_tot_2019 = findTotal(df_2019)
            result_df_2019 = award_tot_2019.merge(filter_tot_2019, how='outer',on='Award Year')
            result_df_2019 = result_df_2019.fillna(0)
            result_df_2019['Award Proportion'] = result_df_2019[key + ' Sum']/result_df_2019['Annual Sum']*100
            result_2019.append(result_df_2019)
        return result, result_2019, keys

    results,result_2019,keys = dataAnalysis()

    y_test = {}
    errors = {}
    i = 0
    for df_result in results:
        y = df_result['Award Proportion']
        X = df_result['Award Year']
        y_2019 = result_2019[i]['Award Proportion']
        errors[keys[i]] = {}

        X_trans = []
        for yr in X:
            X_trans.append([yr])

        est = GridSearchCV(
            DecisionTreeRegressor(), #KNeighborsRegressor - similar results
            {'max_depth':range(1,10)},
            cv = 2,
            n_jobs = 2,
        )
        est.fit(X_trans, y)

        pred = est.predict([[2019]])
        errors[keys[i]]['DecisionTree'] = abs(pred[0] - y_2019)/y_2019*100

        pipe = Pipeline([
            ('poly', PolynomialFeatures()),
            ('standard_scaler', StandardScaler()),
            ('linear_reg', LinearRegression())
        ])
        est2 = GridSearchCV(
            pipe,
            {'poly__degree': range(2,5)},
            cv = 2,
            n_jobs = 2,
        )
        est2.fit(X_trans, y)

        pred2 = est2.predict([[2019]])
        errors[keys[i]]['Polynomial'] = abs(pred2[0] - y_2019)/y_2019*100

        est3 = GridSearchCV(
            SVR(),
            {'C':[5,50,100], 'gamma':[0.1,0.15,0.2]},
            cv = 2,
            n_jobs = 2,
        )

        est3.fit(X_trans, y)
        pred3 = est3.predict([[2019]])
        errors[keys[i]]['SVR'] = abs(pred3[0] - y_2019)/y_2019*100

        y_comb = y.copy()
        y_comb.loc[len(y_comb)] = y_2019[0]
        y_test[keys[i]] = y_comb

        i += 1

    dt_error = {}
    poly_error = {}
    svr_error = {}
    for k in errors:
        dt_error[k] = errors[k]['DecisionTree'][0]
        poly_error[k] = errors[k]['Polynomial'][0]
        svr_error[k] = errors[k]['SVR'][0]

    def findKey(dicts, val):
        for k,v in dicts.items():
            if val == v[0]:
                return k

    X_transNew = X_trans + [[2019]]
    X_pred = [[2019],[2020],[2021],[2022]]
    y_pred = {}
    low_mode = {}
    for key in keys:
        low_error = min(dt_error[key], poly_error[key], svr_error[key])
        method = findKey(errors[key], low_error)
        low_mode[key] = method

        if method == 'DecisionTree':
            est = GridSearchCV(
            DecisionTreeRegressor(),
            {'max_depth':range(1,10)},
            cv = 2,
            n_jobs = 2,
            )
        elif method == 'Polynomial':
            pipe = Pipeline([
            ('poly', PolynomialFeatures()),
            ('standard_scaler', StandardScaler()),
            ('linear_reg', LinearRegression())
            ])

            est = GridSearchCV(
            pipe,
            {'poly__degree': range(2,5)},
            cv = 2,
            n_jobs = 2,
            )
        elif method == 'SVR':
            est = GridSearchCV(
            SVR(),
            {'C':[5,50,100], 'gamma':[0.1,0.15,0.2]},
            cv = 2,
            n_jobs = 2,
            )

        est.fit(X_transNew, y_test[key])
        y_pred[key] = est.predict(X_pred)

    x_bokeh = []
    for x in X_transNew:
        x_bokeh.append(x[0])

    x_bokeh2 = []
    for x in X_pred:
        x_bokeh2.append(x[0])

    p1 = figure(title = 'Award Proportion', x_axis_label = 'Year',
                y_axis_label = 'Award Proportion (%)',
                plot_width=600, plot_height=600)
    colors = list(palette)
    i = 0
    for result in results:
        y_bokeh = []
        for val in y_test[keys[i]]:
            y_bokeh.append(val)

        y_bokeh2 = []
        for val in y_pred[keys[i]]:
            y_bokeh2.append(val)

        p1.line(x_bokeh, y_bokeh, legend_label = keys[i],
                line_color = colors.pop(), width = 3)
        p1.line(x_bokeh2, y_bokeh2, legend_label = keys[i]+' prediction '+low_mode[keys[i]],
                line_color = 'red',line_dash='dashed')

        i += 1

    p1.title.text_font_size = '16px'
    p1.title.align = 'center'
    p1.legend.location = 'top_left'

    script1, div1 = components(p1)

    key_bokeh = []
    for k in dt_error.keys():
        key_bokeh.append(k)
    dt_bokeh = []
    for e in dt_error.values():
        dt_bokeh.append(e)
    poly_bokeh = []
    for e in poly_error.values():
        poly_bokeh.append(e)
    svr_bokeh = []
    for e in svr_error.values():
        svr_bokeh.append(e)

    data_bar = {'research': key_bokeh,
            'Decision Tree': dt_bokeh,
            'Polynomial': poly_bokeh,
            'SVR': svr_bokeh
            }
    source = ColumnDataSource(data=data_bar)
    colors = list(palette)

    p2 = figure(x_range=key_bokeh, title="Prediction Error",
               y_axis_label='Error (%)', plot_height=400, plot_width=600)

    p2.vbar(x=dodge('research', -0.25, range=p2.x_range), top='Decision Tree', width=0.2, source=source,
           color=colors.pop(), legend_label="Decision Tree")
    p2.vbar(x=dodge('research',  0.0,  range=p2.x_range), top='Polynomial', width=0.2, source=source,
           color=colors.pop(), legend_label="Polynomial")
    p2.vbar(x=dodge('research',  0.25,  range=p2.x_range), top='SVR', width=0.2, source=source,
           color=colors.pop(), legend_label="SVR")

    p2.title.text_font_size = '16px'
    p2.title.align = 'center'
    p2.legend.location = "top_left"
    p2.legend.orientation = "horizontal"
    p2.xgrid.grid_line_color = None

    script2, div2 = components(p2)

    html = render_template(
        'index.html',
        script1=script1,
        div1=div1,
        script2=script2,
        div2=div2
    )

    return html

if __name__ == '__main__':
    app.run(debug=True)
