from flask import Flask, send_file, render_template
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../../')
from datk import visualizer
import pymysql


app = Flask(__name__)
host_addr = '0.0.0.0'
port_num = '8080'


@app.route('/example/index', methods=['GET'])
def test_index():
    return render_template('example/index.html')


@app.route('/example/login', methods=['GET', 'POST'])
def test_login():
    return render_template('example/login.html')


@app.route('/example/register', methods=['GET', 'POST'])
def test_register():
    return render_template('example/register.html')


@app.route('/example/forgot-password', methods=['GET', 'POST'])
def test_forgot_password():
    return render_template('example/forgot-password.html')


@app.route('/example/404', methods=['GET'])
def test_404():
    return render_template('example/404.html')


@app.route('/example/blank', methods=['GET'])
def test_blank():
    return render_template('example/blank.html')


@app.route('/example/tables', methods=['GET'])
def test_tables():
    return render_template('example/tables.html')


@app.route('/example/charts', methods=['GET'])
def test_charts():
    return render_template('example/charts.html')


@app.route('/example/show_figure_alone/<int:mean>_<int:var>_<int:bins>', methods=['GET'])
def show_figure_alone(mean, var, bins):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    xs = np.random.normal(mean, var, 100)
    visualizer.draw_pdf(ax, xs, bins=bins)
    img = BytesIO()
    fig.savefig(img, format='png', dpi=200)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/example/show_figure_in_html/<m_v_b>', methods=['GET'])
def show_figure_in_html(m_v_b):
    m, v, b = m_v_b.split('_')
    m, v, b = int(m), int(v), int(b)
    return render_template('example/show_figure_in_html.html', mean=m, var=v, bins=b, width=800, height=600)


@app.route('/example/show_table/<table>', methods=['GET'])
def show_table(table):
    conn = pymysql.connect(host='localhost', user='user1', password='12345678', db='test', charset='utf8')
    curs = conn.cursor()
    sql = "select * from " + str(table)
    curs.execute(sql)
    rows = curs.fetchall()
    ret = pd.read_sql_query(sql, conn)
    conn.close()
    return ret.to_html()


if __name__ == '__main__':
    app.run(host=host_addr, port=port_num, debug=True)
