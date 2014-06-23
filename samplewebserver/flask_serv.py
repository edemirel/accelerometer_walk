# coding: utf-8
import logging
from datetime import datetime

from flask import Flask, render_template


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


app = Flask(__name__)


@app.route("/")
def landhere():
    now = datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
        'title': 'Ege Demirel',
        'time': timeString
    }

    return render_template('main.html', **templateData)


@app.route("/test")
def landhere2():
    now = datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
        'title': 'Ege Demirel',
        'time': timeString
    }

    return render_template('test.html', **templateData)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
