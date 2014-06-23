# coding: utf-8
import logging
from datetime import datetime

from flask import Flask, render_template


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


app = Flask(__name__)


@app.context_processor
def timestamp():
    return {
        'time': datetime.now().isoformat()
    }


@app.route("/")
def index():
    return render_template('main.html')


@app.route("/test")
def test():
    return render_template('test.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
