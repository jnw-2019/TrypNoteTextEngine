from flask import Flask
from flask import redirect, request, jsonify, url_for
from flask_cors import CORS
import ldafunc
import sentimentfunc

app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/')
def works():
    params = { 'text' : "Works!" }
    return jsonify(params)

@app.route('/analyze', methods = ['POST'])
def analyze():
    # read json + reply
    data = request.get_json()
    results = ldafunc.runlda(data['postObj'])
    return jsonify({ 'results' : results })

@app.route('/sentiment', methods = ['POST'])
def sentiment():
    # read json + reply
    data = request.get_json()
    results = sentimentfunc.runmodel(data['postObj'])
    return results

if __name__ == '__main__':
    app.run()