import ai_app
from flask import Flask, json, render_template, request

api = Flask(__name__)

@api.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@api.route('/ai', methods=['GET'])
def get_companies():
  query_text = request.args.get('query')
  query = [query_text]
  response = ai_app.run_ai(query)
  
  return {
      "query": query,
      "response": response
  }

if __name__ == '__main__':
    api.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    api.run(debug=True)