import ai_app
from flask import Flask, json, render_template, request

api = Flask(__name__)

@api.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@api.route('/ai', methods=['GET'])
def get_companies():
  query_text = request.args.get('query')
  question_text = request.args.get('question')

  query = [query_text]
  records = ai_app.run_ai(query, question_text)
  
  return {
      "query": query,
      "records": records
  }

if __name__ == '__main__':
    api.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    api.run(debug=True, port=5000, host='0.0.0.0')