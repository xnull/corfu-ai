import ai_app
from flask import Flask, json, render_template, request
import os

api = Flask(__name__)

@api.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@api.route('/ai', methods=['GET'])
def run_ai_proceessing():
  query_text = request.args.get('query')
  question_text = request.args.get('question')

  with open('questions.txt', 'a') as f:
    f.write(query_text + '\n')
    f.write(question_text + '\n')
    f.write('\n')

  query = [query_text]
  records = ai_app.run_ai(query, question_text)
  
  return {
      "query": query_text,
      "question": question_text,
      "records": records
  }

if __name__ == '__main__':
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' 

    api.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    api.run(debug=True, port=5005, host='0.0.0.0')