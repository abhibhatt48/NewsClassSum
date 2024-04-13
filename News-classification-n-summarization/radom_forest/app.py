from flask import Flask, request
from main import prediction
app = Flask(__name__)



@app.route('/classification',methods=["POST"])
def hello_world():
    data = request.json
    news_title = str(data["title"])
    result = prediction(news_title)[0]
    print(result)
    if result == 1:
        return "Real news"
    else:
        return "Fake news"


# app
if __name__ == "__main__":
    app.run(debug=True)