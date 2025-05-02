from flask import Flask, render_template, request
from modules.ir import run_ir
from modules.lda import run_lda, get_topic_detail, get_all_topics, search_word_in_topics
from modules.bertopic_module import run_bertopic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ir/form')
def ir_form():
    return render_template('ir_form.html')

@app.route('/lda/form')
def lda_form():
    return render_template('lda_form.html')

@app.route('/bertopic/form', methods=['GET', 'POST'])
def bertopic_form():
    from modules.bertopic_module import run_bertopic

    predicted_topic = None
    if request.method == 'POST':
        query = request.form['query']
        predicted_topic = run_bertopic(query)

    return render_template("bertopic_form.html", predicted_topic=predicted_topic)


def get_all_bertopic_topics():
    topics_info = bertopic_model.get_topic_info()
    return topics_info.to_dict(orient="records")


@app.route('/ir', methods=['POST'])
def ir():
    query = request.form['query']
    results = run_ir(query)
    return render_template('result.html', method='IR', results=results)

# @app.route('/lda', methods=['POST'])
# def lda():
#     query = request.form['query']
#     topic, distribution = run_lda(query)
#     return render_template('result.html', method='LDA', topic=topic, distribution=distribution)



@app.route('/lda/topics', methods=['GET', 'POST'])
def lda_topics():
    keyword = None
    search_results = []

    if request.method == 'POST':
        keyword = request.form['keyword']
        search_results = search_word_in_topics(keyword)

    topics = get_all_topics()
    return render_template(
        'lda_topics.html',
        topics=topics,
        keyword=keyword,
        search_results=search_results
    )


@app.route('/lda/topic/<int:topic_id>')
def lda_topic_detail(topic_id):
    word_scores = get_topic_detail(topic_id)
    return render_template('lda_topic_detail.html', topic_id=topic_id, word_scores=word_scores)

@app.route('/bertopic', methods=['POST'])
def bertopic():
    query = request.form['query']
    from modules.bertopic_module import run_bertopic
    topic, confidence, documents, topic_words = run_bertopic(query)
    return render_template(
        'result.html',
        method='BERTopic',
        topic=topic,
        confidence=confidence,
        documents=documents,
        topic_words=topic_words
    )


if __name__ == '__main__':
    print("✅ Server running at http://127.0.0.1:5000")
    app.run(debug=True)
