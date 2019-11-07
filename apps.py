from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from utils import create_model, load_config, get_logger
from data_utils import load_word2vec, input_from_line
from model import Model
import tensorflow as tf

app = Flask(__name__)
api = Api(app)

TODOS = {
    'single_sentence': {'task': 'do NER for a single sentence'},
    'list_sentence': {'task': 'do NER for a list of sentences'},

}


def abort_if_todo_doesnt_exist(todo_id):
    """Abort request if todo_id does not exist in TODOS"""
    if todo_id not in TODOS:
        abort(404, message="Todo {} doesn't exist".format(todo_id))


# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('task')


# http://127.0.0.1:5000/todos/single_sentence?task=%E4%B8%AD%E5%9B%BD%E7%89%9B%E9%80%BC
def evaluate_line(user_query):
    config = load_config("config_file")
    logger = get_logger("train.log")
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open("maps.pkl", "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, "ckpt", load_word2vec, config, id_to_char, logger)

        try:
            line = user_query
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            print(result)
        except Exception as e:
            logger.info(e)

    return result


def evaluate_lines(user_query):
    config = load_config("config_file")
    logger = get_logger("train.log")
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open("maps.pkl", "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, "ckpt", load_word2vec, config, id_to_char, logger)

        try:
            results = []
            lines = user_query.split('<SEP>')
            for item in lines:
                result = model.evaluate_line(sess, input_from_line(item, char_to_id), id_to_tag)
                results.append(result)
            print(result + '\n')
        except Exception as e:
            logger.info(e)


    return results


# Todo
# shows a single todo item and lets you updatae or delete a todo item
class Todo(Resource):
    def get(self, todo_id):

        abort_if_todo_doesnt_exist(todo_id)

        args = parser.parse_args()
        user_query = args['task']
        if todo_id == 1:
            results = evaluate_line(user_query)
        else:
            results = evaluate_lines(user_query)

        results = {"results": results}

        return jsonify(results)

    def delete(self, todo_id):

        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        # 204: SUCCESS; NO FURTHER CONTENT
        return '', 204

    def put(self, todo_id):

        # parser
        abort_if_todo_doesnt_exist(todo_id)

        args = parser.parse_args()
        task = {'task': args['task']}
        TODOS[todo_id] = task
        # 201: CREATED
        return task, 201


# TodoList
# shows a list of all todos, and lets you POST to add new tasks
class TodoList(Resource):
    def get(self):
        return TODOS

    def post(self):
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
        todo_id = 'todo%i' % todo_id
        TODOS[todo_id] = {'task': args['task']}
        # 201: CREATED
        return TODOS[todo_id], 201


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(TodoList, '/todos')
api.add_resource(Todo, '/todos/<todo_id>')

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)
