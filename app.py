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


# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        result = evaluate_line(user_query)
        print(result)
        # create JSON object
        output = {'result': result}

        return jsonify(output)


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)
