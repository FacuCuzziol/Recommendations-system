import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import render_template
import pickle
from scipy import spatial 

print('Server corriendo a todo vapor!')


app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    encoded_id_to_check=int(data['item_1'])

    dist, encoded_id = tree.query(embeddings[encoded_id_to_check], k=k_elements+1)

    df_most_similar=pd.DataFrame({'dist':dist, 'encoded':encoded_id})
    

    df_most_similar=df_most_similar[df_most_similar['encoded']!=encoded_id_to_check].copy()

    df_most_similar['rank']=list(range(1, k_elements+1))


    similar_item_id=le_x.inverse_transform(df_most_similar['encoded'])
    df_most_similar['item_id']=similar_item_id
    
    # Input item
    df_item_in=df_items_selec[df_items_selec['item_id'].isin(le_x.inverse_transform([encoded_id_to_check]))][['item_id','title', 'domain_id', 'price']]

    # Target items (recommendations)
    df_top_items=df_items_selec[df_items_selec['item_id'].isin(df_most_similar['item_id'])][['item_id','title', 'domain_id', 'price']]

    df_items_rel=pd.merge(df_top_items, df_most_similar, on='item_id').sort_values('rank')
    

    
    # making a json for our api
    data_res={
    'item_base': 'Item ID: ' + str(df_item_in.iloc[0]['item_id']) + ' | ' + str(df_item_in.iloc[0]['title']), 
    
    'item_reco_1': 'Item ID: ' + str(df_items_rel['item_id'][0]) + ' | Encoded: ' + str(df_items_rel['encoded'][0]) + ' | ' + str(df_items_rel['title'][0]), 
    'item_reco_2': 'Item ID: ' + str(df_items_rel['item_id'][1]) + ' | Encoded: ' + str(df_items_rel['encoded'][1]) + ' | ' + str(df_items_rel['title'][1]), 
    'item_reco_3': 'Item ID: ' + str(df_items_rel['item_id'][2]) + ' | Encoded: ' + str(df_items_rel['encoded'][2]) + ' | ' + str(df_items_rel['title'][2])


    
    }

    return data_res


if __name__ == "__main__":

    print("**** Running Keras Model ****")
    
    
    # Load the model
    model_final = keras.models.load_model('meli_files/final_modelNEW.h5')

    # Load selected item's metadata
    prep_files='meli_files/prep_files.pkl'

    with open(prep_files, "rb") as f:
        df_items_selec, le_x, embeddings=pickle.load(f)
    
    #tree parameters
    tree = spatial.KDTree(embeddings)
    k_elements=3


    app.run()