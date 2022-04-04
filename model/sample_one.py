import os
import numpy as np
import pandas as pd
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")
 

def predict(userId_chosed, movieId_chosed):
    trained_datapath = "./"

    npy_path = os.path.join(trained_datapath, "cosine_sim.npy")
    assert os.path.exists(npy_path), "No data Found"
    cosine_sim = np.load(npy_path)

    serialized_model_path = os.path.join(trained_datapath, "dnmf_model_final.h5")
    assert os.path.exists(serialized_model_path), "No model Found"
    dnmf_model_final = keras.models.load_model(serialized_model_path)

    userIdChosed_vector = np.asarray([userId_chosed]).astype(np.int32)
    movieIdChosed_vector = np.asarray([movieId_chosed]).astype(np.int32)

    predicted_rating = dnmf_model_final.predict(
        [userIdChosed_vector, movieIdChosed_vector])[0][0]
        
    print("User#{} would give *{:2.2} to the Movie#{}".format(userId_chosed, predicted_rating, movieId_chosed))
    
    score_series = pd.Series(cosine_sim[movieId_chosed]).sort_values(ascending = False)   # similarity scores in descending order
    top_10_indices = list(score_series.iloc[1:11].index)   # to get the indices of top 10 most similar movies
    print("Recommand Top10:", top_10_indices)

if __name__ == "__main__":
    predict(1, 10)