import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


tf.random.set_seed(10)

def reduce_item_dim(df_ratings):
    df_user_item = df_ratings.pivot(
        index='userId', columns='movieId', values='rating')
    # reset movieId
    df_user_item = df_user_item.T.reset_index(drop=True).T
    # undo pivot/melt - compress data frame
    df_ratings_new = df_user_item \
        .reset_index('userId') \
        .melt(
            id_vars='userId',
            value_vars=df_user_item.columns,
            var_name='movieId',
            value_name='rating')
    # drop nan and final clean up
    return df_ratings_new.dropna().sort_values(['userId', 'movieId']).reset_index(drop=True)

def train(data, serialized_model_path):
    # calling the function
    df_ratings_reduced = reduce_item_dim(data)

    """calculating the exact numbers of different userIds and movieIds"""
    n_users = len(df_ratings_reduced["userId"].unique())
    n_movies = len(df_ratings_reduced["movieId"].unique())

    """number of latent factors that will be used in our model"""
    n_latent_factors_mf = 20
    n_latent_factors_mlp = 20

    """creating the input user and movie vectors"""
    userIds_vector = np.asarray(df_ratings_reduced.userId).astype(np.int32)
    movieIds_vector = np.asarray(df_ratings_reduced.movieId).astype(np.int32)
    ratings_vector = np.asarray(df_ratings_reduced.rating).astype(np.float32)


    """implementing the model architecture and fitting it to the input user and movie vectors"""
    # Users matrix factorization embedding path
    users_input = keras.layers.Input(shape=[1], dtype='int32', name="users_input")
    users_mf_embedding = keras.layers.Embedding(
        input_dim=n_users + 1,
        output_dim=n_latent_factors_mf,
        name='users_mf_embedding')
    users_flattened_mf = keras.layers.Flatten()(users_mf_embedding(users_input))

    # Users multi-layer perceptron embedding path
    users_mlp_embedding = keras.layers.Embedding(
        input_dim=n_users + 1,
        output_dim=n_latent_factors_mlp,
        name='users_mlp_embedding')
    users_flattened_mlp = keras.layers.Flatten()(users_mlp_embedding(users_input))

    # Movies matrix factorization embedding path
    movies_input = keras.layers.Input(
        shape=[1], dtype='int32', name="movies_input")
    movies_mf_embedding = keras.layers.Embedding(
        input_dim=n_movies + 1,
        output_dim=n_latent_factors_mf,
        name='movies_mf_embedding')
    movies_flattened_mf = keras.layers.Flatten()(movies_mf_embedding(movies_input))

    # Movies multi-layer perceptron embedding path
    movies_mlp_embedding = keras.layers.Embedding(
        input_dim=n_movies + 1,
        output_dim=n_latent_factors_mlp,
        name='movies_mlp_embedding')
    movies_flattened_mlp = keras.layers.Flatten()(
        movies_mlp_embedding(movies_input))

    # Dot product of users and movies matrix factorization embeddings
    interaction_matrix = keras.layers.Dot(name="interaction_matrix", axes=1)([
        movies_flattened_mf, users_flattened_mf])

    # Concatenation of users and movies multi-layer peceptron embeddings
    concatenation_vector = keras.layers.Concatenate(
        name="concatenation_vector")([movies_flattened_mlp, users_flattened_mlp])

    # Adding dense layers
    dense_1 = keras.layers.Dense(
        50, activation='elu', kernel_initializer="he_normal")(concatenation_vector)
    dense_2 = keras.layers.Dense(
        25, activation='elu', kernel_initializer="he_normal")(dense_1)
    dense_3 = keras.layers.Dense(
        12, activation='elu', kernel_initializer="he_normal")(dense_2)
    dense_4 = keras.layers.Dense(
        6, activation='elu', kernel_initializer="he_normal")(dense_3)
    dense_5 = keras.layers.Dense(
        3, activation='elu', kernel_initializer="he_normal")(dense_4)

    # concatenation of the matrix factorization and multi-layer perceptron parts
    final_concatenation = keras.layers.Concatenate(
        name="final_concatenation")([interaction_matrix, dense_5])

    # Adding the output layer
    output_layer = keras.layers.Dense(1)(final_concatenation)

    # Stitching input and output
    dnmf_model_final = keras.models.Model(
        inputs=[users_input, movies_input], outputs=output_layer)

    # Model compilation and saving of its implementation and weights
    dnmf_model_final.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(
        lr=0.01, momentum=0.9, nesterov=True, clipvalue=1.0), metrics=[keras.metrics.RootMeanSquaredError()])

    history = dnmf_model_final.fit(
        [userIds_vector, movieIds_vector], ratings_vector, epochs=1)
    
    dnmf_model_final.save(serialized_model_path)

    return history


def predict_one(userId_chosed, movieId_chosed):
    # predicting the rating that the userId_chosed would give to the movieId_chosed according to the DNMF model
    

    userIdChosed_vector = np.asarray([userId_chosed]).astype(np.int32)


    movieIdChosed_vector = np.asarray([movieId_chosed]).astype(np.int32)

    predicted_rating = dnmf_model_final.predict(
        [userIdChosed_vector, movieIdChosed_vector])
    return predicted_rating[0][0]
    
        
if __name__ == "__main__":
    data_path = os.environ['DATA_PATH']
    movies_datapath = data_path
    trained_datapath = "./"

    """loading the ratings dataset"""
    # int32 (and float32) instead of int64 in order to use less memory
    df_ratings = pd.read_csv(
        os.path.join(movies_datapath, 'ratings.csv'),
        usecols=['userId', 'movieId', 'rating'],
        dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

    """function 'reduce_item_dim', necessary for the ratings dataset to be fed to our Neural Network model"""

    serialized_model_path = os.path.join(trained_datapath, "dnmf_model_final.h5")
    if not os.path.exists(serialized_model_path):
        _ = train(df_ratings, serialized_model_path)
    
    dnmf_model_final = keras.models.load_model(serialized_model_path)

    # choosing a (userId, movieId) couple not already existent in the ratings.csv file, for exemple (1, 10)
    chosed_tuple = (1, 10)
    predicted_rating = predict_one(*chosed_tuple)
    userId_chosed = chosed_tuple[0]
    movieId_chosed = chosed_tuple[1]
    print("User#{} would give *{:2.2} to the Movie#{}".format(userId_chosed, predicted_rating, movieId_chosed))
