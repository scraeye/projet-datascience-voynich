import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.preprocessing import Normalizer

st.set_page_config(page_title='Projet Data Science - Manuscript de Voynich - 2 Word2Vec')
st.title('Projet Data Science - Manuscript de Voynich')

st.subheader('Sommaire')
st.markdown("<a href='https://share.streamlit.io/scraeye/projet-datascience-voynich/main/01_exploration.py'>1 Exploration</a>", unsafe_allow_html=True)
st.markdown("2 Word2Vec", unsafe_allow_html=True)
st.markdown("<a href='https://share.streamlit.io/scraeye/projet-datascience-voynich/main/03_RNN.py'>3 RNN</a>", unsafe_allow_html=True)

st.header('2 - Word2Vec')

st.markdown("On importe le texte")
with st.echo():
    file = open("voynich.txt", "r") 
    txt = file.read()
    file.close()

st.text_area('Transcription du manuscrit de Voynich', value=txt)

st.markdown("On transforme le texte en vecteur et on affiche sa longueur")
with st.echo():
    mots = txt.split(" ")

    tokenizer = RegexpTokenizer("[a-zA-Z]{2,}")
    word_list = tokenizer.tokenize(txt.lower())

    vectorizer = CountVectorizer()
    vectorizer.fit(word_list)

    word2idx = vectorizer.vocabulary_
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    vocab_size = len(idx2word)

st.info(vocab_size)

st.markdown("On crée une fonction pour transformer notre vecteur en jeux de donnée X et y.")
with st.echo():
    def sentenceToData(tokens,WINDOW_SIZE):
        window = np.concatenate((np.arange(-WINDOW_SIZE,0),np.arange(1,WINDOW_SIZE+1)))
        X,Y=([],[])
        for word_index, word in enumerate(tokens) :
            if ((word_index - WINDOW_SIZE >= 0) and (word_index + WINDOW_SIZE <= len(tokens) - 1)) :
                X.append(word2idx[word])
                Y.append([word2idx[tokens[word_index-i]] for i in window])
        return X, Y

with st.echo():
    WINDOW_SIZE = 5

    X, Y = ([], [])

    X1, Y1 = sentenceToData(word_list, WINDOW_SIZE//2)
    X.extend(X1)
    Y.extend(Y1)
        
    X = np.array(X).astype(int).reshape([-1,1])
    y = np.array(Y).astype(int)

st.info('Shape of X :' + str(X.shape))
st.info('Shape of Y :' + str(y.shape))

st.markdown("On crée une classe Word2vec qui dérive de la classe Model de Tensorflow Keras")
with st.echo():
    class Word2vec(tf.keras.Model):
        def __init__(self, N_DIM):
            super(Word2vec, self).__init__()
            self.W1 = tf.Variable(tf.random.uniform([vocab_size, N_DIM], -1.0, 1.0))
            self.W2 = tf.Variable(tf.random.uniform([N_DIM, vocab_size], -1.0, 1.0))
        def __call__(self, X, training=True):
            X = tf.one_hot(X, depth=vocab_size, axis=-1, on_value=None, off_value=None)
            X = tf.squeeze(X, axis=1)
            h = tf.linalg.matmul(X, self.W1)
            u = tf.linalg.matmul(h, self.W2)
            return u

st.markdown("On choisit la taille de notre batch, ici on prend 64")
with st.echo():
    batch_size = 64

st.markdown("On crée notre fonction de perte pour minimiser la distance entre nos données et notre modèle.")
with st.echo():
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.one_hot(y_true, depth=vocab_size, on_value=None, off_value=None)
        return -tf.tensordot(y_pred, tf.reduce_sum(y_true, axis=[1]),2)/batch_size + \
                    tf.reduce_sum(4*tf.math.log(tf.reduce_sum(tf.exp(y_pred), axis=[1])))/batch_size
    
st.markdown("On crée une fonction SaveFitHistory pour sauvegarder la valeur de la perte à chaque étape. "
    +"Puis une fonction PlotFitHistories pour comparer visuellement les différentes méthodes d'entraînement.")
with st.echo():
    def SaveFitHistory(history, hist_csv_file):
        hist_df = pd.DataFrame(history.history) 
        with open(hist_csv_file+".csv", mode='w') as f:
            hist_df.to_csv(f)
        return hist_df
    def PlotFitHistories(hist_csv_files):
        fig = plt.figure(figsize=(14,7))
        xmax = 0
        for hist_csv_file in hist_csv_files:
            df_hist = pd.read_csv(hist_csv_file+".csv")
            plt.plot(df_hist['loss'], label=hist_csv_file)
            if len(df_hist) > xmax:
                xmax = len(df_hist)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xlim(0,xmax)
        plt.title('Courble de descente de gradient par epoch')
        plt.legend()
        plt.show()
        st.pyplot(fig)


st.markdown("On instancie notre classe Word2vec.")
with st.echo():
    word2vec = Word2vec(100)
    
st.markdown("On affiche nos courbes d'entraînement avec un taux d'entraînement de 0.01 et 0.001.")
with st.echo():
    PlotFitHistories(['word2vec_lr1e-2_epochs20', 'word2vec_lr1e-3_epochs20'])
st.info("Le taux d'entraînement de 0.001 donne un meilleur résultat.")

st.markdown("On charge notre modèle que l'on a déjà entraîné.")
with st.echo():
    word2vec.load_weights('word2vec_lr1e-3_epochs20')
    
st.markdown("On normalise notre vecteur.")
with st.echo():
    vectors = word2vec.W1.numpy()
    normalizer = Normalizer()
    vectors = normalizer.fit_transform(vectors, 'l2')

st.markdown("On crée des fonctions pour afficher les relations entre les mots.")
with st.echo():
    def dot_product(vec1, vec2):
        return np.sum((vec1*vec2))

    def cosine_similarity(vec1, vec2):
        return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

    def find_closest(word_index, vectors, number_closest):
        list1=[]
        query_vector = vectors[word_index]
        for index, vector in enumerate(vectors):
            if not np.array_equal(vector, query_vector):
                dist = cosine_similarity(vector, query_vector)
                list1.append([dist,index])
        return np.asarray(sorted(list1,reverse=True)[:number_closest])

    def compare(index_word1,index_word2,index_word3,vectors,number_closest):
        list1=[]
        query_vector = vectors[index_word1]-vectors[index_word2]+vectors[index_word3]
        normalizer = Normalizer()
        query_vector =  normalizer.fit_transform([query_vector], 'l2')
        query_vector= query_vector[0]
        for index, vector in enumerate(vectors):
            if not np.array_equal(vector, query_vector):
                dist = cosine_similarity(vector, query_vector)
                list1.append([dist,index])
        return np.asarray(sorted(list1,reverse=True)[:number_closest])

    def print_closest(word, number=10):
        index_closest_words = find_closest(word2idx[word], vectors, number)
        for index_word in index_closest_words :
            st.info(str(idx2word[index_word[1]]) + " -- " + str(index_word[0]))

st.markdown("On affiche les mots les plus proche de \"daiin\" qui est le mot de plus fréquent du texte.")
with st.echo():
    print_closest('daiin')

st.markdown("On compare les mots les trois mots les plus fréquent du texte.")
with st.echo():
    points = compare(word2idx['daiin'],word2idx['ol'],word2idx['chedy'],vectors,10)

    fig = plt.figure()
    plt.scatter([i[0] for i in points], [i[1] for i in points])
    st.pyplot(fig)

st.markdown("On compare \"daiin\" avec deux mots proches.")
with st.echo():
    points = compare(word2idx['daiin'],word2idx['otod'],word2idx['kchydy'],vectors,10)
    
    fig = plt.figure()
    plt.scatter([i[0] for i in points], [i[1] for i in points])
    st.pyplot(fig)

st.success("On constate que dans le deuxième cas, les points sont un peu plus proches donc les mots ont a sens plus proche. Ce qui est cohérent avec les mots choisis")

st.markdown("<a href='https://share.streamlit.io/scraeye/projet-datascience-voynich/main/03_RNN.py'>3 RNN</a>", unsafe_allow_html=True)