import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(page_title='Projet Data Science - Manuscript de Voynich - 3 RNN')
st.title('Projet Data Science - Manuscript de Voynich')

st.subheader('Sommaire')
st.markdown("<a href='https://share.streamlit.io/scraeye/projet-datascience-voynich/main/01_exploration.py'>1 Exploration</a>", unsafe_allow_html=True)
st.markdown("<a href='https://share.streamlit.io/scraeye/projet-datascience-voynich/main/02_Word2Vec.py'>2 Word2Vec</a>", unsafe_allow_html=True)
st.markdown("3 RNN", unsafe_allow_html=True)

st.header('3 - RNN')

st.markdown("On importe le texte")
with st.echo():
    file = open("voynich.txt", "r") 
    text = file.read()
    file.close()

st.text_area('Transcription du manuscrit de Voynich', value=text)

with st.echo():
    vocab = sorted(set(text))
    st.info('{} caractères uniques'.format(len(vocab)))

st.markdown("On transforme notre texte en tensor")
with st.echo():
    char2idx = {j:i for i, j in enumerate(vocab)}
    text_as_int = np.array([char2idx[c] for c in text])

    st.info('{')
    for char, _ in zip(char2idx, range(10)):
        st.info('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    st.info('  ...\n}')

with st.echo():
    seq_length = 100
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    idx2char = np.array(vocab)
    for i in char_dataset.take(5):
        st.info(str(i) + " : " + str(idx2char[i.numpy()]))

st.markdown("On crée des séquences pour pouvoir entraîner notre modèle")
with st.echo():
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    for i in sequences.take(5):
        st.info(repr(''.join(idx2char[i.numpy()])))

st.markdown("On crée une fonction pour séparer les données")
with st.echo():
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

st.markdown("On sépare les données")
with st.echo():
    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        st.info('Input data: '+ repr(''.join(idx2char[input_example.numpy()])))
        st.info('Target data:'+ repr(''.join(idx2char[target_example.numpy()])))

st.markdown("On mélange les données pour éviter les fénoméne de sur-entraînemeent")
with st.echo():
    dataset = dataset.shuffle(10000).batch(64, drop_remainder = True)

st.markdown("On crée une fonction pour crée notre modèle RNN")
with st.echo():
    from tensorflow.keras.layers import Lambda, RNN, GRU, GRUCell, Dense

    vocab_size = len(vocab)

    def build_model(batch_size):
        model = tf.keras.Sequential()
        model.add(Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32), depth=vocab_size, on_value=None, off_value=None), batch_input_shape=[batch_size, None]))
        model.add(RNN(GRUCell(512), 
                    return_sequences=True, 
                    stateful=True))
        model.add(Dense(vocab_size))
        return model

    def loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

st.markdown("On choisit des batch de 64")
with st.echo():
    model = build_model(64)
#    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss)

#with st.echo():
#    model.fit(dataset, epochs=10)

st.markdown("On charge le modèle RNN que l'on a déjà entraîné sur 10 epochs")
with st.echo():
    model = build_model(1)
    model.load_weights('model_rnn_epochs10')

st.markdown("On crée une fonction pour générer le texte à partir du modèle RNN fourni en paramètre")
with st.echo():
    def generate_text(model, start_string, num_generate = 500):
        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        # Reset initial state
        model.reset_states()
        for _ in range(num_generate):
            # Probability prediction
            prediction = tf.nn.softmax(model(input_eval), axis=-1)
            # Index prediction
            index = tf.argmax(prediction, axis=-1).numpy()[0]
            input_eval = tf.expand_dims([index[-1]], 0)
            # Save letter in text_generated list
            text_generated.append(idx2char[index[-1]])
        return (start_string + ''.join(text_generated))

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

st.info(generate_text(model, start_string="chtaiin "))

st.markdown("On charge le modèle RNN que l'on a déjà entraîné sur 100 epochs")
with st.echo():
    model = build_model(1)
    model.load_weights('model_rnn_epochs100')
st.markdown("On affiche le texte généré par ce modèle")
st.info(generate_text(model, start_string="chtaiin "))

st.markdown("On charge le modèle RNN que l'on a déjà entraîné sur 200 epochs")
with st.echo():
    model = build_model(1)
    model.load_weights('model_rnn_epochs200')
st.markdown("On affiche le texte généré par ce modèle")
st.info(generate_text(model, start_string="chtaiin "))

st.markdown("On charge le modèle RNN que l'on a déjà entraîné sur 400 epochs")
with st.echo():
    model = build_model(1)
    model.load_weights('model_rnn_epochs400')
st.markdown("On affiche le texte généré par ce modèle")
st.success(generate_text(model, start_string="chtaiin "))

st.markdown("On affiche les courbes d'entraînement selon le nombre d'epochs")
PlotFitHistories(['model_rnn_epochs10', 'model_rnn_epochs100', 'model_rnn_epochs200', 'model_rnn_epochs400'])

st.success("On remarque que le modèle RNN minimise rapidement la perte jusqu'à 150 epochs puis la progression est très lente. "
    +"En choisissant 200 epochs, on a le meilleur compromis entre vitesse d'apprentissage et perte minimisé.")