import streamlit as st
from PIL import Image
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.regexp import RegexpTokenizer
from wordcloud import WordCloud


st.set_page_config(page_title='Projet Data Science - Manuscrit de Voynich - 1 Exploration')
st.title('Projet Data Science - Manuscrit de Voynich')

st.subheader('Sommaire')
st.markdown("1 Exploration", unsafe_allow_html=True)
st.markdown("<a href='https://share.streamlit.io/scraeye/projet-datascience-voynich/main/02_Word2Vec.py'>2 Word2Vec</a>", unsafe_allow_html=True)
st.markdown("<a href='https://share.streamlit.io/scraeye/projet-datascience-voynich/main/03_RNN.py'>3 RNN</a>", unsafe_allow_html=True)

st.subheader('Introduction')

st.markdown('Le manuscrit de Voynich est un livre illustré anonyme rédigé dans une écriture à ce jour non déchiffrée et une langue non identifiée. '
    +'Le livre doit son nom à Wilfrid M. Voynich, qui l\'a découvert en 1912 à Frascati, près de Rome, dans la bibliothèque d\'une communauté de jésuites. '
    +'la datation par le carbone 14 du parchemin utilisé pour la confection du livre a permis d\'établir que le vélin a été fabriqué entre 1404 et 1438. '
    +'Il contient du texte et des illustrations présentées selon le style européen des herbiers de l\'époque.')

st.markdown('Ici, nous allons étudier uniquement le texte à partir de la transcription de <a href=\"https://github.com/voynichese/voynichese/blob/wiki/DataSets.md\" target=\"_blank\">The Voynichese project</a>', 
    unsafe_allow_html=True)
    
st.markdown('On allons voir ce que la Data Science peut nous apprendre et essayer de répondre à quelques questions. ')

st.text('Illustration du manuscrit de Voynich')
col1, col2 = st.beta_columns(2)
with col1:
    st.image(Image.open('1024px-Voynich_Manuscript_(119).jpg'), use_column_width=True)
with col2:
    st.image(Image.open('1024px-Voynich_Manuscript_(141).jpg'), use_column_width=True)

st.header('1 - Exploration de donnée')

st.markdown("Commençons par importer le texte")
with st.echo():
    file = open("voynich.txt", "r") 
    txt = file.read()
    file.close()

st.text_area('Transcription du manuscrit de Voynich', value=txt)


st.markdown("On met chaques mots dans un tableau appelé \"mots\"")
with st.echo():
    mots = txt.split(" ")
st.info("Nombre de mots : " + str(len(mots)))

st.subheader('Fréquence des mots')

st.markdown("On crée une fonction \"frequenceMot\" où on met chaque mot différent dans un dictionnaire avec le nombre d'd'occurrence puis on transforme ce dictionnaire en DataFrame et on en profite pour ajouter une colonne avec la longueur du mot. ")
with st.echo():
    def frequenceMot(mots):
        mot_unique = {}
        for mot in mots:
            if mot in mot_unique :
                mot_unique[mot] += 1
            else:
                mot_unique[mot] = 1
        
        df = pd.DataFrame.from_dict(mot_unique, orient='index')
        df['mot'] = df.index
        df['count'] = df[0].astype('int')
        df['longueur'] = df['mot'].apply(lambda m: len(m))
        df = df.reset_index().drop(['index', 0], axis=1)
        return df   

st.markdown("On affiche le DataFrame, on peut le trier cliquant sur l'entête de la colonne. ")
with st.echo():
    df = frequenceMot(mots)

st.dataframe(df)
st.info("Nombre de mots différents : " + str(len(df)))
st.info('Le nombre de mots est cohérent avec un texte écrit dans une autre langue car touts les mots ne sont pas différents, ils se répéte.')

#st.write(df.info())
st.markdown("On trie le DataFrame du mot le plus fréquent au moins fréquent.")
with st.echo():
    df = df.sort_values(by='count', ascending=False)

st.markdown("On affiche la distribution de tous les mots")
with st.echo():
    fig = plt.figure(figsize=(20,10))
    plt.bar(range(len(df)), height=df['count'])
    plt.xlabel('Mots')
    plt.ylabel('Fréquence')
    plt.title('Fréquence des mots')
    plt.xlim(0, 8079)
    plt.show()
st.pyplot(fig)

st.markdown("On affiche la distribution des mots 200 mots les plus fréquents.")
with st.echo():
    fig = plt.figure(figsize=(20,10))
    plt.bar(range(len(df)), height=df['count'])
    plt.xlabel('Mots')
    plt.ylabel('Fréquence')
    plt.title('Fréquence des mots - les 200 mots les plus fréquents')
    plt.xlim(0, 200)
    plt.show()
st.pyplot(fig)

st.info("On constate que la distribution n'est pas régulière comme dans une langue.")

st.markdown("On affiche les 100 mots le plus fréquent, plus le mot est grand, plus il est fréquent.")
with st.echo():
    fig = plt.figure(figsize=(20,10))
    wc = WordCloud(background_color='white', 
                    max_words=100, 
                    max_font_size=50, 
                    random_state=42)
    wc.generate(txt)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(wc)
    plt.show()
st.pyplot(fig)

st.markdown("On affiche les 100 mots le plus fréquent en reprennent la police de caractère d'origine. "
    +"En utilisant le même random_state, on peut garder la même couleur et pratiquement disposition des mots.")
with st.echo():
    fig = plt.figure(figsize=(20,10))
    wc = WordCloud(background_color='white', 
                    max_words=100, 
                    max_font_size=50, 
                    random_state=42,
                    font_path='eva1.ttf')
    wc.generate(txt)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(wc)
    plt.show()
st.pyplot(fig)

st.subheader('Fréquence des N-grammes')

st.markdown("On crée deux fonctions pour générer un tableau de deux consécutif et de trois mots consécutif. ")
with st.echo():
    def deuxMots(mots):
        result = []
        mot1 = ''
        mot2 = ''
        for mot in mots:
            mot2 = mot1
            mot1 = mot
            if mot1 != '' and mot2 != '':
                result.append(mot2 + ' ' + mot1)
        return result

    def troisMots(mots):
        result = []
        mot1 = ''
        mot2 = ''
        mot3 = ''
        for mot in mots:
            mot3 = mot2
            mot2 = mot1
            mot1 = mot
            if mot1 != '' and mot2 != '' and mot3 != '':
                result.append(mot3 + ' ' + mot2 + ' ' + mot1)
        return result

st.markdown("On affiche les 10 séquences des deux mots les plus fréquentes.")
with st.echo():
    df_f2m = frequenceMot(deuxMots(mots))
    df_f2m = df_f2m.sort_values(by='count', ascending=False)

st.dataframe(df_f2m.head(10))

st.markdown("On affiche les 10 séquences des trois mots les plus fréquentes.")
with st.echo():
    df_f3m = frequenceMot(troisMots(mots))
    df_f3m = df_f3m.sort_values(by='count', ascending=False)

st.dataframe(df_f3m.head(10))
st.info("On constate que les séquence se répéte comme dans toutes les langues.")

st.markdown("On trie le DataFrame du mot le plus long au plus court.")
with st.echo():
    df = df.sort_values(by='longueur', ascending=False)

st.markdown("On affiche le DataFrame des 10 mots les plus long. ")
st.dataframe(df.head(10))



st.markdown("On affiche la distribution suivant la longueur du mot.")
st.write(df['longueur'].value_counts())
with st.echo():
    fig = plt.figure(figsize=(20,10))
    plt.hist(df['longueur'])
    plt.xlabel('Longueur des mots')
    plt.ylabel('Fréquence')
    plt.title('Fréquence de la longueur des mots')
    plt.show()
st.pyplot(fig)
st.warning("Il est inhabituel dans une langue d'avoir si peu de mots de deux et trois caractères")


st.info("L'exploration des données montre que le texte de Voynich a bien les même caractéristique qu'un texte écrit dans une autre langue. "
    +"Cela n'est pas suffisant pour prouver le texte n'est pas une supercherie mais ici rien ne prouve le contraire")


st.markdown("<a href='https://share.streamlit.io/scraeye/projet-datascience-voynich/main/02_Word2Vec.py'>2 Word2Vec</a>", unsafe_allow_html=True)