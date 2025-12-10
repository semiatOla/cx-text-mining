import streamlit as st
import pandas as pd
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from textblob import TextBlob
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
FR_STOPWORDS = set(stopwords.words('french'))
EN_STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = [
    "nan", "ras", "r.a.s", "RAS", "Ras", "ok", "okay", "Ok", "Okay",
    "cool", "Cool", "daccord", "d", "accord", "merci", "parfait",
    "cest", "c'est", "ca", "ça", "bon", "bien", "ai", "a", "va", "deja", 
    "déjà", "cette", "être", "suis", "svp", "avoir", "alors", "vers", "puis",
    "faire", "quand", "sans", "peut", "aussi", "non", "après", "car", "faut",
    "lors", "si", "sorte", "aller", "neant", "fait",
    "veux", "veut", "leur", "leurs", "ya", "plus"
]
STOPWORDS = list(FR_STOPWORDS | EN_STOPWORDS | set(CUSTOM_STOPWORDS))


from groq import Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def load_data(path='data.csv'):
    df = pd.read_csv(path, sep=";")
    df = df[df["Comments"].notna()]
    df["text"] = df["Comments"].astype(str).map(clean_comment)
    df = df[df["text"].str.strip() != ""]
    df["Status"] = df["Status"].replace("Solved", "Résolu")

    # 5=ok, 0=non ok
    df["FCR_score"] = df["FCR"].map({1: 5, 3: 2})
    return df

def clean_comment(text):
    if pd.isnull(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)
    # return text

def word_cloud(df, max_words=150, title=None):
    text = "".join(df['text'].dropna().tolist())
    wc = WordCloud(width=800, height=400, max_words=max_words, stopwords=STOPWORDS).generate(text)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    if title: ax.set_title(title)
    return fig

def sentiment_score2(text):
    pos = ["super","bien","rapide","efficace","professionnel","satisfait","résolu","sympa"]
    neg = ["attente","long","lent","pas","jamais","impossible","déçu","incompétent","problème","colère"]
    s = 0
    for p in pos: s += text.count(p)
    for n in neg: s -= text.count(n)
    return np.sign(s)

def sentiment_score(text):
    blob = TextBlob(str(text))
    polar = blob.sentiment.polarity
    if polar > 0:
        return 'positive'
    elif polar < 0:
        return 'negative'
    else:
        return 'neutre'
    

SYSTEM_PROMPT = """
Tu es un modèle spécialisé en analyse du sentiment pour des commentaires de clients évaluant le service client dans le secteur des télécommunications.
Ton objectif est d'évaluer uniquement le sentiment exprimé dans le commentaire fourni, en te basant sur le ton, l'émotion, la satisfaction ou l'insatisfaction du client.

Tu dois impérativement répondre par une seule des trois étiquettes suivantes :
- positive
- negative
- neutre

Exemple:
commentaire: "essayez de mettre les agents aux bout des files au même niveau d'information."
réponse: negative

Voici le commentaire à analyser :
<<< COMMENTAIRE >>>

Réponse :
"""

#"openai/gpt-oss-120b"
def sentiment_score_llm(commentaire, model_id="llama-3.1-8b-instant"):
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.replace("<<< COMMENTAIRE >>>", commentaire)},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip().lower()
        if "posit" in content:
            return "positive"
        elif "negat" in content:
            return "negative"
        elif "neutr" in content:
            return "neutre"
        return "neutre"
    except Exception as e:
        #yield f"\n\n(Erreur: {e})"
        return "Erreur"
    
SENTI_PROMPT = """
Tu es un assistant spécialisé en analyse des commentaires négatifs des clients d'un service de télécommunications.

On te fournit un ensemble de commentaires négatifs des clients :

<<< COMMENTAIRES >>>

Tâches :
1. Identifier et lister les **préoccupations principales** des clients (ex : délai trop long, agent peu disponible, problème réseau, facturation, etc.)
2. Fournir des **recommandations concrètes** pour améliorer le service (ex : formation agents, réduction du temps d'attente, amélioration réseau, etc.)
3. Répondre de manière claire et structurée en **deux sections distinctes** :
   - Préoccupations des clients : liste claire
   - Recommandations : liste simple et brève

Réponse uniquement sous ce format.
"""

def analyse_llm(commentaires, model_id="llama-3.1-8b-instant"):
    try:
        commentaires = "\n".join(commentaires)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SENTI_PROMPT.replace("<<< COMMENTAIRES >>>", commentaires)},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip().lower()
        return content
    except Exception as e:
        return "Erreur"
         

def top_n_words_by_group(df, n=25):
    vect = CountVectorizer(stop_words=STOPWORDS, ngram_range=(1,3), min_df=10)
    X = vect.fit_transform(df['text'].fillna(''))
    cols = vect.get_feature_names_out()
    df_counts = pd.DataFrame(X.toarray(), columns=cols)
    df_counts['sentiment'] = df['sentiment'].values
    result = {}
    for g in df['sentiment'].dropna().unique():
        sub = df_counts[df_counts['sentiment']==g].drop(columns=['sentiment']).sum().sort_values(ascending=False).head(n)
        result[g] = sub
    return result

sentiment_colors = {
    "negative": "red",
    "neutral": "gray",
    "positive": "green"
}

def plot_sentiment_table(table, title=""):
    fig, ax = plt.subplots(figsize=(6,2))

    x = range(len(table.columns))
    bar_width = 0.25

    for i, sentiment in enumerate(table.index):
        ax.bar(
            [p + i*bar_width for p in x],
            table.loc[sentiment],
            width=bar_width,
            label=sentiment.capitalize(),
            color=sentiment_colors.get(sentiment, "black")
        )

    ax.set_title(title)
    ax.set_ylabel("Pourcentage (%)")
    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(table.columns, rotation=45)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    #plt.tight_layout()
    return fig

def generate_insights(table_reason, table_channel, table_status, table_fcr, table_effort, table_scores):
    insights = []

    # Raison du contact
    st.dataframe(table_reason)
    worst_reason = table_reason.loc["negative"].idxmax()
    best_reason = table_reason.loc["positive"].idxmax()
    if worst_reason == best_reason:
        txt = f"La raison **'{worst_reason}'** est paradoxalement celle qui génère à la fois"
        insights.append(txt)
        st.markdown(txt)
    else:
        txt = f"La raison la plus génératrice de sentiment négatif est : {worst_reason}."
        insights.append(txt)
        st.markdown(txt)
        txt = f"La raison la plus positive est : {best_reason}."
        insights.append(txt)
        st.markdown(txt)

    # Canal
    st.write()
    fig = plot_sentiment_table(table_channel, "Satisfaction par canal")
    st.pyplot(fig)
    worst_channel = table_channel.loc["negative"].idxmax()
    best_channel = table_channel.loc["positive"].idxmax()
    if worst_channel == best_channel:
        txt = f"L'assistance via le canal **'{worst_channel}'** est plus ou moins apprécié par les clients"
        insights.append(txt)
        st.markdown(txt)
    else:
        txt = f"L'assistance via le canal '{worst_channel}' est le plus problématique."
        insights.append(txt)
        st.markdown(txt)

        txt = f"L'assistance via le canal '{best_channel}' est le mieux perçu."
        insights.append(txt)
        st.markdown(txt)

    # Status résolution
    st.write()
    fig = plot_sentiment_table(table_status, "Satisfaction par statut de la requête")
    st.pyplot(fig)
    worst_status = table_status.loc["negative"].idxmax()
    txt = f"La majorité des clients insatisfaits ont leur cas : **{worst_status}**."
    insights.append(txt)
    st.markdown(txt)
    if "Résolu" not in table_status.columns:
        neg_non_resolu = table_status.loc["negative"]["Non résolu"]
        txt = f"{neg_non_resolu:.1f}% des commentaires négatifs sont dû à la non résolution de ces cas."
        insights.append(txt)
        st.markdown(txt)


    # FCR
    st.write()
    if 3 in table_fcr.columns:
        fig = plot_sentiment_table(table_fcr, "Satisfaction par FCR")
        st.pyplot(fig)
        low_fcr = table_fcr.loc["negative"][3]
        if low_fcr > 70:
            txt = f"La majorité des commentaires négatifs ({low_fcr:.1f}%) concernent des cas non résolus au premier contact."
        else:
            txt = f"Les commentaires négatifs ne sont pas dû aux cas non résolus au premier contact."
        insights.append(txt)
        st.markdown(txt)

    # WaitTime / ResTime / CES
    st.write()
    fig = plot_sentiment_table(table_effort, "Evaluation des CC sur le temps")
    st.pyplot(fig)
    low_effort = table_effort.loc["negative"].idxmin()
    high_effort = table_effort.loc["positive"].idxmax()
    if low_effort == high_effort:
        txt = f"**{high_effort}** a le meilleur score d'effort"
        insights.append(txt)
        st.markdown(txt)
    else:
        txt = f"Le meilleur score effort est sur : **{high_effort}**."
        insights.append(txt)
        st.markdown(txt)
        txt = f"Le plus faible score effort est : **{low_effort}**."
        insights.append(txt)
        st.markdown(txt)

    # 
    st.write()
    fig = plot_sentiment_table(table_scores, "Evaluation des CC sur l'aptitude")
    st.pyplot(fig)
    best_score = table_scores.loc["positive"].idxmax()
    worst_score = table_scores.loc["negative"].idxmin()
    if worst_score == best_score:
        txt = f"**{best_score}** est la dimension la mieux perçue."
        insights.append(txt)
        st.markdown(txt)
    else:
        txt = f"La dimension la mieux perçue est : **{best_score}**."
        insights.append(txt)
        st.markdown(txt)

        txt = f"La dimension la plus faible est : **{worst_score}**."
        insights.append(txt)
        st.markdown(txt)
    

    return insights


