import streamlit as st
from functions import *


st.set_page_config(page_title="Text mining")
st.title("Analyse CX - Digital")

st.markdown("""
- HPS: help& problem solving
- AP: attention & professionnalism	
""")
features = ["AP", "HPS", "WaitTime", "ResTime", "CES", "FCR_score"]

file = st.file_uploader("Chargez le fichier", type=["csv"])

if file:
    df= load_data(file)

    df['sentiment'] = pd.cut(
        df["CSAT"], bins=[1,2,3,5],
        labels=['negative','neutre','positive']
    )

    required_cols = ["CSAT","HPS","AP","Res_CSAT","WaitTime","ResTime","CES","Contact Reason","FCR","Channel","Status","Contact Reason L1","Comments"]
    if not all(c in df.columns for c in required_cols):
        st.error(f"Colonnes manquantes: {required_cols}")
        st.stop()

    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce")


    tab1, tab2, tab3 = st.tabs(["Analyse Sentiment", "Nuage de mots", "Analyse Croisée"])

    with tab1:
        st.subheader("Appréciation du temps de résolution par la raison de contact")
        st.caption(f"Au moins 80% des clients doivent être satisfait.")

        df["ResTime_neg"] = df["ResTime"] < 3
        res_time_summary = df.groupby("Contact Reason L1").agg(
            Nbr_total = ("ResTime", "count"),
            Nbr_neg = ("ResTime_neg", "sum"),
            Score_moyen_ResTime = ("ResTime", "mean")
        ).sort_values(by="Score_moyen_ResTime")
        res_time_summary["%_negatif"] = (res_time_summary["Nbr_neg"] / res_time_summary["Nbr_total"]) * 100

        st.dataframe(res_time_summary[["Score_moyen_ResTime", "%_negatif"]])

        to_review = res_time_summary[res_time_summary["%_negatif"] > 20]
        if not to_review.empty:
            st.markdown(f"Il faut revoir le temps de résolution des problèmes liés à : **{list(to_review.index)}**")

        st.markdown("#### Contact Reason 2")
        df["ResTime_neg"] = df["ResTime"] < 3
        res_time_summary = df.groupby("Contact Reason").agg(
            Nbr_total = ("ResTime", "count"),
            Nbr_neg = ("ResTime_neg", "sum"),
            Score_moyen_ResTime = ("ResTime", "mean")
        ).sort_values(by="Score_moyen_ResTime")
        res_time_summary["%_negatif"] = (res_time_summary["Nbr_neg"] / res_time_summary["Nbr_total"]) * 100

        st.dataframe(res_time_summary[["Score_moyen_ResTime", "%_negatif"]])

        to_review = res_time_summary[res_time_summary["%_negatif"] > 20]
        if not to_review.empty:
            st.markdown(f"Il faut revoir le temps de résolution des problèmes liés à : **{list(to_review.index)}**")


        st.subheader("Corrélation entre CSAT et les autres critères de satisfaction")
        corr_with_csat = df[features + ["CSAT"]].corr()["CSAT"].drop("CSAT")
        corr_with_csat.sort_values(ascending=False)
        st.dataframe(corr_with_csat.to_frame("Correlation").style.format("{:.2f}"))
        to_review = corr_with_csat[corr_with_csat >= 0.7]
        st.caption(f"La satisfaction client est fortement liée à *la capacité du CC à résoudre son problème et temps de résolution*.")
        st.markdown(f"La satisfaction client est fortement liée à : **{list(to_review.index)}**.")

        # st.subheader("Heatmap des corrélations")
        # plt.figure(figsize=(6, 4))
        # sns.heatmap(df[features + ["CSAT"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
        # st.pyplot(plt)

        satisfied_wait_pct = (df["WaitTime"] >= 3).mean() * 100
        txt = f"Satisfaction globale par rapport au temps d'attente "
        if satisfied_wait_pct < 50 :
            txt += f"*très faible* ({satisfied_wait_pct:.2f}%). Il faudrait augmenter l'effectif ou chercher le problème"
        elif satisfied_wait_pct < 80 :
            txt += f"*acceptable* ({satisfied_wait_pct:.2f}%) mais peut faire mieux"
        else:
            txt += f"*excellente* ({satisfied_wait_pct:.2f}%)"

        st.markdown(txt)

        st.subheader("Commentaires négatifs")
        negative_comments = df[df["sentiment"] == "negative"]
        st.dataframe(negative_comments[["text", "Comments"]])
        st.caption(f"Les clients se plaignent par rapport au temps d'attente, à la formation des CC et à la complexité du parcours client (réclamations & app).")
        
        n = min(150, len(negative_comments))
        analyse = analyse_llm(negative_comments["Comments"].sample(n=n).tolist())
        st.subheader(f"Analyse LLM de {n} commentaires négatives")
        st.markdown(analyse)

        st.subheader("Commentaires positifs")
        positive_comments = df[df["sentiment"] == "positive"]
        st.dataframe(positive_comments[["text", "Comments"]])


        # st.subheader("Prédiction???")
        # df2 = load_data(path='data2.csv')
        # # df['sentiment_blob'] = df['text'].map(sentiment_score)
        # # df["sentiment_llm"] = df["Comments"].apply(sentiment_score_llm)
        # fig, ax = plt.subplots(figsize=(6,4))
        # sns.countplot(x=df2['sentiment_llm'], ax=ax, legend =['negative','neutre','positive'])
        # st.pyplot(fig)
        # st.write(df2[['Comments', 'sentiment_llm']])

        # y_true = df2["sentiment"].astype(str).str.lower()
        # y_pred = df2["sentiment_llm"].astype(str).str.lower()

        # cm = confusion_matrix(y_true, y_pred, labels=["positive", "neutre", "negative"])
        # cm_df = pd.DataFrame(cm, index=["positive", "neutre", "negative"], columns=["positive", "neutre", "negative"])
        # st.subheader("Matrice de confusion")
        # st.dataframe(cm_df)

        #df.to_csv("data2.csv", sep=';', index=False, encoding='utf-8')


    with tab2:
        fig = word_cloud(df)
        st.pyplot(fig)
        # Wordcloud by rating group
        for g in ['negative','neutre','positive']:
            sub = df[df['sentiment']==g]
            if not sub.empty:
                st.subheader(f"Wordcloud {g} (n={sub.shape[0]})")
                fig = word_cloud(sub, title=f'Wordcloud {g}')
                st.pyplot(fig)

    
    with tab3:
        table_reason = pd.crosstab(df.sentiment, df["Contact Reason"], normalize='index') * 100
        table_channel = pd.crosstab(df.sentiment, df["Channel"], normalize='index') * 100
        table_status = pd.crosstab(df.sentiment, df["Status"], normalize='index') * 100
        table_fcr = pd.crosstab(df.sentiment, df["FCR"], normalize='index') * 100
        table_effort = df.groupby("sentiment")[["WaitTime", "ResTime", "CES"]].mean()
        table_scores = df.groupby("sentiment")[["AP", "HPS"]].mean()
        
        insights = generate_insights(
            table_reason, table_channel, table_status, table_fcr, table_effort, table_scores
        )


        df_features = df.groupby("sentiment")[features].mean()
        st.subheader("Moyenne des critères par sentiment")
        st.dataframe(df_features.style.format("{:.2f}"))

        # st.subheader("Heatmap Sentiment × critères")
        # plt.figure(figsize=(10, 4))
        # sns.heatmap(df_features, annot=True, fmt=".2f", cmap="coolwarm")
        # st.pyplot(plt)
        # st.caption(f"Les clients sont globalement satisfait sauf par rapport au temps d'attente.")

        # st.subheader("Heatmap Sentiment × Raison du contact")
        # plt.figure(figsize=(10, 4))
        # sns.heatmap(table_reason, annot=True, fmt=".2f", cmap="coolwarm")
        # st.pyplot(plt)
        # st.caption(f"Les clients sont globalement satisfait sauf par rapport au temps d'attente.")

        # st.subheader("Heatmap Sentiment × Canal")
        # plt.figure(figsize=(10, 4))
        # sns.heatmap(table_channel, annot=True, fmt=".2f", cmap="coolwarm")
        # st.pyplot(plt)



    

# Confusion matrix


#st.markdown(top_n_words_by_group(df))
#WNlemmatizer = WordNetLemmatizer()
#lem = WNlemmatizer.lemmatize("essayez de mettre les agents aux bout des files au même niveau d'information.", pos="a") 
#st.markdown(lem)

