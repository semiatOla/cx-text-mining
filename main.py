import streamlit as st
from functions import *
from sklearn.feature_extraction.text import TfidfVectorizer


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
        df["CSAT"], bins=[1,3,5],
        labels=['negative', 'positive']
    )

    required_cols = ["CSAT","HPS","AP","Res_CSAT","WaitTime","ResTime","CES","Contact Reason","FCR","Channel","Status","Contact Reason L1","Comments"]
    if not all(c in df.columns for c in required_cols):
        st.error(f"Colonnes manquantes: {required_cols}")
        st.stop()

    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    st.dataframe(df.describe().loc[['count', 'mean', 'std', 'min', 'max']])

    tab4,tab1, tab2, tab3 = st.tabs(["Stats", "Analyse Sentiment", "Nuage de mots", "Analyse Croisée"])

    with tab4:
        negative_comments = df[
            (df["sentiment"] == "negative") &
            (df["Comments"].notna())
        ]
        negative_comments["len"] = negative_comments["Comments"].str.len()
        top10_negative = (
            negative_comments
            .sort_values("len", ascending=False)
            .head(10)
        )
        top10_negative["Comments"].tolist()
        incoherent = df[
            ((df["CSAT"] >= 4) & (df["sentiment"] == "negative")) |
            ((df["CSAT"] <= 2) & (df["sentiment"] == "positive"))
        ]
        top10_incoherent = incoherent.head(10)


        mois_fr = {
            1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
            5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
            9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
        }

        st.subheader("Moyennes mensuelles des indicateurs")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["year_month"] = df["Date"].dt.to_period("M")
        df["mois"] = df["Date"].dt.month.map(mois_fr)
        df = df.sort_values("mois")

        df_monthly_mean = (
            df.groupby("year_month")
            .mean(numeric_only=True)
            .reset_index()
        )
        df_monthly_mean["mois"] = df_monthly_mean["year_month"].dt.month.map(mois_fr)
        df_monthly_mean = df_monthly_mean.sort_values("year_month")
        # df_monthly_mean.index = df_monthly_mean.index.to_timestamp()
        st.dataframe(df_monthly_mean[["mois", "CSAT", "WaitTime", "ResTime"]].round(2))

        

        st.subheader("Top 50 commentaires informatifs")
        vectorizer = TfidfVectorizer(stop_words=STOPWORDS, max_features=100)
        df_novembre = df[df["mois"] == "Octobre"]

        X = vectorizer.fit_transform(df_novembre["Comments"].dropna())
        y = vectorizer.fit_transform(df["Comments"].dropna())

        df_novembre["importance"] = X.sum(axis=1).A1
        df["importance"] = y.sum(axis=1).A1
        st.dataframe(df_novembre.describe())
        top10_tfidf = df_novembre.sort_values("importance", ascending=False).head(50)
        st.dataframe(top10_tfidf[["Comments", "CSAT", "importance"]])
        if st.button("Analyser les commentaires ici"):
            n = min(150, len(top10_tfidf))
            analyse = analyse_llm(top10_tfidf["Comments"].sample(n=n).tolist())
            st.markdown(f"#### Analyse LLM")
            st.markdown(analyse)

        st.subheader("Prédiction - sentiments")
        df["sentiment2"] = df["Comments"].apply(lambda x: map_sentiment(sentiment_analyzer(x)[0]['label']))
        st.dataframe(df[["Comments", "CSAT", "sentiment2", "OverallSup", "Res_CSAT", "importance"]])



        st.subheader("Appréciation du temps de résolution par la raison de contact")
        df["ResTime_neg"] = df["ResTime"] < 3
        res_time_summary = df.groupby(["Contact Reason L1", "year_month"]).agg(
            Nbr_total = ("ResTime", "count"),
            Nbr_neg = ("ResTime_neg", "sum"),
            Score_moyen_ResTime = ("ResTime", "mean")
        ).reset_index()
        res_time_summary["%_negatif"] = (res_time_summary["Nbr_neg"] / res_time_summary["Nbr_total"]) * 100
        res_time_summary["%_positive"] = 100 - res_time_summary["%_negatif"]
        res_time_summary = res_time_summary[res_time_summary["Score_moyen_ResTime"] > 0]

        res_time_summary["mois"] = res_time_summary["year_month"].dt.month.map(mois_fr)
        res_time_summary = res_time_summary.sort_values("year_month")
        st.dataframe(res_time_summary[["mois", "Contact Reason L1", "Score_moyen_ResTime", "Nbr_total", "Nbr_neg"]])




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
        st.text(res_time_summary.to_csv(sep='\t', index=False))


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
        st.text(corr_with_csat.to_csv(sep='\t', index=False))

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
        
        if st.button("Analyser les commentaires"):
            n = min(150, len(negative_comments))
            analyse = analyse_llm(negative_comments["Comments"].sample(n=n).tolist())
            st.markdown(f"#### Analyse LLM de {n} commentaires négatives")
            st.markdown(analyse)

        st.subheader("Commentaires positifs")
        positive_comments = df[df["sentiment"] == "positive"]
        st.dataframe(positive_comments[["text", "Comments"]])


    with tab2:
        fig = word_cloud(df)
        st.pyplot(fig)
        # Wordcloud by rating group
        for g in ['negative','positive']:
            sub = df[df['sentiment']==g]
            if not sub.empty:
                st.subheader(f"Wordcloud {g} (n={sub.shape[0]})")
                fig = word_cloud(sub, title=f'Wordcloud {g}')
                st.pyplot(fig)

        st.subheader(f"Words bag")
        st.dataframe(top_n_words_by_group(df))

    
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

        threshold = 3
        df_features_pct = (
            df.groupby("sentiment")[features]
            .apply(lambda x: (x >= threshold).sum() / len(x) * 100)
            .round(1)
        )

        st.subheader("Pourcentage de l’effectif satisfait par critère (%)")
        st.dataframe(df_features_pct)
        st.markdown(df_features_pct)


        df_features = df.groupby("sentiment")[features].mean()
        st.subheader("Moyenne des critères par sentiment")
        st.dataframe(df_features.style.format("{:.2f}"))


