import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Olympics Data Analysis", layout="wide")
st.title("🏅 Olympics Data Analysis (1976–2008)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Summer-Olympic-medals-1976-to-2008.csv", encoding="latin1")
    df.drop(['Event_gender', 'Country_Code'], axis=1, inplace=True)
    df.dropna(how='all', inplace=True)
    df.dropna(inplace=True)
    df['Year'] = df['Year'].astype(int)
    return df

df = load_data()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🏆 Leaders", "🔮 Prediction", "ℹ️ About"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.subheader("1. Top 10 Countries by Medal Count")
    top_countries = df['Country'].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    top_countries.plot(kind='bar', ax=ax1, color='gold')
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("2. Medals Over the Years")
    medals_by_year = df['Year'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    sns.lineplot(x=medals_by_year.index, y=medals_by_year.values, marker='o', ax=ax2)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Total Medals")
    st.pyplot(fig2)

    st.subheader("3. Gender Distribution")
    fig3, ax3 = plt.subplots()
    df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], explode=[0.05, 0], ax=ax3)
    ax3.set_ylabel('')
    st.pyplot(fig3)

with tab2:
    st.header("Top Performers")
    st.subheader("1. Top 10 Athletes by Medal Count")
    top_athletes = df['Athlete'].value_counts().head(10)
    st.bar_chart(top_athletes)

    st.subheader("2. Top Sports by Events")
    sport_counts = df['Sport'].value_counts().head(10)
    st.bar_chart(sport_counts)

    st.subheader("3. Top Cities by Events Hosted")
    city_counts = df['City'].value_counts().head(10)
    st.bar_chart(city_counts)

with tab3:
    st.header("🎯 Predict Medal Win (Logistic Regression)")
    df_ml = df.copy()
    df_ml['Medal_Flag'] = df_ml['Medal'].apply(lambda x: 0 if pd.isna(x) else 1)

    # Encode categorical features
    le = LabelEncoder()
    for col in ['Country', 'Sport', 'Gender']:
        df_ml[col] = le.fit_transform(df_ml[col])

    X = df_ml[['Country', 'Sport', 'Gender']]
    y = df_ml['Medal_Flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown("**Model Accuracy:** {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

    st.subheader("📍 Try a Prediction")
    country = st.selectbox("Select Country", df['Country'].unique())
    sport = st.selectbox("Select Sport", df['Sport'].unique())
    gender = st.selectbox("Select Gender", ['Male', 'Female'])

    input_data = pd.DataFrame({
        'Country': [le.transform([country])[0]],
        'Sport': [le.transform([sport])[0]],
        'Gender': [le.transform(['Men' if gender == 'Male' else 'Women'])[0]]
    })

    pred = model.predict(input_data)[0]
    st.success("🏅 Likely to win a medal!" if pred == 1 else "❌ Not likely to win a medal.")

with tab4:
    st.header("About the Project")
    st.markdown("""
    - 🔍 **Dataset**: Medal winners in Summer Olympics from 1976 to 2008.
    - 📈 **Tools**: Python, Streamlit, pandas, seaborn, scikit-learn.
    - 🤖 **ML Task**: Predict whether an athlete will win a medal based on gender, country, and sport.

    Made with ❤️ for data storytelling.
    """)
