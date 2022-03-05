import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import nltk
import numpy as np
from PIL import Image


STREAMLIT_AGGRID_URL = "https://github.com/PablocFonseca/streamlit-aggrid"
st.set_page_config(
    layout="centered", page_icon="🖱️", page_title="Lexis Nexis Dashboard"
)
st.title("Russia After The Cold War")

st.write("""After World War II, there was a period of geopolitical tension between the United States and Russia.
        This dashboard allows you to explore the post-Cold War view of Russia.""")

@st.cache(suppress_st_warning=True)
def getMetaData():
    countryDF = pd.read_csv("country.csv")
    cityDF = pd.read_csv("city.csv")
    companyDF = pd.read_csv("company.csv")
    industryDF = pd.read_csv("industry.csv")
    personDF = pd.read_csv("person.csv")
    subjectDF = pd.read_csv("subject.csv")

    metadata = {"country": countryDF,
                "city": cityDF,
                "company" : companyDF,
                "industry" : industryDF,
                "person" : personDF,
                "subject" : subjectDF
                }
    return metadata


@st.cache(suppress_st_warning=True)
def getData():
    df = pd.read_csv("core.csv", index_col=0)
    df['text_token'] = df["content"].astype(str).apply(nltk.stem.SnowballStemmer("english").stem)
    df.dropna()
    return df

@st.cache(suppress_st_warning=True)
def getStopwords():
    return STOPWORDS.update(["said", "talk", "u", "now", "say","must","one","will","us","s"])


df = getData()
metadata = getMetaData()

st.write("## News Data from 1995 to 2010")
st.dataframe(df)

st.write("## Reference Based Visualizations")

st.write("### Who's mentioned the most in the data?")
fig = px.choropleth(metadata['country'], locations="value",
                    color="count", 
                    hover_name="value", 
                    locationmode = 'country names',
                    title = "News Reference Count by Country",
                    color_continuous_scale=px.colors.sequential.Plasma)
st.plotly_chart(fig, use_container_width=True)


st.write("### Interactive Pie Chart")
pieChoose = st.selectbox(
     'Select a topic to visualize:',
     metadata.keys())

fig = px.pie(metadata[pieChoose], values='count', names='value', 
             title = f"News References Makeup by {pieChoose.capitalize()}",
            labels = {"count": "references",
                    "value":"subject"})

fig.update_traces(textposition='inside')
st.plotly_chart(fig, use_container_width=True)

st.write("## Language Analysis")
st.write("### Wordcloud by Year")
yearChoice = st.slider('Select a Year', min_value = 1995,  max_value =2010, value = 1995)
mask = np.array(Image.open('russia.png'))
source = list(df.loc[df["year"] == yearChoice]['text_token'].values)
long_string = ','.join(source)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='red', stopwords=getStopwords(), width=5000, height=5000,)
wordcloud.generate(long_string)
st.image(wordcloud.to_image(), use_column_width = "auto" )

st.write("## Sentiment Tracking")

fig = px.histogram(df, x = "year")

st.plotly_chart(fig, use_container_width=True)

df["Positive_Count"] = df.loc[df["title_sentiment"] == 'POS', 'title_sentiment'].count()
df["Negative_Count"] = df.loc[df["title_sentiment"] == 'NEG', 'title_sentiment'].count()
df["Neutral_Count"] = df.loc[df["title_sentiment"] == 'NEU', 'title_sentiment'].count()
fig = px.histogram(df, x="year", 
             y=['Positive_Count','Negative_Count','Neutral_Count'], 
             barmode = 'group',
             title="Sentiment Labels by Year")

st.plotly_chart(fig, use_container_width=True)

fig = px.scatter_3d(df, x='year', y='title_score', z='word_count',
                    title = "Country Subscription Price vs Library Size vs GDP per Capita")
st.plotly_chart(fig, use_container_width=True)
