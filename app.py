import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import nltk
import numpy as np
from PIL import Image
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

st.set_page_config(
    layout="centered", page_icon="🌐", page_title="Lexis Nexis Dashboard"
)
st.title("Russia After The Cold War")

st.write("""After World War II, there was a period of geopolitical tension between the United States and Russia.
        This dashboard allows you to explore the post-Cold War view of Russia.""")

@st.cache(suppress_st_warning=True)
def getMetaData():
    countryDF = pd.read_csv("country.csv", index_col=0)
    cityDF = pd.read_csv("city.csv", index_col=0)
    companyDF = pd.read_csv("company.csv", index_col=0)
    industryDF = pd.read_csv("industry.csv", index_col=0)
    personDF = pd.read_csv("person.csv", index_col=0)
    subjectDF = pd.read_csv("subject.csv", index_col=0)

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
    df.drop(df.tail(4).index,inplace=True)
    return df

@st.cache(suppress_st_warning=True)
def getIndustry():
    df = pd.read_csv("industryDF.csv",index_col=0)
    df.drop(df.tail(4).index,inplace=True)
    return df

@st.cache(suppress_st_warning=True)
def getSubject():
    df = pd.read_csv("subjectDF.csv",index_col=0)
    df.drop(df.tail(4).index,inplace=True)
    return df

@st.cache(suppress_st_warning=True)
def getStopwords():
    return STOPWORDS.update(["said", "talk", "u", "now", "say","must","one","will","us","s","russia","russian","russians","even","says"])


df = getData()
metadata = getMetaData()

st.write("## News Referencing Russia from 1995 to 2010")
st.dataframe(df.head(1000))

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
yearChoice = st.slider('Select a Year', min_value = 1995,  max_value =2010, value = 2000)
mask = np.array(Image.open('russia.png'))
source = list(df.loc[df["year"] == yearChoice]['text_token'].values)
long_string = ','.join(source)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='black', stopwords=getStopwords(), mask = mask)
wordcloud.generate(long_string)
st.image(wordcloud.to_image(), use_column_width = "auto" )



st.write("## Sentiment Tracking")


sentChoice = st.selectbox(
     "Choose one of the following:", 
     ('subject', 'industry'))
choiceDF = getIndustry() if sentChoice == 'industry' else getSubject()

fig = px.scatter_3d(choiceDF.dropna(subset=[sentChoice]), x='year', y='title_score', z='word_count', color=sentChoice)
st.plotly_chart(fig, use_container_width=True)

subTopic = st.selectbox(
     'Select a subtopic to visualize:',
     metadata[sentChoice].value)

subChoice = choiceDF.loc[choiceDF[sentChoice] == subTopic]
subChoice = subChoice.sort_values(by='publication_date')
fig = px.line(subChoice, x="publication_date", 
                y='title_score', color = sentChoice ,hover_data=["title_sentiment"],
             title = "Topic sentiment over time")
st.plotly_chart(fig, use_container_width=True)

posNegNeu = st.selectbox(
     "Choose one of the following:", 
     ('Positive_Count','Negative_Count','Neutral_Count'))
fig = px.histogram(choiceDF, x=sentChoice, 
             y=posNegNeu, 
             barmode = 'group', 
             title="Sentiment Labels by Topic")
st.plotly_chart(fig, use_container_width=True)
