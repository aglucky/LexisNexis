import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import plotly.express as px
from wordcloud import WordCloud


STREAMLIT_AGGRID_URL = "https://github.com/PablocFonseca/streamlit-aggrid"
st.set_page_config(
    layout="centered", page_icon="🖱️", page_title="Interactive table app"
)
st.title("🖱️ Interactive table app")


countryDF = pd.read_csv("country.csv")
cityDF = pd.read_csv("city.csv")
companyDF = pd.read_csv("company.csv")
industryDF = pd.read_csv("industry.csv")
personDF = pd.read_csv("person.csv")
subjectDF = pd.read_csv("subject.csv")

metadata = {"country": countryDF,
            "city": cityDF,
            "company" : companyDF,
            "industy" : industryDF,
            "person" : personDF,
            "subject" : subjectDF
            }

df = pd.read_csv("core.csv", index_col=0)

st.write("## News Data")
st.dataframe(df)

st.write("## Reference Based Visualizations")

st.write("### Chloropeth Reference Chart")
fig = px.choropleth(countryDF, locations="value",
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
             title = f"News References Makeup by {pieChoose}",
            labels = {"count": "references",
                    "value":"subject"})

fig.update_traces(textposition='inside')
st.plotly_chart(fig, use_container_width=True)

st.write("## Sentiment Tracking")

yearChoice = st.slider('Choose a year', 1900, 2022, 1)
st.write("## Language Analyisis")
source = list(df.loc[df["year"] == yearChoice]['content'].values)
long_string = ','.join(source)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
st.image(wordcloud.to_image())