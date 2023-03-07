import numpy as np
import pandas as pd
from textblob import TextBlob
import plotly_express as px

df = pd.read_csv('netflix_titles.csv')

print(df.shape) #8807 rows 12 columns
print(df.columns) #the name of columns
print(df.head) #first 5 rows

data_rating = df.groupby(['rating']).size().reset_index(name='counts')
piechart = px.pie(data_rating, values='counts', names='rating', title='Distribution of Content Ratings on Netflix', color_discrete_sequence=px.colors.qualitative.Set2)
piechart.show()

#top 5 directors by total count of shows/movies
df['director'] = df['director'].fillna('Not Specified')
directors = pd.DataFrame()
directors=df['director'].str.split(',',expand=True).stack()
directors=directors.to_frame()
directors.columns=['Director']
directors_1 = directors.groupby(['Director']).size().reset_index(name='Total Content')
directors_1 = directors_1[directors_1.Director !='Not Specified']
directors_1 = directors_1.sort_values(by=['Total Content'],ascending=False)
directorsTop5 = directors_1.head()
directorsTop5 = directorsTop5.sort_values(by=['Total Content'])
fig1 = px.bar(directorsTop5,x='Total Content',y='Director',title='Top 5 Directors on Netflix')
fig1.show()

#top 5 actors by total count of movies/shows
df['cast'] = df['cast'].fillna('No Cast Specified')
filtered_cast = pd.DataFrame()
filtered_cast = df['cast'].str.split(',',expand=True).stack()
filtered_cast = filtered_cast.to_frame()
filtered_cast.columns = ['Actor']
actors = filtered_cast.groupby(['Actor']).size().reset_index(name='Total Content')
actors = actors[actors.Actor !='No Cast Specified']
actors = actors.sort_values(by=['Total Content'],ascending=False)
actorsTop5 = actors.head()
actorsTop5 = actorsTop5.sort_values(by=['Total Content'])
fig2 = px.bar(actorsTop5,x='Total Content',y='Actor', title='Top 5 Actors on Netflix')
fig2.show()

#analysing content
df1 = df[['type','release_year']]
df1 = df1.rename(columns={"release_year": "Release Year"})
df2 = df1.groupby(['Release Year','type']).size().reset_index(name='Total Content')
df2 = df2[df2['Release Year']>=2010]
fig3 = px.line(df2, x="Release Year", y="Total Content", color='type',title='Trend of content produced over the years on Netflix')
fig3.show()

#analysing the reviews (sentiments)
dfx = df[['release_year','description']]
dfx = dfx.rename(columns={'release_year':'Release Year'})
for index,row in dfx.iterrows():
    z = row['description']
    testimonial = TextBlob(z)
    p = testimonial.sentiment.polarity
    if p == 0:
        sent = 'Neutral'
    elif p > 0:
        sent = 'Positive'
    else:
        sent = 'Negative'
    dfx.loc[[index,2],'Sentiment'] = sent


dfx = dfx.groupby(['Release Year','Sentiment']).size().reset_index(name='Total Content')

dfx = dfx[dfx['Release Year']>=2010]
fig4 = px.bar(dfx, x="Release Year", y="Total Content", color="Sentiment", title="Sentiment of content on Netflix")
fig4.show()