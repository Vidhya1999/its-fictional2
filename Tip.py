import os
os.chdir("/home/vidhya/Project_final")
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix

def books(s):
	stringedTags=pd.read_csv('/home/vidhya/Documents/book/Booktags.csv')
	countVec = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
	tagMatrix = countVec.fit_transform(stringedTags['all_tags'])
	tagMatrix
	cosineSim = cosine_similarity(tagMatrix, tagMatrix)
	cosineSim
	stringedTags = stringedTags.reset_index()
	bookTitles = stringedTags['title']
	indices = pd.Series(stringedTags.index, index = bookTitles)
	def topRecommendations(title):
	    index = indices[title]
	    similarityScore = list(enumerate(cosineSim[index]))
	    similarityScore = sorted(similarityScore, key = lambda x: x[1], reverse = True)
	    similarityScore = similarityScore[1:10]
	    bookIndex = [i[0] for i in similarityScore]
	    return bookTitles.iloc[bookIndex]

	TR=topRecommendations(s).head(10)
	TR_dict=TR.to_dict()
	for x in TR_dict.values():
  		print(x) 

def movie(s):
	df = pd.read_csv('/home/vidhya/Documents/Tip/movies.csv')
	features = ['keywords','cast','genres','director']
	for feature in features:
		df[feature] = df[feature].fillna('') 
	df['combined_features'] = df['keywords']+ ' ' + df['cast'] + ' ' + df['genres'] + ' ' + df['director']
	df.to_csv('movie_features.csv')
	cv = CountVectorizer()	
	count_matrix = cv.fit_transform(df['combined_features'])
	cosine_sim = cosine_similarity(count_matrix)

	def get_title_from_index(index):
		return df[df.index == index]['title'].values[0]

	def get_index_from_title(title):
		return df[df.title == title]['index'].values[0]
	movie_user_likes = s
	movie_index = get_index_from_title(movie_user_likes)

	similar_movies = list(enumerate(cosine_sim[movie_index]))
	sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
	i=0
	for element in sorted_similar_movies:
	    print(get_title_from_index(element[0]))
	    i=i+1
	    if i>10:
	        break

def movietobook(s):
	def get_index_from_title(title):
		return movies[movies.title == title]['index'].values[0]

	movies=pd.read_csv('movie_features.csv')
	moviec=pd.read_csv('moviec.csv')
	bookc=pd.read_csv('bookc.csv')
	movie_user_likes = s
	movie_index = get_index_from_title(movie_user_likes)
	mov=moviec.iloc[[movie_index]]
	bookc=bookc.append(mov)
	cv = CountVectorizer() 
	count_matrix = cv.fit_transform(bookc['all_tags'])
	cosine_sim = cosine_similarity(count_matrix)
	def get_title_from_index(index):
		return bookc[bookc.index == index]['title'].values[0]

	similar_books = list(enumerate(cosine_sim[10000]))
	sorted_similar = sorted(similar_books,key=lambda x:x[1],reverse=True)[1:]
	i=0
	for element in sorted_similar:
		print(get_title_from_index(element[0]))
		i=i+1
		if i>10:
			break

s1=raw_input('\nEnter the book name for recommendation of books  :')
books(s1)
s2=raw_input('\nEnter the movie name for recommendation of movies  :')
movie(s2)
s3=raw_input('\nEnter the movie name for recommendation of books  :')
movietobook(s3)
