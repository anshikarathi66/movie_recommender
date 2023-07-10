from django.http import HttpRequest
from django.shortcuts import render, HttpResponse
from django import urls
from django.template import Template, Context

# Create your views here.
def index(request):
    return render(request, 'index.html')

#emotion detector
def external(request):
    choice1=request.POST.get('choice1')
    choice2= request.POST.get('choice2')
    choice3=request.POST.get('choice3')

    choice1=choice1.lower()
    choice2=choice2.lower()
    choice3=choice3.lower()

    joy=["yellow","light orange","blue","green"]
    love=["light red"]
    anger=["dark red"]
    sadness=["brown","grey","black"]
    joy_love=joy + love
    anger_sadness=anger + sadness

    if((choice1 in joy) and (choice2 in joy) and (choice3 in joy)):
        user_emotion="Joy"

    elif((choice1 in love) and (choice2 in love) and (choice3 in love)):
        user_emotion="Love"

    elif((choice1 in anger) and (choice2 in anger) and (choice3 in anger)):
        user_emotion="Anger"

    elif((choice1 in sadness) and (choice2 in sadness) and (choice3 in sadness)):
        user_emotion="Sadness"

    elif((choice1 in joy and choice2 in joy and choice3 not in joy) or (choice1 in joy and choice3 in joy and choice2 not in joy) or (choice3 in joy and choice2 in joy and choice1 not in joy)):
        user_emotion="Joy"

    elif((choice1 in love and choice2 in love and choice3 not in love) or (choice1 in love and choice3 in love and choice2 not in love) or (choice3 in love and choice2 in love and choice1 not in love)):
        user_emotion="Love"

    elif((choice1 in anger and choice2 in anger and choice3 not in anger) or (choice1 in anger and choice3 in anger and choice2 not in anger) or (choice3 in anger and choice2 in anger and choice1 not in anger)):
        user_emotion="Anger"

    elif((choice1 in sadness and choice2 in sadness and choice3 not in sadness) or (choice1 in sadness and choice3 in sadness and choice2 not in sadness) or (choice3 in sadness and choice2 in sadness and choice1 not in sadness)):
        user_emotion="Sadness"

    elif((choice1 in joy_love and choice2 in joy_love and choice3 not in joy_love) or (choice1 in joy_love and choice3 in joy_love and choice2 not in joy_love) or (choice3 in joy_love and choice2 in joy_love and choice1 not in joy_love)):
        user_emotion="Joy and Love"

    elif((choice1 in anger_sadness and choice2 in anger_sadness and choice3 not in anger_sadness) or (choice1 in anger_sadness and choice3 in anger_sadness and choice2 not in anger_sadness) or (choice3 in anger_sadness and choice2 in anger_sadness and choice1 not in anger_sadness)):
        user_emotion="Anger and Sadness"

    context={
        'user_emotion': user_emotion
    }
    return render(request, 'index.html', context=context)

#recommender
def recommender(request):
    import numpy as np #linear algebra
    import pandas as pd #data processing

    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    #read datasets
    movies=pd.read_csv(r'D:\acer\documents\webdevelopmentcourse\atom\recommendersystem\imdb_top_1000.csv')

    movies= movies[['Series_Title','Genre','IMDB_Rating','Overview']]

    # Checking for NULL Values
    movies.isnull().sum()
    # Checking for Duplicated samples
    movies.duplicated().sum()

    #convert genre from string to list
    def Convert(string):
        string = string.replace(',', '')
        li = list(string.split(" "))
        return li
    movies['Genre'] = movies['Genre'].apply(Convert)

    #converting IMDB-rating from float to list
    def Float_to_list(float):
        string=str(float)
        li = list(string.split(" "))
        return li
    movies['IMDB_Rating']= movies['IMDB_Rating'].apply(Float_to_list)

    # Converting Overview into list
    movies['Overview'] = movies['Overview'].apply(lambda x:x.split())

    # Making a new column 'tags' which is made by combining all features
    movies['tags'] = movies['Overview'] + movies['Genre'] + movies['IMDB_Rating']

    #final dataset for model training
    final_dataset = movies[['Series_Title','tags']]

    #joining back all list elements of tags
    final_dataset['tags'] = final_dataset['tags'].apply(lambda x: " ".join(x))

    # Converting all the tags to lowercase
    final_dataset['tags'] = final_dataset['tags'].apply(lambda x : x.lower())

    #stemming
    #nltk-Natural Language Tool Kit
    #nltk.stem is a package to perform stemming using different classes
    #PorterStemmer is one of that class which performs suffix stemming
    stemmer = PorterStemmer()

    def stemming(text):
        stemmed_output = []
        for i in text.split():
            stemmed_output.append(stemmer.stem(i))
        string = " ".join(stemmed_output)
        return string
    final_dataset['tags'] = final_dataset['tags'].apply(stemming)

    #training model
    #vectorize our tags which are in text format.
    #use CosineSimillarity to find similar vectors so that we make recommendation
    #count vectorization involves counting the number of occurrences each words appears in a document
    cv = CountVectorizer(max_features=4787,stop_words='english')
    cv_vector = cv.fit_transform(final_dataset['tags']).toarray()
    cv_vector.shape

    cv_similarity_matrix = cosine_similarity(cv_vector)
    cv_similarity_matrix.shape

    cv_similarity_matrix

    #recommender function
    def recommend(movie):
        index = final_dataset[final_dataset['Series_Title'] == movie].index[0]

        similarity_score = sorted(list(enumerate(cv_similarity_matrix[index])),reverse=True,key = lambda x: x[1])

        ls=[]
        for i in similarity_score[1:6]:
            ls.append(final_dataset.iloc[i[0]].Series_Title)
        return ls

    movie=request.POST.get('movie')
    ans = recommend(movie)
    context={
        'movie1' : ans[0],
        'movie2' : ans[1],
        'movie3' : ans[2],
        'movie4' : ans[3],
        'movie5' : ans[4]
    }
    return render(request, 'index.html', context=context)
