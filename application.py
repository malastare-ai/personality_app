# Import required packages
from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file
import pandas as pd
import numpy as np
import json
import xlrd
from sklearn.feature_extraction.text import TfidfVectorizer

import io

# Create flask application
application= Flask(__name__)

class DataStore():
    data1=None
    datadisplay=None
data=DataStore()

# Create application route
@application.route("/main",methods=["GET","POST"])

@application.route("/",methods=["GET","POST"])
def homepage():
  # Call value from html form
  # Start- Call value from HTML form
  
  # data1=main()
  data1 = request.form.get('question_field', 'Education')
  params = data1
  data.data1=data1
  # session['my_var']=data1
  # End
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.options.mode.chained_assignment = None
  
  # Read raw data as a DataFrame
  datasource1 = pd.read_csv("Personalities.csv")
  datasource1 = pd.DataFrame(datasource1)

  # Start- Here divide inputs into multiple words and filter data using words to narrow sample size
  # Separate words into multiple entries
  words = data1.split() # This will be stored as separate items in a list

  # Filter data for posts that contain either of these words
  datasource2 = []
  for i in words:
      d2 = datasource1[datasource1['posts'].str.contains(str(i))]
      datasource2.append(d2)

  datasource2 = pd.concat(datasource2, ignore_index=True)
  datasource2 = pd.DataFrame(datasource2)
  datasource2=datasource2.drop_duplicates()
  datasource2=datasource2.reset_index()
  # print(datasource.head())
  # End
  datasource=datasource2
  index_source = 0
  max_simi = 0


  # Create an empty DataFrame called documents which we will use for text analysis
  # We clear the DataFrame each time the user text changes
  # Called within a for loop
  for i in range(0, len(datasource)):
      documents = []
      documents.append(params)
      documents.append(datasource.loc[i, 'posts'])
      ## Start- add stop_words and ngram_range to the TfidVectorizer below
      tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 4)).fit_transform(documents)
      ## End
      # No need to normalize, since Vectorizer will return normalized tf-idf
      pairwise_similarity = tfidf * tfidf.T
      # Add stop word, Add ngram

      temp = pairwise_similarity.A[0][1]

      simiscore = "{0:.4f}".format(temp)
      datasource.loc[i, 'Simiscore'] = simiscore

  ## Compute the rank and add it to the DataFrame as its own column
  # Sort by the rank
  datasource['Rank'] = datasource['Simiscore'].rank(method='average', ascending=False)
  datasource['Search Term']=data1
  datasource = datasource.sort_values('Rank', ascending=True)
  
  # Select top 5 observations
  datasource = datasource.head(5)
  datasource = datasource.reset_index()

  # Select some needed columns to display
  datadisplay = datasource[
      ["type" ,"Rank",
       "Simiscore","Search Term","Introversion/Extraversion","Intuitive/Observant","Thinking/Feeling","Judging/Perceiving"]]
  datadisplay.Rank = datadisplay.Rank.astype(np.int64)

  datadisplay=datadisplay.to_json(orient="records")


  return render_template('Pdocuments.html', docs=json.loads(datadisplay),data1=data1)


# Completion of application with flask
if __name__ == "__main__":
    app.run(debug=True)
