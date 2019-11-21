
![](https://media.giphy.com/media/8F9uH7lTO9AcAawBRu/giphy.gif)

In this blog, I'll describe an application that helps users predict their own personality type on the **Myers Briggs spectrum** (of 16 personalities) using text analytics in python. The final application is hosted on PaaS - heroku. The application makes use of *flask* for app development and the *tfidvectorizer* from *sklearn* for text analytics.

### Data analysis and key question
The data contains posts by reddit users corresponding to different personality types. The data can be accessed on [kaggle](https://www.kaggle.com/datasnaek/mbti-type). We'll create a personality finder application where a user can enter any set of words that best describes them. The app will output a **similarity score** of the user entered text with the various entries from the personality type data to predict the personality type (1 of 16 Myers Briggs personalities. It will also predict the likelihood of the four Myers Briggs characteristics such as:

* Introverted/Extraverted, 
* Intuitive/Observant, 
* Thinking/Feeling,
* Judging/Perceiving. 

[Learn more about these personality types](https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/home.htm?bhcp=1)

For example, someone who prefers introversion, intuition, thinking and perceiving would be labelled an INTP in the MBTI system, and there are lots of personality based components that would model or describe this person’s preferences or behaviour based on the label.

>It is one of, if not the, the most popular personality test in the world. It is used in businesses, online, for fun, for research and lots more. A simple google search reveals all of the different ways the test has been used over time. It’s safe to say that this test is still very relevant in the world in terms of its use.

>From scientific or psychological perspective it is based on the work done on [cognitive functions](http://www.cognitiveprocesses.com/Cognitive-Functions/) by Carl Jung i.e. Jungian Typology. This was a model of 8 distinct functions, thought processes or ways of thinking that were suggested to be present in the mind. Later this work was transformed into several different personality systems to make it more accessible, the most popular of which is of course the MBTI.

>Recently, its use/validity has come into question because of unreliability in experiments surrounding it, among other reasons. But it is still clung to as being a very useful tool in a lot of areas, and the purpose of this dataset is to help see if any patterns can be detected in specific types and their style of writing, which overall explores the validity of the test in analysing, predicting or categorising behaviour.

The app will showcase the top 5 predictions with the **similarity scores** of the users' personality that fits them.

### Application Structure
The structure will use a standard flask application structure with an index.html file that will contain our UI and host our results as a python file with the code.

Below is a diagrammatic representation of the flow of data in the app.

![](https://r-variawa.rstudio.cloud/1c9cbbba765c4e5bbb6953afd5615006/file_show?path=%2Fcloud%2Fproject%2FPersonality-type-finder-20-11-19%2Fstatic%2Fdata_flow.png)

### Building the UI (User Interface)
The first part - index.html file is easy. This is just some basic text that describes the application itself and a simple image that we'll host in our ‘static’ folder.

```
<head>
<section>
  <img class="mySlides" src="/static/pt.jpg"
  style="height:60%" style="width:110%">
  </section> 
   <h1>Personality type using text analysis</h1>
   <p>Find your personality type using this application that makes use of a text vectorizer to analyze posts by individuals identifying as different personality types. This app will yield your top 5 personalities from the 16 Myers Briggs personality types.</p>
   <p>Kindly note that the script might take a few seconds to run as it parse through over 8 000 entries.
</head>

```

Next, we create a form where the user can submit **words or sentences** that we will use for the text analysis. This is a simple html *post form* with a submit button. We will keep track of the name we assign to this form since this will be used later in the backend.

```
<form method="post" >
    <input name="question_field" placeholder="enter search term here">
    <input type="submit">
</form>

```

Finally, we'll create a table which outputs the top 5 entries of the user. We will also print out various headers such as the Personality Type, Similarity Score, Search Term, etc. We will need to define table heads (‘th’) and table data (‘td’) in html.

```
</style>
  <h2>Relevant personality types displayed below</h2>
<table>
  <tr>
    <th>Personality Type</th>
    <th>SimiScore</th>
    <th>Search Term</th>
    <th>Rank</th>
    <th>Introversion/Extraversion</th>
    <th>Intuitive/Observant</th>
    <th>Thinking/Feeling</th>
    <th>Judging/Perceiving</th>
    
</tr>  

```

Now, the data will be brought in from the python side. So we can use some code to effectively access our python dataframe variables. We'll call our python dataframe in the backend ‘docs’. The code will allow us to iterate over this python dataframe. The order in which each variable is called should be similar to the order of the heads we defined in the block above.

```
{% for doc in docs %}
<tr>
<td>{{doc["Personality Type"]}}</td>
<td>{{doc["Simiscore"]}}</td>
<td>{{doc["Search Term"]}}</td>
<td>{{doc["Rank"]}}</td>
<td>{{doc["Introversion/Extraversion"]}}</td>
<td>{{doc["Intuitive/Observant"]}}</td>
<td>{{doc["Thinking/Feeling"]}}</td>
<td>{{doc["Judging/Perceiving"]}}</td>

</tr>
{% endfor %}
</table>

```

### Building the Backend
The first step is to import all of the required python libraries. As we'll be using text analytics, we'll need to import the *Tfidvectorizer* function from the sklearn package. Also, let’s go ahead and create the application itself in flask. We will also create a standard get and post route for the application so we can effectively send data to it and receive data from it.

```
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

# Create application route
@application.route("/main",methods=["GET","POST"])

```

Now let’s define what will happen on the homepage using a homepage function within the flask application. Now, here is where we'll first try to call the data that the user has entered in the form we defined above in html. As we know the form is labeled ‘question1_field’. We can access this data using a simple request function that will invoke the data. We can also specify a default word that will be called in case the user has not selected any words (The default word for this example is ‘Education’). Let’s also read in the raw data itself as a dataframe using pandas.

```
def homepage():
  # Call value from HTML form  
  words = request.form.get('question_field', 'Education')
  # Read raw data as a dataframe
  datasource1 = pd.read_csv("Personalities.csv")

```

**Dealing with multiple words.** -- A user may enter multiple words as a part of a single entry. We would split these words into multiple items so that we can run the text analysis for each word separately before running the analysis for all the words together. We'll filter the posts in the dataframe for each of the words and bring those filtereted posts into a different dataframe we'll use. So, let’s create another dataframe for that purpose. After creating this new dataframe, let’s drop duplicates just in case the same posts gets picked twice.

```
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

```

Cool. Let’s talk about what a **similarity score** is and how and why we are going to compute it in the given context. A *similarity score* or a *pairwise similarity score* is basically the distance of a particular word from other words in an entry. So basically if a user says ‘I love ice-cream’, these words would have a very high similarity with a post that mentions anything related to ice-cream. Since each post corresponds to a personality type, we can technically ‘predict’ that the user’s personality is similar to the personality type of the user that made that post.

The **tfidfVectorizer** will compute the frequency and significance of words that appear within a particular post. So, if in the above example, if the word ice-cream in the post just occurs in passing, the tfid score would be very low thus yielding a low similarity score. the *tfid vectorizer* would compute its score on the basis of the frequency of the word and its uniqueness. I won’t go into the details of how the *tfidvectorizer* works in this article. So, to compute the similarity between the user text and the post text, we would need the two values side by side in a dataframe. We already have a filtered dataframe with the relevant posts for the user. Let’s create an empty dataframe called documents which we'll use for the text analysis. We want to clear the dataframe each time the user text changes. So, we can call it within the for loop below.

```
for i in range(0, len(datasource)):
  documents = []
  documents.append(params)
  documents.append(datasource.loc[i, 'posts'])

```

Now for each entry in the dataframe, we want to compute the tfidf score for the post with respect to the user entry. This can be accomplished with the *tfidvectorizer* function. The function allows us to specify stop words i.e. words that the function should ignore articulate (a, an, the etc.) and prepositions (over, under, between etc.). So, if we just set the stop words parameter in the function to ‘english’, these words get ignored automatically. Adding the *tfidvectorizer* to compute a score for each entry, our function now looks like.

```
for i in range(0, len(datasource)):
  documents = []
  documents.append(params)
  documents.append(datasource.loc[i, 'posts'])
  ## Start- add stop_words and ngram_range to the TfidVectorizer below
  tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 4)).fit_transform(documents)

```

Great. We know the significance of the user entered words within each of the selected posts. However, we want to compute a *similarity score* using this tfidf score i.e. how similar are the words entered by the user to the words entered in the post. The similarity can also be defined as the cosine distance between the words. 

>A [post](https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents) from stack overflow explains the computation of the pairwise similarity in detail. 

The short explanation is that you can compute the pairwise similarity by multiplying the tfidf score by its transpose. This yields a similarity matrix and accessing the first element of this matrix will give us the similarity score. We'll also format the similarity score to display only the first four decimals and add it to the dataframe as its own column. So, the above code with the pairwise similarity computation becomes

```
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

```

Now that the loop is done, we'll have similarity scores for all the posts in our dataset computed for the text entered by the user. The next steps are ranking the similarity scores, picking the top 5, adding the search term as its own column to the dataframe and selecting the relevant columns that we would like to display. The final output will be assigned to a dataframe called docs that we will pass to our html table.

```
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

```

Since we are sending this data to a html table, we'll have to convert it to a json format. Thankfully, this can be easily achieved in pandas using the *to_json* function. We can also set the orientation to records, so that each entry becomes a record. We can complete the homepage function in flask by returning an object called ‘docs’ which our frontend is expecting. We'll use the *render_template* function in flask to send the data to our index.html file.

```
# Convert to json
docs = docs.to_json(orient="records")
# Return data to frontend for output
return render_template('index.html',docs=json.loads(docs))

```

Finally, we'll finish our application in flask style,

```
# Completion of application with flask
if __name__ == "__main__":
    app.run(debug=True)

```

### Acknowledgements
This data was collected through the [PersonalityCafe forum,](https://www.personalitycafe.com/forum/) as it provides a large selection of people and their MBTI personality type, as well as what they have written.


