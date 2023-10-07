import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request

# dummy intrest data 
interest_data = pd.DataFrame({
    'projects': ['project0', 'project1', 'project2', 'project3', 'project4', 'project5', 'project6'],
    'Interests': ['space', 'Environment', 'Society', 'Web', 'mobile', 'Android', 'Programming']
})

# Sample user preferences
initial_user_preferences = {
    'space': 1,
    'Environment': 0,
    'Society': 1,
    'Web': 0,
    'mobile': 1,
    'Android': 0,
    'Programming': 1,
}

# Demo for the project to be given according to the intrest above
project_interests = {
    'project0': ['space', 'Programming'],
    'project1': ['Environment', 'Society'],
    'project2': ['Society', 'Web'],
    'project3': ['Web', 'mobile'],
    'project4': ['Android', 'Programming'],
    'project5': ['space', 'Society'],
    'project6': ['Environment', 'Programming'],
}

# Create a TF-IDF vectorizer to convert interest text into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(interest_data['Interests'])

# Calculate the cosine similarity between interests and user preferences
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user preferences from the form
        user_form_preferences = {}
        for interest in interest_data['Interests']:
            user_form_preferences[interest] = int(request.form.get(interest, 0))

        # Create a weighted user profile based on form preferences
        user_profile = [user_form_preferences[interest] for interest in interest_data['Interests']]

        # Calculate the weighted sum of interest similarity scores
        weighted_scores = cosine_sim.dot(user_profile)

        # Create a DataFrame to store interest titles and their weighted scores
        interest_scores = pd.DataFrame({'Interest': interest_data['Interests'], 'Weighted_Score': weighted_scores})

        # Sort interests by weighted score in descending order
        recommended_interests = interest_scores.sort_values(by='Weighted_Score', ascending=False)

        # Suggest an interest to explore based on the most relevant interest
        suggested_interest = recommended_interests.iloc[0]['Interest']

        # Find projects related to the suggested interest
        suggested_projects = []
        for project, interests in project_interests.items():
            if suggested_interest in interests:
                suggested_projects.append(project)

        return render_template('index1.html', interest_data=interest_data, suggested_projects=suggested_projects)

    return render_template('index1.html', interest_data=interest_data)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
