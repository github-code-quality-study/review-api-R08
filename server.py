import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # perform sentiment analysis on each review and add 'sentiment' key to each review and get valid locations
        self.valid_locations = set()
        for review in reviews:
            review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])
            self.valid_locations.add(review['Location'])



    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            
            # receive the query parameters
            query_string = environ["QUERY_STRING"]
            query_params = parse_qs(query_string)

            filtered_reviews = reviews
            # filter reviews by the parameters
            if query_params:
                if 'location' in query_params:
                    filtered_reviews = [review for review in filtered_reviews if review['Location'] == query_params['location'][0]]
                if 'start_date' in query_params:
                    start_date = datetime.strptime(query_params['start_date'][0], '%Y-%m-%d').date()
                    filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S').date() >= start_date]
                if 'end_date' in query_params:
                    end_date = datetime.strptime(query_params['end_date'][0], '%Y-%m-%d').date()
                    filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S').date() <= end_date]

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # read the request body and convert it to a dictionary
            request_body_size = int(environ.get("CONTENT_LENGTH", 0))
            request_body = environ["wsgi.input"].read(request_body_size)

            # b'Location=San Diego, California&ReviewBody=I love this place!' parse the request body
            response_body = parse_qs(request_body.decode("utf-8"))

            # check if the request body contains the required parameters
            if response_body.get('Location') is None:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Location is required"]
            if response_body.get('ReviewBody') is None:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"ReviewBody is required"]
            
            # location has to be a valid location
            if response_body['Location'][0] not in self.valid_locations:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Location is not valid"]

            # add a unique identifier to the review and timestamp
            response_body['Location'] = response_body['Location'][0]
            response_body['ReviewBody'] = response_body['ReviewBody'][0]
            response_body['ReviewId'] = str(uuid.uuid4())
            response_body['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # return response which is the same request params
            response_body = json.dumps(response_body).encode("utf-8")

            # Set the appropriate response headers
            start_response("201 Created", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()