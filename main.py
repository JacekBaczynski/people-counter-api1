from flask import Flask, request
from flask_restful import Resource, Api
import cv2
import requests
import numpy as np

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('foto1.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(4, 4))

        return {'count in disk': len(boxes)}


class PeopleCounterLink(Resource):
    def get(self):
        url = request.args.get('url')
        if not url:
            return {'error': 'url is required'}

        try:
            response = requests.get(url)
            response.raise_for_status()

            # Sprawdzenie, czy odpowiedź nie jest pusta
            if not response.content:
                return {'error': 'Empty response from the URL'}

            # Odczytanie obrazu z treści odpowiedzi
            foto2 = cv2.imdecode(
                np.frombuffer(response.content, np.uint8),
                cv2.IMREAD_COLOR
            )

            # Sprawdzenie, czy udało się odczytać obraz
            if foto2 is None:
                return {'error': 'Unable to decode the image'}

            # Detekcja ludzi na obrazie
            boxes, weights = hog.detectMultiScale(foto2, winStride=(5, 5))
            return {'count in link': len(boxes)}

        except Exception as e:
            return {'error': f'Error processing image from URL: {str(e)}'}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(PeopleCounterLink, '/remote')
api.add_resource(PeopleCounter, '/')
api.add_resource(HelloWorld, '/test')

if __name__ == '__main__':
    app.run(debug=True)
