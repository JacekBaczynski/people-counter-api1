from flask import Flask
from flask_restful import Resource, Api
import cv2


app = Flask(__name__)
api = Api(app)


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('foto1.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(4, 4))

        return {'count' : len(boxes)}

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(PeopleCounter , '/')
api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)
