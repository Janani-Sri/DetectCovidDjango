from django.shortcuts import render
from keras.preprocessing import image
import numpy as np
from .deeplearning import graph, model, output_list
import base64

def index(request):
    return render(request,'theme/home.html')

def classify(request):
    if request.method == 'POST' and request.FILES.get('myfile'):
        myfile = request.FILES['myfile']
        b64_img = base64.b64encode(myfile.file.read()).decode('ascii')
        img = image.load_img(myfile,target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        img = img/255

        with graph.as_default():
            prediction = model.predict(img)

        prediction_flatten = prediction.flatten()
        max_val_index = np.argmax(prediction_flatten)
        result = output_list[max_val_index]
        return render(request, "theme/classify.html", {
            'result': result, 'file_url':b64_img
        })
        # return render(request, "classify/index.html", {
        #     'result': result, 'file_url':b64_img
        # })
    return render(request, "theme/classify.html")

def about(request):
    return render(request,'theme/about.html')