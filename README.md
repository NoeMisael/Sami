# Sami

## Instrucciones de funcionamiento

#### para poder utilizar el Sistema sami, se ha agregado el siguiente apartado.
 
### Requisitos
#### Los requisitos para poder correr el sistema de Sami en el equipo son

<img src="Media/requisitos.jpeg">

#### Enlace del funcionamiento
##### Se anexa enlace del funcionamiento del modelo SAMi alojado en Youtube:

[Funcionamiento de Sami](https://www.youtube.com/watch?v=26n6SWY3jUc)

#### primeramente se establece el entrenamiento del mismo.

##### Se importan las bibliotecas necesarias para poder correr el codigo

```python
import numpy as np
import IPython
import keras
import librosa
import whisper
from nltk.sentiment import SentimentIntensityAnalyzer
from translate import Translator
import openai as analizer
```

##### Se procede con el proceso de declaracion de funciones y de variables para el funcionamiento del sistema.

```python
translator= Translator(from_lang="es",to_lang="en")

modelo = whisper.load_model("medium")

class Predictions:

    def __init__(self, path, file):
        self.path = path
        self.file = file
    def cargar_modelo(self):
        self.cargar_modelo = keras.models.load_model(self.path)
        return self.cargar_modelo.summary()
    def predicciones(self):
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = np.argmax(self.cargar_modelo.predict(x),axis=-1)
        print("Emocion identificada:", self.convertclasstoemotion(predictions))    
    @staticmethod
    def convertclasstoemotion(pred):

        label_conversion = {'0': 'triste',
                            '1': 'feliz',
                            '2': 'neutral',
                         }
        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
                global emotion
                emotion=value
        return label

```

##### Aqui podremos visualizar los datos con los que el modelo fue entrenado

<img src="Media/trainedModel.jpeg">

##### Creamos las variables que seran necesarias para procesarse, en este caso se trabajaron con 3 resultados
```Python
pred = Predictions(path='../trainedModel/emotion.h5',file='../AudioPrueba/noeneutral.wav')
pred2= Predictions(path='../trainedModel/emotion.h5',file='../AudioPrueba/noefeliz.wav')
pred3= Predictions(path='../trainedModel/emotion.h5',file='../AudioPrueba/maria2.wav')
```

##### Procedemos a cargar los modelos y comenzar con la ejecución de algunas variables necesarias

```Python
pred.cargar_modelo()
pred2.cargar_modelo()
pred3.cargar_modelo()
    
result = modelo.transcribe("../AudioPrueba/noeneutral.wav")
texto= result["text"]
result2 = modelo.transcribe("../AudioPrueba/noefeliz.wav")
texto2=result2["text"]
result3 = modelo.transcribe("../AudioPrueba/maria2.wav")
texto3=result3["text"]
ingles = translation = translator.translate(texto)
ingles2= translation = translator.translate(texto2)
ingles3= translation = translator.translate(texto3)
sia = SentimentIntensityAnalyzer()
sia2= SentimentIntensityAnalyzer()
sia3= SentimentIntensityAnalyzer()
sia.polarity_scores(ingles)
sia2.polarity_scores(ingles2)
sia3.polarity_scores(ingles3)

ltk= sia.polarity_scores(ingles)
ltk2= sia2.polarity_scores(ingles2)
ltk3= sia3.polarity_scores(ingles3)
```

##### por ultimo ejecutamos las funciones con las cuales obtenemos los resultados

```Python
print(" En este apartado se imprimen los resultados obtenidos\n")

print("prueba 1 \n")
pred.predicciones()
emocion_detectada = "Recomendaciones si me siento" + emotion
analizer.api_key = "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"
completion = analizer.Completion.create(engine="text-davinci-003",
                                      prompt=emocion_detectada,
                                      max_tokens=2048)

print("\nSegún las palabras detectadas en el audio fueron:\n")
print(texto +"\n")
if(list(ltk.values())[0] > list(ltk.values())[1] and list(ltk.values())[0] > list(ltk.values())[2]):
    print("Emocion detectada triste\n")
elif(list(ltk.values())[1] > list(ltk.values())[0] and list(ltk.values())[1] > list(ltk.values())[2]):
    print("Emocion detectada neutra\n")
else:
    print("Emocion detectada feliz\n")
print("Las recomendaciones para el estado de animo presentado son los siguientes:")
print(completion.choices[0].text)
print("_________________________________________________________________________________")

print("\n\nprueba 2 \n")
pred2.predicciones()
emocion_detectada = "Recomendaciones si me siento" + emotion
analizer.api_key = "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"
completion = analizer.Completion.create(engine="text-davinci-003",
                                      prompt=emocion_detectada,
                                      max_tokens=2048)

print("\nSegún las palabras detectadas en el audio fueron:\n")
print(texto2 +"\n")
if(list(ltk2.values())[0] > list(ltk2.values())[1] and list(ltk2.values())[0] > list(ltk2.values())[2]):
    print("Emocion detectada triste\n")
elif(list(ltk2.values())[1] > list(ltk2.values())[0] and list(ltk2.values())[1] > list(ltk2.values())[2]):
    print("Emocion detectada neutra\n")
else:
    print("Emocion detectada feliz\n")
print("Las recomendaciones para el estado de animo presentado son los siguientes:")
print(completion.choices[0].text)
print("_________________________________________________________________________________")

print("\n\nprueba 3 \n")
pred3.predicciones()
emocion_detectada = "Recomendaciones si me siento" + emotion
analizer.api_key = "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"
completion = analizer.Completion.create(engine="text-davinci-003",
                                      prompt=emocion_detectada,
                                      max_tokens=2048)

print("\nSegún las palabras detectadas en el audio fueron:\n")
print(texto3 +"\n")
if(list(ltk3.values())[0] > list(ltk3.values())[1] and list(ltk3.values())[0] > list(ltk3.values())[2]):
    print("Emocion detectada triste\n")
elif(list(ltk3.values())[1] > list(ltk3.values())[0] and list(ltk3.values())[1] > list(ltk3.values())[2]):
    print("Emocion detectada neutra\n")
else:
    print("Emocion detectada feliz\n")
print("Las recomendaciones para el estado de animo presentado son los siguientes:")
print(completion.choices[0].text)
```

#### Resultados obtenidos

##### Prueba 1
<img src="Media/prueba1.jpeg">

##### Prueba 2
<img src="Media/prueba2.jpeg">

##### Prueba 3
<img src="Media/prueba3.jpeg">



##### Modelo entrenado

##### El modelo entrenado para estas preubas se encuentra en la ruta de 
[emotion.h5](./trainedModel/emotion.h5)

##### El dataset utilizado para el entreamiento es fue el de [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) alojado en Kaggle.com


