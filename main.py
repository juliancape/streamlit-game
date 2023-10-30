import streamlit as st
from streamlit_text_label import label_select
from utils import code_input
from sklearn.model_selection import train_test_split
from streamlit_tags import st_tags
import numpy as np
import time
import pandas as pd

# Just add it after st.sidebar:
#   sidebar = st.sidebar.radio('Choose:',[1,2])

st.title('Aventura de Datos: Domina el Aprendizaje Supervisado')
st.markdown("""
            En un mundo impulsado por datos y tecnología, el aprendizaje supervisado se ha convertido en una de las piedras angulares 
            del análisis de datos y la inteligencia artificial. ¡Bienvenidos a "Aventura de Datos: Domina el Aprendizaje Supervisado"! 
            Este emocionante juego te llevará a un viaje educativo que cambiará la forma en que ves y comprendes los datos.

            El aprendizaje supervisado es una técnica fundamental en la que se entrena un modelo de inteligencia artificial utilizando 
            datos etiquetados. Estos datos etiquetados actúan como ejemplos para enseñar al modelo a realizar predicciones precisas. 
            Ya sea para clasificar correos electrónicos como spam o no spam, predecir el precio de las acciones o diagnosticar enfermedades, 
            el aprendizaje supervisado está detrás de muchas de las aplicaciones inteligentes que utilizamos a diario.
""")







st.header('_Desafio 1 -_ :blue[Titanic] 🚢', divider='grey')
img_titanic = 'media/titanic.jpg'
st.image(img_titanic)
st.markdown("""
            El objetivo principal al trabajar con el conjunto de datos del Titanic es predecir si un pasajero sobreviviría al naufragio en
             función de las características disponibles. Esto se hace a través de un modelo de aprendizaje supervisado de clasificación, 
            donde el resultado deseado es predecir si un pasajero sobrevivio o no.
""")




st.subheader('1. Cargue el Dataset del Titanic', divider='rainbow')
updated_file = st.file_uploader('👀 Cargue el Archivo CSV del titanic', type = ['csv'], accept_multiple_files = False)
if updated_file is not None:
    st.markdown("_cambia la ruta del archivo :green['ruta_archivo.csv'] por la que deberia de ser :blue['files/titanic.csv']_")
    _, return_code = code_input(code_string="""# Para leer un csv en python y generar su DataFrame se utiliza 
import pandas as pd
df_titanic =  pd.read_csv("files/titanic.csv")#pd.read_csv("ruta_archivo.csv")
#print(df_titanic)""", height= '200px')
    
    if "files/titanic.csv" in return_code: 
        df_titanic = pd.read_csv(updated_file)
        st.success('Correcto! ✔️✔️')
        bar = st.progress(10)
        st.dataframe(data = df_titanic)
        st.markdown("felicitaciones, acabas de cargar tu primer dataset")
        st.markdown('Esto es la visualizacion del df_titanic, podras observar los datos y  columnas que trae 🧐')
        st.markdown("""Aquí tienes una explicación de cada una de las variables en el conjunto de datos del Titanic (df_titanic):

1. **PassengerId**: Este es un identificador único asignado a cada pasajero del Titanic. Cada pasajero tiene un número de identificación diferente.

2. **Survived**: Esta variable indica si un pasajero sobrevivió o no. Puede tomar dos valores:
   - 0: El pasajero no sobrevivió.
   - 1: El pasajero sobrevivió.

3. **Pclass**: Representa la clase en la que viajaba el pasajero. Puede tomar uno de los tres valores:
   - 1: Primera clase.
   - 2: Segunda clase.
   - 3: Tercera clase.

4. **Name**: El nombre del pasajero, que incluye su título, nombre, y apellidos.

5. **Sex**: El género del pasajero. Puede ser "male" (hombre) o "female" (mujer).

6. **Age**: La edad del pasajero en años. Esta variable representa la edad del pasajero en el momento del viaje en el Titanic.

7. **SibSp**: El número de hermanos o cónyuges a bordo del Titanic. Indica cuántos hermanos o cónyuges viajaban con el pasajero.

8. **Parch**: El número de padres o hijos a bordo del Titanic. Indica cuántos padres o hijos viajaban con el pasajero.

9. **Ticket**: El número de boleto del pasajero, que es un identificador único para cada boleto.

10. **Fare**: El costo del boleto del pasajero. Representa el precio pagado por el boleto.

11. **Cabin**: La cabina en la que se alojaba el pasajero. Esta variable contiene información sobre la ubicación de la cabina del pasajero en el barco.

12. **Embarked**: El puerto de embarque del pasajero. Puede tomar uno de tres valores:
    - "C": Cherbourg
    - "Q": Queenstown
    - "S": Southampton

Estas variables son parte de un conjunto de datos ampliamente utilizado en proyectos de análisis de datos y aprendizaje automático para predecir si un pasajero sobrevivió al desastre del Titanic en función de sus características personales y el boleto que tenían. El análisis de este conjunto de datos a menudo se utiliza como un ejercicio introductorio en el campo del aprendizaje automático y la ciencia de datos.""")

    st.subheader('2. Selecciona las variables que no aportan valor al problema🛠️', divider='rainbow')
    st.markdown("""Respecto al dataset del titanic hay variables que no aportarian valor al modelo, por ejemplo
                el nombre (Name) de una persona no es relevante para predecir si una persona sobrevivio o no, el 
                PassengerId no es relevante para predecir si una persona sobrevive o no""")
    st.markdown('Con esa logica, seleciona las variables que no estan relacionadas con si sobrevivio o no')
    keywords = st_tags(
    label='# Ingresa Variables Irrelevantes a Eliminar:',
    text='Ingresar 4 variables',
    value=["Name", "Ticket", "Cabin", "PassengerId"],
    suggestions=['Cabin', 'Name', 'Ticket', 'Sex', 
                 'Embarked', 'Survived', 'Pclass', 
                 'SibSp', 'Cabin', 'PassengerId'],
    maxtags = 4,
    key='1')
    
    if set(keywords) == set(["Name", "Ticket", "Cabin", "PassengerId"]):
        
        #["PassengerId", "Name", "Ticket", "Cabin"] variables irrelevantes
        df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
        st.success('Correcto! ✔️✔️')
        bar =  st.progress(20)
        code_input(code_string= """# Este es el codigo de lo que acabaste de hacer
df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)""" , height='100px', action="Copy")




    st.subheader('3. Codificar variables categóricas', divider='rainbow')
    st.markdown('TEXTO EN PROCESO xd')
    _, return_code3 = code_input(code_string="""#Codificar variables categóricas
df_titanic = pd.get_dummies(df_titanic)
""", height="200px", cache="""df_titanic = pd.read_csv("files/titanic.csv")
df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
""")
    if "df_titanic=pd.get_dummies(df_titanic)" in return_code3:
        st.success('Correcto! ✔️✔️')
        bar = st.progress(30)
        df_titanic = pd.get_dummies(df_titanic)
        st.dataframe(data = df_titanic)





    #Actividad 4
    st.subheader('4. Seleccione las variables predictorias y Objetivo', divider='rainbow')
    st.markdown('_Para borrar una anotacion, dar click en el texto resaltado y presionar backspace_')
    labels_selection = """Los datos contienen información sobre cada pasajero a bordo del Titanic, incluyendo su 
                        PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked"""
    selections = label_select(body=labels_selection, labels=["Variables predictoras", "Variable objetivo"])

    text_input_list = []
    labels_input_list = []
    for item in selections:
        text = item.text
        label = item.labels
        text_input_list.append(text)
        labels_input_list.append(label)
    #TO DO: comprobar que las listas tengan los elementos que deben tener y que la variable objetivo en text_input_list este en la misma posicion (index) que labels_input_list
    
    
    real_text_list = ['Pclass','Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                        'Fare', 'Embarked']
    msg1 = ""
    
    if len(text_input_list) == 8:
        for i in range(len(text_input_list)):
            if any(text_input_list[i] in j for j in real_text_list):
            #if real_text_list.contains():
                print(text_input_list[i])    
                if text_input_list[i] == 'Survived':
                    if labels_input_list[i][0] == 'Variable objetivo':
                        msg1 = "**Correcto!** ✔️✔️"
                    else: 
                        msg1 = "**Verifica tus elecciones.** ❌"
            else:
                msg1 = "**Alguna eleccion no coincide, lee detenidamente.** 👁️‍🗨️"
    else:
        msg1 = "**Selecciona todas las variables.** 👁️‍🗨️"
        
    if msg1 == "**Correcto!** ✔️✔️":
        st.success('Correcto! ✔️✔️')
        bar =  st.progress(40)
    else:
        st.markdown(msg1)
    
    st.subheader('5. Divide tu df_titanic en _X_ y _Y_', divider='rainbow')
    st.markdown("""En un proyecto de aprendizaje supervisado, se divide el conjunto de datos en dos partes principales: 
                X y Y, que representan las características (variables predictorias o independientes) y la variable objetivo (variable dependiente) respectivamente.""")
    
    _, return_code4 = code_input(code_string="""# Seleccionando features(X) y target(y)
#Utiliza el .drop para eliminar la variable que no pertenece a las variables independientes
X = df_titanic.drop("Survived", axis = 1) 
# Crea "y" con la unica columna que necesitas
y = df_titanic["Survived"]""", height='200px', cache="""df_titanic = pd.read_csv("files/titanic.csv")
df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
df_titanic = pd.get_dummies(df_titanic)
""")
    X = df_titanic.drop("Survived", axis=1) 
    y = df_titanic["Survived"]
    if 'X=df_titanic.drop("Survived",axis=1)' in return_code4 and 'y=df_titanic["Survived"]' in return_code4:
        st.success('Correcto! ✔️✔️')
        bar = st.progress(50)


    st.subheader('6. Crea el df de train (entrenamiento) y de prueba(test)', divider='rainbow')
    st.markdown("""Hacer la división entre conjuntos de entrenamiento (train) y prueba (test) 
                es una de las prácticas más importantes en el aprendizaje supervisado, los datos de entrenamiento (train)
                son los primeros datos con los que se entrena el modelo para ya luego probar si verdaderamente funciona el modelo
                es necesario probar con datos con los que el modelo no se ha entrenado, esos son los datos de prueba (test)""")
    st.markdown('Para este desafio es necesario que leas la documentacion de esta libreria para hacer el train y test: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html')
    _, return_code4 = code_input(code_string="""# Dividir en train y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)""", height='200px', cache="""df_titanic = pd.read_csv("files/titanic.csv")
df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
X = df_titanic.drop("Survived", axis=1) 
y = df_titanic["Survived"]""")
    if 'X_train,X_test,y_train,y_test=train_test_split(X,y' in return_code4:
        st.success('Correcto! ✔️✔️')
#X_train, ..., ..., y_test = train_test_split(..., ..., test_size=..., random_state=42)    
    

    st.subheader('7. Crea tu primer modelo de aprendizaje supervisado!🛠️', divider='rainbow')
    st.markdown("""Construiremos lo que se conoce como modelo de bosque aleatorio. Este modelo 
                está construido con varios "árboles" (hay tres árboles en la imagen siguiente, 
                ¡pero construiremos 100!) que considerarán individualmente los datos de cada pasajero 
                y votarán si el individuo sobrevivió. Luego, el modelo de bosque aleatorio toma una 
                decisión democrática: ¡gana el resultado con más votos!""")
    img_trees = 'media/trees.PNG'
    st.image(img_trees)



# Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )


















# Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!") 