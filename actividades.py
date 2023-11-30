import streamlit as st
from streamlit_text_label import label_select
from utils import code_input
from streamlit_tags import st_tags
import time
import os
import subprocess

# Imports actividades
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Imports adicionales
from main import actualizar
import utils

# Validacion de si esta correcta la actividad, y el resultado esperado
valores = [
    [False, ["import pandas as pd", 
             "import numpy as np", 
             "from sklearn.model_selection import train_test_split",
             "from sklearn.tree import DecisionTreeClassifier",
             "from sklearn.metrics import accuracy_score"]], 
    [False, "files/titanic.csv"],
    [False, "df_titanic.describe()"], 
    [False, "df_titanic.info()"], 
    [False, "df_titanic = df_titanic.dropna()"], 
    [False, ""], 
    [False, "df_titanic=pd.get_dummies(df_titanic)"], 
    [False, ""], 
    [False, ['X=df_titanic.drop("Survived",axis=1)', 
             'y=df_titanic["Survived"]']], 
    [False, "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"], 
    [False, "DecisionTreeClassifier(random_state=42)"], 
    [False, "clf.fit(X_train, y_train)"], 
    [False, "clf.predict(X_test)"], 
    [False, "accuracy = accuracy_score(y_test, y_pred)"], 
    [False, "accuracy"]
    ]

# Actualizar la validacion de una actividad
def asig_valores(i, val):
    valores[i][0] = val

# Enviar valores
def ret_valores():
    return valores

puntaje_juego = 0

def asig_puntaje_juego(val):
    global puntaje_juego
    puntaje_juego = val
    
def ret_puntaje_juego():
    return puntaje_juego

# Actividad 1
def act_1():
    st.subheader('1. Importación de Bibliotecas', divider='rainbow')
    st.markdown("Se importan las bibliotecas necesarias")
    _, return_code1 = code_input(code_string="""# Importa pandas como pd
                                 \n# Importa numpy como np
                                 \n# Importa train_test_split de sklearn.model_selection
                                 \n# Importa DecisionTreeClassifier de sklearn.tree import
                                 \n# Importa accuracy_score de sklearn.metrics""", height= '300px')
    if ret_valores()[0][1][0].replace(' ', '') in return_code1 and ret_valores()[0][1][1].replace(' ', '') in return_code1 and ret_valores()[0][1][2].replace(' ', '') in return_code1 and ret_valores()[0][1][3].replace(' ', '') in return_code1 and ret_valores()[0][1][4].replace(' ', '') in return_code1:
        st.success('Correcto! ✔️✔️')
        bar = st.progress(6)
        
        actualizar(0, True, '''import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score\n''')   

# Actividad 2
def act_2():
    st.subheader('2. Cargue el Dataset del Titanic', divider='rainbow')
    updated_file = st.file_uploader('👀 Cargue el Archivo CSV del titanic', type = ['csv'], accept_multiple_files = False)
    if updated_file is not None:
        st.markdown("_cambia la ruta del archivo :green['ruta_archivo.csv'] por la que deberia de ser :blue['files/titanic.csv']_")
        _, return_code = code_input(code_string="""# Para leer un csv en python y generar su DataFrame se utiliza\ndf_titanic =  pd.read_csv("ruta_de_tu_archivo.csv")""", cache=utils.ret_cache(), height= '200px')
        if ret_valores()[1][1].replace(' ', '') in return_code: 
            
            df_titanic = pd.read_csv('files/titanic.csv')
            
            st.success('Correcto! ✔️✔️')
            bar = st.progress(12)
            st.dataframe(data = df_titanic)
            actualizar(1, True, '''df_titanic = pd.read_csv('files/titanic.csv')\n''')
            
            st.markdown("Felicitaciones, acabas de cargar tu primer dataset!")
            st.markdown('Esto es la visualizacion del df_titanic, podras observar los datos y  columnas que trae 🧐')
            st.markdown('Aquí tienes una explicación de cada una de las variables en el conjunto de datos del Titanic (df_titanic):')
            with st.expander("Ver las variables del dataset:"):
                st.markdown("""
                            \n1. **PassengerId**: Este es un identificador único asignado a cada pasajero del Titanic. Cada pasajero tiene un número de identificación diferente.
                            \n2. **Survived**: Esta variable indica si un pasajero sobrevivió o no. Puede tomar dos valores:
                            \n    - 0: El pasajero no sobrevivió.
                            \n    - 1: El pasajero sobrevivió.
                            \n3. **Pclass**: Representa la clase en la que viajaba el pasajero. Puede tomar uno de los tres valores:
                            \n    - 1: Primera clase.
                            \n    - 2: Segunda clase.
                            \n    - 3: Tercera clase.
                            \n4. **Name**: El nombre del pasajero, que incluye su título, nombre, y apellidos.
                            \n5. **Sex**: El género del pasajero. Puede ser "male" (hombre) o "female" (mujer).
                            \n6. **Age**: La edad del pasajero en años. Esta variable representa la edad del pasajero en el momento del viaje en el Titanic.
                            \n7. **SibSp**: El número de hermanos o cónyuges a bordo del Titanic. Indica cuántos hermanos o cónyuges viajaban con el pasajero.
                            \n8. **Parch**: El número de padres o hijos a bordo del Titanic. Indica cuántos padres o hijos viajaban con el pasajero.
                            \n9. **Ticket**: El número de boleto del pasajero, que es un identificador único para cada boleto.
                            \n10. **Fare**: El costo del boleto del pasajero. Representa el precio pagado por el boleto.
                            \n11. **Cabin**: La cabina en la que se alojaba el pasajero. Esta variable contiene información sobre la ubicación de la cabina del pasajero en el barco.
                            \n12. **Embarked**: El puerto de embarque del pasajero. Puede tomar uno de tres valores:
                            \n    - "C": Cherbourg
                            \n    - "Q": Queenstown
                            \n    - "S": Southampton
                            \nEstas variables son parte de un conjunto de datos ampliamente utilizado en proyectos de análisis de datos y aprendizaje automático para predecir si un pasajero sobrevivió al desastre del Titanic en función de sus características personales y el boleto que tenían. El análisis de este conjunto de datos a menudo se utiliza como un ejercicio introductorio en el campo del aprendizaje automático y la ciencia de datos."""
                )

# Actividad 3
def act_3():
    st.subheader('3. Exploracion de Datos', divider='rainbow')
    st.markdown('''Es el proceso de analizar y visualizar datos para comprender sus características, patrones y distribuciones. 
                Ayuda a identificar posibles problemas, tendencias y relaciones en los datos.
                \nVamos a emplear la funcion .describe() en df_titanic, para obtener un resumen estadistico de los datos.''')
    _, return_code = code_input(code_string="""# describe()
                                \ndf_titanic""", height='200px')
    if ret_valores()[2][1].replace(' ', '') in return_code:
        
        df_titanic = pd.read_csv('files/titanic.csv')
        #df_titanic.describe()
        
        asig_valores(2, True)
        st.dataframe(data = df_titanic.describe())
        st.success('Correcto! ✔️✔️')
        bar = st.progress(18)

# Actividad 4
def act_4():
    st.subheader('4. Información del dataframe', divider='rainbow')
    st.markdown('''Vamos a emplear la funcion .info() en df_titanic. Esta se refiere a los detalles y propiedades asociados con un DataFrame, 
                una estructura de datos tabular utilizada en análisis de datos, donde las filas y columnas representan observaciones y variables, respectivamente.''')
    _, return_code = code_input(code_string="""# info()
                                \ndf_titanic""", height='200px')
    if ret_valores()[3][1].replace(' ', '') in return_code:
        
        df_titanic = pd.read_csv('files/titanic.csv')
        #df_titanic.info()
        
        asig_valores(3, True)
        st.image('media/infoDataset.png')
        st.success('Correcto! ✔️✔️')
        bar = st.progress(24)

# Actividad 5
def act_5():
    st.subheader('5. Eliminación de datos nulos', divider='rainbow')
    st.markdown('''Vamos a emplear la funcion dropna() en df_titanic. Esta implica la exclusión de filas o columnas que contienen valores faltantes en un conjunto de datos o datos nulos. 
                Esto se realiza para evitar problemas durante el análisis y modelado, asegurando la integridad de los datos.''')
    _, return_code = code_input(code_string="""# dropna()
                                \ndf_titanic = df_titanic""", height='200px')
    if ret_valores()[4][1].replace(' ', '') in return_code:
        
        df_titanic = pd.read_csv('files/titanic.csv')
        
        st.dataframe(data = df_titanic.dropna())
        df_titanic = df_titanic.dropna()
        
        actualizar(4, True, '''df_titanic = df_titanic.dropna()\n''')
        st.success('Correcto! ✔️✔️')
        bar = st.progress(30)

# Actividad 6
def act_6():
    st.subheader('6. Selecciona las variables que no aportan valor al problema 🛠️', divider='rainbow')
    st.markdown("""Respecto al dataset del titanic hay variables que no aportarian valor al modelo, por ejemplo
                el nombre (Name) de una persona no es relevante para predecir si una persona sobrevivio o no, el 
                PassengerId no es relevante para predecir si una persona sobrevive o no.""")
    st.markdown('Con esa logica, seleciona las variables que no estan relacionadas con si sobrevivio o no')
    keywords = st_tags(
    label='Ingresa Variables Irrelevantes a Eliminar:',
    text='Ingresar 4 variables',
    value=[],
    suggestions=['cabin', 'name', 'ticket', 'sex', 
                'embarked', 'survived', 'pclass', 
                'sibsp', 'cabin', 'passengerid'],
    maxtags = 4,
    key='1')
    if set(keywords) == set(["name", "ticket", "cabin", "passengerid"]):
        
        df_titanic = pd.read_csv('files/titanic.csv')
        df_titanic = df_titanic.dropna()
        df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
        
        actualizar(5, True, '''df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)\n''')
        
        st.success('Correcto! ✔️✔️')
        bar =  st.progress(36)
        st.code("""# Este es el codigo de lo que acabaste de hacer
                \ndf_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)""", language='python')

# Actividad 7
def act_7():         
    st.subheader('7. Codificar variables categóricas', divider='rainbow')
    st.markdown('''Codificar variables categóricas con get_dummies es un proceso de conversión de variables 
                categóricas en variables dummy o indicadoras. Este método es comúnmente utilizado en el 
                preprocesamiento de datos antes de aplicar algoritmos de aprendizaje automático que requieren 
                datos numéricos en lugar de categóricos.''')
    _, return_code3 = code_input(code_string="""# Codificar variables categóricas
                                    \n# En los parentesis ingresa nuestro df_titanic
                                    \ndf_titanic = pd.get_dummies(_)""", cache=utils.ret_cache(), height="200px")
    if ret_valores()[6][1].replace(' ', '') in return_code3:
        st.success('Correcto! ✔️✔️')
        bar = st.progress(42)
        
        df_titanic = pd.read_csv('files/titanic.csv')
        df_titanic = df_titanic.dropna()
        df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
        df_titanic = pd.get_dummies(df_titanic)
        
        st.dataframe(data = df_titanic)
        actualizar(6, True, '''df_titanic = pd.get_dummies(df_titanic)\n''')

# Actividad 8
def act_8():
    st.subheader('8. Seleccione las variables predictorias y Objetivo', divider='rainbow')
    st.markdown('_Para borrar una anotacion, dar click en el texto resaltado y presionar backspace_')
    st.markdown('Los datos contienen información sobre cada pasajero a bordo del Titanic, incluyendo:')
    labels_selection = """Survived\nPclass\nSex\nAge\nSibSp\nParch\nFare\nEmbarked"""
    selections = label_select(body=labels_selection, labels=["Variables predictoras", "Variable objetivo"])
    text_input_list = []
    labels_input_list = []
    for item in selections:
        text = item.text
        label = item.labels
        text_input_list.append(text)
        labels_input_list.append(label)                   
    real_text_list = ['Pclass','Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                        'Fare', 'Embarked']
    msg1 = ""
    if len(text_input_list) == 8:
        for i in range(len(text_input_list)):
            if any(text_input_list[i] in j for j in real_text_list): 
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
        bar =  st.progress(48)
        
        # Validacion 
        asig_valores(7, True)
    else:
        st.markdown(msg1)

# Actividad 9     
def act_9():
    st.subheader('9. Divide tu df_titanic en _X_ y _Y_', divider='rainbow')
    st.markdown("""En un proyecto de aprendizaje supervisado, se divide el conjunto de datos en dos partes principales: 
                X y Y, que representan las características (variables predictorias o independientes) y la variable objetivo (variable dependiente) respectivamente.""")
    _, return_code4 = code_input(code_string="""# Seleccionando features(X) y target(y)
                                    \n# Utiliza el .drop para eliminar la variable que no pertenece a las variables independientes (la variable objetivo)
                                    \nX = df_titanic.drop("", axis = 1)
                                    \n# Crea "y" con la unica columna que necesitas (con la variable objetivo)
                                    \ny = df_titanic[""]""", cache=utils.ret_cache(), height='300px')
    if ret_valores()[8][1][0].replace(' ', '') in return_code4 and ret_valores()[8][1][1].replace(' ', '') in return_code4:
        
        df_titanic = pd.read_csv('files/titanic.csv')
        df_titanic = df_titanic.dropna()
        df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
        df_titanic = pd.get_dummies(df_titanic)
        X = df_titanic.drop("Survived", axis=1) 
        y = df_titanic["Survived"]
        
        st.success('Correcto! ✔️✔️')
        bar = st.progress(56)
        actualizar(8, True, '''X = df_titanic.drop("Survived",axis=1)\ny = df_titanic["Survived"]\n''')

# Actividad 10
def act_10():
    st.subheader('10. Crea el df de train (entrenamiento) y de prueba(test)', divider='rainbow')
    st.markdown("""Hacer la división entre conjuntos de entrenamiento (train) y prueba (test) es una de las prácticas 
                más importantes en el aprendizaje supervisado, los datos de entrenamiento (train) son los primeros datos 
                con los que se entrena el modelo para ya luego probar si verdaderamente funciona el modelo es necesario 
                probar con datos con los que el modelo no se ha entrenado, esos son los datos de prueba (test)""")
    st.markdown('Para este desafio es necesario que leas la documentacion de esta libreria para hacer el train y test: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html')
    st.markdown('Para esta sección, usa un tamaño de test del 20% y un random_state de 42')
    _, return_code4 = code_input(code_string="""# Dividir en train y test
                                    \nX_train, X_test, y_train, y_test = train_test_split()""", cache=utils.ret_cache(), height='200px')
    if ret_valores()[9][1].replace(' ', '') in return_code4:
        st.success('Correcto! ✔️✔️')
        bar = st.progress(60)
        actualizar(9, True, '''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n''')
        
        df_titanic = pd.read_csv('files/titanic.csv')
        df_titanic = df_titanic.dropna()
        df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
        df_titanic = pd.get_dummies(df_titanic)
        X = df_titanic.drop("Survived", axis=1) 
        y = df_titanic["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Actividad 11        
def act_11():
    st.subheader('11. Creación del árbol de decisión', divider='rainbow')
    img_trees = 'media/trees.PNG'
    st.image(img_trees)
    st.markdown('''Implica la construcción de un modelo de árbol que representa decisiones basadas en características del conjunto de datos. 
                Es una técnica común en algoritmos de clasificación.\nVamos a emplear DecisionTreeClassifier para la creacion
                del arbol de decision.\nPuedes averiguar más de esta clase de arboles de decision en: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
                \nRecuerda usar un random_state de 42''')
    _, return_code = code_input(code_string="""# Crea un objeto DecisionTreeClassifier()
                                \nclf = DecisionTreeClassifier""", height='200px')
    if ret_valores()[10][1].replace(' ', '') in return_code:
        clf = DecisionTreeClassifier(random_state=42)
        
        actualizar(10, True, '''clf = DecisionTreeClassifier(random_state=42)\n''')
        st.success('Correcto! ✔️✔️')
        bar = st.progress(66)

# Actividad 12
def act_12():
    st.subheader('12. Entrenar el modelo en los datos de entrenamiento', divider='rainbow')
    st.markdown('''Implica alimentar el algoritmo con datos de entrenamiento para que ajuste sus parámetros internos y 
                aprenda patrones en los datos. Este proceso permite que el modelo pueda hacer predicciones en nuevos datos.
                \nVamos a emplear la funcion fit() al objeto clf. Y le vamos a enviar los datos de entrenamiento de X y Y, en ese orden.''')
    _, return_code = code_input(code_string="""# fit(conjunto de entrenamiento de x, conjunto de entrenamiento de y)
                                \nclf.fit(_, _)""", height='200px')
    if ret_valores()[11][1].replace(' ', '') in return_code:
        
        df_titanic = pd.read_csv('files/titanic.csv')
        df_titanic = df_titanic.dropna()
        df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
        df_titanic = pd.get_dummies(df_titanic)
        X = df_titanic.drop("Survived", axis=1) 
        y = df_titanic["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        actualizar(11, True, '''clf.fit(X_train, y_train)\n''')
        st.success('Correcto! ✔️✔️')
        bar = st.progress(72)

# Actividad 13
def act_13():
    st.subheader('13. Predicciones en los datos de prueba', divider='rainbow')
    st.markdown('''Después de entrenar el modelo, se utiliza para hacer predicciones en un conjunto de datos de prueba independiente. 
                Esto evalúa la capacidad del modelo para generalizar y realizar predicciones precisas en datos no vistos durante el entrenamiento.
                \nVamos a emplear la funcion predict() al objeto clf. Y le vamos a enviar los datos de testeo de X.''')
    _, return_code = code_input(code_string="""# predict(conjunto de testeo de X)
                                \n# y_pred va a ser nuestro conjunto de datos predecidos
                                \ny_pred = clf.predict(_)""", height='200px')
    if ret_valores()[12][1].replace(' ', '') in return_code:
        
        df_titanic = pd.read_csv('files/titanic.csv')
        df_titanic = df_titanic.dropna()
        df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
        df_titanic = pd.get_dummies(df_titanic)
        X = df_titanic.drop("Survived", axis=1) 
        y = df_titanic["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        actualizar(12, True, '''y_pred = clf.predict(X_test)\n''')
        st.success('Correcto! ✔️✔️')
        bar = st.progress(78)

# Actividad 14
def act_14():
    st.subheader('14. Precisión del modelo en los datos de prueba', divider='rainbow')
    st.markdown('''Es una métrica que evalúa qué tan bien el modelo hace predicciones correctas en los datos de prueba. Se calcula dividiendo 
                el número de predicciones correctas entre el número total de predicciones realizadas.
                \nVamos a emplear accuracy_score para validar el desempeño y capacidad de clasificacion de nuestro modelo
                de decisión.\nEsta función realiza la precisión del modelo en los datos de prueba''')
    _, return_code = code_input(code_string="""# accuracy(datos de testeo de y, datos predecidos)
                                \naccuracy = accuracy_score(_, _)""", height='200px')
    if ret_valores()[13][1].replace(' ', '') in return_code:
        
        df_titanic = pd.read_csv('files/titanic.csv')
        df_titanic = df_titanic.dropna()
        df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
        df_titanic = pd.get_dummies(df_titanic)
        X = df_titanic.drop("Survived", axis=1) 
        y = df_titanic["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        actualizar(13, True, '''accuracy = accuracy_score(y_test, y_pred)\n''')
        st.success('Correcto! ✔️✔️')
        bar = st.progress(84)

# Actividad 15
def act_15():
    st.subheader('15. Imprimir la precisión', divider='rainbow')
    st.markdown('''Vamos a mostrar la precisión de nuestro modelo. para esto vamos a imprimir el accuracy obtenido en el paso anterior.''')
    _, return_code = code_input(code_string="""# Imprime accuracy
                                \nprint(f"Precisión del modelo: {_:.2f}")""", height='200px')
    if ret_valores()[14][1].replace(' ', '') in return_code:
        
        df_titanic = pd.read_csv('files/titanic.csv')
        df_titanic = df_titanic.dropna()
        df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
        df_titanic = pd.get_dummies(df_titanic)
        X = df_titanic.drop("Survived", axis=1) 
        y = df_titanic["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        #print(f"Precisión del modelo: {accuracy:.2f}")
        
        st.text("Resultado:")
        st.write(accuracy)

        # Validacion 
        asig_valores(14, True)
        #actualizar(14, True, '''accuracy = accuracy_score(y_test, y_pred)\n''')
        st.success('Correcto! ✔️✔️')
        bar = st.progress(90)

# Activar Juego
def act_juego():
    st.subheader('JUEGO DEL TITANIC', divider='rainbow')
    st.markdown('''Haz completado de forma correcta tu primer modelo de decisión. ¡Ahora es momento de juegues una breve partida de un juego!. Te lo mereces :).''')
    if st.button("JUGAR JUEGO TITANIC SURVIVOR"):
        try:
            ruta_script = os.path.realpath(__file__)
            # Obtiene la carpeta del script
            ruta_del_juego = os.path.dirname(ruta_script) + "/juego/juego_integrado.py"
            subprocess.run(["pip", "install", "pygame"])
            puntaje = subprocess.run(["python3", ruta_del_juego], capture_output=True, text=True)
            
            # Captura la salida estándar del script, que debería ser el puntaje
            output_lines = puntaje.stdout.strip().split('\n')
            puntaje_str = output_lines[-1]
            # variable que contiene el puntaje ganado en el juego
            puntaje_juego = int(puntaje_str)
            
            asig_puntaje_juego(puntaje_juego)
            
            st.text("Felicidades!")
            #st.write(puntaje_juego)
            
        except Exception as e:
            st.error(f"Error de ejecución: {e}")