from code_editor import code_editor
import pandas as pd
from sklearn.model_selection import train_test_split
def code_input(code_string: str = '', height: int = '500px', cache = '', action = "Run"):

    if action == "Run":
        button_run = [{
        "name": action,
        "feather": "Play",
        "primary": True,
        "hasText": True,
        "showWithIcon": True,
        "commands": ["submit"],
        "style": {"bottom": "0.44rem", "right": "0.4rem"}
        }]
        response_dict = code_editor(code_string, height=height, buttons = button_run, theme="contrast")
        try:
            exec(cache+response_dict['text'])


            return (exec(response_dict['text'])), response_dict['text'].replace(' ', '')
        except Exception as e:
            print(f"Ocurrió un error: {e}")

    elif action == "Copy":
        custom_btns = [{
        "name": "Copy",
        "feather": "Copy",
        "hasText": True,
        "alwaysOn": False,       
        }]     
        response_dict = code_editor(code_string, lang="python", height=height, buttons=custom_btns, theme="contrast")


    

# df_titanic = pd.read_csv("files/titanic.csv")
# df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
# df_titanic = pd.get_dummies(df_titanic)
# print(df_titanic)





# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Cargar el conjunto de datos
# df_titanic = pd.read_csv("files/titanic.csv")

# # Excluir variables que no aportan valor al modelo
# df_titanic = df_titanic.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)

# # Codificar variables categóricas
# df_titanic = pd.get_dummies(df_titanic, columns=["Sex", "Embarked"])

# # Separar características (X) y etiquetas (y)
# X = df_titanic.drop("Survived", axis=1)
# y = df_titanic["Survived"]

# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Crear un modelo de árbol de decisión
# clf = DecisionTreeClassifier(random_state=42)

# # Entrenar el modelo en los datos de entrenamiento
# clf.fit(X_train, y_train)

# # Realizar predicciones en los datos de prueba
# y_pred = clf.predict(X_test)

# # Calcular la precisión del modelo en los datos de prueba
# accuracy = accuracy_score(y_test, y_pred)

# # Imprimir la precisión
# print(f"Precisión del modelo: {accuracy:.2f}")









