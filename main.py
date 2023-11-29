import streamlit as st
from streamlit_text_label import label_select
from streamlit_tags import st_tags
import time
from utils import code_input
import actividades
import utils

def actualizar(i, booleano, cache):
    # Cache
    utils.asig_cache(cache)
    # Validacion
    actividades.asig_valores(i, booleano)

def main():

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
    
    actividades.act_1()
    if actividades.ret_valores()[0][0] == True:
        actividades.act_2()
        if actividades.ret_valores()[1][0] == True:
            actividades.act_3()        
            if actividades.ret_valores()[2][0] == True:
                actividades.act_4()        
                if actividades.ret_valores()[3][0] == True:
                    actividades.act_5()
                    if actividades.ret_valores()[4][0] == True:
                        actividades.act_6()        
                        if actividades.ret_valores()[5][0] == True:
                            actividades.act_7()        
                            if actividades.ret_valores()[6][0] == True:
                                actividades.act_8()
                                if actividades.ret_valores()[7][0] == True:
                                    actividades.act_9()
                                    if actividades.ret_valores()[8][0] == True:
                                        actividades.act_10()
                                        if actividades.ret_valores()[9][0] == True:
                                            actividades.act_11()
                                            if actividades.ret_valores()[10][0] == True:
                                                actividades.act_12()
                                                if actividades.ret_valores()[11][0] == True:
                                                    actividades.act_13()
                                                    if actividades.ret_valores()[12][0] == True:
                                                        actividades.act_14()
                                                        if actividades.ret_valores()[13][0] == True:
                                                            actividades.act_15()
                                                            if actividades.ret_valores()[14][0] == True:
                                                                actividades.act_juego()
                                                                st.text("Puntaje del juego:")
                                                                st.write(actividades.ret_puntaje_juego())

if __name__ == "__main__":
    main()