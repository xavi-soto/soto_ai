from chat_soto import crear_indice, responder

indice = crear_indice()

while True:
    pregunta = input("Hazme una pregunta: ")
    if pregunta.lower() in ["salir", "exit"]:
        break
    respuesta = responder(indice, pregunta)
    print("soto dice:", respuesta)
