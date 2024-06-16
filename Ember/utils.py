def format_model_list(model_list):
    formatted_list = "Available Models:\n"
    for model in model_list:
        formatted_list += f"Empresa: {model['empresa']}, Nombre: {model['nombre']}\n"
    return formatted_list
