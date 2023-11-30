from code_editor import code_editor

cache_val = ''''''

def ret_cache():
    return cache_val

def asig_cache(val):
    global cache_val
    cache_val = cache_val + val
    

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
            global cache_val
            cache = cache_val
            exec(cache+response_dict['text'])


            return (exec(cache + response_dict['text'])), response_dict['text'].replace(' ', '')
        except Exception as e:
            #print(f"Ocurrió un error: {e}")
            return "Error"

    elif action == "Copy":
        custom_btns = [{
        "name": "Copy",
        "feather": "Copy",
        "hasText": True,
        "alwaysOn": False,       
        }]     
        response_dict = code_editor(code_string, lang="python", height=height, buttons=custom_btns, theme="contrast")
            

    #elif action == "Copy":
    #    custom_btns = [{
    #    "name": "Copy",
    #    "feather": "Copy",
    #    "hasText": True,
    #    "alwaysOn": False,       
    #    }]     
    #    response_dict = code_editor(code_string, lang="python", height=height, buttons=custom_btns, theme="contrast")
