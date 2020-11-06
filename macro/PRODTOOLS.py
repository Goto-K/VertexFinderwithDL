
def yinx(y, x):
    for _y in y:
        if _y not in x: return False
    return True

def listremove(y, x):
    for _y in y: 
        x.remove(_y)
    return x
