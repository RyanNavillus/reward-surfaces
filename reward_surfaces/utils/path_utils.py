
def strip_lagging_slash(f):
    if f[-1] == '/':
        return f[:-1]
    else:
        return f
