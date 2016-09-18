import time


def timing(f):
    def wrap(*args):
        print('%s function start' % (f.__name__,))
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        elapse = (time2 - time1) * 1000.0
        seconds = elapse / 1000
        millis = elapse % 1000
        print('%s function elapse %i sec %i ms' % (f.__name__, seconds, millis))
        return ret

    return wrap
