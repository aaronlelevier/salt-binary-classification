import datetime


def timestamp(dt=None):
    dt = dt or datetime.datetime.now()
    return dt.strftime('%Y%m%d_%H%M')
