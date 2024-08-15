
class vocab:
    sos = 'bos'
    eos = 'eos'
    unk = 'unk'
    sos_idx = 0
    eos_idx = 1
    unk_idx = 2

save = 'last'
assert save in ['all', 'last', 'best']  # TODO: last and best

data_instances = -1


class sample_methods:
    argmax = 'argmax'
    multinomial = 'multinomial'


class inits:
    xavier = 'xavied'
    zeros = 'zeros'


logging_cfg = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        },
        'django.request': {
            'handlers': ['default'],
            'level': 'WARN',
            'propagate': False
        },
    }
}
