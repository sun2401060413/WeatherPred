import models.WeatherClsCNN.flags as flags
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

def get_optimizer(info_dict = {}):
    if 'op' in info_dict.keys():
        if info_dict['op'] not in ['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
            opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        if info_dict['op'] == 'sgd':
            if info_dict['lr']:
                opt = SGD(lr=info_dict['lr'], decay=1e-6, momentum=0.9, nesterov=True)
            else:
                opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        if info_dict['op'] == 'RMSprop':
            if info_dict['lr']:
                opt = RMSprop(lr=info_dict['lr'], rho=0.9, epsilon=1e-06)
            else:
                opt = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-06)
        if info_dict['op'] == 'Adagrad':
            if info_dict['lr']:
                opt = Adagrad(lr=info_dict['lr'], rho=0.9, epsilon=1e-06)
            else:
                opt = Adagrad(lr=0.01, rho=0.9, epsilon=1e-06)
        if info_dict['op'] == 'Adadelta':
            if info_dict['lr']:
                opt = Adadelta(lr=info_dict['lr'], rho=0.9, epsilon=1e-06)
            else:
                opt = Adadelta(lr=0.01, rho=0.9, epsilon=1e-06)
        if info_dict['op'] == 'Adam':
            if info_dict['lr']:
                opt = Adam(lr=info_dict['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            else:
                opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        if info_dict['op'] == 'Adamax':
            if info_dict['lr']:
                opt = Adamax(lr=info_dict['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            else:
                opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        if info_dict['op'] == 'Nadam':
            if info_dict['lr']:
                opt = Nadam(lr=info_dict['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            else:
                opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    else:
        opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    return opt
