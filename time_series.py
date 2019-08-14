# define generator
def get_gen(set_x, set_y, n_input = 200):
    return TimeseriesGenerator(set_x, set_y, length=n_input, batch_size=128*2)

train_generator = get_gen(trainX, trainY)

val_generator = get_gen(valX, valY)

# number of samples
print('Samples: %d' % len(train_generator))

x, y = train_generator[0]

features = x.shape[2]
