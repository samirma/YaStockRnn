from tqdm import tqdm

dt.rewind()

path = "drive/My Drive/model/"

def get_set(set_name, data_count):
  trainX = []
  trainY = []
  x_path = path + set_name + "X.npy"
  y_path = path + set_name + "Y.npy"
  
  if (os.path.exists(x_path) and (os.path.exists(y_path))):
    print("Loading data from files")
    trainX = np.load(x_path)
    trainY = np.load(y_path)
  else:
    for i in tqdm(range(data_count)):
        state = dt.next()
        state = get_state(state)
        trainX.append(state[0])
        trainY.append(state[1])
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    np.save(x_path, trainX)
    np.save(y_path, trainY)

  return trainX, trainY

data_count = dt.steps - 1000
       
#data_count = 5000

val_percentage = 0.03

trainX, trainY = get_set("train", int(data_count*(1-val_percentage)))

valX, valY = get_set("val", int(data_count*val_percentage))

print(trainX.shape)
print(trainY.shape)


print(valX.shape)
print(valY.shape)

