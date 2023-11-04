from .main import HeyDittoNet
from keras.layers import Dense
from keras.models import Model

net = HeyDittoNet()

embedddings_layer = net.model.layers[-2].output
embeddings = Dense(256, name='embeddings')(embedddings_layer)

model = Model(inputs=net.model.input, outputs=embeddings)

model.save('models/HeyDittoNetEmbeddings')