from main import HeyDittoNet
from keras.layers import Dense
from keras.models import Model, load_model

class HeyDittoNetEmbeddings:
    def __init__(self):
        self.load_models()
    
    def load_models(self):
        hey_ditto_net = HeyDittoNet(
            train=False,
            tflite=False,
            reinforce=False
        )
        embedddings_layer = hey_ditto_net.model.layers[-2].output
        embeddings = Dense(64, name='embeddings')(embedddings_layer)
        self.embeddings_model = Model(
            inputs=hey_ditto_net.model.input, 
            outputs=embeddings
        )
        self.hey_ditto_net = hey_ditto_net

    def save_embeddings_model(self):
        self.embeddings_model.save('models/hey_ditto_net_embeddings')

    def load_embeddings_model(self):
        self.embeddings_model = load_model('models/hey_ditto_net_embeddings')

