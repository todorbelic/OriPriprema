from abc import abstractmethod
import math
import random
import copy

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from matplotlib import pyplot

random.seed(1337)


def standardize_data(data):

    scale = StandardScaler()
    scaled_data = scale.fit_transform(data)
    return scaled_data


class ComputationalNode(object):

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dz):
        pass


class MultiplyNode(ComputationalNode):

    def __init__(self):
        self.x = [0., 0.]  # x[0] je ulaz, x[1] je tezina

    def forward(self, x):
        self.x = x
        # TODO 1: implementirati forward-pass za mnozac
        return self.x[0] * self.x[1]

    def backward(self, dz):
        # TODO 1: implementirati backward-pass za mnozac
        # z = x * y
        # dz / dx = y => dx = dz * y
        # dz / dy = x => dy = dz * x
        return [dz * self.x[1], dz * self.x[0]]


# MultiplyNode tests
mn_test = MultiplyNode()
assert mn_test.forward([2., 3.]) == 6., 'Failed MultiplyNode, forward()'
assert mn_test.backward(-2.) == [-2. * 3., -2. * 2.], 'Failed MultiplyNode, backward()'
print('MultiplyNode: tests passed')


class SumNode(ComputationalNode):

    def __init__(self):
        self.x = []  # x je vektor, odnosno niz skalara

    def forward(self, x):
        self.x = x
        # TODO 2: implementirati forward-pass za sabirac
        return sum(self.x)

    def backward(self, dz):
        # TODO 2: implementirati backward-pass za sabirac
        # z = x + y
        # dz / dx = 1 => dx = dz
        # dz / dy = 1 => dy = dz
        return [dz for xx in self.x]


# SumNode tests
sn_test = SumNode()
assert sn_test.forward([1., 2., -2, 5.]) == 6., 'Failed SumNode, forward()'
assert sn_test.backward(-2.) == [-2.] * 4, 'Failed SumNode, backward()'
print('SumNode: tests passed')


class SigmoidNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x je skalar

    def forward(self, x):
        self.x = x
        # TODO 3: implementirati forward-pass za sigmoidalni cvor
        return self._sigmoid(self.x)

    def backward(self, dz):
        # TODO 3: implementirati backward-pass za sigmoidalni cvor
        # z = 1 / 1 + e ^ -x
        # dz / dx = sigm(x) * (1 - sigm(x))
        # = > dx = dz * sigm(x) * (1 - sigm(x))
        return dz * self._sigmoid(self.x) * (1 - self._sigmoid(self.x))

    def _sigmoid(self, x):
        # TODO 3: implementirati sigmoidalnu funkciju
        return 1. / (1. + math.exp(-x))


# SigmoidNode tests
sign_test = SigmoidNode()
assert sign_test.forward(1.) == 0.7310585786300049, 'Failed SigmoidNode, forward()'
assert sign_test.backward(-2.) == -2. * 0.7310585786300049 * (1. - 0.7310585786300049), 'Failed SigmoidNode, backward()'
print('SigmoidNode: tests passed')


class ReluNode(ComputationalNode):

    def __init__(self):
        self.x = 0.

    def forward(self, x):
        self.x = x
        return self._relu(self.x)

    def backward(self, dz):
        return dz * (1. if self.x > 0. else 0)

    def _relu(self, x):
        return max(0., x)


class LinNode(ComputationalNode):

    def __init__(self):
        self.x  = 0.

    def forward(self, x):
        self.x = x
        return self._lin(x)

    def backward(self, dz):
        return dz * 1

    def _lin(self, x):
        return x

class NeuronNode(ComputationalNode):

    def __init__(self, n_inputs, activation):
        self.n_inputs = n_inputs  # moramo da znamo kolika ima ulaza da bismo znali koliko nam treba mnozaca
        self.multiply_nodes = []  # lista mnozaca
        self.sum_node = SumNode()  # sabirac

        # TODO 4: napraviti n_inputs mnozaca u listi mnozaca, odnosno mnozac za svaki ulaz i njemu odgovarajucu tezinu
        # za svaki mnozac inicijalizovati tezinu na broj iz normalne (gauss) raspodele sa st. devijacijom 0.1
        for n in range(n_inputs):
            mn = MultiplyNode()
            mn.x = [1, random.gauss(0., 0.1)]  # da ne dobijemo inicijalno velike vrednosti
            self.multiply_nodes.append(mn)

        # TODO 5: dodati jos jedan mnozac u listi mnozaca, za bias
        # bias ulaz je uvek fiksiran na 1.
        # bias tezinu inicijalizovati na broj iz normalne (gauss) raspodele sa st. devijacijom 0.01
        mn = MultiplyNode()
        mn.x = [1., random.gauss(0., 0.1)]
        self.multiply_nodes.append(mn)

        # TODO 6: ako ulazni parametar funckije 'activation' ima vrednosti 'sigmoid',
        # inicijalizovati da aktivaciona funckija bude sigmoidalni cvor
        if activation == 'sigmoid':
            self.activation_node = SigmoidNode()
        elif activation == 'relu':
            self.activation_node = ReluNode()
        elif activation == 'lin':
            self.activation_node = LinNode()
        else:
            raise RuntimeError('Unknown activation function "{0}"'.format(activation))

        self.previous_first_moments = [0.] * (self.n_inputs + 1)
        self.previous_second_moments = [0.] * (self.n_inputs + 1)
        self.gradients = []

    def forward(self, x):  # x je vektor ulaza u neuron, odnosno lista skalara
        x = copy.copy(x)
        x.append(1.)  # uvek implicitino dodajemo bias=1. kao ulaz

        # TODO 7: implementirati forward-pass za vestacki neuron
        # u x se nalaze ulazi i bias neurona
        # iskoristi forward-pass za mnozace, sabirac i aktivacionu funkciju da bi se dobio konacni izlaz iz neurona
        for_sum = []
        for i, xx in enumerate(x):
            inp = [x[i], self.multiply_nodes[i].x[1]]
            for_sum.append(self.multiply_nodes[i].forward(inp))

        summed = self.sum_node.forward(for_sum)
        summed_act = self.activation_node.forward(summed)
        return summed_act

    def backward(self, dz):
        dw = []
        dx = []
        d = dz[0] if type(dz[0]) == float else sum(dz)  # u d se nalazi spoljasnji gradijent izlaza neurona

        # TODO 8: implementirati backward-pass za vestacki neuron
        # iskoristiti backward-pass za aktivacionu funkciju, sabirac i mnozace da bi se dobili gradijenti tezina neurona
        # izracunate gradijente tezina ubaciti u listu dw
        d_activation = self.activation_node.backward(d)
        d_sum = self.sum_node.backward(d_activation)
        d_mul = []
        for d_sumi, mul_node in zip(d_sum, self.multiply_nodes):
            [dxi, dwi] = mul_node.backward(d_sumi)
            dw.append(dwi)
            dx.append(dxi)
        self.gradients = dw
        return dx

    def update_weights(self, learning_rate, beta1, beta2, epsilon=1e-8):
        # azuriranje tezina vestackog neurona
        # learning_rate je korak gradijenta

        # TODO 11: azurirati tezine neurona (odnosno azurirati drugi parametar svih mnozaca u neuronu)
        # gradijenti tezina se nalaze u list self.gradients
        for i, multiply_node in enumerate(self.multiply_nodes):
            g_i = self.gradients[i]
            m_i = beta1 * self.previous_first_moments[i] + (1 - beta1) * g_i
            self.previous_first_moments[i] = m_i
            v_i = beta2 * self.previous_second_moments[i] + (1 - beta2) * (g_i ** 2)
            self.previous_second_moments[i] = v_i
            m_hat = m_i / (1 - beta1)
            v_hat = v_i / (1 - beta2)
            self.multiply_nodes[i].x[1] -= learning_rate * (m_hat/(math.sqrt(v_hat) + epsilon))

        self.gradients = []  # ciscenje liste gradijenata (da sve bude cisto za sledecu iteraciju)


class NeuralLayer(ComputationalNode):

    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs  # broj ulaza u ovaj sloj neurona
        self.n_neurons = n_neurons  # broj neurona u sloju (toliko ce biti i izlaza iz ovog sloja)
        self.activation = activation  # aktivaciona funkcija neurona u ovom sloju

        self.neurons = []
        # konstruisanje sloja nuerona
        for _ in range(n_neurons):
            neuron = NeuronNode(n_inputs, activation)
            self.neurons.append(neuron)

    def forward(self, x):  # x je vektor, odnosno lista "n_inputs" elemenata
        layer_output = []
        # forward-pass za sloj neurona je zapravo forward-pass za svaki neuron u sloju nad zadatim ulazom x
        for neuron in self.neurons:
            neuron_output = neuron.forward(x)
            layer_output.append(neuron_output)

        return layer_output

    def backward(self, dz):  # dz je vektor, odnosno lista "n_neurons" elemenata
        dd = []
        # backward-pass za sloj neurona je zapravo backward-pass za svaki neuron u sloju nad
        # zadatim spoljasnjim gradijentima dz
        for i, neuron in enumerate(self.neurons):
            neuron_dz = [d[i] for d in dz]
            neuron_dz = neuron.backward(neuron_dz)
            dd.append(neuron_dz[:-1])  # izuzimamo gradijent za bias jer se on ne propagira unazad

        return dd

    def update_weights(self, learning_rate, momentum, beta1, beta2):
        # azuriranje tezina slojeva neurona je azuriranje tezina svakog neurona u tom sloju
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum, beta1, beta2)


class NeuralNetwork(ComputationalNode):

    def __init__(self):
        self.layers = []  # neuronska mreza se sastoji od slojeva neurona


    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):  # x je vektor koji predstavlja ulaz u neuronsku mrezu
        # TODO 9: implementirati forward-pass za celu neuronsku mrezu
        # ulaz za prvi sloj neurona je x
        # ulaz za sve ostale slojeve izlaz iz prethodnog sloja
        prev_layer_output = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                prev_layer_output = layer.forward(x)
            else:
                prev_layer_output = layer.forward(prev_layer_output)

        return prev_layer_output

    def backward(self, dz):
        # TODO 10: implementirati forward-pass za celu neuronsku mrezu
        # spoljasnji gradijent za izlazni sloj neurona je dz
        # spoljasnji gradijenti za ostale slojeve su izracunati gradijenti iz sledeceg sloja
        next_layer_dz = None
        for idx, layer in enumerate(self.layers[::-1]):
            if idx == 0:
                next_layer_dz = layer.backward(dz)
            else:
                next_layer_dz = layer.backward(next_layer_dz)

        return next_layer_dz

    def update_weights(self, learning_rate, momentum, beta1, beta2):
        # azuriranje tezina neuronske mreze je azuriranje tezina slojeva
        for layer in self.layers:
            layer.update_weights(learning_rate, momentum, beta1, beta2)

    def fit(self, X, Y, learning_rate=0.1, momentum=0.0, nb_epochs=10, shuffle=False, verbose=0, beta1=0.9, beta2=0.999):
        assert len(X) == len(Y)

        hist = []  # za plotovanje funkcije greske kroz epohe
        for epoch in range(nb_epochs):

            if shuffle:  # izmesati podatke
                random.seed(epoch)
                random.shuffle(X)
                random.seed(epoch)
                random.shuffle(Y)

            total_loss = np.array([0, 0])
            for x, y in zip(X, Y):
                y_pred = self.forward(x)  # forward-pass da izracunamo izlaz
                y_target = y  # zeljeni izlaz
                total_loss = np.add(total_loss, 0.5 * np.power(np.subtract(y_target, y_pred), 2))               #total_loss += 0.5 * (t - p) ** 2.  # funkcija greske je kvadratna greska
                grad = -1 * np.subtract(y_target, y_pred)  # gradijent funkcije greske u odnosu na izlaz
                # backward-pass da izracunamo gradijente tezina
                self.backward([grad])
                # azuriranje tezina na osnovu izracunatih gradijenata i koraka "learning_rate"
                self.update_weights(learning_rate, momentum, beta1, beta2)
            total_loss /= len(X)
            if verbose == 1:
                print('Epoch {0}: loss {1}'.format(epoch + 1, total_loss))

            hist.append(total_loss)

        print('Loss: {0}'.format(total_loss))
        return hist

    def predict(self, x):
        return self.forward(x)

    def get_accuracy(self, testX, testY):

        predicted = []
        for x in testX:
            list = self.predict(x)
            p = list.index(max(list))
            predicted.append(p)
        accurate_count = 0
        for p, y in zip(predicted, testY):
            if y[p]:
                accurate_count += 1

        return accurate_count / len(testX) * 100


if __name__ == '__main__':
    nn = NeuralNetwork()
    # TODO 12: konstruisati neuronsku mrezu za resavanje XOR problema
    nn.add(NeuralLayer(5, 4, 'sigmoid')) # proveri dali relu radi
    nn.add(NeuralLayer(4, 2, 'sigmoid')) # pazi koliko ulaza ima!!!

    # obucavajuci skup
    X = [[0., 0.],
         [1., 0.],
         [0., 1.],
         [1., 1.]]
    Y = [[0.],
         [1.],
         [1.],
         [0.]]

    #https://sparkbyexamples.com/pandas/pandas-replace-values-based-on-condition/



    df = pd.read_csv('../../customer_churn.csv')
    print(df.isnull().sum())  # proverimo da li imaju null vrednosti
    df['churn'] = df['churn'].astype(dtype=float)
    df['international plan'].replace(('yes', 'no'), (1.0, 0.0), inplace=True)
    df['voice mail plan'].replace(('yes', 'no'), (1.0, 0.0), inplace=True)
    df['number vmail messages'] = df['number vmail messages'].astype(dtype=float)
    df['total intl calls'] = df['total intl calls'].astype(dtype=float)
    df['total night calls'] = df['total night calls'].astype(dtype=float)
    df['churn'].fillna(value=df['churn'].mean, inplace=True)

    encoder = OneHotEncoder(handle_unknown='ignore')
    # perform one-hot encoding on 'team' column
    encoder_df = pd.DataFrame(encoder.fit_transform(df[['churn']]).toarray())
    # merge one-hot encoded columns back with original DataFrame
    final_df = df.join(encoder_df)
    # view final df

    final_df.drop('churn', axis=1, inplace=True)
    #print(df['churn'].nunique(dropna=True))
    df.dropna(inplace=True)
    #print(df['churn'].value_counts())

    X = df[['international plan', 'voice mail plan', 'number vmail messages', 'total intl calls', 'total night calls']]
    X = X.values #bitno!!!

    print(final_df.shape)
    Y = final_df.iloc[:, [20, 21]]
    Y = Y.values
    print(Y)

    X_scaled = standardize_data(X)

    print(X_scaled)
    #Y = Y.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.3, shuffle=True)
    X_train = X_train.tolist()
    y_train = y_train.tolist()
    X_test = X_test.tolist()
    y_test = y_test.tolist()
    Y = Y.tolist()
    X_scaled = X_scaled.tolist()
    #obucavanje neuronske mreze
    history = nn.fit(X_scaled, Y, learning_rate=0.1, momentum=0.9, nb_epochs=20, verbose=1)
    print(nn.get_accuracy(X_scaled, Y))
    # plotovanje funkcije greske
    pyplot.plot(history)
    pyplot.show()


    """
    class_3,class_2,class_1 = df.label.value_counts()
    c3 = df[df['label'] == 3]
    c2 = df[df['label'] == 2]
    c1 = df[df['label'] == 1]
    df_3 = c3.sample(class_1)
    df_2 = c2.sample(class_1)

    undersampled_df = pd.concat([df_3,df_2,c1],axis=0)
    
    encoder = OneHotEncoder(handle_unknown='ignore')
    #perform one-hot encoding on 'team' column 
    encoder_df = pd.DataFrame(encoder.fit_transform(df[['team']]).toarray())
    #merge one-hot encoded columns back with original DataFrame
    final_df = df.join(encoder_df)
    #view final df
    print(final_df)
    final_df.drop('team', axis=1, inplace=True)
    """