import numpy
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + numpy.exp(-x))



def loss(y_true,y_pre):
    return ((y_true - y_pre) ** 2).mean()

def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Nerualnetwo():
    def __init__(self):
        self.w1 = numpy.random.normal()
        self.w2 = numpy.random.normal()
        self.w3 = numpy.random.normal()
        self.w4 = numpy.random.normal()
        self.w5 = numpy.random.normal()
        self.w6 = numpy.random.normal()
        self.b1 = numpy.random.normal()
        self.b2 = numpy.random.normal()
        self.b3 = numpy.random.normal()

    def feedforward(self, x):
        h1 = x[0] * self.w1 + x[1] * self.w2 + self.b1
        h1f = sigmoid(h1)
        h2 = x[0] * self.w3 + x[1] * self.w4 + self.b2
        h2f = sigmoid(h2)
        o1 = h1f * self.w5 + h2f * self.w6 + self.b3
        o1f = sigmoid(o1)
        return h1, h1f , h2 , h2f , o1 , o1f

    def simulate(self,x):
        h1 = x[1] * self.w1 + x[1] * self.w2 + self.b1
        h1f = sigmoid(h1)
        h2 = x[1] * self.w3 + x[1] * self.w4 + self.b2
        h2f = sigmoid(h2)
        o1 = h1f * self.w5 + h2f * self.w6 + self.b3
        o1f = sigmoid(o1)
        return o1f

    def train(self,data,all_y_true):
        epochs = 2000
        learn_rate = 0.1
        loss_history = []

        for i in range(epochs):
            for x,y_true in zip(data,all_y_true):
                valcell = self.feedforward(x)
                y_pre = valcell[5]
                der_L_y_pre = -2 * (y_true - y_pre)
                der_y_pre_h1 = der_sigmoid(valcell[4] * self.w5)
                der_y_pre_h2 = der_sigmoid(valcell[4] * self.w6)
                der_h1_w1 = der_sigmoid(valcell[0]) * x[0]
                der_h1_w2 = der_sigmoid(valcell[0]) * x[1]
                der_h2_w3 = der_sigmoid(valcell[2]) * x[0]
                der_h2_w4 = der_sigmoid(valcell[2]) * x[1]
                der_y_pre_w5 = der_sigmoid(valcell[4]) * valcell[1]
                der_y_pre_w6 = der_sigmoid(valcell[4]) * valcell[3]
                der_y_pre_b3 = der_sigmoid(valcell[4])
                der_h1_b1 = der_sigmoid(valcell[0])
                der_h2_b2 = der_sigmoid(valcell[2])

                self.w1 -= learn_rate * der_L_y_pre * der_y_pre_h1 * der_h1_w1
                self.w2 -= learn_rate * der_L_y_pre * der_y_pre_h1 * der_h1_w2
                self.w3 -= learn_rate * der_L_y_pre * der_y_pre_h2 * der_h2_w3
                self.w4 -= learn_rate * der_L_y_pre * der_y_pre_h2 * der_h2_w4
                self.w5 -= learn_rate * der_L_y_pre * der_y_pre_w5
                self.w6 -= learn_rate * der_L_y_pre * der_y_pre_w6
                self.b1 -= learn_rate * der_L_y_pre * der_y_pre_h1 * der_h1_b1
                self.b2 -= learn_rate * der_L_y_pre * der_y_pre_h2 * der_h2_b2
                self.b3 -= learn_rate * der_L_y_pre * der_y_pre_b3

                if i % 10 == 0 :
                    y_pred = numpy.apply_along_axis(self.simulate,1,data)
                    the_loss = loss(all_y_true , y_pred)
                    print(i,the_loss)
                    loss_history.append(the_loss)

        return loss_history


data = numpy.array([[-2, -1],[25, 6],[17, 4],[-15, -6],[-7,-3],[26,3]])
all_y_trues = numpy.array([1,0,0,1,1,0])
ner = Nerualnetwo()

history = ner.train(data,all_y_trues)
plt.plot(history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('training loss')
plt.grid(True)
plt.show()
test1 = numpy.array([-7,-3])
print(ner.simulate(test1))


test2 = numpy.array([26,3])
print(ner.simulate(test2))


