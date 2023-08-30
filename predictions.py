import torch
from train import FeedForwardNet, download_mnist_datasets

class_mapping = [ #mapira index u pripadnu vridnost
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

def predict(model, input, target, class_mapping):
    model.eval() #eval je pytorch metoda koja minja ponasanje modela, kad se pozove npr. neki se slojevi "ugase" jer ne tribamo i, ako ih zelimo upalit ponovno samo model.train()
    with torch.no_grad(): #koristi se kako se ne bi gradient racuna  i smanjilo koristenje memorije
        predictions = model(input) #koriz model se provuce input, a rezultat je tensor 1x10 velicine pri cemo svako polje predstavlja vjerojatnost da je ta znamenka tako npr 60 % da je znamenka 9 jer je na indexu 9 0.6
        # Tensor (1, 10)-> [[0.1, 0.01, ..., 0.6]] 1.dimenzija-> broj inputa,2. dimenzija-> broj klasa koje predvida
        predicted_index = predictions[0].argmax(0) #želimo predikciju za prvi i jedini sample koji smo ode koristili
        #koristimo argmax kako bi dobili index sa najvecon vridnoscu
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

        return predicted, expected

if __name__ == "__main__": #provjerava je li skripta pokrenuta kao  glavni program
    #load back the model
    feed_forward_net = FeedForwardNet() #inicijaliziramo model
    state_dict = torch.load("../cnnnet.pth") #"rjecnik stanja" modela je ucitan iz filea di smo spremili model, sadrzi parametre i naucene tezine treniranog modela
    feed_forward_net.load_state_dict(state_dict) #ucitavamo virdnosti u nas model

    #load MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    #get a sample from the validaton dataset for inference, povezati input sa targeton
    input, target = validation_data[0][0], validation_data[0][1] #ovo ni tribali bit indexi

    #make an inference(zakljucak)
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)

    print(f"Predicted: '{predicted}', expected: '{expected}'")
