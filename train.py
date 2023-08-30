import torch
import torchaudio
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from urbansounddataset import UrbanSoundDataSet
from cnn import CNNNetwork

#Constant
BATCH_SIZE = 2500
EPOCHS = 20
LEARNING_RATE =.001

ANNOTATIONS_FILE = "C:/Users/MarijaGaliatović/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "C:/Users/MarijaGaliatović/Downloads/UrbanSound8K/audio"

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050*3

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs,targets = inputs.to(device), targets.to(device)

        #calculate loss
        predictions = model(inputs)
        loss=loss_fn(predictions, targets)

        #backpropagate loss and update weights
        optimiser.zero_grad() #restarting gradient
        loss.backward()
        optimiser.step() #update weights

        print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("_______________________")
    print("Training is done.")






#ovo je ka neka skripta po kojoj se sve izvršava
if __name__ == "__main__": #provjerava je li trenutna skripta pokrenuta kao glavni modul,
    # to osigurava da blok koda unutar uvjeta je izvršen kad je izvršena skripta direktno, a ne kad je importana ka modul

    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    print(f"Using {device} device")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,  # od koliko se "tocaka" sastoji jedan window
        hop_length=512,  # svakih koliko tocaka je novi window, tj. u koliko se tocaka poklapaju 2
        n_mels=64  # broj mel andova tj. amo rec kategorija frekvencija ovisno o sluhu
    )

    usd = UrbanSoundDataSet(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    #train_data_loader = create_data_loader(usd, BATCH_SIZE)
    train_data, test_data = train_test_split(usd, test_size=0.2, random_state=1234)
    train_data_loader = create_data_loader(train_data, BATCH_SIZE)
    #test_data_loader = create_data_loader(test_data, BATCH_SIZE)

    for i in range(test_data.__len__()):
        input, target = test_data[i][0], test_data[i][1]
        print(f"input: {input} target: {target}")

    #construct model
    cnn = CNNNetwork().to(device)
    print(f"Model: {cnn}")

    #instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    #train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, epochs=EPOCHS)

    #5 save model
    torch.save(cnn.state_dict(), "cnnnet_01.pth")

    print("Model trained and stored at cnnnet_01.pth")