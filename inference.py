import torch
import torchaudio

from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataSet
from sklearn.model_selection import train_test_split
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES


class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]
def predict_all(model, dataset, class_mapping):
    counter = 0
    for i in range(dataset.__len__()):
        input, target = dataset[i][0], dataset[i][1]
        input.unsqueeze_(0)
        predicted, expected = predict(model, input, target, class_mapping)
        if(predicted == expected):
            counter = counter+1
        #print(f"Sample: {i} Predicted: '{predicted}', expected: '{expected}'")
    return counter

def predict(model, input, target, class_mapping):
    model.eval() #eval je pytorch metoda koja minja ponasanje modela, kad se pozove npr. neki se slojevi "ugase" jer ne tribamo i, ako ih zelimo upalit ponovno samo model.train()
    with torch.no_grad():#koristi se kako se ne bi gradient racuna  i smanjilo koristenje memorije
        predictions = model(input) #Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]  1.dimenzija-> broj inputa,2. dimenzija-> broj klasa koje predvidamo
        # koristimo argmax kako bi dobili index sa najvecon vridnoscu tj. not 0.6 u redu iznad
        predicted_index = predictions[0].argmax(0) #želimo predikciju za prvi i jedini sample koji smo ode koristili

        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("C:/py/py1/cnnnet_01.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataSet(ANNOTATIONS_FILE,AUDIO_DIR,mel_spectrogram,SAMPLE_RATE,NUM_SAMPLES,"cpu")
    train_data, test_data = train_test_split(usd, test_size=0.2, random_state=1234)
   # # get a sample from the urban sound dataset for inference
   # input, target = usd[0][0], usd[0][1] # [batch size, num_channels, fr, time]
   # input.unsqueeze_(0) #kako input ima 3 dimenzije a triba 4 sa ovin na 0. index dodamo dimenziju velicine 1

    counter =  predict_all(cnn, test_data, class_mapping)

    print(f"Tocno je predvideno : {counter}  primjeraka od {test_data.__len__()}")