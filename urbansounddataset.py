import os #za operacije patha
import torchaudio
import torch.nn.functional
from torch.utils.data import Dataset #za stvaranje custom datasetova
import pandas as pd #za tablicne podatke
import matplotlib.pyplot as plt


class UrbanSoundDataSet(Dataset):
    #nasljeduje  od dataseta
    #annotations je neki file sa biljeskama, path do csv filea
    #A CSV (comma-separated values) file is a text file that
    # has a specific format which allows data to be saved in a table structured format.
    #file name svakog audio samplea
    # F stupac je folder
    #classID je label

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device): #klasa prima 2 parametra path do CSV filea i direktorij
        self.annotations = pd.read_csv(annotations_file) #pd.read_csv je funkcija iz pandas biblioteke koja uzima
        #annotations_file kao input (path od CDV filea koji sadrzi metapodatke) i vraca DataFrame
        #koji sadrzi metapodatke o audio sampleovima , file nameoviima , labelama itd.
        #tako šta spremimo annotations DataFrame ka atribut unutar klase , postaje jednostavno dohvacat specificne podatke
        #o audio sampleovi prilikom loadanja i procesiranja
        self.audio_dir = audio_dir
        self.device = device
        #self.transformation = transformation prije cude tj.gpu
        self.transformation = transformation.to(self.device) #sa ovin tranformaciju saljemo na zeljeni uredaj cpu/gpu, definirano u mainu
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    #len(usd)
    def __len__(self):
        return len(self.annotations) #the length of the dataset, which is the number of rows in the annotations CSV file

    #a_list[1]  je zapravo a_list.__getitem__(1
    def __getitem__(self, index): #koristimo za dohvacanje elemenata iz dataseta na trazenom indexu
        audio_sample_path = self._get_audio_sample_path(index)
        print(f"audio_sample_path = {audio_sample_path}")
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path) #sa ovin dohvacamo sound sample na osnovu patha (dobijemo sam signal i sample rate)
        signal = signal.to(self.device) #sa ovin signal saljemo na zeljeni uredaj cpu/gpu, definirano u mainu
        # nemaju svi sampleovi isti sample rate (amo rec frekvenciju sa kojima ih sampleamo (ka koliko puta u sekundi uzmemo uzorak tog signala,
        # sta je veca frekvencija bolje smo digitalizirali zvuk, ali trazi vise resursa), zelimo da nan dataset bude ujednacen
        # tj. da svi uzorci zvuka imaju isti samplerate i onda ih izjednacavamo na isti sr ako vec nisu sa tin sample rate-on uzorkovani)
        self._graph_(signal, label)
        signal = self._resample_if_necessary(signal, sr)
        #svaki zvucni uzorak moze imat i razlicit broj kanala npr. kad snimamo podcast imamo jedan kanal,
        # a kod stereo zvuka tj. zvucnika imamo najcesce 2 (broj kanala zapravo predstavlja broj izvora zvuka koji su istovremeni (valjda))
        signal = self._mix_down_if_necessary(signal)
        #sample-ovi nan tribaju biti iste duljine pa triba skratit na zeljenu duljinu one koji su pre dugi
        #duljina j ezapravo "num_samples" tj.broj uzoraka koji smo izjednacili u ovom slucaju sa sample_rateom pa je to zapravo duljina od 1 sekunde
        signal = self._cut_if_necessary(signal)
        #ako su pre kratki triba dodat padding na kraj zvuka kako bi bili trazene duljine
        signal = self._right_pad_if_necessary(signal)
        self._graph_(signal, label)
        #nas signal pretvorimo u mel_spectogram (koji zapravo prikazuje frekvencije tog signala, a pri tome koristi mel skalu koja raspone frekvencija pretvara u raspon
        # sa obziron na to kako te frekvencije pripozna ljudsko uho u smislu bolje cujemo razlik izmedu frekvencija od 1000 i 1500 Hz nego 10000 i 10500 Hz i to mel skala uzima u obzir i grupira na ispravan nacin)
        signal = self.transformation(signal)
        return  signal, label #vracamao signal i labelu tj. u koju klasu(kategoriju) spada taj signal

    #kako bi dosli do nekog filea moramo imat njegov path koji se sastoji od svih direktorija unutar koji se nalaze annotations dataFrame (podatci o zvucnim zapisima),
    #od foldera u kojem se pojedini zvucni zapis nalazi (podiljeni su npr u 10 foldera ovisno sta je koji zvuk i to je zapisano u dataFrameovima) i od imena samog zapisa
    #npr. home/desktop/datasets/urbanSounds8k/audio je zapisano u audio_dir, fold je npr fold5 , i ime je neko random ime zvuka npr. zapis5-3
    #onda ce path bit home/desktop/datasets/urbanSounds8k/audio/fold5/‚zapis5-3
    def _get_audio_sample_path(self, index): #index je zapravo redak u annotation fileu (svaki zapis ima 6-7 stupaca sa podatcima, jedan redak je jedan zvucni zapis)
        fold = f"fold{self.annotations.iloc[index, 5]}" #spajamo broj foldera, kojem pristupamo sa iloc funkcijon priko kordinata (prva je broj retka, druga stupca), sa rici fold jer je to naziv foldera u kojem se nalaze zapisi
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index,0]) #spajamo sve u jedan path koji onda returnamo
        return path

    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,6] #stupac 6 zapravo je class_id pri cemu su  klase npr 0->zvukovi zivotinja, 1-> zvuk sirene itd....

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal


    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: #shape zapravo vrati dimenzije tj. izgled tensora npr. za stereo [2,1600], 2 predstavlja broj kanala,
            # a shape[0] vridnost na indexu 0 sta je u ovom slucaju br. 2, tako da ako je broj kanala veci od 1 radi se mixanje na broj kanala 1
            signal = torch.mean(signal, dim=0, keepdim=True) #mean svede na jednu dimenziju onu dimenziju na koju ju primjenimo u nasem slucaju na 0. dimenziju tj.retke
        return signal

    #signal-> Tensor -> (1,num_samples)  -> broj sampleova nan omogucava odredit duljinu, triba bit jednak defaultno postavljenoj i skratit/ prosirit sve na tu vridnost
    #u tensoru prvu dimenziju tj. u primjeru broj 1 predstavlja broj kanala tj. izvora zvuka, a drugu broj uzoraka (uzorci su ka kolko puta smo snimili amplitudu tog zvuka)

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:  #signal.shape[1] = num_samples od tog tensora
            signal = signal[:, :self.num_samples] #ovo zapravo znaci da prvu dimenziju ne diramo, a uzimamo samo vridnosti druge dimenzije od njenog pocetka do iznosa num_samples koji smo poslali kao parametar klase
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1] #broj sampleova tog primjerka zvuka
        if length_signal < self.num_samples: #ako je vece tribamo prosirit
            #print(f"Duljina prije : {signal.shape}")
            num_missing_samples = self.num_samples - length_signal  #koliko je kraci nas primjerak od defaultne duljine na koju sve primjerke izjednacavamo
            last_dim_padding = (0, num_missing_samples) #definira padding na zadnju dimenziju, 0 elemenata sa live strane jer nan ne triba to i num_missing _samples elemenata sa desne koliko fali da imamo trazenu duljinu audio signala
            signal = torch.nn.functional.pad(signal, last_dim_padding) #dodaje padding na signal
           # print(f"Duljina nakon : {signal.shape}")
        return signal

    def _graph_(self, signal,label):
        print(f"signal.size(1):{signal.size(1)}")
        time_axis = torch.arange(0, signal.size(1))/SAMPLE_RATE

        # Plot the audio signal
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, signal.t().numpy())
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.title(f"Audio Signal - Label: {label}")
        plt.show()


if __name__ == "__main__":
    ANNOTATIONS_FILE = "C:/Users/MarijaGaliatović/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "C:/Users/MarijaGaliatović/Downloads/UrbanSound8K/audio" #ovo promini kad stignes
    SAMPLE_RATE = 22050  #ovo ce bit defaultna sample rate
    NUM_SAMPLES = 22050*3

    if torch.cuda.is_available():
        device="cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    #objekt koji mozemo pozvat
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,  #od koliko se "tocaka" sastoji jedan window
        hop_length=512, #svakih koliko tocaka je novi window, tj. u koliko se tocaka poklapaju 2
        n_mels=64 #broj mel andova tj. amo rec kategorija frekvencija ovisno o sluhu
    )

    usd = UrbanSoundDataSet(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[211]
    print(f"Duljina zadnja : {signal.shape}")

    plt.figure(figsize=(10, 6))
    plt.imshow(signal[0].detach().numpy(), cmap='viridis', origin='lower', aspect='auto',
               extent = [0, 3, 0, signal.shape[1]])

    plt.xlabel("Time")
    plt.ylabel("Mel Frequency Bin")
    plt.title(f"Mel Spectrogram - Label: {label}")
    plt.colorbar(format="%+2.0f dB")
    #plt.xlim([0,3])
    plt.show()

    # tensor-> [1, num_mels, num_frames]

    #ispis nekoliko primjeraka

