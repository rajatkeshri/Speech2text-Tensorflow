import librosa
import os
import json

DATASET_PATH = "Dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # Equals to 1 second in audio | We define to use minimum of 1 second of audio for training

# n_mfcc -> the mfcc length which will be extracted from each audio file
# hop_length -> the window size in frames for which the mfcc will be extracted for each audio file
# n_fft -> fast fourier transform window size

def preproccessing(dataset_path, json_path, n_mfcc = 13, hop_length = 512, n_fft = 2048):

    # data dictionary
    data = {
            "mappings" : [],
            "labels" : [],
            "MFCCs" : [],
            "files" : []
    }

    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):
        print(i,dirpath,dirnames)

        if dirpath is not dataset_path:

            # updating the mappings
            category = dirpath.split("/")[-1] # Dataset/down --> [Dataset, down]
            data["mappings"].append(category)
            print("processing " + category)

            # looping through each file
            for f in filenames:
                file_path = os.path.join(dirpath,f)

                # loading the audio file
                signal,sr = librosa.load(file_path)

                # Only if lenght of audio greater than 1 second
                if (len(signal)) == SAMPLES_TO_CONSIDER:
                    signal = signal[:SAMPLES_TO_CONSIDER] #capping to 1 second to preserve dimensionalilty
                    MFCC = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCC.T.tolist())
                    data["files"].append(file_path)
                    print(file_path + " " + str(i-1))


    with open(json_path,"w") as fp:
        json.dump(data,fp,indent = 4)

if __name__ == "__main__":
    preproccessing(DATASET_PATH,JSON_PATH)
