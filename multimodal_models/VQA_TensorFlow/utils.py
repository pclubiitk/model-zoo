import os
import requests
import zipfile

########################################

def extract(url, base, path, name):
    if not os.path.exists(base + "/zips/" + name + ".zip"):
        
        print(f"Downloading from {url}")
        r = requests.get(url) 
        with open(base + "/zips/" + name + ".zip",'wb') as f: 
            f.write(r.content)

        if not os.path.exists(base + path):
            os.makedirs(base + path)

        with zipfile.ZipFile(base + "/zips/" + name + ".zip", 'r') as zip_ref:
            zip_ref.extractall(base + path) 
        
        print("Saved extracted files at " + base + path)


#########################################

def get_data(path):
    base = path + "/data"
    if not os.path.exists(base + "/zips"):
        os.makedirs(base + "/zips")
    urls = ["https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip",
            "http://nlp.stanford.edu/data/glove.6B.zip", 
            "https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/data_train_val.zip",]
    paths = ["/val_annotations", "/train_ques", "/glove", "/image_data"]
    
    for i in range(4):
        extract(urls[i], base, paths[i], str(i))

