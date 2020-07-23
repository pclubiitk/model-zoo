def load_file(path):
    
    with open(path, "r") as file:
        data = file.read()
    data = data.split('\n')
    file.close()
    return data


def load_all_descriptions(desc_data):
    temp_descriptions = dict()
    
    for i in range(len(desc_data)):
        line = desc_data[i].split('#')
        if len(line) < 2:
            continue
       
        image_name = line[0].split('.')[0]
        
        if image_name not in temp_descriptions.keys():
            temp_descriptions[image_name] = list()
       
        temp_descriptions[image_name].append( line[1][2:] )
        
    return temp_descriptions


def save_descriptions(description, path):
    all_data = list()
    for key, key_des in description.items():
        for i in range(len(key_des)):
            data_to_write = key + " " + key_des[i] 
            all_data.append(data_to_write)
    all_data = '\n'.join(all_data)
    with open(path, "w") as file:
        file.write(all_data)
    file.close()


def load_dataset(path):
    
    with open(path, "r") as file:
        data = file.read()
    data = data.split('\n')
    file.close()
    
    temp_d = list()
    for i in range(len(data)):
        temp_d.append(data[i].split('.')[0])
        
    return temp_d


def load_selected_captions(dataset, entire_dataset):
    
    train_temp = {}
    length = 0
    for i in range(len(entire_dataset)):
        line = entire_dataset[i].split()
        
        img_id = line[0]
        
        if img_id in dataset:
            
            desc = " ".join(line[1:])
            desc = "<sos> " + desc + " <eos>"
            
            if(length<len(desc.split())):
                length = len(desc.split())
                
            if img_id not in train_temp.keys():
                train_temp[img_id] = list()

            train_temp[img_id].append(desc)
        
    return length, train_temp 
    
