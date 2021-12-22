import os




def choose_best_model(models_dir):

    MAX_LOSS = 9999
    BEST_MODEL = None

    for model in os.listdir(models_dir):
        
        file = os.path.join(models_dir, model)
        loss = round(float(file.split('_')[-1].split('.')[0]), 3)
        if loss< MAX_LOSS:
            MAX_LOSS = loss
            BEST_MODEL = file
    print(f"The Best Model is {BEST_MODEL.split('/')[-1]}")
    
    return file
         
