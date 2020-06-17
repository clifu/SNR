from prerequisites import *
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import re
import seaborn as sns
import pandas as pd

def print_confusion_matrix(_test_generator, _test_steps, feature='learning_rate_'):
    y_preds = dict()
    files_list = [f'results/{f}' for f in os.listdir(RESULTS_DIR) if isfile(join(RESULTS_DIR, f)) and 'h5' in f]
    for file_name in files_list:
        research = re.search(rf'({feature}.*)\.h5', file_name)
        if research is None:
            continue
        name = research.group(1)
        model = models.load_model(file_name)
        Y_pred = model.predict_generator(_test_generator, _test_steps)
        y_preds[name] = np.argmax(Y_pred, axis=1)

        del model
        K.clear_session()
        gc.collect()

    # print Confusion Matrix
    classes = np.arange(class_count)
    for name, y_pred in y_preds.items():
        con_mat = confusion_matrix(_test_generator.classes, y_pred)
        print(f'\n\n{name}')
        print('Confusion Matrix')
        print(con_mat)
        
        # normalization & heat map
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

        con_mat_df = pd.DataFrame(con_mat_norm,
                            index = classes, 
                            columns = classes)

        figure = plt.figure(figsize=(8, 8))
        sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


# get dataset
batch_size = 15
class_count, _, _, _, _, test_datagen, test_generator = get_data(batch_size)
test_steps = len(test_generator.filenames) // batch_size

print_confusion_matrix(test_generator, test_steps, 'adam_2c_batch_v1_')