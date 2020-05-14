from prerequisites import *
import re


def draw_plots(feature='learning_rate_'):
    files_list = [f'results/{f}' for f in os.listdir(RESULTS_DIR) if isfile(join(RESULTS_DIR, f)) and 'pickle' in f]
    for file_name in files_list:
        research = re.search(rf'({feature}.*)\.pickle', file_name)
        if research is None:
            continue
        name = research.group(1)
        with open(file_name, 'rb') as f:
            training_history = pickle.load(f)
        draw_training_info(training_history, name)
        del training_history
        K.clear_session()
        gc.collect()


def print_statistics(_test_generator, _test_steps, feature='learning_rate_'):
    test_score = dict()
    files_list = [f'results/{f}' for f in os.listdir(RESULTS_DIR) if isfile(join(RESULTS_DIR, f)) and 'h5' in f]
    for file_name in files_list:
        research = re.search(rf'({feature}.*)\.h5', file_name)
        if research is None:
            continue
        name = research.group(1)
        model = models.load_model(file_name)
        test_score[name] = model.evaluate_generator(_test_generator, steps=_test_steps)
        del model
        K.clear_session()
        gc.collect()

    # print statistics
    for name, score in test_score.items():
        print(f'\n\n{name}')
        print(f'Test loss: {score[0]}')
        print(f'Test acc:  {score[1]}')


# get dataset
batch_size = 15
class_count, _, _, _, _, test_datagen, test_generator = get_data(batch_size)
test_steps = len(test_generator.filenames) // batch_size

draw_plots('adam_2c_')
print_statistics(test_generator, test_steps, 'adam_2c_')
