from prerequisites import *


def train_model(epoch_count, batch_size, file_suffix, optimizer):

    # get dataset
    class_count, train_datagen, train_generator, val_datagen, val_generator, test_datagen, test_generator \
        = get_data(batch_size)
    train_steps = len(train_generator.filenames) // batch_size
    val_steps = len(val_generator.filenames) // batch_size
    test_steps = len(test_generator.filenames) // batch_size

    # get model with trainable: last convolutional layer & all fully connected layers
    model = get_prepared_model(class_count, first_trainable_layer='block1_conv1')

    # compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    # train model
    training_history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epoch_count,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, min_delta=1e-4)]
    )
    model.save(f'results/learned_vgg16_Z2a{file_suffix}.h5')

    # test model
    test_score = model.evaluate_generator(test_generator, steps=test_steps)
    print(f'\n\n{file_suffix}')
    print(f'Test loss: {test_score[0]}')
    print(f'Test acc:  {test_score[1]}')

    # pickle learning history
    with open(f'results/history_vgg16_Z2a{file_suffix}.pickle', 'wb') as file:
        pickle.dump(training_history, file)

    del training_history
    del model
    K.clear_session()
    gc.collect()


def experiment_learning_rate():
    batch_size = 20
    epoch_count = 50
    for learning_rate in [0.001, 0.0005, 0.0001, 0.00005]:
        train_model(
            epoch_count=epoch_count,
            batch_size=batch_size,
            file_suffix=f'_learning_rate_{learning_rate}',
            optimizer=optimizers.RMSprop(learning_rate=learning_rate)
        )


def experiment_batch_size():
    learning_rate = 0.00005
    epoch_count = 70
    optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    for batch_size in [5, 10, 15, 20]:
        train_model(
            epoch_count=epoch_count,
            batch_size=batch_size,
            file_suffix=f'_batch_size_{batch_size}',
            optimizer=optimizer
        )


train_model(
    epoch_count=70,
    batch_size=10,
    file_suffix=f'_adam_2c_batch_v1_',
    optimizer=optimizers.Adam(lr=0.00001)
)
