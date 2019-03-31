import sys
import os
import argparse

import pandas
import keras
import keras.applications


def parse_args(argv):
    """parse input arguments"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_train', type=str,
                        help='Input path to train data file')
    parser.add_argument('image_dir', type=str,
                        help='Input path to image directory')

    parser.add_argument('out_dir', type=str,
                        help='Output path to generated files')

    args = parser.parse_args(args=argv)
    return args


def preproc(data):
    data /= 127.5
    data -= 1.
    return data


def main(args):
    """main function"""

    # Parse args
    args = parse_args(args)
    print('Args: %s' % str(args))

    base_model = keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224, 224, 1),
        include_top=False,
        weights=None,
        pooling='avg'
        )

    out = base_model.output
    pred = keras.layers.Dense(1, activation='sigmoid')(out)

    model = keras.models.Model(
        inputs=base_model.input,
        outputs=pred
        )

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Metrics
    metrics = [keras.metrics.binary_accuracy]

    optimizer = keras.optimizers.Adam()

    # Loss
    loss = keras.losses.binary_crossentropy

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
        )

    model.summary()

    # Read data
    df_train = pandas.read_csv(args.data_train, dtype=str)
    mask = df_train['Finding Labels'] != 'No Finding'
    df_train.loc[mask, 'Finding Labels'] = 'Potential Disease'

    # Data generators
    datagen_train = keras.preprocessing.image.ImageDataGenerator(
       # rotation_range=40,
       # width_shift_range=0.2,
       # height_shift_range=0.2,
       # shear_range=0.2,
       # zoom_range=0.2,
       # horizontal_flip=True,
       # fill_mode='nearest',
        preprocessing_function=preproc
        )

    flow_train = datagen_train.flow_from_dataframe(
        dataframe=df_train,
        x_col='Image Index',
        y_col='Finding Labels',
        batch_size=32,
        shuffle=True,
        directory=args.image_dir,
        target_size=(224,224),
        color_mode='grayscale',
        class_mode='binary',
        )

    # Callbacks
    checkpoints = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.out_dir, 'model-checkpoint-{epoch:03d}.hdf5'),
        verbose=1
        )

    save_best = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.out_dir, 'model-best-{epoch:03d}.hdf5'),
        monitor='binary_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
        )

    callbacks = [checkpoints, save_best]

    # Fit
    model.fit_generator(
        generator=flow_train,
        steps_per_epoch=flow_train.n // flow_train.batch_size,
        epochs=200,
        callbacks=callbacks,
        verbose=1,
        max_queue_size=flow_train.batch_size * 2,
        workers=10
        )

    print('Done!')


if __name__ == '__main__':
    main(sys.argv[1:])
