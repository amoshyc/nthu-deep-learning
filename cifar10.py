import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

tfe = tf.contrib.eager
tf.enable_eager_execution()


def prepare_cifar(src_dir, dst_dir, n_train=50_000):
    src_dir = Path(src_dir)
    bin_paths = sorted(list(src_dir.glob('*.bin')))
    dst_dir = Path(dst_dir)
    train_dir = dst_dir / 'train'
    valid_dir = dst_dir / 'valid'
    for dir_ in [train_dir, valid_dir]:
        dir_.mkdir(exist_ok=True, parents=True)

    train_anns = []
    valid_anns = []

    def process_img_lbl(img, lbl):
        if len(train_anns) >= n_train:
            img_path = valid_dir / f'{len(valid_anns):06d}.jpg'
            img_path = img_path.resolve().absolute()
            Image.fromarray(img).save(img_path, quality=100)
            valid_anns.append({'img_path': str(img_path), 'label': int(lbl)})
        else:
            img_path = train_dir / f'{len(train_anns):06d}.jpg'
            img_path = img_path.resolve().absolute()
            Image.fromarray(img).save(img_path, quality=100)
            train_anns.append({'img_path': str(img_path), 'label': int(lbl)})

    for bin_path in tqdm(bin_paths, desc='Prepare'):
        data = np.fromfile(str(bin_path), dtype=np.uint8)
        data = data.reshape(10_000, 3073)
        lbls = data[:, 0]
        imgs = data[:, 1:].reshape(-1, 3, 32, 32)
        imgs = imgs.transpose(0, 2, 3, 1)
        for img, lbl in zip(imgs, lbls):
            process_img_lbl(img, lbl)

    with (train_dir / 'ann.json').open('w') as f:
        json.dump(train_anns, f, indent=2, ensure_ascii=False)
    with (valid_dir / 'ann.json').open('w') as f:
        json.dump(valid_anns, f, indent=2, ensure_ascii=False)


# src_dir = './data/cifar-10-batches-bin/'
# dst_dir = './data/cifar10/'
# prepare_cifar(src_dir, dst_dir)


def py_func_decorator(output_types=None,
                      output_shapes=None,
                      stateful=True,
                      name=None):
    def decorator(func):
        def call(*args, **kwargs):
            return tf.contrib.framework.py_func(
                func=func,
                args=args,
                kwargs=kwargs,
                output_types=output_types,
                output_shapes=output_shapes,
                stateful=stateful,
                name=name)

        return call

    return decorator





with open('./data/cifar10/train/ann.json') as f:
    train_anns = json.load(f)
with open('./data/cifar10/valid/ann.json') as f:
    valid_anns = json.load(f)


@py_func_decorator(
    output_types=(tf.float32, tf.float32), output_shapes=([32, 32, 3], [10]))
def load_train(idx):
    return get_item(train_anns, idx, [32, 32])


@py_func_decorator(
    output_types=(tf.float32, tf.float32), output_shapes=([32, 32, 3], [10]))
def load_valid(idx):
    return get_item(valid_anns, idx, [32, 32])


n_train = len(train_anns) // 256 * 256
train_set = tf.data.Dataset.range(n_train)
train_set = train_set.map(load_train, num_parallel_calls=1)
train_set = train_set.prefetch(512).batch(256)

n_valid = len(valid_anns) // 256 * 256
valid_set = tf.data.Dataset.range(n_valid)
valid_set = valid_set.map(load_valid, num_parallel_calls=1)
valid_set = valid_set.prefetch(512).batch(256)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D([2, 2], strides=2),
    tf.keras.layers.Conv2D(64, [3, 3], padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D([2, 2], strides=2),
    tf.keras.layers.Conv2D(128, [3, 3], padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D([2, 2], strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

optim = tf.train.AdamOptimizer(learning_rate=1e-3)
model.compile(
    loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
model.summary()

for epoch in range(10):
    print('Epoch', epoch)

    metrics = {
        'loss': tfe.metrics.Mean(),
        'acc': tfe.metrics.Mean(),
    }
    pbar = tqdm(total=n_train, desc='  Train', ascii=True)
    for img_batch, lbl_batch in train_set:
        img_batch, lbl_batch = np.float32(img_batch), np.float32(lbl_batch)
        loss, acc = model.train_on_batch(img_batch, lbl_batch)
        metrics['loss'](loss)
        metrics['acc'](acc)
        postfix = {k: f'{v.result():.3f}' for k, v in metrics.items()}
        pbar.set_postfix(postfix)
        pbar.update(img_batch.shape[0])
    pbar.close()

    metrics = {
        'loss': tfe.metrics.Mean(),
        'acc': tfe.metrics.Mean(),
    }
    pbar = tqdm(total=n_valid, desc='  Valid', ascii=True)
    for img_batch, lbl_batch in valid_set:
        img_batch, lbl_batch = np.float32(img_batch), np.float32(lbl_batch)
        loss, acc = model.test_on_batch(img_batch, lbl_batch)
        metrics['loss'](loss)
        metrics['acc'](acc)
        postfix = {k: f'{v.result():.3f}' for k, v in metrics.items()}
        pbar.set_postfix(postfix)
        pbar.update(img_batch.shape[0])
    pbar.close()
