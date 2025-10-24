# train.py
import os
import argparse
from ocr_model_tf import build_crnn, CTCModel
from utils import load_labels, data_generator, CHARS, NUM_CLASSES
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", default="data/labels/train_labels.csv")
parser.add_argument("--train_dir", default="data/train")
parser.add_argument("--val_csv", default="data/labels/val_labels.csv")
parser.add_argument("--val_dir", default="data/val")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--out_weights", default="crnn_weights.h5")
args = parser.parse_args()

# load rows
train_rows = load_labels(args.train_csv, args.train_dir)
val_rows = load_labels(args.val_csv, args.val_dir)

# build model
num_classes = len(CHARS) + 1
base_net = build_crnn(input_shape=(32,128,1), num_classes=num_classes)
model = CTCModel(base_net, CHARS)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer)

# dataset as tf.data for speed
train_gen = data_generator(train_rows, batch_size=args.batch_size, shuffle=True)
val_gen = data_generator(val_rows, batch_size=args.batch_size, shuffle=False)

steps_per_epoch = max(1, len(train_rows) // args.batch_size)
validation_steps = max(1, len(val_rows) // args.batch_size)

# fit using keras fit (generator produces tuples expected by our train_step)
model.fit(
    x=train_gen,
    epochs=args.epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps
)

# save weights
model.net.save_weights(args.out_weights)
print(f"Saved weights to {args.out_weights}")
