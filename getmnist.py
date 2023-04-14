import tensorflow_datasets as tfds
import tensorflow as tf

ds = tfds.load('mnist', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)
