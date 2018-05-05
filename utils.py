import re
import hashlib
import tensorflow as tf
import os
from scipy import ndimage, misc
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from os.path import basename


def load_images(image_dir, testing_percentage, shape):
    """Загружает изображения из папки с картинками, делит на трейн и тест,
       масштабирует выбранного размера
       На выходе возвращает словарь с обучающим и тестовым множеством, 
       картинками и метками классов для них на основе имен файлов
    """

    def load_single_image(file_path):
        image = ndimage.imread(file_path, mode="RGB")
        return misc.imresize(image, shape)

    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    image_list = {"train": {"labels": [], "images": []}, 
                  "test":  {"labels": [], "images": []}}
    labels = dict()

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(tf.gfile.Glob(file_glob))

    for file_name in file_list:
        base_file_name = basename(file_name).lower().split(".")[0]
        label_name = re.sub(r'[^a-z]+', '', base_file_name)
        label_index = labels.setdefault(label_name, len(labels))

        dataset_type = "test" if is_test(file_name, testing_percentage) else "train"
        image_list[dataset_type]["labels"].append(label_index)
        image_list[dataset_type]["images"].append(load_single_image(file_name))

    enc = OneHotEncoder()
    enc.fit(np.array(image_list["train"]["labels"]).reshape((-1, 1)))
    for dataset_type in ["train", "test"]:
        labels_array = np.array(image_list[dataset_type]["labels"], dtype='float32').reshape((-1, 1))
        image_list[dataset_type]["one_hot_labels"] = enc.transform(labels_array).toarray()
        image_list[dataset_type]["images"] = np.array(image_list[dataset_type]["images"], dtype='float32')

    return image_list


def is_test(file_name, testing_percentage):
    max_num_images_per_class = 2 ** 27 - 1  # ~134M
    hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(file_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (max_num_images_per_class + 1)) *
                       (100.0 / max_num_images_per_class))
    return percentage_hash < testing_percentage
