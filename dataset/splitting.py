import os


def save_cls_file(classes, filename):
    with open(filename, 'w') as file:
        for cls in classes:
            file.write(cls)
            file.write('\n')


def save_ref_file(flattened_dataset, filename):
    if os.path.exists(filename):
        raise FileExistsError(filename)
    
    with open(filename, 'w') as file:
        for fn, cls_id in flattened_dataset:
            file.write(fn)
            file.write(',')
            file.write(str(cls_id))
            file.write('\n')


def load_ref_file(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            fn, cls_id = line.split(',')
            cls_id = int(cls_id)
            dataset.append((fn, cls_id))
    return tuple(dataset)


def load_ref_files(dataset_dir):
    with open(os.path.join(dataset_dir, 'classes.txt')) as file:
        classes = tuple(line.strip() for line in file)
    
    train_set = load_ref_file(os.path.join(dataset_dir, 'train_set.txt'))
    test_set = load_ref_file(os.path.join(dataset_dir, 'test_set.txt'))
    valid_set = load_ref_file(os.path.join(dataset_dir, 'valid_set.txt'))
    
    return classes, train_set, test_set, valid_set
