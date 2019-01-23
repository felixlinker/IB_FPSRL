from os import makedirs, path

def ensure_can_write(file_path):
    dir_path, file_name = path.split(file_path)
    assert file_name != ''
    try:
        makedirs(dir_path)
    except FileExistsError:
        pass
    return file_path
