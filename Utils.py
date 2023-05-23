import os.path


def folder_creation(file_name: str, verbose: bool = True):
    """ Create the required file if it doesn't already exist."""

    if not os.path.exists(file_name):
        os.mkdir(file_name)
        if verbose:
            print('Created a /{}/ directory in {}/'.format(file_name, os.getcwd()))
                