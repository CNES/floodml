#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""



from __future__ import print_function
import os
import shutil
import logging

log = logging.getLogger(__name__)


def create_directory(path):
    """
    Creates a Directory with the given path. Throws a warning if it's already existing and an error if
    a file with the same name already exists.
    :param path: The full path to the new directory
    """
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError("Cannot create the output directory. There is a file with the same name: %s" % path)
        else:
            logging.debug("Directory already existing: %s" % path)
    else:
        try:
            os.makedirs(path)
        except OSError:
            raise IOError("Cannot create directory %s" % path)
    return 0


def remove_file(filename):
    """
    Removes File of given path
    :param filename: Path to File
    """
    try:
        os.remove(filename)
    except OSError:
        log.debug("Cannot remove file %s" % filename)



def remove_directory(directory):
    """
    Removes Directory of given path
    :param directory: Path to Directory
    """
    try:
        shutil.rmtree(directory)
    except OSError:
        log.debug("Cannot remove directory {0}".format(directory))


def find(pattern, path, case_sensitive=False, depth=None, ftype="all"):
    """
    Find a file or dir in a directory-tree of given depth.

    :param pattern: The filename to be searched for
    :param path: The path to the root directory
    :param case_sensitive: Do a case sensitive comparison. Default is False.
    :param depth: Search only up to a specified depth. Default is None, signifying a maximum limit of 20.
    :param ftype: Can be "file", "folder" or "all".
    :return: The file/directory if found. AssertionError if not.
    """
    import re
    result = []

    reg_to_find = pattern.replace("*", ".*")

    if not case_sensitive:
        reg_to_find = reg_to_find.lower()

    path = os.path.abspath(path)
    if not depth:
        depth = 20  # Limit depth in case it is not specified.
    for root, dirs, files in os.walk(path):
        if root[len(path):].count(os.sep) < depth:
            if ftype == "all":
                names = files + dirs
            elif ftype == "file":
                names = files
            elif ftype == "folder":
                names = dirs
            else:
                raise ValueError("Unknown type %s" % ftype)
            for name in names:
                if re.search(reg_to_find, name if case_sensitive else name.lower()):
                    result.append(os.path.join(root, name))
    if not result:
        raise ValueError("Cannot find %s in %s" % (pattern, path))
    return result


def find_single(pattern, path, case_sensitive=False, depth=None, ftype="all"):
    """
    Find a single file or dir in a directory-tree.

    :param pattern: The filename to be searched for
    :param path: The path to the root directory
    :param case_sensitive: Do a case sensitive comparison.
    :param depth: Search only up to a specified depth. Default is None, signifying a maximum limit of 20.
    :param ftype: Can be "file", "folder" or "all".
    :return: The file/directory if found. ValueError if not.
    """
    return find(pattern, path, case_sensitive=case_sensitive, depth=depth, ftype=ftype)[0]


def symlink(src, dst):
    """
    Create symlink from src to dst and raise Exception if it didnt work
    :param src: The full path to the source file or folder
    :param dst: The destination folder
    :return: None. OSError if symlink cannot be created.
    """
    if os.path.exists(dst) and os.path.islink(dst):
        logging.debug("File already existing: %s" % dst)
        return

    try:
        os.symlink(src, dst)
    except OSError:
        raise OSError("Cannot create symlink for %s at %s."
                      "Does your plaform support symlinks?" % (src, dst))


def __get_return_code(proc, log_level):
    """
    Read the stdout of a subprocess while also processing its return code if available
    :param proc: The subprocess
    :param log_level: The log level for the messages displayed.
    :return: The return code of the app
    """
    while proc.poll() is None:
        line = proc.stdout.readline()  # This blocks until it receives a newline.
        if log_level == logging.DEBUG:
            print(line.decode('utf-8'), end="")
    proc.stdout.close()
    return proc.wait()


def find_in_file(filename, pattern):
    """
    Find a pattern inside a file
    :param filename: The filename
    :param pattern: The pattern to be searched for
    :return: The pattern (if found). None if not.
    """
    import re
    assert os.path.isfile(filename)
    with open(filename, "r") as f:
        content = "".join(f.read().splitlines())
        lut_url = re.search(pattern, content)
    if lut_url:
        return lut_url.group()
    return None
