from __future__ import print_function

import os
from random import SystemRandom


class SECRET_KEY:
    value = None
    directory = os.path.dirname(os.path.abspath( __file__ ))
    filename = '.SECRET_KEY'
    length = 50
    allowed_chars = 'abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)'

    def __init__(self, directory=None, filename=None):
        if directory is not None:
            self.directory = directory
        if filename is not None:
            self.filename = filename
        self.path = os.path.join(self.directory, self.filename)
        if self.exist_secret_key():
            self.get_secret_key()
        else:
            self.set_secret_key()

    def exist_secret_key(self):
        if os.path.exists(self.path):
            return True
        return False

    def get_secret_key(self):
        with open(self.path, 'r') as f:
            self.value = f.read(self.length)
        if len([x for x in self.value if x not in self.allowed_chars]) > 0:
            raise ValueError('The secret key must contain only the following characters: {}'. format(self.allowed_chars))
        if len(self.value) != self.length:
            raise ValueError('The secret key must be 50 characters.')

    def set_secret_key(self):
        system_random = SystemRandom()
        self.value = ''.join([system_random.choice(self.allowed_chars) for _ in range(self.length)])
        with open(self.path, 'w') as f:
            f.write(self.value)

    def __str__(self):
        return self.value


if __name__ == '__main__':
    print(SECRET_KEY())
