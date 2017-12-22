class Progress(object):
    def __init__(self, iterable):
        self._length = len(iterable)
        self._iterator = iter(iterable)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            value = next(self._iterator)
            print('\r{} / {}'.format(self._index, self._length), end='', flush=True)
            self._index += 1
            return value
        except StopIteration:
            print('\r', end='', flush=True)
            raise
