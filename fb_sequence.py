
class FBGenerator:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ptr = low+1
        self.direction = -1

    def next(self):
        self.ptr += self.direction
        if (self.ptr == self.high) or (self.ptr == self.low):
            self.direction *= -1
        return (self.ptr, self.ptr+self.direction)