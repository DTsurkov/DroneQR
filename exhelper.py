class GenDataNotValid(Exception):
    def __init__(self, message="DataNotValid. Must be le 15"):
        self.message = message
        super().__init__(self.message)

    pass


class MatrixDataNotValid(Exception):
    def __init__(self, message="MatrixDataNotValid. Dimension must be eq 8"):
        self.message = message
        super().__init__(self.message)

    pass


class MatrixDimensionNotValid(Exception):
    def __init__(self, message="Custom dimensions are not implemented"):
        self.message = message
        super().__init__(self.message)

    pass
