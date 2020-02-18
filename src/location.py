import abc


class BaseLocation(abc.ABC):
    pass


class BaseKnownLocation(BaseLocation, abc.ABC):
    pass


class UnknownLocation(BaseLocation):
    pass


class BBox(BaseKnownLocation):
    __slots__ = ('left', 'top', 'right', 'bottom')

    def __init__(self, left, top, *, right=None, bottom=None, width=None, height=None):
        self.left = left
        self.top = top
        if right is not None:
            self.right = right
            if width is not None:
                raise ValueError("Either argument 'right' or 'width' may be passed, not both!")
        elif width is not None:
            self.width = width
        else:
            raise ValueError("Argument 'right' or 'width' must be passed!")
        if bottom is not None:
            self.bottom = bottom
            if height is not None:
                raise ValueError("Either argument 'bottom' or 'height' may be passed, not both!")
        elif height is not None:
            self.height = height
        else:
            raise ValueError("Argument 'bottom' or 'height' must be passed!")

    def __str__(self):
        return f"left={self.left} top={self.top} right={self.right} bottom={self.bottom}"

    @property
    def width(self):
        return self.right - self.left

    @width.setter
    def width(self, val):
        self.right = self.left + val

    @property
    def height(self):
        return self.bottom - self.top

    @height.setter
    def height(self, val):
        self.bottom = self.top + val
