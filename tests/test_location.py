import pytest

from src.location import BBox


class TestBBox:

    def test_error_on_both_right_and_with_passed(self):
        with pytest.raises(ValueError) as excinfo:
            BBox(1, 2, right=3, width=4, bottom=5)

    def test_error_on_both_bottom_and_height_passed(self):
        with pytest.raises(ValueError) as excinfo:
            BBox(1, 2, right=3, bottom=4, height=5)

    def test_error_on_neither_right_or_width_passed(self):
        with pytest.raises(ValueError) as excinfo:
            BBox(1, 2, bottom=3)

    def test_error_on_neither_bottom_or_height_passed(self):
        with pytest.raises(ValueError) as excinfo:
            BBox(1, 2, right=3)

    def test_valid_initalization(self):
        BBox(1, 2, right=3, bottom=4)
        BBox(1, 2, right=3, height=4)
        BBox(1, 2, width=3, bottom=4)
        BBox(1, 2, width=3, height=4)

    def test_width_getter(self):
        assert BBox(1, 2, right=7, bottom=11).width == 6

    def test_height_getter(self):
        assert BBox(1, 2, right=7, bottom=11).height == 9

    def test_width_setter(self):
        bbox = BBox(1, 2, right=3, bottom=4)
        bbox.width = 6
        assert bbox.right == 7

    def test_height_setter(self):
        bbox = BBox(1, 2, right=3, height=4)
        bbox.height = 9
        assert bbox.bottom == 11
