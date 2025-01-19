import shutil
from collections import defaultdict
from typing import Dict, Union, Any

class MultiSet:
    def __init__(self, init_data=None):
        self.data = defaultdict(int)
        if init_data is not None:
            if isinstance(init_data, dict):
                self.data.update(init_data)
            else:
                for item in init_data:
                    self.data[item] += 1

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def clear(self):
        self.data.clear()

    def copy(self):
        return MultiSet(self.data.copy())

    def __getitem__(self, key: Any) -> int:
        return self.data[key]

    def __setitem__(self, key: Any, value: int):
        self.data[key] = value

    def __repr__(self) -> str:
        return f"MultiSet({dict(self.data)})"

    def __eq__(self, other) -> bool:
        if isinstance(other, MultiSet):
            return self.data == other.data
        return False

    def _operate(self, other, operation):
        result = MultiSet()
        keys = set(self.data.keys()) | set(other.data.keys())
        for key in keys:
            result[key] = operation(self[key], other[key])
        return result

    def _operate_scalar(self, scalar, operation):
        result = MultiSet()
        for key in self.data:
            result[key] = operation(self[key], scalar)
        return result

    def __add__(self, other):
        if isinstance(other, MultiSet):
            return self._operate(other, lambda x, y: x + y)
        return self._operate_scalar(other, lambda x, y: x + y)

    def __sub__(self, other):
        if isinstance(other, MultiSet):
            return self._operate(other, lambda x, y: x - y)
        return self._operate_scalar(other, lambda x, y: x - y)

    def __mul__(self, other):
        if isinstance(other, MultiSet):
            return self._operate(other, lambda x, y: x * y)
        return self._operate_scalar(other, lambda x, y: x * y)

    def __truediv__(self, other):
        if isinstance(other, MultiSet):
            return self._operate(other, lambda x, y: x / y if y != 0 else 0)
        return self._operate_scalar(other, lambda x, y: x / y if y != 0 else 0)

    def __iadd__(self, other):
        if isinstance(other, MultiSet):
            for key in other.data:
                self.data[key] += other[key]
        else:
            for key in self.data:
                self.data[key] += other
        return self

    def __isub__(self, other):
        if isinstance(other, MultiSet):
            for key in other.data:
                self.data[key] -= other[key]
        else:
            for key in self.data:
                self.data[key] -= other
        return self

    def __imul__(self, other):
        if isinstance(other, MultiSet):
            for key in other.data:
                self.data[key] *= other[key]
        else:
            for key in self.data:
                self.data[key] *= other
        return self

    def __itruediv__(self, other):
        if isinstance(other, MultiSet):
            for key in other.data:
                self.data[key] /= other[key] if other[key] != 0 else 0
        else:
            for key in self.data:
                self.data[key] /= other if other != 0 else 0
        return self

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        result = MultiSet()
        for key in self.data:
            result[key] = other / self[key] if self[key] != 0 else 0
        return result

    def __neg__(self):
        return self._operate_scalar(-1, lambda x, y: x * y)

def format_dict_pretty(d: Dict[str, Any], min_spacing: int = 2) -> str:
    """
    Format dictionary key-value pairs in aligned columns that fit the terminal width.
    Keys are sorted alphabetically and arranged in columns from top to bottom.
    Floating point values are limited to 4 decimal places.

    Args:
        d: Dictionary to format
        min_spacing: Minimum number of spaces between columns

    Returns:
        Formatted string representation of the dictionary
    """

    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns

    # Convert all items to strings, handling floats specially
    items = []
    for k, v in d.items():
        key_str = str(k)
        if isinstance(v, float):
            val_str = f"{v:.4f}" #.rstrip('0').rstrip('.')
        else:
            val_str = str(v)
        items.append((key_str, val_str))

    # Sort items by key
    items.sort(key=lambda x: x[0])

    # Find the maximum lengths
    max_key_len = max((len(k) for k, v in items), default=0)
    max_val_len = max((len(v) for k, v in items), default=0)

    # Calculate the width needed for each pair
    pair_width = max_key_len + max_val_len + 4  # 4 accounts for ': ' and minimum spacing

    # Calculate how many pairs can fit in one row
    pairs_per_row = max(1, terminal_width // pair_width)

    # Calculate number of rows needed
    num_rows = (len(items) + pairs_per_row - 1) // pairs_per_row

    # Reorder items to go down columns instead of across rows
    column_ordered_items = []
    for row in range(num_rows):
        for col in range(pairs_per_row):
            idx = row + col * num_rows
            if idx < len(items):
                column_ordered_items.append(items[idx])

    # Build the output string
    result = []
    for i in range(0, len(column_ordered_items), pairs_per_row):
        row_items = column_ordered_items[i:i + pairs_per_row]
        format_str = ""
        for _ in row_items:
            format_str += f"{{:<{max_key_len}}}: {{:<{max_val_len}}}{' ' * min_spacing}"
        result.append(format_str.format(*[item for pair in row_items for item in pair]))

    return "\n".join(result)

