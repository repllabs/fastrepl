def map_number_range(value, from_min, from_max, to_min, to_max) -> float:
    assert from_min < from_max
    assert to_min < to_max

    return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min
