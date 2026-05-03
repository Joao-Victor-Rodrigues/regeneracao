def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def _get(row, key):
    val = row.get(key)
    if val is None:
        return None
    return float(val)


def calculate_indices(row):
    b2 = _get(row, 'B2_median')
    b4 = _get(row, 'B4_median')
    b8 = _get(row, 'B8_median')
    b11 = _get(row, 'B11_median')
    b12 = _get(row, 'B12_median')

    if any(v is None for v in (b2, b4, b8, b11, b12)):
        return {'NDVI': None, 'NDMI': None, 'EVI': None, 'SAVI': None, 'NBR': None}

    ndvi = safe_divide(b8 - b4, b8 + b4)
    ndmi = safe_divide(b8 - b11, b8 + b11)
    evi = safe_divide(b8 - b4, b8 + 6 * b4 - 7.5 * b2 + 1)

    savi = safe_divide(b8 - b4, b8 + b4 + 0.5)
    if savi is not None:
        savi *= 1.5

    nbr = safe_divide(b8 - b12, b8 + b12)

    return {
        'NDVI': ndvi,
        'NDMI': ndmi,
        'EVI': evi,
        'SAVI': savi,
        'NBR': nbr,
    }
