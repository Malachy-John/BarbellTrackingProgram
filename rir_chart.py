# Define ranges and their corresponding outputs
ranges = [
    ((0.16, 0.23), 0),
    ((0.23, 0.26), 1),
    ((0.26, 0.30), 2),
    ((0.30, 0.34), 3),
    ((0.34, 0.38), 4),
    ((0.38, 0.42), 5),
    ((0.42, 1.00), 6),
]

def get_rir(value):
    for value_range, output in ranges:
        if value_range[0] <= value < value_range[1]:
            return output
    return None  # Default for out-of-range inputs

# Example usage
values = [0.22, 0.24, 0.27, 0.19]
rir_values = [get_rir(value) for value in values]
print(rir_values)  # Output: [0, 1, 2, 0]