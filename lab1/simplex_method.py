def read_lp_from_file(filename):
    """ File format:
    max
    1 2 4 1
    3
    1 1 1 0 <= 10
    0 1 2 1 = 6
    1 0 0 1 >= 2
    accuracy=6
    """

    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    obj_type = lines[0].lower().strip()
    obj_coeffs = list(map(float, lines[1].split()))
    m = int(lines[2])

    constraints_raw = []
    for i in range(3, 3 + m):
        parts = lines[i].split()
        b = float(parts[-1])
        sign = parts[-2]
        coeffs = list(map(float, parts[:-2]))
        constraints_raw.append((coeffs, sign, b))

    accuracy_line = [line for line in lines if line.startswith('accuracy=')]
    if accuracy_line:
        accuracy = int(accuracy_line[0].split('=')[1])
    else:
        accuracy = 6

    return obj_type, obj_coeffs, constraints_raw, accuracy


def is_linear(coefficients):
    """Check if all coefficients are numbers (linear)."""
    return all(isinstance(c, (int, float)) for c in coefficients)


def preprocess_lp(obj_type, obj_coeffs, constraints_raw, accuracy=6):
    """ Converts problem to canonical form
    obj_type: 'max' / 'min'
    obj_coeffs: coefficients [c1, c2, c3, ...]
    constraints_raw: список ограничений в формате:
        [
          ([coeffs], sign, right_side),
          ...
        ]
        sign: ('<=', '>=', '=')
    accuracy: number of digits after the decimal point
    :returns:
        obj, constraints, rhs, accuracy, is_maximization
    """

    obj = obj_coeffs.copy()
    constraints = []
    rhs = []

    is_maximization = (obj_type.lower() == 'max')

    for coeffs, sign, b in constraints_raw:
        if sign == '<=':
            constraints.append(coeffs)
            rhs.append(b)
        elif sign == '>=':
            constraints.append([-c for c in coeffs])
            rhs.append(-b)
        elif sign == '=':
            constraints.append(coeffs)
            rhs.append(b)
            constraints.append([-c for c in coeffs])
            rhs.append(-b)
        else:
            raise ValueError(f"Wrong constraint sign: {sign}")

    return obj, constraints, rhs, accuracy, is_maximization


def round_value(val, accuracy):
    return round(val, accuracy)


def simplex(obj, constraints, rhs, accuracy, is_maximization):
    """Main simplex function"""
    n = len(obj)
    m = len(constraints)

    if not is_maximization:
        obj = [-x for x in obj]

    # Building the simplex tableau
    table = [[0 for _ in range(n + 1 + m)] for _ in range(m + 1)]  # Create table

    for i in range(n):
        table[0][i] = -obj[i]  # Fill in z-row

    for i in range(1, m + 1):  # Fill in constraints coefficients
        for j in range(n):
            if j < len(constraints[i - 1]):
                table[i][j] = constraints[i - 1][j]

    table[0][-1] = 0  # RHS of z
    for i in range(1, m + 1):  # Other RHS
        table[i][-1] = rhs[i - 1]

    for i in range(1, m + 1):  # Fill in slack variables
        for j in range(n, n + m):
            if j - n == i - 1:
                table[i][j] = 1

    # Initialize basis variables
    basis = [n + i for i in range(m)]  # [n, n+1, ..., n+m-1]

    # Initialize answers and z_value
    # Removed initial answers assignment as it will be handled after optimization
    z_value = 0  # This will store the value of the objective function

    while any(round_value(x, accuracy) < 0 for x in table[0][:-1]):  # Exclude RHS in z-row
        key_col = -1
        min_val = float('inf')

        # Find the column to pivot on (most negative coefficient in z-row)
        for i in range(n + m):
            if round_value(table[0][i], accuracy) < round_value(min_val, accuracy):
                min_val = table[0][i]
                key_col = i

        if key_col == -1:
            print("No valid pivot column found. The method is not applicable!")
            exit()

        key_row = -1
        min_ratio = float('inf')

        # Find the row to pivot on using the minimum ratio test
        for i in range(1, m + 1):
            if round_value(table[i][key_col], accuracy) > 0:
                ratio = table[i][-1] / table[i][key_col]
                if 0 <= round_value(ratio, accuracy) < round_value(min_ratio, accuracy):
                    min_ratio = ratio
                    key_row = i

        if key_row == -1:
            print("Unbounded solution. The method is not applicable!")
            exit()

        # Perform the pivot
        pivot = table[key_row][key_col]
        for i in range(n + m + 1):
            table[key_row][i] = round_value(table[key_row][i] / pivot, accuracy)

        for i in range(m + 1):  # Make all zeroes in key-column except key-row
            if i != key_row:
                divisor = table[i][key_col]
                for j in range(n + m + 1):
                    table[i][j] = round_value(table[i][j] - divisor * table[key_row][j], accuracy)

        # Update basis
        basis[key_row - 1] = key_col  # Update the basis with the new basic variable

        # Check for degeneracy
        if round_value(table[key_row][-1], accuracy) == 0:
            print(f"Degeneracy detected in row {key_row}. A basic variable is zero.")

        z_value = round_value(table[0][-1], accuracy)  # Update the current value of z

    # After the loop, determine the values of the decision variables
    answers = [0] * n  # Initialize all decision variables to zero
    for i in range(m):
        if basis[i] < n:  # Only assign values to decision variables, not slack variables
            answers[basis[i]] = round_value(table[i + 1][-1], accuracy)  # Assign the RHS value

    return z_value, answers


def output_values(z_value, answers, is_maximization):
    """ Output Function """
    # 1. Print the optimization problem
    if is_maximization:
        optimization_type = "\nMaximize"
    else:
        optimization_type = "\nMinimize"

    # Construct the objective function string
    objective_terms = []
    for i, coeff in enumerate(obj):
        term = f"{coeff}*x{i + 1}"
        objective_terms.append(term)
    objective_str = " + ".join(objective_terms)

    print(f"{optimization_type} z = {objective_str}")

    # Print the constraints
    print("subject to the constraints:")
    for i, constraint in enumerate(constraints):
        constraint_terms = []
        for j, coeff in enumerate(constraint):
            term = f"{coeff}*x{j + 1}"
            constraint_terms.append(term)
        constraint_str = " + ".join(constraint_terms)
        print(f"{constraint_str} <= {rhs[i]}")

    print()  # Add an empty line for better readability

    # 2. Print the solution
    if is_maximization:
        print("Maximum z =", z_value)
    else:
        print("Minimum z =", -z_value)

    # Output the values of decision variables
    for i in range(len(answers)):
        print(f"x{i + 1} =", answers[i])

if __name__ == '__main__':
    filename = "task1.txt"
    obj_type, obj_coeffs, constraints_raw, accuracy = read_lp_from_file(filename)

    """Problem preparation to simplex (converting to canonical form)"""
    obj, constraints, rhs, accuracy, is_maximization = preprocess_lp(
        obj_type, obj_coeffs, constraints_raw, accuracy
    )

    z_value, answers = simplex(obj, constraints, rhs, accuracy, is_maximization)
    output_values(z_value, answers, is_maximization)


