file = 'data.txt'
out = 'result.txt'
with open(file, 'r') as f_in, open(out, 'w') as f_out:
    for line in f_in:
        if line.strip():  # skip empty lines
            x_str, y_str = line.split()
            x = float(x_str)
            y = float(y_str)
            if x > y:
                f_out.write(f'0\n')
            else:
                f_out.write(f'1\n')


