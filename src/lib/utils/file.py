def write_numList_to_file(file_name, list):
    with open(file_name, "w") as f:
        for ele in list:
            f.write(str(ele) + "\n")

def read_numList_from_file(file_name, format="int"):
    if format == "int":
        format_func = lambda x: int(x)
    elif format == "float":
        format_func = lambda x: float(x)

    num_list = []
    with open(file_name, "r") as f:
        lines = f.readlines()
    for line in lines:
        num_list.append(
            format_func(line.strip())
        )
    return num_list