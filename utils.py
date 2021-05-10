import os
import csv


def write_to_csv(scores, params, outputfile):
    """This function writes the parameters and the scores with their names in a
    csv file."""
    # creates the file if not existing.
    file = open(outputfile, 'a')
    # If file is empty writes the keys to the file.
    params_dict = vars(params)
    if os.stat(outputfile).st_size == 0:
        # Writes the configuration parameters
        for key in params_dict.keys():
            file.write(key+";")
        for i, key in enumerate(scores.keys()):
            ending = ";" if i < len(scores.keys())-1 else ""
            file.write(key+ending)
        file.write("\n")
    file.close()

    # Writes the values to each corresponding column.
    with open(outputfile, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)

    # Iterates over the header names and write the corresponding values.
    with open(outputfile, 'a') as f:
        for i, key in enumerate(headers):
            ending = ";" if i < len(headers)-1 else ""
            if key in params_dict:
                f.write(str(params_dict[key])+ending)
            elif key in scores:
                f.write(str(scores[key])+ending)
            else:
                raise AssertionError("Key not found in the given dictionary")
        f.write("\n")

