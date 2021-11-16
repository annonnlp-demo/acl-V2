def test():
    line_list=[]
    source_object = open('./1000 basic words', 'r')
    for line in source_object:
        line = line.rstrip()
        line_list = line.split(' ')
        break

    result_list = [value[:-1] for value in line_list]
    return result_list

