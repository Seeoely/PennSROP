with open('/Users/colevogt/Downloads/PennSROP/s82drw/s82drw_r.dat', 'r') as file:
    data = file.readlines()
    with open('/Users/colevogt/Downloads/PennSROP/AGN.txt', 'a') as f:
        for x in range(120):
            for i in range(7):
                f.write(data[x+3][i+71])
            f.write('\n')
            for i in range(9):
                f.write(data[x+3][i+82])
            f.write('\n')
    f.close()
file.close()

def convert(s):
    new = ""
    for k in s:
        new += k
    num = float(new)
    return num


with open("/Users/colevogt/Downloads/PennSROP/AGN.txt", 'r') as file:
    data = file.readlines()
    for x in range(100):
        tauStr = [data[2*x][i] for i in range(7)]
        tau = convert(tauStr)
        ampStr = [data[2*x+1][j] for j in range(9)]
        amp = convert(ampStr)
file.close
