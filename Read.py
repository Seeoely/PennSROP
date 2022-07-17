def convert(s):
    new = ""
    for x in s:
        new += x
    num = float(new)
    return num


for x in range(120):
    with open("/Users/colevogt/Downloads/PennSROP/AGN.txt", 'r') as file:
        data = file.readlines()
        tauStr = [data[2 * x][i] for i in range(7)]
        TempTau = convert(tauStr)
        if (TempTau < 4):
            tau = TempTau
            ampStr = [data[2 * x + 1][j] for j in range(9)]
            amp = convert(ampStr)
            print(tau, amp)
    file.close