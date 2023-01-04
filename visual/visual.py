import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Creating dataset

    data_1 = [91.44, 91.18, 92.4]
    data_2 = [92.55, 92.31, 92.32]
    data = [data_1, data_2]

    fig = plt.figure(figsize=(10, 7))

    # Creating plot
    plt.boxplot(data)

    # show plot
    plt.savefig("./var.png",dpi=800)
    # plt.show()