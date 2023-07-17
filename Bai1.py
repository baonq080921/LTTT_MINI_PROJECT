import  numpy as np
from scipy.stats import entropy

def input_matrix_probabilities_matrix():
    M = int(input("Nhập số hàng M: "))
    N = int(input("Nhập số cột N: "))

    probabilites = []

    for i in range(M):
        row = []
        for j in range(N):
            while True:
                probabilites_str = input(f"Nhập vào xác xuất P({i},{j}):")

                try:
                    probability = float(probabilites_str)
                    if probability >= 0:
                        break
                    else:
                        print("Xác suất không được am. Vui lòng nhập lại.")
                except ValueError:
                    #Kiểm tra nếu gía trị nhập vào là số thâo phâ:
                    numerator,denominator = map(int,probabilites_str.split('/'))
                    if numerator >= 0 and denominator > 0:
                        probability = numerator / denominator
                        break
                    else:
                        print("Xác suất không hơp lệ.Vui lòng nhập lại!")
            row.append(probability)
        print(row)
        probabilites.append(row)
    return  probabilites

#Hiển thị ma trận:
def print_probability_matrix(matrix):
    print("\nMa trận kết hợp P(x,y)")
    for row in matrix:
        print(' \t'.join(str(prob) for prob in row))

def calculate_conditional_entropy(probabilities_joint, probabilities_y):
    """
    Use to caculate Conditional Entropy
    """
    conditional_entropy = 0
    for i in range(len(probabilities_joint)):
        for j in range(len(probabilities_joint[i])):
            if probabilities_joint[i][j] > 0:
                conditional_prob = probabilities_joint[i][j] / probabilities_y[j]
                conditional_entropy -= probabilities_joint[i][j] * np.log2(conditional_prob)
    return conditional_entropy
def caculate_mutual_information(probabilities_joint,probabilities_x,probabilities_y):
    """
    Use to caculate Mutual_information:
    """
    mutual_information = 0
    for i in range(len(probabilities_joint)):
        for j in range(len(probabilities_joint[i])):
            if probabilities_joint[i][j] > 0:
                mutual_prob = probabilities_joint[i][j] / (probabilities_x[j]*probabilities_y[j])
                mutual_information += probabilities_joint[i][j] * np.log2(mutual_prob)
    return  mutual_information
def calculate_KL_divergence(probabilities_x,probabilities_y):
    return entropy(probabilities_x,probabilities_y,base=2)

def  main():
    probabilites_matrix = input_matrix_probabilities_matrix()
    print_probability_matrix(probabilites_matrix)

    joint_probabilities = np.array(probabilites_matrix)

    #Tính Xác suất biên(P(x), P(y)):
    marginal_probabilities_y = joint_probabilities.sum(axis=1)
    marginal_probabilities_x = joint_probabilities.sum(axis=0)

    #Tinh entropy:
    entropy_x = entropy(marginal_probabilities_x,base=2)
    entropy_y = entropy(marginal_probabilities_y,base=2)
    #Tinh mutual infomation :
    conditional_entropy_x_given_y = calculate_conditional_entropy(probabilites_matrix,marginal_probabilities_y)
    conditional_entropy_y_given_x = calculate_conditional_entropy(probabilites_matrix,marginal_probabilities_x)
    #Tinh JOINT ENTROPY:
    joint_entropy = entropy(joint_probabilities.flatten(),base=2)

    #Mututal information:
    mutual_information = caculate_mutual_information(probabilites_matrix,marginal_probabilities_x,marginal_probabilities_y)

    # Tính D(P(x)||P(y)) và D(P(y)||P(x))
    kl_divergence_xy = calculate_KL_divergence(marginal_probabilities_x, marginal_probabilities_y)
    kl_divergence_yx = calculate_KL_divergence(marginal_probabilities_y, marginal_probabilities_x)

    #Hien thi ket qua:
    print("\nHien thi ket qua:")
    print(probabilites_matrix)
    print(f"H(x) = {round(entropy_x,3)}(bits)")
    print(f"H(y)= {round(entropy_y,3)}(bits)")
    print(f"H(X|Y)= {round(conditional_entropy_x_given_y,3)}(bits)")
    print(f"H(Y|X)= {round(conditional_entropy_y_given_x,3)}(bits)")
    print(f"H(X,Y)= {round(joint_entropy,3)}(bits)")
    print(f"H(Y)- H(Y|X)= {round(entropy_y - round(conditional_entropy_y_given_x),3)}(bits)")
    print(f"I(X;Y)={mutual_information}(bits)")
    print(f"D(P(X)||P(Y))= {round(kl_divergence_xy,3)}(bits)")
    print(f"D(P(Y)||P(X))= {round(kl_divergence_yx,3)}(bits)")
if __name__ == '__main__':
    main()




