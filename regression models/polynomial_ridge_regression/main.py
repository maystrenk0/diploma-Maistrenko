# Майстренко Олександр ДО-4, 2021
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Завантаження даних

def load_data():
    df = pd.read_csv('hospitalizations_number_06_12.csv', sep=';')
    data = np.zeros((int(df.size/2.), 2))
    for index, row in df.iterrows():
        d = datetime.strptime(row['date'], '%d.%m.%Y')
        data[index, 0] = (d.month * 30.44 + d.day - 218.08) / 121.76
        data[index, 1] = row['hospitalizations']
    x = data[:, 0]
    x = x.reshape((x.shape[0], 1))
    y = data[:, 1]

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x, y, test_size=0.33, random_state=42)

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x.T, test_set_x.T, train_set_y, test_set_y, x.T


train_set_x, test_set_x, train_set_y, test_set_y, full_feature_set_for_plot = load_data()


m_train = len(train_set_x[0])
m_test = len(test_set_x[0])
print("Кількість тренувальних прикладів: m_train = " + str(m_train))
print("Кількість тестових прикладів: m_test = " + str(m_test))

# Палітра кольорів
cmap = plt.get_cmap('viridis')

# Візуіалізація даних
m1 = plt.scatter(121.76 * train_set_x, train_set_y, color=cmap(0.9), s=10)
m2 = plt.scatter(121.76 * test_set_x, test_set_y, color=cmap(0.5), s=10)
plt.xlabel('День')
plt.ylabel('Госпіталізації')
plt.legend((m1, m2), ("Тренувальні дані", "Тестові дані"), loc='lower right')
plt.show()


def polynomial_features(X, degree):
    from itertools import combinations_with_replacement
    # комбінації_з_повторами('ABC', 2) --> AA AB AC BB BC CC

    n_features, n_samples = np.shape(X)

    def index_combinations():  # (1, 2) => [(1),(2),(1,1),(1,2),(2,2)]
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        # comb = [(),((1),(2)),((1,1),(1,2),(2,2))]
        flat_combs = [item for sublist in combs for item in sublist]
        # flat_combs = [(1),(2),(1,1),(1,2),(2,2)]
        return flat_combs

    combinations = index_combinations()

    n_output_features = len(combinations)

    X_new = np.empty((n_output_features, n_samples))

    for i, index_combs in enumerate(combinations):
        X_new[i, :] = np.prod(X[index_combs, :], axis=0)
        # if index_combs == (1,2,3) =>  X_new[:,i] = X[:,1] * X[:,2] * X[:,3]
    return X_new


def mean_squared_error(y_true, y_pred):
    """Повертає середньоквадратичну помилку між y_true та y_pred

    Аргументи:
    y_true -- масив справжніх значень
    y_pred -- масив спрогнозованих значень

    Повертає:
    mse -- середньоквадратична помилка
    """
    mse = (1 / len(y_true.T)) * np.sum((y_true - y_pred) ** 2)

    return mse


class L2Regularization:
    """ Регуляризація для рідж-регресії """

    def __init__(self, alpha):
        """ Встановлює alpha """
        self.alpha = alpha

    def __call__(self, w):
        """
        Обчислює штраф l2 регуляризації

        Аргументи:
        w -- вагові коефіцієнти

        Повертає:
        term -- 1/2 * alpha * norm(w)^2
        """
        term = 1 / 2 * self.alpha * np.linalg.norm(w) ** 2
        return term

    def grad(self, w):
        """
        Обчислює похідну штрафа l2 регуляризації

        Аргументи:
        w -- вагові коефіцієнти

        Повертає:
        vector -- alpha * w
        """
        derivative = self.alpha * w

        return derivative


class PolynomialRidgeRegression(object):
    """
    Параметри:
    -----------
    degree: int
        Степінь полінома, на який буде перетворено незалежну змінну X
    reg_factor: float
        Коефіцієнт який визначає кількість регуляризації та звуження незалежних змінних
    n_iterations: int
        Кількість тренувальних ітерацій алгоритму
    learning_rate: float
        Розмір кроку, який буде застосований при оновленні вагових коефіцієнтів
    """

    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, print_error=False):
        self.degree = degree
        self.regularization = L2Regularization(alpha=reg_factor)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.print_error = print_error

    def initialize_with_zeros(self, n_features):
        """
        Ця функція створює вектор нулів формою (n_features, 1)

        Аргументи:
        n_features -- кількість незалежних змінних
        """
        self.w = np.zeros((n_features, 1))

    def fit(self, X, Y):
        # Генеруємо змінні полінома
        X = polynomial_features(X, self.degree)

        # Вставляємо одиниці для вагових коефіцієнтів випадкової похибки
        X = np.concatenate((np.ones((1, len(X[0]))), X), axis=0)

        # Створюємо масив
        self.initialize_with_zeros(n_features=X.shape[0])

        # Виконуємо градієнтний спуск для n_iterations ітерацій
        for i in range(self.n_iterations):
            # Прогнозуємо дані
            H = self.w.T.dot(X)

            # Градієнт штрафу l2
            grad_w = np.dot(X, (H - Y).T) + self.regularization.grad(self.w)

            # Оновлюємо вагові коефіцієнти
            self.w = self.w - self.learning_rate * grad_w

            if self.print_error and i % 1000 == 0:
                # Обчислюємо l2 штраф
                mse = mean_squared_error(Y, H)
                print("MSE після ітерації %i: %f" % (i, mse))

    def predict(self, X):
        # Генеруємо поля полінома
        X = polynomial_features(X, self.degree)

        # Вставляємо одиниці для вагових коефіцієнтів випадкової похибки
        X = np.concatenate((np.ones((1, len(X[0]))), X), axis=0)

        # Прогнозуємо дані
        y_pred = self.w.T.dot(X)

        return y_pred


poly_degree = 15
learning_rate = 0.001
n_iterations = 40000
reg_factor = 0.1
model = PolynomialRidgeRegression(
    degree=poly_degree,
    reg_factor=reg_factor,
    learning_rate=learning_rate,
    n_iterations=n_iterations,
    print_error=True
)
model.fit(train_set_x, train_set_y)
y_predictions = model.predict(test_set_x)
mse = mean_squared_error(test_set_y, y_predictions)
print("Середньоквадратична помилка на тестовому сеті: %s (фактор регуляризації: %s)" % (mse, reg_factor))

# Прогнозуємо для усіх точок у наборі даних
y_val = model.predict(full_feature_set_for_plot)

# Візуалізуємо результати
m1 = plt.scatter(121.76 * train_set_x, train_set_y, color=cmap(0.9), s=10)
m2 = plt.scatter(121.76 * test_set_x, test_set_y, color=cmap(0.5), s=10)
plt.plot(121.76 * full_feature_set_for_plot.T, y_val.T, color='black', linewidth=2, label="Прогноз")
plt.suptitle("Поліноміальна рідж-регресія")
plt.title("MSE: %.2f" % mse, fontsize=10)
plt.xlabel('День')
plt.ylabel('Госпіталізації')
plt.legend((m1, m2), ("Тренувальні дані", "Тестові дані"), loc='lower right')
plt.show()
