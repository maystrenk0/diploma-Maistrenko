# Майстренко Олександр ДО-4, 2021
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Завантаження даних

def load_data():
    data = pd.read_csv('COVID-19-in-Ukraine-from-April.csv', usecols=['n_confirmed', 'n_deaths', 'n_recovered'])

    x = data[['n_confirmed', 'n_recovered']].to_numpy()
    y = data['n_deaths'].to_numpy()

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x, y, test_size=0.33, random_state=42)
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x.T, train_set_y, test_set_x.T, test_set_y, data


train_set_x, train_set_y, test_set_x, test_set_y, visualization_set = load_data()

m_train = train_set_x.shape[1]
m_test = test_set_x.shape[1]
print("Кількість тренувальних прикладів: m_train = " + str(m_train))
print("Кількість тестових прикладів: m_test = " + str(m_test))


plt.hist(visualization_set['n_deaths'])
plt.xlabel("Нові смерті")
plt.ylabel("Кількість")
plt.tight_layout()
plt.show()

plt.scatter(visualization_set['n_confirmed'], visualization_set['n_deaths'])
plt.xlabel("Нові зафіксовані випадки")
plt.ylabel("Нові смерті")
plt.show()

plt.scatter(visualization_set['n_recovered'], visualization_set['n_deaths'])
plt.xlabel("Нові одужання")
plt.ylabel("Нові смерті")
plt.show()

all_set_x = np.concatenate([train_set_x, test_set_x], axis=1)

mean = all_set_x.mean(axis=1, keepdims=True)
std = all_set_x.std(axis=1, keepdims=True)

train_set_x = (train_set_x - mean) / std
test_set_x = (test_set_x - mean) / std


def initialize_with_zeros(dim):
    """
    Ця функція створює вектор нулів розмірністю (dim, 1) для w ті ініціалізує b як 0.

    Аргумент:
    dim -- розмірність вектора w (або кількість незалежних змінних у цьому випадку)

    Повертає:
    w -- ініціалізований вектор розмірністю (dim, 1)
    b -- ініціалізований скаляр (відповідає випадковій похибці)
    """

    w = np.zeros([dim, 1])
    b = 0

    return w, b


def propagate(w, b, X, Y):
    """
    Імплементує функцію витрат та її градієнт для подальшого градієнтного спуску

    Arguments:
    w -- вагові коефіцієнти, numpy array розміру (кількість полів, 1)
    b -- випадкова похибка, скаляр
    X -- матриця значень незалежних змінних (кількість полів, кількість прикладів)
    Y -- значення залежної змінної (1, кількість прикладів)

    Повертає:
    cost -- функція витрат для лінійної регресії
    dw -- градієнт по w, тієї самої розмірності, що й w
    db -- градієнт по b, тієї самої розмірності, що й b
    """

    m = X.shape[1]

    H = np.dot(w.T, X) + b  # підставляємо поточні w та b
    cost = np.dot((H - Y)[0], (H - Y)[0]) / (2 * m)  # рахуємо значення функції витрат

    dw = np.dot(X, (H - Y).T) / m
    db = np.sum(H - Y) / m

    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    Ця функція оптимізує w та b за допомогою алгоритму градієнтного спуску

    Arguments:
    w -- вагові коефіцієнти, numpy array розміру (кількість полів, 1)
    b -- випадкова похибка, скаляр
    X -- матриця значень незалежних змінних розміру (кількість полів, кількість прикладів)
    Y -- значення залежної змінної (1, кількість прикладів)
    num_iterations -- кількість ітерацій для оптимізуючого циклу
    learning_rate -- розмір кроку для оновлення градієнтного спуску
    print_cost -- встановлюється на True для того щоб надрукувати витрати кожні 100 ітерацій

    Повертає:
    params -- dictionary з ваговими коефіцієнтами theta та b
    grads -- dictionary з градієнтами вагових коефіцієнтів та випадкової похибки із урахуванням функції витрат
    costs -- list усіх значень функції витрат протягом оптимізації, це знадобидться для того,
    щоб побудувати криву навчання.
    """

    costs = []

    for i in range(num_iterations):

        # Обчислення функції витрат та градієнту
        grads, cost = propagate(w, b, X, Y)

        # Отримуємо похідні з градієнтів
        dw = grads["dw"]
        db = grads["db"]

        # правило спуску
        w -= learning_rate * dw
        b -= learning_rate * db

        # Записуємо витрати
        if i % 100 == 0:
            costs.append(cost)

        # Друкуємо витрати кожні 100 ітерацій
        if print_cost and i % 100 == 0:
            print("Витрати після ітерації %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    Прогнозує використовуючи отримані параметри лінійної регресії (w, b)

    Аргументи:
    w -- вагові коефіцієнти, numpy array розміру (кількість прикладів, 1)
    b -- випадкова похибка, скаляр
    X -- дані розміру (кількість полів, кількість прикладів)

    Повертає:
    H -- numpy array (вектор), що містить усі прогнози для прикладів в X
    """

    m = X.shape[1]

    # Обчислюємо вектор "H"
    H = np.dot(w.T, X) + b

    return H


def model(X_train, Y_train, X_test, Y_test, num_iterations=3000, learning_rate=0.5, print_cost=False):
    """
    Будує модель лінійної регресії, використовуючи функції, що були імплементовані  раніше

    Аргументи:
    X_train -- тренувальний сет представлений numpy array розміром (кількість полів, m_train)
    Y_train -- тренувальні значення представлені numpy array (вектором) розміру (1, m_train)
    X_test -- тестовий сет представлений numpy array розміром (кількість полів, m_test)
    Y_test -- тестові значення представлені numpy array (вектором) розміру (1, m_test)
    num_iterations -- параметр, що позначає кількість ітерацій для оптимізації параметрів
    learning_rate -- параметр, що позначає розмір кроку для правила спуску у optimize()
    print_cost -- встановлюється на True для того щоб надрукувати витрати кожні 100 ітерацій

    Повертає:
    d -- dictionary, що містить інформацію про модель
    """

    # ініціалізуємо параметри нулями
    w, b = initialize_with_zeros(X_train.shape[0])

    # Градієнтний спуск
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Отримуємо значення w та b з dictionary
    w = parameters["w"]
    b = parameters["b"]

    # Прогнозуємо значення на тестовому та тренувальному сетах
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Виводимо помилки тестового та тренувального сетів
    print("Тренувальний RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_train - Y_train) ** 2))))
    print("Тестовий RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_test - Y_test) ** 2))))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.05, print_cost=True)

# Тренувальний сет даних
plt.title("Тренувальний сет")
plt.scatter(train_set_y, d["Y_prediction_train"])
plt.plot([0, 100], [0, 100], "--k")
plt.xlabel("Реальні нові смерті")
plt.ylabel("Спрогнозовані нові смерті")
plt.show()

# Тестовий сет даних
plt.title("Тестовий сет")
plt.scatter(test_set_y, d["Y_prediction_test"])
plt.plot([0, 100], [0, 100], "--k")
plt.xlabel("Реальні нові смерті")
plt.ylabel("Спрогнозовані нові смерті")
plt.show()
