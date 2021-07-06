# Майстренко Олександр ДО-4, 2021
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Завантаження даних

def load_data():
    data = pd.read_csv('COVID-19-in-Ukraine-from-April.csv', usecols=['n_confirmed', 'n_deaths'])

    x = data['n_confirmed']
    y = data['n_deaths']

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x, y, test_size=0.33, random_state=42)

    return train_set_x, test_set_x, train_set_y, test_set_y

train_set_x, test_set_x, train_set_y, test_set_y = load_data()

m_train = train_set_x.shape
m_test = test_set_x.shape
print("Кількість тренувальних прикладів: m_train = " + str(m_train))
print("Кількість тестових прикладів: m_test = " + str(m_test))

# Тренувальний сет даних
xmax = max(max(train_set_x), max(test_set_x))
plt.scatter(train_set_x, train_set_y)
plt.xlim([-0.05*xmax, xmax*1.05])
plt.title("Тренувальний сет")
plt.xlabel("Нові зафіксовані випадки")
plt.ylabel("Нові смерті")
plt.show()
# Тестовий сет даних
plt.scatter(test_set_x, test_set_y)
plt.xlim([-0.05*xmax, xmax*1.05])
plt.title("Тестовий сет")
plt.xlabel("Нові зафіксовані випадки")
plt.ylabel("Нові смерті")
plt.show()

mean = np.concatenate([train_set_x, test_set_x]).mean()
std = np.concatenate([train_set_x, test_set_x]).std()

train_set_x = (train_set_x - mean) / std
test_set_x = (test_set_x - mean) / std

# Тренувальний сет даних (після стандартизації)
plt.scatter(train_set_x, train_set_y)
plt.title("Тренувальний сет (після стандартизації)")
plt.xlabel("Нові зафіксовані випадки")
plt.ylabel("Нові смерті")
plt.show()
# Тестовий сет даних (після стандартизації)
plt.scatter(test_set_x, test_set_y)
plt.title("Тестовий сет (після стандартизації)")
plt.xlabel("Нові зафіксовані випадки")
plt.ylabel("Нові смерті")
plt.show()


def initialize_with_zeros():
    """
    Ця функція ініціалізує theta та b як нулі.

    Повертає:
    theta -- ініціалізований скалярний параметр
    b -- ініціалізований скаляр (відповідає випадковій похибці)
    """

    theta = 0
    b = 0

    return theta, b

theta, b = initialize_with_zeros()


def propagate(theta, b, X, Y):
    """
    Імплементує функцію витрат та її градієнт для подальшого градієнтного спуску

    Аргументи:
    theta -- параметр, скаляр
    b -- випадкова похибка, скаляр
    X -- вектор значень незалежної змінної розміру (кількість прикладів, )
    Y -- значення залежної змінної (кількість прикладів, )

    Повертає:
    cost -- функція витрат для лінійної регресії
    dt -- градієнт по theta, тієї самої розмірності, що й theta
    db -- градієнт по b, тієї самої розмірності, що й b
    """

    m = X.shape[0]

    H = theta * X + b  # підставляємо поточні theta та b
    cost = np.dot(H - Y, H - Y) / (2 * m)  # рахуємо значення функції витрат

    dt = np.dot(X, H - Y) / m
    db = np.sum(H - Y) / m

    cost = np.squeeze(cost)

    grads = {"dt": dt,
             "db": db}

    return grads, cost


def optimize(theta, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    Ця функція оптимізує theta та b за допомогою алгоритму градієнтного спуску

    Аргументи:
    theta -- параметр, скаляр
    b -- випадкова похибка, скаляр
    X -- вектор значень незалежної змінної розміру (кількість прикладів, )
    Y -- значення залежної змінної (кількість прикладів, )
    num_iterations -- кількість ітерацій для оптимізуючого циклу
    learning_rate -- розмір кроку для оновлення градієнтного спуску
    print_cost -- встановлюється на True для того щоб надрукувати витрати кожні 100 ітерацій

    Повертає:
    params -- dictionary з ваговими коефіцієнтами theta та b
    grads -- dictionary з градієнтами вагових коефіцієнтів та випадкової похибки із урахуванням функції витрат
    costs -- list усіх значень функції витрат протягом оптимізації.

    """

    costs = []

    for i in range(num_iterations):

        # Обчислення функції витрат та градієнту
        grads, cost = propagate(theta, b, X, Y)

        # Отримуємо похідні з градієнтів
        dt = grads["dt"]
        db = grads["db"]

        # правило спуску
        theta -= learning_rate * dt
        b -= learning_rate * db

        # Записуємо витрати
        if i % 100 == 0:
            costs.append(cost)

        # Друкуємо витрати кожні 100 ітерацій
        if print_cost and i % 100 == 0:
            print("Витрати після ітерації %i: %f" % (i, cost))

    params = {"theta": theta,
              "b": b}

    grads = {"dt": dt,
             "db": db}

    return params, grads, costs


def predict(theta, b, X):
    """
    Прогнозує використовуючи отримані параметри лінійної регресії (theta, b)

    Аргументи:
    theta -- параметр, скаляр
    b -- випадкова похибка, скаляр
    X -- вектор значень незалежної змінної розміру (кількість прикладів, )

    Повертає:
    Y_prediction -- numpy array (вектор), що містить усі прогнози для прикладів в X
    """

    # Обчислюємо вектор "Y_prediction" прогнозуючи кількість нових смертей
    Y_prediction = theta * X + b

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Будує модель лінійної регресії, використовуючи функції, що були імплементовані  раніше

    Аргументи:
    X_train -- тренувальний сет представлений numpy array розміром (m_train, )
    Y_train -- тренувальні значення представлені numpy array (вектором) розміру (m_train, )
    X_test -- тестовий сет представлений numpy array розміром (m_test, )
    Y_test -- тестові значення представлені numpy array (вектором) розміру (m_test, )
    num_iterations -- параметр, що позначає кількість ітерацій для оптимізації параметрів
    learning_rate -- параметр, що позначає розмір кроку для правила спуску у optimize()
    print_cost -- встановлюється на True для того щоб надрукувати витрати кожні 100 ітерацій

    Повертає:
    d -- dictionary, що містить інформацію про модель
    """

    # ініціалізуємо параметри нулями
    theta, b = initialize_with_zeros()

    # Градієнтний спуск
    parameters, grads, costs = optimize(theta, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Отримуємо значення theta та b з dictionary
    theta = parameters["theta"]
    b = parameters["b"]

    # Прогнозуємо значення на тестовому та тренувальному сетах
    Y_prediction_test = predict(theta, b, X_test)
    Y_prediction_train = predict(theta, b, X_train)

    # Виводимо помилки тестового та тренувального сетів
    print("Тренувальний RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_train - Y_train) ** 2))))
    print("Тестовий RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_test - Y_test) ** 2))))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "theta": theta,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=300, learning_rate=0.05, print_cost=True)

# Тренувальний сет даних
plt.title("Тренувальний сет")
plt.scatter(train_set_x, train_set_y)
x = np.array([min(train_set_x), max(train_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y, color='orange')
plt.xlabel("Нові зафіксовані випадки")
plt.ylabel("Нові смерті")
plt.show()

# Тестовий сет даних
plt.title("Тестовий сет")
plt.scatter(test_set_x, test_set_y)
x = np.array([min(test_set_x), max(test_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y, color='orange')
plt.xlabel("Нові зафіксовані випадки")
plt.ylabel("Нові смерті")
plt.show()

