import math
import re
import time
from typing import Callable

import numpy as np
import sympy as sp
from memory_profiler import memory_usage
from prettytable import PrettyTable
from sympy import SympifyError, Pow, Symbol


def measure_function_execution(
        method_function: Callable[[Pow, float, float, float, Symbol], tuple[float, float, int]],
        *args: [Pow, float, float, float, Symbol]) -> tuple[float, float, tuple[float, float, int]]:
    """
    Функция, которая производит запуск передаваемой ей функции и
    замеряет время ее выполнения, а так же расходуемую память.

    :param method_function: Функция для которой должны производиться замеры.
    :param args: Аргументы которые должна принимать функция ``method_function``
    :return: Время работы функции, максимальный расход RAM, результат работы функции
    """

    start_time = time.time()
    mem_usage, result = memory_usage((method_function, args), retval=True, max_usage=True)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time, mem_usage, result


def fibonacci_iterative(n: int) -> int:
    """
    Функция для поиска N-го числа Фибоначчи.
    :param n: Номер искомого числа Фибоначчи.
    :return: N-ое число Фибоначчи.
    """
    if n <= 0:
        print("Ошибка: n должно быть положительным числом")
        return -1
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n):
            a, b = b, a + b
        return b


def fibonacci_method(function: Pow, start: float, end: float, epsilon: float, variable: float) -> tuple[float, float, int]:
    """
    Функция для нахождения минимума функции на заданном отрезке с использованием метода Фибоначчи.

    :param function: Функция, минимум которой нужно найти.
    :param start: Начало отрезка.
    :param end: Конец отрезка.
    :param epsilon: Требуемая точность поиска.
    :param variable: Строка хранящая переменную от которой зависит функция
    :return: Точка минимума и значение минимума.
    """
    count_step = 13
    x1 = start + fibonacci_iterative(count_step) / fibonacci_iterative(count_step + 2) * (end - start)
    x2 = start + fibonacci_iterative(count_step + 1) / fibonacci_iterative(count_step + 2) * (end - start)
    for _ in range(count_step):
        value_x1 = function.subs(variable, x1).evalf(9)
        value_x2 = function.subs(variable, x2).evalf(9)
        if abs(value_x1 - value_x2) < 10 ** -8:
            end = x2
            x2 = x1
            x1 = start + end - x1
        else:
            start = x1
            x1 = x2
            x2 = start + end - x1
    return (start + end) / 2, function.subs(variable, (start + end) / 2), count_step


def golden_section(function: Pow, start: float, end: float, epsilon: float, variable: float) -> tuple[float, float, int]:
    """
    Функция для нахождения минимума функции на заданном отрезке с использованием метода золотого сечения.

    :param function: Функция, минимум которой нужно найти.
    :param start: Начало отрезка.
    :param end: Конец отрезка.
    :param epsilon: Требуемая точность поиска.
    :param variable: Строка хранящая переменную от которой зависит функция
    :return: Точка минимума и значение минимума.
    """
    x1 = start + ((3 - 5 ** 0.5) / 2) * (end - start)
    x2 = start + ((5 ** 0.5 - 1) / 2) * (end - start)
    number_of_iterations = 0
    while abs(end - start) > epsilon:
        number_of_iterations += 1
        value_x1 = function.subs(variable, x1).evalf(9)
        value_x2 = function.subs(variable, x2).evalf(9)
        if abs(value_x1 - value_x2) < 10 ** -8:
            end = x2
            x2 = x1
            x1 = start + end - x1
        else:
            start = x1
            x1 = x2
            x2 = start + end - x1
    return (start + end) / 2, function.subs(variable, (start + end) / 2), number_of_iterations


def binary_search(function: Pow, start: float, end: float, epsilon: float, variable: float) -> tuple[float, float, int]:
    """
    Функция для нахождения минимума функции на заданном отрезке с использованием бинарного поиска.

    :param function: Функция, минимум которой нужно найти.
    :param start: Начало отрезка.
    :param end: Конец отрезка.
    :param epsilon: Требуемая точность поиска.
    :param variable: Строка хранящая переменную от которой зависит функция
    :return: Точка минимума и значение минимума.
    """
    delta = epsilon / 10
    number_of_iterations = 0
    while (end - start) / 2 > epsilon:
        number_of_iterations += 1
        mid_left = (start + end - delta) / 2
        mid_right = (start + end + delta) / 2
        mid_value_left = function.subs(variable, mid_left)
        mid_value_right = function.subs(variable, mid_right)

        if mid_value_left > mid_value_right:
            start = mid_left
        elif mid_value_left < mid_value_right:
            end = mid_right
        else:
            start = mid_left
            end = mid_right
    return (start + end) / 2, function.subs(variable, (start + end) / 2), number_of_iterations


def bitwise_method(function: Pow, start: float, end: float, epsilon: float, variable: float) -> tuple[float, float, int]:
    """
    Функция для нахождения минимума функции на заданном отрезке с использованием поразрядного поиска.

    :param function: Функция, минимум которой нужно найти.
    :param start: Начало отрезка.
    :param end: Конец отрезка.
    :param epsilon: Требуемая точность поиска.
    :param variable: Строка хранящая переменную от которой зависит функция
    :return: Точка минимума и значение минимума.
    """
    step = 0.5
    min_x = current_x = start
    number_of_iterations = 1
    min_val = past_val = function.subs(variable, start)
    while step > epsilon:
        number_of_iterations += 1
        for x in np.arange(start, end + step, step):
            current_val = function.subs(variable, x)
            current_x = x
            if current_val < min_val:
                min_val = current_val
                min_x = x
            if current_val > past_val:
                past_val = current_val
                break
            past_val = current_val
        start = current_x
        end = min_x - step
        step /= -4
    return min_x, min_val, number_of_iterations


def brute_force_method(function: Pow, start: float, end: float, epsilon: float, variable: float) -> tuple[float, float, int]:
    """
        Функция для нахождения минимума функции на заданном отрезке с использованием метода перебора.

        :param function: Функция, минимум которой нужно найти.
        :param start: Начало отрезка.
        :param end: Конец отрезка.
        :param epsilon: Требуемая точность поиска.
        :param variable: Строка хранящая переменную от которой зависит функция
        :return: Точка минимума и значение минимума.
        """

    min_x = start
    number_of_iterations = math.ceil((end - start) / epsilon)
    min_val = past_val = function.subs(variable, start)
    step = (end - start) / number_of_iterations
    for val in np.arange(start + step, end + step, step):
        current_val = function.subs(variable, val)
        if current_val < min_val:
            min_val = current_val
            min_x = val
        if current_val > past_val:
            break
        past_val = current_val
    return min_x, min_val, number_of_iterations


def initializing_function_parameters() -> tuple[Pow, float, float, float, Symbol] | None:
    """
    Функция для инициализации всех необходимых стартовых параметров.

    :returns:
        Функция, которую нужно оптимизировать.
        Левая граница интервала.
        Правая граница интервала.
        Точность.
        Переменная, от которой зависит функция.
    """

    function_str = input("Введите функцию (например, x**2): ").lower()
    checking, symbol = checking_the_number_of_variables(function_str)
    if not checking:
        print(f"Программа работает с функциями от 1 переменной\nВы использовали {symbol}")
    try:
        variable = sp.symbols(*symbol)
        func = sp.sympify(function_str)
    except (SympifyError, ValueError, SyntaxError, TypeError):
        print("Произошла ошибка при обработке символов или строк")
        return

    checking, start, end, epsilon = checking_search_interval()
    if not checking:
        return
    return func, start, end, epsilon, variable


def checking_search_interval() -> [bool, float, float, float]:
    """
    Проверяет корректность введенных пользователем параметров для поиска минимума на заданном интервале.

    Вводит начало и конец интервала, а также точность вычисления минимума. Проверяет, что введенные значения
    соответствуют требованиям: начало интервала меньше или равно концу, точность находится в диапазоне от 0 до 1.
    Если введенные данные некорректны, функция выводит соответствующие сообщения и возвращает флаг, указывающий на
    ошибку, а также некорректные значения.

    :returns:
        - flag (bool): Флаг, указывающий на корректность введенных данных. True, если данные корректны, иначе False.
        - start (float): Начало интервала, если данные корректны.
        - end (float): Конец интервала, если данные корректны.
        - epsilon (float): Точность вычисления минимума, если данные корректны.

    Пример использования:
        flag, start, end, epsilon = checking_search_interval()

        if flag:
            print(f"Начало интервала: {start}, Конец интервала: {end}, Точность: {epsilon}")
        else:
            print("Введенные данные некорректны.")
    """
    flag = True
    start, end, epsilon = [None] * 3
    try:
        start = float(input("Введите начало отрезка: "))
        end = float(input("Введите конец отрезка: "))
        epsilon = float(input("Введите точность: "))
    except ValueError:
        flag = False
    if flag and (epsilon <= 0 or epsilon >= 1):
        print("Точность вычисления минимума должна соответствовать условиям \"0 < epsilon < 1\"")
        flag = False
    if flag and start > end:
        print("Введён не корректный интервал для поиска")
        flag = False
    return flag, start, end, epsilon


def removing_math_operations(input_str: str) -> str:
    # Полный список всех математических операций для исключения
    math_operations = [
        'sin', 'cos', 'tan', 'cot',
        'asin', 'acos', 'atan', 'acot',
        'sinh', 'cosh', 'tanh', 'coth',
        'asinh', 'acosh', 'atanh', 'acoth',
        'ln', 'log', 'log10',
        'exp', 'sqrt', 'factorial',
        'binomial', 'gamma', 'beta',
        'elliptic_f', 'elliptic_e', 'elliptic_k',
        'elliptic_pi'
    ]

    # Создаём регулярное выражение для поиска математических операций
    pattern = '|'.join(map(re.escape, math_operations))

    # Используем регулярное выражение для удаления математических операций из строки
    cleaned_str = re.sub(pattern, '', input_str)

    return cleaned_str


def checking_the_number_of_variables(_function: str) -> tuple[bool, str]:
    """
    Проверяет, содержит ли переданная строка только одну уникальную букву.

    Функция принимает строку, представляющую собой имя функции, и проверяет, содержит ли эта строка только одну
    уникальную букву. Это может быть полезно для проверки, что имя функции соответствует определенному формату или
    соглашению.

    :param _function: Введенная пользователем строка
    :returns:
        - bool: True, если строка содержит только одну уникальную букву, иначе False.
        - set[str]: Множество уникальных буквенных символов, найденных в строке.

    Пример использования:
        valid, letters = checking_the_number_of_variables("f(x)")
        if valid:
            print("Строка содержит только одну уникальную букву.")
        else:
            print("Строка содержит более одной уникальной буквы.")
    """
    _function = removing_math_operations(_function)
    letters = re.findall(r'[a-z]', _function)
    unique_letters_count = len(set(letters))
    return unique_letters_count <= 1, set(letters)


def run():
    """
        Выполняет различные методы оптимизации и собирает результаты в таблицу.

        Функция инициализирует параметры для каждого метода оптимизации, затем выполняет каждый метод с этими
        параметрами. Результаты выполнения каждого метода (время выполнения, использование памяти, результат функции)
        собираются в таблицу `PrettyTable` для удобства представления.

        В случае возникновения ошибки синтаксиса во время выполнения метода, функция выводит сообщение об ошибке и
        завершает работу.

        Пример использования:

        Пример 1:

            >>> 5*x**2 - 8*x**(5/4) - 20*x
            >>> 3
            >>> 3.5
            >>> 0.025
            +---------------------+----------+----------+---------+-----------+
            |        метод        | время, с | RAM, Mb  |    x    |    f(x)   |
            +---------------------+----------+----------+---------+-----------+
            | Поразрядного поиска | 1.64383  | 72.03125 |   3.5   | -47.04791 |
            |      Дихотомии      | 1.58068  | 72.21094 | 3.35883 | -47.14475 |
            |   Золотого сечения  | 1.56883  | 72.31641 | 3.49139 | -47.05896 |
            |      Фибоначчи      | 1.55134  | 72.4375  | 3.49801 | -47.05052 |
            +---------------------+----------+----------+---------+-----------+


        Пример 2:

            >>> 5*x**2 - 8*x**(5/4) - 20*x**4
            >>> 3
            >>> 10
            >>> 0.00025
            +---------------------+----------+----------+---------+---------------+
            |        метод        | время, с | RAM, Mb  |    x    |      f(x)     |
            +---------------------+----------+----------+---------+---------------+
            | Поразрядного поиска | 1.57118  | 72.15234 |   10.0  | -199642.26235 |
            |      Дихотомии      | 1.62242  | 72.47266 | 9.99977 | -199624.19177 |
            |   Золотого сечения  | 1.53975  | 72.71484 | 9.99991 | -199635.19954 |
            |      Фибоначчи      | 1.56135  | 72.85938 | 9.97215 | -197425.72298 |
            +---------------------+----------+----------+---------+---------------+

        Пример 3:

            >>> 5*x**2 - 8*x**(5/4) - 20*x**4 + 12*x
            >>> 0
            >>> 10000
            >>> 0.0000025

            ошибка синтаксиса
        """
    parameters = initializing_function_parameters()
    methods = {
        "Перебора": brute_force_method,
        "Поразрядного поиска": bitwise_method,
        "Дихотомии": binary_search,
        "Золотого сечения": golden_section,
        "Фибоначчи": fibonacci_method
    }
    table = PrettyTable(["метод", "время, с", "RAM, Mb", "кол-во итераций", "x", "f(x)"])

    for method_name, method in methods.items():
        try:
            time_spent, ram_usage, result = measure_function_execution(method, *parameters)
            table.add_row(
                [
                    method_name,
                    round(time_spent, 5),
                    round(ram_usage, 5),
                    result[2],
                    round(result[0], 5),
                    round(result[1], 5)
                ]
            )
        except TypeError:
            print("ошибка синтаксиса")
            return

    print(table)


if __name__ == '__main__':
    run()


    # 5*x**2 - 8*x**(5/4) - 20*x
    # 3
    # 3.5
    # 0.025
    #
    # 5*x**2 - 8*x**(5/4) - 20*x**4
    # 3
    # 10
    # 0.00025
    #
    # 5*x**2 - 8*x**(5/4) - 20*x**4 + 12*x
    # 0
    # 10000
    # 0.0000025
