import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Безопасное пространство имён для eval
_safe_namespace = {}
import numpy as _np, math as _math

for name in dir(_np):
    if not name.startswith("_"):
        _safe_namespace[name] = getattr(_np, name)
for name in dir(_math):
    if not name.startswith("_"):
        if name not in _safe_namespace:
            _safe_namespace[name] = getattr(_math, name)
_safe_namespace['pi'] = math.pi
_safe_namespace['np'] = _np
_safe_namespace['math'] = _math



def make_function_from_string(s: str):
    """
    Converts string, e.g. 'f(x)=x+sin(pi*x)' or 'x+sin(pi*x)',
    to function f(x).
    """
    s = s.strip()
    if s.startswith('f(') or '=' in s:
        if '=' in s:
            s = s.split('=', 1)[1].strip()
    s = s.replace('^', '**')
    code = compile(s, "<string>", "eval")

    def f(x):
        local_ns = {'x': x}
        return eval(code, _safe_namespace, local_ns)

    return f, s


def estimate_L(f, a, b, n=1000):
    """
    Lipschitz constant estimation
    """
    xs = np.linspace(a, b, n)
    ys = np.asarray(f(xs), dtype=float)
    dy = np.abs(np.diff(ys))
    dx = (b - a) / (n - 1)
    deriv_est = dy / dx
    L = deriv_est.max()
    return max(L, 1e-8) * 1.2



def piyavskii_shubert(f, a, b, eps=0.01, L=None, max_iters=10000):
    """
    Piyavskii–Shubert algorithm
    """
    t0 = time.time()
    xs = [a, b]
    ys = [float(f(a)), float(f(b))]
    fmin = min(ys)
    xmin = xs[np.argmin(ys)]
    it = 0
    if L is None:
        L = estimate_L(f, a, b, n=1000)

    def lower_envelope_value(x):
        vals = [yi - L * abs(x - xi) for xi, yi in zip(xs, ys)]
        return max(vals)

    history = []

    while it < max_iters:
        it += 1
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        candidates = []
        for i in range(len(xs) - 1):
            xi, xj = xs[i], xs[i + 1]
            fi, fj = ys[i], ys[i + 1]
            x_star = 0.5 * (xi + xj) - (fi - fj) / (2 * L)
            x_star = min(max(x_star, xi), xj)
            m_val = lower_envelope_value(x_star)
            candidates.append((m_val, x_star, i))
        lower_bound = max([c[0] for c in candidates]) if candidates else -np.inf
        gap = fmin - lower_bound
        history.append((it, fmin, lower_bound, gap))
        if gap <= eps:
            break
        m_val, x_new, idx = max(candidates, key=lambda t: t[0])
        y_new = float(f(x_new))
        xs.append(float(x_new))
        ys.append(float(y_new))
        if y_new < fmin:
            fmin = y_new
            xmin = x_new

    t_elapsed = time.time() - t0
    hist_df = pd.DataFrame(history, columns=["iter", "f_upper", "f_lower", "gap"])

    return {
        "xmin": xmin,
        "fmin": fmin,
        "iterations": it,
        "time": t_elapsed,
        "xs": np.array(xs),
        "ys": np.array(ys),
        "L": L,
        "history": hist_df,
    }


def plot_results(f, a, b, result, func_str, fname_png="result_plot.png"):
    """
    Making visualizations
    """
    xs_dense = np.linspace(a, b, 2000)
    ys_dense = np.asarray(f(xs_dense), dtype=float)
    xi_samples = result["xs"]
    yi_samples = result["ys"]
    L = result["L"]
    envelope = np.max(
        [yi_samples[i] - L * np.abs(xs_dense - xi_samples[i]) for i in range(len(xi_samples))],
        axis=0,
    )
    plt.figure(figsize=(10, 6))
    plt.plot(xs_dense, ys_dense, label="f(x)")
    plt.plot(xs_dense, envelope, label="нижняя огибающая", linestyle="--")
    plt.scatter(xi_samples, yi_samples, color="orange", label="испытанные точки")
    plt.scatter([result["xmin"]], [result["fmin"]], color="red", marker="x", s=80, label="найденный минимум")
    plt.title(
        "Метод Пиявского–Шуберта\n"
        + func_str
        + f"\nНайдено x≈{result['xmin']:.6g}, f≈{result['fmin']:.6g}, итераций={result['iterations']}, время={result['time']:.3f}s"
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname_png)
    plt.close()


def make_txt_report(f, func_str, a, b, eps, result, txt_file="result.txt"):
    """
    .txt report generation
    """
    lines = []
    lines.append("ОТЧЁТ: Глобальный поиск минимума (метод Пиявского–Шуберта)\n")
    lines.append(f"Функция: {func_str}")
    lines.append(f"Отрезок: [{a}, {b}]")
    lines.append(f"Точность eps: {eps}")
    lines.append(f"Оценка L (Липшиц): {result['L']:.6g}\n")
    lines.append("РЕЗУЛЬТАТЫ:")
    lines.append(f"  Приближённый аргумент минимума x ≈ {result['xmin']:.8g}")
    lines.append(f"  Приближённое значение f(x) ≈ {result['fmin']:.8g}")
    lines.append(f"  Число итераций: {result['iterations']}")
    lines.append(f"  Время вычислений (s): {result['time']:.6f}\n")

    # iterations table
    lines.append("История итераций (iter, f_upper, f_lower, gap):")
    for row in result["history"].itertuples(index=False):
        lines.append(f"  {row.iter:4d} | {row.f_upper:10.6f} | {row.f_lower:10.6f} | {row.gap:10.6f}")

    text = "\n".join(lines)
    with open(txt_file, "w", encoding="utf-8") as f_out:
        f_out.write(text)


if __name__ == "__main__":
    func_input = "10 + x**2 - 10*cos(2*pi*x)"  # Функция Растригина (1D)
    f, func_str = make_function_from_string(func_input)
    a, b = -5.12, 5.12
    eps = 0.01

    # print("Введите одномерную функцию f(x), например: f(x) = x + sin(3.14159*x)")
    # func_input = input("f(x) = ").strip()
    # if not func_input:
    #     func_input = "x + sin(3.14159*x)"  # значение по умолчанию
    #
    # f, func_str = make_function_from_string(func_input)
    #
    # try:
    #     a = float(input("Введите левую границу отрезка a: "))
    #     b = float(input("Введите правую границу отрезка b: "))
    # except ValueError:
    #     print("Ошибка ввода, используются значения по умолчанию a=-5, b=5")
    #     a, b = -5.0, 5.0
    #
    # try:
    #     eps = float(input("Введите точность eps: "))
    # except ValueError:
    #     print("Ошибка ввода, используется значение по умолчанию eps=0.01")
    #     eps = 0.01

    L_est = estimate_L(f, a, b, n=2000)
    res = piyavskii_shubert(f, a, b, eps=eps, L=L_est, max_iters=2000)

    plot_results(f, a, b, res, func_str, fname_png="result_plot.png")
    make_txt_report(f, func_str, a, b, eps, res, txt_file="result.txt")

    print('-'*50)
    print("Файлы успешно сохранены:")
    print(" - result_plot.png — график функции и итераций")
    print(" - result.txt — результат поиска")
    print(f"Найденный минимум: x ≈ {res['xmin']:.8g}, f ≈ {res['fmin']:.8g}")
    print(f"Итераций: {res['iterations']}, время: {res['time']:.6f} s")
