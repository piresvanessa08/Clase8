"""Entrena AND, OR y NOT, muestra sus resultados y guarda las gráficas."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perceptron import Example, TrainingResult, predict, train


GATES: dict[str, tuple[Example, ...]] = {
    "AND": (((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)),
    "OR": (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)),
    "NOT": (((0,), 1), ((1,), 0)),
}
MAX_EPOCHS = 100
OUTPUT_DIRECTORY = Path(__file__).parent / "graficos"


def save_error_plot(name: str, result: TrainingResult) -> Path:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"error_{name.lower()}.png"
    epochs = range(1, len(result.errors) + 1)

    figure, axis = plt.subplots()
    axis.plot(epochs, result.errors, marker="o")
    axis.set(title=f"Error por época - {name}", xlabel="Época", ylabel="Error absoluto")
    axis.set_xticks(list(epochs))
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def show_results(name: str, examples: tuple[Example, ...], result: TrainingResult) -> bool:
    print(f"\n{name}: {'aprendida' if result.converged else 'NO convergió'}")
    print(f"Épocas ejecutadas: {result.epochs}")
    print(f"Pesos finales: {result.weights}; bias final: {result.bias:g}")
    print("Entrada | Esperada | Predicha | Coincide")
    correct = True
    for inputs, expected in examples:
        predicted = predict(inputs, result.weights, result.bias)
        matches = predicted == expected
        correct &= matches
        print(f"{str(inputs):7} | {expected:8} | {predicted:8} | {'Sí' if matches else 'No'}")
    if not result.converged:
        print(f"FALLO: no clasificó correctamente todos los casos en {MAX_EPOCHS} épocas.")
    return correct and result.converged


def main() -> int:
    all_converged = True
    for name, examples in GATES.items():
        result = train(examples, max_epochs=MAX_EPOCHS)
        print(f"\nErrores por época ({name}): {list(result.errors)}")
        all_converged &= show_results(name, examples, result)
        print(f"Gráfica: {save_error_plot(name, result)}")

    if not all_converged:
        print("\nUna o más compuertas no convergieron; revisa los parámetros o el límite.")
        return 1
    print("\nLas tres compuertas aprendieron todas sus combinaciones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
