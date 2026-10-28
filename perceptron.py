"""Entrenamiento de compuertas lógicas con un perceptrón de una neurona."""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


Pattern = tuple[int, ...]
Example = tuple[Pattern, int]


@dataclass(frozen=True)
class TrainingResult:
    weights: tuple[float, ...]
    bias: float
    epochs: int
    errors: tuple[int, ...]
    converged: bool


def step(weighted_sum: float) -> int:
    """Devuelve 1 si la suma ponderada alcanza el umbral cero; si no, 0."""
    return int(weighted_sum >= 0)


def predict(inputs: Pattern, weights: Sequence[float], bias: float) -> int:
    """Calcula la salida de una neurona usando suma ponderada, bias y escalón."""
    if len(inputs) != len(weights):
        raise ValueError("La cantidad de entradas debe coincidir con la de pesos.")
    return step(sum(weight * value for weight, value in zip(weights, inputs)) + bias)


def train(
    examples: Sequence[Example],
    *,
    learning_rate: float = 1.0,
    initial_weights: Sequence[float] | None = None,
    initial_bias: float = 0.0,
    max_epochs: int = 100,
) -> TrainingResult:
    """Entrena en línea hasta clasificar correctamente todos los ejemplos."""
    if not examples:
        raise ValueError("Se requiere al menos un ejemplo de entrenamiento.")
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("La tasa de aprendizaje debe ser un número positivo y finito.")
    if not isfinite(initial_bias):
        raise ValueError("El bias inicial debe ser finito.")
    if max_epochs < 1:
        raise ValueError("El límite de épocas debe ser al menos 1.")

    input_count = len(examples[0][0])
    if input_count < 1 or any(len(inputs) != input_count for inputs, _ in examples):
        raise ValueError("Todos los ejemplos deben tener la misma cantidad de entradas.")
    if any(value not in (0, 1) for inputs, _ in examples for value in inputs):
        raise ValueError("Las entradas deben ser binarias (0 o 1).")
    if any(target not in (0, 1) for _, target in examples):
        raise ValueError("Las salidas esperadas deben ser binarias (0 o 1).")

    weights = list(initial_weights if initial_weights is not None else [0.0] * input_count)
    if len(weights) != input_count or any(not isfinite(weight) for weight in weights):
        raise ValueError("Debe proporcionarse un peso inicial finito por entrada.")
    bias = initial_bias
    errors: list[int] = []

    for epoch in range(1, max_epochs + 1):
        epoch_error = 0
        for inputs, target in examples:
            prediction = predict(inputs, weights, bias)
            error = target - prediction
            epoch_error += abs(error)
            weights = [
                weight + learning_rate * error * value
                for weight, value in zip(weights, inputs)
            ]
            bias += learning_rate * error

        errors.append(epoch_error)
        if epoch_error == 0:
            return TrainingResult(tuple(weights), bias, epoch, tuple(errors), True)

    return TrainingResult(tuple(weights), bias, max_epochs, tuple(errors), False)
