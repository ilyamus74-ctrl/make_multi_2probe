#!/bin/bash

echo "=== Сборка SimpleGlobalTracker ==="

# Проверка зависимостей
if ! pkg-config --exists opencv4; then
    echo "❌ OpenCV4 не найден"
    echo "Установите: sudo apt-get install libopencv-dev"
    exit 1
fi

# Очистка предыдущей сборки
make clean

# Компиляция
echo "Компиляция..."
if make; then
    echo "✅ Сборка успешна"
    echo ""
    echo "=== Запуск тестов ==="
    ./test_simple_tracker
else
    echo "❌ Ошибка сборки"
    exit 1
fi