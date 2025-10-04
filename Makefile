# Makefile для SimpleGlobalTracker
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -I/usr/include/opencv4
LIBS = -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_imgcodecs -lstdc++fs

TARGET = test_simple_tracker
SOURCES = simple_global_tracker.cpp test_simple_tracker.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
    @echo "=== Линковка $(TARGET) ==="
    $(CXX) $(OBJECTS) -o $(TARGET) $(LIBS)
    @echo "✅ Сборка завершена успешно"

%.o: %.cpp
    @echo "Компиляция $<..."
    $(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
    @echo "Очистка..."
    rm -f $(OBJECTS) $(TARGET)
    @echo "✅ Очистка завершена"

install:
    @echo "Установка зависимостей..."
    sudo apt-get update
    sudo apt-get install -y libopencv-dev pkg-config

test: $(TARGET)
    @echo "=== Запуск тестов ==="
    ./$(TARGET)
    @echo "✅ Тесты завершены"

