# Проект: Multi-Camera Tracking на Rock5B

## Цель
Детекция объектов с мультикамерами (30% перекрытие), глобальный трекинг, вывод G#ID (камер/всего)

## Готово
- Калибровка стерео (calibrate_web)
- Детекция RKNN
- Camera Manager
- Config.json структура

## Текущие файлы
- global_tracker.cpp (3300 строк, сложный)
- camera_manager.cpp (вывод оверлеев - НЕ ПРИЛОЖЕН из-за лимита)
- config.json (hemisphere_single, 3 камеры: id=1,3,5)
- results/calibration_results.json (стерео калибровка)

## Проблемы
1. Расстояние до объекта = 0 или некорректное
2. Нет вывода (камер/всего) для Global ID

## Решение
Написать simple_global_tracker.cpp (~500 строк) вместо исправления текущего

## Вопрос к новому чату
Как сейчас выводятся оверлеи в camera_manager.cpp? (файл не приложен)