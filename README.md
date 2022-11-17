## УПРАВЛЕНИЕ ДВИЖЕНИЕМ СФЕРИЧЕСКОГО РОБОТА ПРИ ПОМОЩИ Q-ОБУЧЕНИЯ

[![Видео]](https://www.youtube.com/watch?v=pCcGI_YZO_s)

![Alt Text](paper/trajectory.gif)

## 1. Настройка
- для того, чтобы запустить обучение или симуляцию с уже обученной моделью, необходимо выполнить инструкции по установке и настройке физического движка ```MuJoCo```,
настройки валидны для ```Ubuntu 18.04``` и выше

- инструкции по найстройке движка лежат в ```./server/setup_part_2.sh```

## 2. Воспроизведение результатов
- результаты из статьи лежат в папке ```./paper```
в файле ```result_paper.json``` лежат сгенерированные случайные траектории, по которым выводятся все метрики в статье
и строятся 3 графика, чтобы их воспроизвести достаточно просто запустить скрипт ```paper_result.py```

- для того чтобы увидеть, как робот едет по этим траекториям, нужно запустить скрипт ```./test/sphere.py```



#### UPD: поскольку гит не пускает файлы выше 100Мб и настроить git lfs не получилось, то модели доступны по ссылкам, после скачивания переместить в папку ``./models``

 - [Q-network](https://disk.yandex.ru/d/v1yUVwlFXtMcYw)

 - [policy-network](https://disk.yandex.ru/d/zYff7QSoP7cdAQ)
