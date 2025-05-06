Особенности сборки: 

- Возможность переключения с CUDA режима (по умолчанию) на режим DirectML для работы на видеокартах AMD и Intel (и старых Nvidia).
- Возможность выбора языка интерфейса - русский или английский
Также что нового:
- Обновлены библиотеки до актуальных версий, незначительное ускорение работы.
- Все обновления, вошедшие в версию 3.2.0, например новая модель xseg для маски лица.
Для смены CUDA режима на режим для остальных карт, запустите change_backend.bat и следуйте инструкции.

Скачать:

С моего сервера (https://disk.yandex.ru/d/2G8XMILgzEpMIg)
С Яндекс Диска (https://disk.yandex.ru/d/2G8XMILgzEpMIg)
Universal FaceFusion build:
- Ability to switch from CUDA mode (default) to DirectML mode to work on AMD and Intel (and older Nvidia) graphics cards.
- Ability to select interface language - Russian or English
Also what's new:
- Updated libraries to current versions, minor speedup.
- All updates included in version 3.2.0, e.g. new xseg model for face mask.
To change CUDA mode to the mode for the rest of the maps, run change_backend.bat and follow the instructions.

Download:

From my server (https://disk.yandex.ru/d/2G8XMILgzEpMIg)
From Yandex Disk (https://disk.yandex.ru/d/2G8XMILgzEpMIg) 

FaceFusion
==========

> Industry leading face manipulation platform.

[![Build Status](https://img.shields.io/github/actions/workflow/status/facefusion/facefusion/ci.yml.svg?branch=master)](https://github.com/facefusion/facefusion/actions?query=workflow:ci)
[![Coverage Status](https://img.shields.io/coveralls/facefusion/facefusion.svg)](https://coveralls.io/r/facefusion/facefusion)
![License](https://img.shields.io/badge/license-OpenRAIL--AS-green)


Preview
-------

![Preview](https://raw.githubusercontent.com/facefusion/facefusion/master/.github/preview.png?sanitize=true)


Installation
------------

Be aware, the [installation](https://docs.facefusion.io/installation) needs technical skills and is not recommended for beginners. In case you are not comfortable using a terminal, our [Windows Installer](http://windows-installer.facefusion.io) and [macOS Installer](http://macos-installer.facefusion.io) get you started.


Usage
-----

Run the command:

```
python facefusion.py [commands] [options]

options:
  -h, --help                                      show this help message and exit
  -v, --version                                   show program's version number and exit

commands:
    run                                           run the program
    headless-run                                  run the program in headless mode
    batch-run                                     run the program in batch mode
    force-download                                force automate downloads and exit
    job-list                                      list jobs by status
    job-create                                    create a drafted job
    job-submit                                    submit a drafted job to become a queued job
    job-submit-all                                submit all drafted jobs to become a queued jobs
    job-delete                                    delete a drafted, queued, failed or completed job
    job-delete-all                                delete all drafted, queued, failed and completed jobs
    job-add-step                                  add a step to a drafted job
    job-remix-step                                remix a previous step from a drafted job
    job-insert-step                               insert a step to a drafted job
    job-remove-step                               remove a step from a drafted job
    job-run                                       run a queued job
    job-run-all                                   run all queued jobs
    job-retry                                     retry a failed job
    job-retry-all                                 retry all failed jobs
```


Documentation
-------------

Read the [documentation](https://docs.facefusion.io) for a deep dive.
