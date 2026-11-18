# World Model Demo

Demonstrator modelu świata dla środowiska `CarRacing-v3` z Gymnasium. Celem jest
interaktywne generowanie krótkich sekwencji obrazu na podstawie sterowania,
z użyciem VAE i rekurencyjnego modelu dynamiki MDN-RNN.

## Zakres pierwszej wersji

- zbieranie klatek 64×64 wraz z akcjami i granicami epizodów,
- VAE kodujący klatki do wektorów latentnych o wymiarze 32,
- MDN-RNN przewidujący następny stan latentny,
- proste demo webowe ze sterowaniem, resetem i temperaturą,
- diagnostyczne porównanie z Gymnasium oraz pomiar wydajności.

Ręczna edycja przestrzeni latentnej i długie, stabilne sekwencje są poza zakresem
minimalnej wersji.

## Wymagania

- Python 3.10 lub nowszy
- Zależności projektu zainstalowane przez `pip install -e .`

## Struktura

Kod pakietu znajduje się w `src/world_model/`, a skrypty uruchomieniowe w
`scripts/`. Dane i wytrenowane wagi nie są częścią repozytorium.
