# Raport z Projektu: Produkcyjny Klasyfikator Żywności w formacie ONNX

## 1. Cel i Opis Projektu
Celem projektu było stworzenie kompletnego, produkcyjnego potoku (pipeline) uczenia maszynowego dla zadania klasyfikacji obrazów (Computer Vision). Projekt demonstruje pełną ścieżkę inżynierską: od trenowania w chmurze przy użyciu akceleracji GPU, przez optymalizację formatu modelu, aż po konteneryzację mikroserwisu i stworzenie niezależnego interfejsu graficznego (GUI).

Aplikacja implementuje architekturę klient-serwer (separacja backendu i frontendu), co pozwala na pełną skalowalność i niezależność komponentów.

---

## 2. Wybór Modelu i Zbioru Danych
* **Model Bazowy:** `MobileNetV2` – lekka architektura sieci neuronowej, zoptymalizowana pod kątem szybkiej inferencji na urządzeniach o ograniczonych zasobach (np. procesorach CPU).
* **Zbiór Danych:** Przefiltrowany podzbiór 20 najpopularniejszych klas z oficjalnej bazy danych **Food101** (*'apple_pie', 'beef_tartare', 'caesar_salad', 'cheesecake', 'chicken_wings','chocolate_cake', 'club_sandwich', 'dumplings', 'french_fries', 'garlic_bread','hamburger', 'ice_cream', 'lasagna', 'macaroni_and_cheese', 'omelette','pancakes', 'pizza', 'spaghetti_bolognese', 'sushi', 'waffles'*).
* **Metodologia Treningu:** Wykorzystano technikę **Transfer Learningu**. Wszystkie warstwy ekstrakcji cech wyuczone na zbiorze ImageNet zostały zamrożone (`requires_grad=False`). Treningowi poddano wyłącznie nową, końcową warstwę klasyfikatora liniowego dopasowaną do 20 wybranych potraw. Trening przeprowadzono w chmurze Google Colab z użyciem karty graficznej **Nvidia T4 GPU** przez 3 epoki.

---

## 3. Eksport do formatu ONNX
Po zakończeniu procesu uczenia model został wyeksportowany z frameworka PyTorch do uniwersalnego formatu **ONNX (Open Neural Network Exchange)** z parametrem `opset_version=12`. 

W procesie eksportu zdefiniowano osie dynamiczne (`dynamic_axes`) dla wymiaru `batch_size`. Dzięki temu model potrafi przetwarzać w locie pojedyncze zdjęcia przesyłane przez użytkowników końcowych, zachowując minimalne zużycie pamięci RAM.

---

## 4. Wyniki Testów Zgodności i Wydajności (Benchmark)
Lokalne testy porównawcze zostały przeprowadzone na procesorze (CPU) maszyny klienckiej

* **Status Zgodności Predykcji (Punkt 4):** Predykcje wyjściowe z biblioteki PyTorch oraz ONNX Runtime dla tego samego obrazu testowego okazały się matematycznie zgodne (różnice w wartościach prawdopodobieństw poniżej tolerancji `1e-05`).
* **Średni czas inferencji PyTorch CPU (Punkt 5):** 13.12 ms / obraz
* **Średni czas inferencji ONNX Runtime CPU (Punkt 5):** 2.54 ms / obraz
* **Wnioski dotyczące wydajności:** ONNX Runtime okazał się w teście benchmarkowym **5.17x szybszy** niż standardowy framework PyTorch uruchomiony na CPU. Wynik ten w pełni uzasadnia konwersję modelu w celach produkcyjnych.

---

## 5. Struktura Plików Projektu
Wdrożona aplikacja lokalna składa się z następującej struktury katalogów:
```text
projekt-ml/
├── food_classifier_20.onnx        # Zoptymalizowany model ONNX
├── app.py                         # Serwer produkcyjny (FastAPI backend)
├── gui.py                         # Interfejs użytkownika (Streamlit frontend)
├── requirements.txt               # Zależności biblioteczne dla Pythona
└── Dockerfile                     # Przepis budowania obrazu kontenera

```

# Instrukcja Uruchomienia Aplikacji

Postępuj zgodnie z poniższymi krokami, aby uruchomić cały system (Serwer w Dockerze + GUI).

### Krok 1: Uruchomienie Serwera (Backend w Dockerze)
1. Otwórz terminal (np. PowerShell) w folderze `dockerApp`.
2. Zbuduj obraz kontenera (wymagane przy pierwszym uruchomieniu lub po zmianach w kodzie):
```bash
docker build -t food-classifier .
```

3. Uruchom kontener:
```bash
docker run --name food-classifier -p 8000:8000 food-classifier
```

### Krok 2: Uruchomienie Frontend-u (GUI w Tkinter)

Aplikacja okienkowa uruchamiana jest bezpośrednio na Twoim komputerze w osobnym procesie i komunikuje się z Dockerem przez sieć.

1. Otwórz **drugie, zupełnie osobne okno terminala** (nie zamykaj okna z Dockerem!).
2. Zainstaluj wymagane pakiety do obsługi okna, przeciągania plików oraz komunikacji HTTP:
```bash
pip install tkinterdnd2 requests pillow
```
3. Uruchom aplikację graficzną komendą:
```bash
python gui.py
```

### Krok 3: Obsługa programu

1. Po wykonaniu komendy z Kroku 2 na ekranie pojawi się klasyczne, szare okienko systemowe z dużym obszarem oznaczonym jako `[ UPUŚĆ PLIK TUTAJ ]`.
2. Chwyć myszką dowolne zdjęcie potrawy z folderu na swoim komputerze (np. zdjęcie pizzy lub frytek) i przeciągnij je nad szary obszar okna, a następnie puść lewy przycisk myszy.
3. Program automatycznie:
* Wyświetli miniaturkę zdjęcia w okienku.
* Prześle plik do kontenera Docker.
* Odbierze wynik i wyświetli duży, zielony napis z nazwą rozpoznanej potrawy na samym dole okna (np. `WYNIK: PIZZA`).