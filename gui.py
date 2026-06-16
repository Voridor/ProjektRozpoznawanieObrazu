import tkinter as tk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import requests
from PIL import Image, ImageTk

# Adres Twojego serwera FastAPI działającego w kontenerze Docker
API_URL = "http://localhost:8000/predict"


def handle_drop(event):
    # Pobranie ścieżki pliku i oczyszczenie jej z ewentualnych klamer Windowsa {}
    file_path = event.data.strip('{}')

    # Sprawdzenie czy upuszczony plik to obrazek
    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        messagebox.showerror("Błąd", "To nie jest poprawny plik graficzny (.jpg, .jpeg, .png)!")
        return

    # Zmiana statusu na czas wysyłania żądania
    label_status.config(text="Status: Przetwarzanie obrazu...", fg="blue")
    root.update_idletasks()

    try:
        # 1. Wczytanie i wyświetlenie miniaturki zdjęcia w okienku Tkintera
        img = Image.open(file_path)
        img.thumbnail((200, 200))  # Skalowanie obrazu, aby zmieścił się w UI
        img_tk = ImageTk.PhotoImage(img)
        label_image.config(image=img_tk)
        label_image.image = img_tk  # Zapobiega usunięciu obiektu przez Garbage Collector

        # 2. Wysyłanie zdjęcia jako strumień bajtów do API w Dockerze
        with open(file_path, "rb") as f:
            files = {"file": f.read()}
            response = requests.post(API_URL, files=files)

        # 3. Odczytanie odpowiedzi i wyświetlenie wyniku użytkownikowi
        if response.status_code == 200:
            result = response.json()
            dish_name = result.get("dish_name", "Nieznany")
            # TUTAJ WYŚWIETLA SIĘ WYNIK MODELU:
            label_status.config(text=f"WYNIK: {dish_name.upper()}", fg="green")
        else:
            label_status.config(text="Błąd: Serwer Docker zwrócił błąd wewnętrzny.", fg="red")

    except requests.exceptions.ConnectionError:
        label_status.config(text="Błąd: Brak połączenia! Czy Docker działa?", fg="red")
    except Exception as e:
        label_status.config(text=f"Wystąpił błąd: {str(e)}", fg="red")


# --- Inicjalizacja Głównego Okna Programu z obsługą Drag & Drop ---
root = TkinterDnD.Tk()
root.title("Klasyfikator Potraw - 20 Klas")
root.geometry("400x500")
root.resizable(False, False)

# Nagłówek okna
label_title = tk.Label(root, text="Przeciągnij i upuść zdjęcie potrawy tutaj:", font=("Arial", 12, "bold"), pady=15)
label_title.pack()

# Poprawiony Szary Kwadrat (Strefa Upuszczania Pliku) - bez błędnego "dashed"
drop_zone = tk.Label(
    root,
    text="\n\n[ UPURŚ PLIK TUTAJ ]\n\n",
    bg="#e0e0e0",
    fg="#555555",
    width=40,
    height=10,
    relief="solid",
    bd=1,
    highlightbackground="#aaaaaa",
    highlightthickness=1
)
drop_zone.pack(pady=10)

# Rejestracja strefy jako odbiorcy plików z systemu operacyjnego
drop_zone.drop_target_register(DND_FILES)
drop_zone.dnd_bind('<<Drop>>', handle_drop)

# Etykieta przeznaczona na podgląd załadowanego zdjęcia
label_image = tk.Label(root)
label_image.pack(pady=10)

# Główna etykieta statusu oraz wyniku działania modelu
label_status = tk.Label(root, text="Status: Oczekiwanie na plik...", font=("Arial", 12, "bold"), fg="gray")
label_status.pack(pady=15)

# Start pętli aplikacji okienkowej
root.mainloop()