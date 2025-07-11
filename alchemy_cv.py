import tkinter as tk
from tkinter import ttk

class AdvancedFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AlchemyCV")
        # Get screen dimensions and set window size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        # --- Member Variables (to be added later) ---
        self.original_cv_image = None
        self.processed_cv_image = None
        
        self._create_widgets()

    def _create_widgets(self):
        # Create a label to show something is happening
        ttk.Label(self.root, text="AlchemyCV Initialized. Widgets to be added.", font=("Helvetica", 16)).pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedFilterApp(root)
    root.mainloop()