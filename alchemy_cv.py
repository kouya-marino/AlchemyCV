import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk

class AdvancedFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AlchemyCV")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        # --- Member Variables ---
        self.original_cv_image = None
        self.processed_cv_image = None
        self.tk_image = None

        self._create_widgets()

    def _create_widgets(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # --- Left Panel (Controls) ---
        controls_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(controls_frame, width=480, stretch="never")
        
        # --- Right Panel (Image Display) ---
        right_container = ttk.Frame(main_pane, padding="10")
        right_container.grid_rowconfigure(0, weight=1)
        right_container.grid_columnconfigure(0, weight=1)
        self.canvas_right = tk.Canvas(right_container, background="gray")
        self.canvas_right.grid(row=0, column=0, sticky="nsew")
        self.image_on_canvas = self.canvas_right.create_image(0, 0, anchor="nw")
        main_pane.add(right_container)

        # --- Control Widgets ---
        top_button_frame = ttk.Frame(controls_frame)
        top_button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Button(top_button_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=2)
        self.save_button = ttk.Button(top_button_frame, text="Save Image", command=self.save_image, state='disabled')
        self.save_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(controls_frame, text="Filter controls will be added here.").pack(pady=20)

    def open_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp"), ("All files", "*.*")])
        if not filepath: return
        self.original_cv_image = cv2.imread(filepath)
        if self.original_cv_image is None: 
            messagebox.showerror("Error", f"Failed to open image: {filepath}")
            return
        
        # For now, the processed image is just a copy of the original
        self.processed_cv_image = self.original_cv_image.copy()
        self.update_image_display(self.processed_cv_image)
        self.save_button.config(state='normal')

    def save_image(self):
        if self.processed_cv_image is None:
            messagebox.showwarning("No Image", "There is no image to save.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if not filepath: return
        try:
            cv2.imwrite(filepath, self.processed_cv_image)
            messagebox.showinfo("Success", f"Image saved successfully to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save image: {e}")

    def update_image_display(self, cv_image):
        if cv_image is None: return

        # Convert from BGR (OpenCV) to RGB (PIL)
        if len(cv_image.shape) == 2:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(rgb_image)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        
        self.canvas_right.itemconfig(self.image_on_canvas, image=self.tk_image)
        self.canvas_right.config(scrollregion=self.canvas_right.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedFilterApp(root)
    root.mainloop()