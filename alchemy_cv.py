import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class AdvancedFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AlchemyCV")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        self.original_cv_image = None
        self.processed_cv_image = None
        self.tk_image = None

        self.preprocessing_data = {
            'None': {'type': 'none'},
            'Gaussian Blur': {'type': 'gaussian', 'params': {'Kernel Size': {'range': (1, 51), 'default': 5}}},
            'Median Blur': {'type': 'median', 'params': {'Kernel Size': {'range': (1, 51), 'default': 5}}},
            'Bilateral Filter': {'type': 'bilateral', 'params': {'Diameter': {'range': (1, 25), 'default': 9}, 'Sigma Color': {'range': (1, 150), 'default': 75}, 'Sigma Space': {'range': (1, 150), 'default': 75}}},
        }
        self.selected_preproc = tk.StringVar(value='None')
        self.param_vars = {}
        self.preproc_param_widgets = []

        self._create_widgets()
        self.on_preproc_change()

    def _create_widgets(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(controls_frame, width=480, stretch="never")
        
        right_container = ttk.Frame(main_pane, padding="10")
        right_container.grid_rowconfigure(0, weight=1)
        right_container.grid_columnconfigure(0, weight=1)
        self.canvas_right = tk.Canvas(right_container, background="gray")
        self.canvas_right.grid(row=0, column=0, sticky="nsew")
        self.image_on_canvas = self.canvas_right.create_image(0, 0, anchor="nw")
        main_pane.add(right_container)

        top_button_frame = ttk.Frame(controls_frame)
        top_button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Button(top_button_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=2)
        self.save_button = ttk.Button(top_button_frame, text="Save Image", command=self.save_image, state='disabled')
        self.save_button.pack(side=tk.LEFT, padx=2)

        preproc_frame = ttk.LabelFrame(controls_frame, text="1. Pre-processing", padding="10")
        preproc_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.OptionMenu(preproc_frame, self.selected_preproc, self.selected_preproc.get(), *self.preprocessing_data.keys(), command=self.on_preproc_change).pack(fill=tk.X)
        self.preproc_parameter_frame = ttk.Frame(preproc_frame, padding=(0, 10, 0, 0))
        self.preproc_parameter_frame.pack(fill=tk.X)

    def open_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp"), ("All files", "*.*")])
        if not filepath: return
        self.original_cv_image = cv2.imread(filepath)
        if self.original_cv_image is None:
            messagebox.showerror("Error", f"Failed to open image: {filepath}")
            return
        self.apply_filter()
        self.save_button.config(state='normal')

    def save_image(self):
        if self.processed_cv_image is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if not filepath: return
        cv2.imwrite(filepath, self.processed_cv_image)
        messagebox.showinfo("Success", f"Image saved to {filepath}")

    def on_preproc_change(self, event=None):
        self._create_dynamic_panel(self.preprocessing_data, 'selected_preproc', self.preproc_parameter_frame, self.preproc_param_widgets, 'preproc')
        self.apply_filter()

    def apply_filter(self, event=None):
        if self.original_cv_image is None: return
        self.processed_cv_image = self._process_preprocessing(self.original_cv_image)
        self.update_image_display(self.processed_cv_image)

    def _process_preprocessing(self, image):
        preproc_type = self.selected_preproc.get()
        if preproc_type == 'None': return image.copy()
        
        try:
            if preproc_type == 'Gaussian Blur':
                ksize = self.param_vars['preproc_Kernel_Size'].get()
                if ksize > 0 and ksize % 2 == 0: ksize += 1
                return cv2.GaussianBlur(image, (ksize, ksize), 0)
            elif preproc_type == 'Median Blur':
                ksize = self.param_vars['preproc_Kernel_Size'].get()
                if ksize > 0 and ksize % 2 == 0: ksize += 1
                return cv2.medianBlur(image, ksize)
            elif preproc_type == 'Bilateral Filter':
                d = self.param_vars['preproc_Diameter'].get()
                sc = self.param_vars['preproc_Sigma_Color'].get()
                ss = self.param_vars['preproc_Sigma_Space'].get()
                return cv2.bilateralFilter(image, d, sc, ss)
        except (KeyError, tk.TclError): # Handle case where UI is updating
            return image.copy()
        return image.copy()

    def _create_dynamic_panel(self, data, selector_var_name, parent_frame, widget_list, param_prefix):
        for widget in widget_list: widget.destroy()
        widget_list.clear()
        
        selection_name = getattr(self, selector_var_name).get()
        config = data.get(selection_name, {})
        params = config.get('params', {})
        
        for k in list(self.param_vars.keys()):
            if k.startswith(param_prefix):
                del self.param_vars[k]

        for p_name, p_data in params.items():
            frame = self._create_param_row(p_name, parent=parent_frame)
            widget_list.append(frame)
            var_key = f"{param_prefix}_{p_name.replace(' ', '_')}"
            
            var = tk.IntVar(value=p_data['default'])
            slider, entry = self._create_slider_and_entry(frame, var, p_data['range'][0], p_data['range'][1])
            widget_list.extend([slider, entry])
            self.param_vars[var_key] = var

    def _create_param_row(self, label, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky='w', padx=5)
        return frame

    def _create_slider_and_entry(self, parent, variable, min_val, max_val):
        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=variable, command=self.apply_filter)
        slider.grid(row=0, column=1, sticky='ew', padx=5)
        entry = ttk.Entry(parent, textvariable=variable, width=7)
        entry.grid(row=0, column=2)
        entry.bind("<Return>", self.apply_filter)
        return slider, entry

    def update_image_display(self, cv_image):
        if cv_image is None: return
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