import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json

# Optional import for the Histogram feature
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
        self.reset_button = None
        
        # --- Zoom and Pan State ---
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # --- DATA STRUCTURES ---
        self.preprocessing_data = {
            'None': {'type': 'none'},
            'Gaussian Blur': {'type': 'gaussian', 'params': {'Kernel Size': {'range': (1, 51), 'default': 5}}},
            'Median Blur': {'type': 'median', 'params': {'Kernel Size': {'range': (1, 51), 'default': 5}}},
            'Bilateral Filter': {'type': 'bilateral', 'params': {'Diameter': {'range': (1, 25), 'default': 9}, 'Sigma Color': {'range': (1, 150), 'default': 75}, 'Sigma Space': {'range': (1, 150), 'default': 75}}},
        }
        self.enhancement_data = {
            'None': {'type': 'none'},
            'Histogram Equalization': {'type': 'hist_equal'},
            'CLAHE': {'type': 'clahe', 'params': {'Clip Limit': {'range': (1, 40), 'default': 2}, 'Tile Grid Size': {'range': (2, 32), 'default': 8}}},
            'Contrast Stretching': {'type': 'contrast_stretch'},
            'Gamma Correction': {'type': 'gamma', 'params': {'Gamma x100': {'range': (10, 300), 'default': 100}}},
            'Log Transform': {'type': 'log'},
            'Single-Scale Retinex': {'type': 'ssr', 'params': {'Sigma': {'range': (1, 250), 'default': 30}}},
            'Unsharp Masking': {'type': 'unsharp', 'params': {'Kernel Size': {'range': (1, 51), 'default': 5}, 'Alpha x10': {'range': (1, 50), 'default': 15}}},
        }
        self.frequency_data = {
            'None': {'type': 'none'},
            'Ideal Low-Pass': {'type': 'ilpf', 'params': {'Cutoff Freq (D0)': {'range': (1, 250), 'default': 30}}},
            'Gaussian Low-Pass': {'type': 'glpf', 'params': {'Cutoff Freq (D0)': {'range': (1, 250), 'default': 30}}},
            'Butterworth Low-Pass': {'type': 'blpf', 'params': {'Cutoff Freq (D0)': {'range': (1, 250), 'default': 30}, 'Order (n)': {'range': (1, 10), 'default': 2}}},
            'Ideal High-Pass': {'type': 'ihpf', 'params': {'Cutoff Freq (D0)': {'range': (1, 250), 'default': 30}}},
            'Gaussian High-Pass': {'type': 'ghpf', 'params': {'Cutoff Freq (D0)': {'range': (1, 250), 'default': 30}}},
            'Butterworth High-Pass': {'type': 'bhpf', 'params': {'Cutoff Freq (D0)': {'range': (1, 250), 'default': 30}, 'Order (n)': {'range': (1, 10), 'default': 2}}},
        }
        self.channel_data = {
            'Grayscale': {'code': cv2.COLOR_BGR2GRAY, 'channels': ['Intensity']},
            'RGB/BGR': {'code': None, 'channels': ['B', 'G', 'R']},
            'HSV': {'code': cv2.COLOR_BGR2HSV, 'channels': ['H', 'S', 'V']},
            'HLS': {'code': cv2.COLOR_BGR2HLS, 'channels': ['H', 'L', 'S']},
            'Lab': {'code': cv2.COLOR_BGR2LAB, 'channels': ['L', 'a', 'b']},
            'YCrCb': {'code': cv2.COLOR_BGR2YCrCb, 'channels': ['Y', 'Cr', 'Cb']}
        }
        self.filter_data = {
            'RGB/BGR (Color Filter)': {'type': 'color', 'channels': ['B', 'G', 'R'], 'ranges': [(0, 255), (0, 255), (0, 255)]},
            'HSV': {'type': 'color', 'channels': ['H', 'S', 'V'], 'ranges': [(0, 179), (0, 255), (0, 255)]},
            'HLS': {'type': 'color', 'channels': ['H', 'L', 'S'], 'ranges': [(0, 179), (0, 255), (0, 255)]},
            'Lab': {'type': 'color', 'channels': ['L', 'a', 'b'], 'ranges': [(0, 255), (0, 255), (0, 255)]},
            'YCrCb': {'type': 'color', 'channels': ['Y', 'Cr', 'Cb'], 'ranges': [(0, 255), (0, 255), (0, 255)]},
            'Grayscale Range': {'type': 'grayscale_range', 'params': {'Min Value': {'range': (0, 255), 'default': 0}, 'Max Value': {'range': (0, 255), 'default': 127}}},
            'Adaptive Threshold': {'type': 'adaptive_thresh', 'params': {'Adaptive Method': {'options': ['Mean C', 'Gaussian C'], 'default': 'Gaussian C'}, 'Threshold Type': {'options': ['Binary', 'Binary Inverted'], 'default': 'Binary'}, 'Block Size': {'range': (3, 55), 'default': 11}, 'C (Constant)': {'range': (-30, 30), 'default': 2}}},
            "Otsu's Binarization": {'type': 'otsu', 'params': {'Threshold Type': {'options': ['Binary', 'Binary Inverted'], 'default': 'Binary'}}}
        }
        self.edge_detection_data = {
            'Canny': {'type': 'canny', 'params': {'Threshold 1': {'range': (0, 255), 'default': 50}, 'Threshold 2': {'range': (0, 255), 'default': 150}}},
            'Sobel': {'type': 'sobel', 'params': {'Kernel Size': {'range': (1, 31), 'default': 3}, 'Direction': {'options': ['X', 'Y', 'Magnitude'], 'default': 'Magnitude'}}},
            'Prewitt': {'type': 'prewitt', 'params': {'Direction': {'options': ['X', 'Y', 'Magnitude'], 'default': 'Magnitude'}}},
            'Roberts': {'type': 'roberts', 'params': {'Direction': {'options': ['X', 'Y', 'Magnitude'], 'default': 'Magnitude'}}}
        }
        
        # --- TKinter Variables ---
        self.selected_preproc = tk.StringVar(value='None')
        self.selected_enhancement = tk.StringVar(value='None')
        self.selected_frequency_filter = tk.StringVar(value='None')
        self.channel_enabled = tk.BooleanVar(value=False)
        self.selected_color_space = tk.StringVar(value=list(self.channel_data.keys())[0])
        self.selected_channel = tk.StringVar()
        self.selected_filter = tk.StringVar(value=list(self.filter_data.keys())[0])
        self.edge_enabled = tk.BooleanVar(value=False)
        self.selected_edge_filter = tk.StringVar(value='Canny')
        self.morph_enabled = tk.BooleanVar(value=False)
        self.param_vars = {}
        
        self.contours_enabled = tk.BooleanVar(value=False)
        self.draw_contours = tk.BooleanVar(value=True)
        self.contour_min_area = tk.IntVar(value=50)
        self.contour_max_area = tk.IntVar(value=1000000)
        self.object_count_text = tk.StringVar(value="Objects Found: --")
        self.display_mode = tk.StringVar(value="Final Result")
        self.zoom_level_text = tk.StringVar(value="Zoom: 100%")

        # --- Widget Storage ---
        self.preproc_param_widgets = []
        self.enhancement_param_widgets = []
        self.frequency_param_widgets = []
        self.channel_param_widgets = []
        self.param_widgets = []
        self.edge_param_widgets = []
        self.morph_param_widgets = []

        self._create_widgets()
        self._initialize_ui_states()
        self.selected_filter.trace_add('write', self.on_filter_change)

    def _create_widgets(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_container = ttk.Frame(main_pane)
        self.canvas_left = tk.Canvas(left_container)
        v_scroll_left = ttk.Scrollbar(left_container, orient="vertical", command=self.canvas_left.yview)
        h_scroll_left = ttk.Scrollbar(left_container, orient="horizontal", command=self.canvas_left.xview)
        self.canvas_left.configure(yscrollcommand=v_scroll_left.set, xscrollcommand=h_scroll_left.set)
        
        left_container.grid_rowconfigure(0, weight=1)
        left_container.grid_columnconfigure(0, weight=1)
        self.canvas_left.grid(row=0, column=0, sticky="nsew")
        v_scroll_left.grid(row=0, column=1, sticky="ns")
        h_scroll_left.grid(row=1, column=0, sticky="ew")

        controls_frame = ttk.Frame(self.canvas_left, padding="10")
        self.canvas_left.create_window((0, 0), window=controls_frame, anchor="nw")
        
        controls_frame.bind("<Configure>", lambda e: self.canvas_left.configure(scrollregion=self.canvas_left.bbox("all")))
        self.canvas_left.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas_left.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

        main_pane.add(left_container, width=480, stretch="never")
        
        right_container = ttk.Frame(main_pane, padding="10")
        right_container.grid_rowconfigure(0, weight=1)
        right_container.grid_columnconfigure(0, weight=1)

        self.canvas_right = tk.Canvas(right_container, background="gray")
        v_scroll_right = ttk.Scrollbar(right_container, orient="vertical", command=self.canvas_right.yview)
        h_scroll_right = ttk.Scrollbar(right_container, orient="horizontal", command=self.canvas_right.xview)
        self.canvas_right.configure(yscrollcommand=v_scroll_right.set, xscrollcommand=h_scroll_right.set)
        
        self.canvas_right.grid(row=0, column=0, sticky="nsew")
        v_scroll_right.grid(row=0, column=1, sticky="ns")
        h_scroll_right.grid(row=1, column=0, sticky="ew")

        self.image_on_canvas = self.canvas_right.create_image(0, 0, anchor="nw")
        
        self.canvas_right.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas_right.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.canvas_right.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)

        zoom_frame = ttk.Frame(right_container, padding=5)
        zoom_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(zoom_frame, text="Zoom In (+)", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Zoom Out (-)", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Fit to Screen", command=self.fit_to_screen).pack(side=tk.LEFT, padx=5)
        ttk.Label(zoom_frame, textvariable=self.zoom_level_text).pack(side=tk.RIGHT, padx=5)
        
        main_pane.add(right_container)

        top_button_frame = ttk.Frame(controls_frame)
        top_button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        top_button_frame.columnconfigure((0,1,2,3), weight=1)
        
        ttk.Button(top_button_frame, text="Open Image", command=self.open_image).grid(row=0, column=0, sticky='ew', padx=(0, 2))
        self.save_button = ttk.Button(top_button_frame, text="Save Image", command=self.save_image, state='disabled')
        self.save_button.grid(row=0, column=1, sticky='ew', padx=2)
        ttk.Button(top_button_frame, text="Load Settings", command=self.load_settings).grid(row=0, column=2, sticky='ew', padx=2)
        ttk.Button(top_button_frame, text="Save Settings", command=self.save_settings).grid(row=0, column=3, sticky='ew', padx=(2, 0))
        
        self.reset_button = ttk.Button(controls_frame, text="Reset All to Defaults", command=self.reset_all_to_defaults, state='disabled')
        self.reset_button.pack(fill=tk.X, pady=(0, 10))

        preproc_frame = ttk.LabelFrame(controls_frame, text="1. Pre-processing", padding="10")
        preproc_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.OptionMenu(preproc_frame, self.selected_preproc, self.selected_preproc.get(), *self.preprocessing_data.keys(), command=self.on_preproc_change).pack(fill=tk.X)
        self.preproc_parameter_frame = ttk.Frame(preproc_frame, padding=(0, 10, 0, 0))
        self.preproc_parameter_frame.pack(fill=tk.X)

        enhancement_frame = ttk.LabelFrame(controls_frame, text="2. Image Enhancement", padding="10")
        enhancement_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.OptionMenu(enhancement_frame, self.selected_enhancement, self.selected_enhancement.get(), *self.enhancement_data.keys(), command=self.on_enhancement_change).pack(fill=tk.X)
        self.enhancement_parameter_frame = ttk.Frame(enhancement_frame, padding=(0, 10, 0, 0))
        self.enhancement_parameter_frame.pack(fill=tk.X)
        
        freq_frame = ttk.LabelFrame(controls_frame, text="3. Frequency Domain Filters", padding="10")
        freq_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.OptionMenu(freq_frame, self.selected_frequency_filter, self.selected_frequency_filter.get(), *self.frequency_data.keys(), command=self.on_frequency_filter_change).pack(fill=tk.X)
        self.frequency_parameter_frame = ttk.Frame(freq_frame, padding=(0, 10, 0, 0))
        self.frequency_parameter_frame.pack(fill=tk.X)
        self.show_spectrum_button = ttk.Button(freq_frame, text="Show Frequency Spectrum", command=self.show_frequency_spectrum, state='disabled')
        self.show_spectrum_button.pack(fill=tk.X, pady=(5,0))

        channel_frame = ttk.LabelFrame(controls_frame, text="4. Channel Extractor", padding="10")
        channel_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(channel_frame, text="Enable", variable=self.channel_enabled, command=self.on_channel_viewer_toggle).pack(anchor='w')
        self.channel_controls_container = ttk.Frame(channel_frame)
        self.channel_controls_container.pack(fill=tk.X, pady=5)
        self._create_channel_controls()

        self.filter_frame = ttk.LabelFrame(controls_frame, text="5. Mask Generation", padding="10")
        self.filter_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self.filter_menubutton = ttk.Menubutton(self.filter_frame, textvariable=self.selected_filter, direction='flush')
        filter_menu = tk.Menu(self.filter_menubutton, tearoff=False)
        for name in self.filter_data.keys():
            filter_menu.add_radiobutton(label=name, variable=self.selected_filter)
        self.filter_menubutton['menu'] = filter_menu
        self.filter_menubutton.pack(fill=tk.X)
        self.parameter_frame = ttk.Frame(self.filter_frame, padding=(0, 10, 0, 0))
        self.parameter_frame.pack(fill=tk.X)

        self.refinement_frame = ttk.LabelFrame(controls_frame, text="6. Mask Refinement", padding="10")
        self.refinement_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self._create_refinement_controls()

        self._create_contour_controls(controls_frame)
        self._create_display_controls(controls_frame)

    def _on_mousewheel(self, event):
        event.widget.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shift_mousewheel(self, event):
        event.widget.xview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def _on_ctrl_mousewheel(self, event):
        if event.widget == self.canvas_right:
            if event.delta > 0: self.zoom_in()
            else: self.zoom_out()

    def _initialize_ui_states(self):
        self.on_preproc_change()
        self.on_enhancement_change()
        self.on_frequency_filter_change()
        self.on_color_space_change()
        self.on_filter_change()
        self.on_edge_filter_change()
        self.on_channel_viewer_toggle()
        self.on_edge_toggle()
        self.on_morph_toggle()
        self.on_contour_toggle()

    def open_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp"), ("All files", "*.*")])
        if not filepath: return
        self.original_cv_image = cv2.imread(filepath)
        if self.original_cv_image is None: 
            messagebox.showerror("Error", f"Failed to open image: {filepath}")
            return
            
        self.save_button.config(state='normal')
        self.reset_button.config(state='normal')
        self.reset_all_to_defaults()
        self.fit_to_screen()

    def reset_all_to_defaults(self):
        if self.original_cv_image is None: return

        self.selected_preproc.set('None')
        self.selected_enhancement.set('None')
        self.selected_frequency_filter.set('None')
        self.channel_enabled.set(False)
        self.selected_color_space.set(list(self.channel_data.keys())[0])
        self.selected_filter.set(list(self.filter_data.keys())[0]) 
        self.edge_enabled.set(False)
        self.selected_edge_filter.set(list(self.edge_detection_data.keys())[0])
        self.morph_enabled.set(False)
        if 'Morph Operation' in self.param_vars:
            self.param_vars['Morph Operation'].set('Dilate')
            self.param_vars['Morph Kernel Shape'].set('Rectangle')
            self.param_vars['Morph Kernel Size'].set(5)
            self.param_vars['Morph Iterations'].set(1)
        self.contours_enabled.set(False)
        self.draw_contours.set(True)
        self.contour_min_area.set(50)
        self.contour_max_area.set(1000000)
        self.display_mode.set("Final Result")
        
        self._initialize_ui_states()
        self.fit_to_screen()

    # --- UI Event Handlers ---
    def on_preproc_change(self, event=None):
        self._create_dynamic_panel(self.preprocessing_data, 'selected_preproc', self.preproc_parameter_frame, self.preproc_param_widgets, 'preproc')
        self.apply_filter()

    def on_enhancement_change(self, event=None):
        self._create_dynamic_panel(self.enhancement_data, 'selected_enhancement', self.enhancement_parameter_frame, self.enhancement_param_widgets, 'enhancement')
        self.apply_filter()
        
    def on_frequency_filter_change(self, event=None):
        self._create_dynamic_panel(self.frequency_data, 'selected_frequency_filter', self.frequency_parameter_frame, self.frequency_param_widgets, 'frequency')
        state = 'normal' if self.selected_frequency_filter.get() != 'None' else 'disabled'
        self.show_spectrum_button.config(state=state)
        self.apply_filter()

    def on_channel_viewer_toggle(self, event=None):
        is_enabled = self.channel_enabled.get()
        channel_state = 'normal' if is_enabled else 'disabled'
        for child in self.channel_controls_container.winfo_children():
            try: child.config(state=channel_state)
            except tk.TclError: pass
        self._update_filter_menu_states()
        
        current_filter_name = self.selected_filter.get()
        current_filter_type = self.filter_data.get(current_filter_name, {}).get('type')
        if is_enabled and current_filter_type == 'color':
            for name, data in self.filter_data.items():
                if data['type'] != 'color':
                    self.selected_filter.set(name)
                    break
        else:
            self.apply_filter()

    def on_color_space_change(self, event=None):
        space = self.selected_color_space.get()
        channels = self.channel_data[space]['channels']
        if hasattr(self, 'channel_menu'):
            menu = self.channel_menu['menu']
            menu.delete(0, 'end')
            for channel in channels:
                menu.add_command(label=channel, command=lambda value=channel: (self.selected_channel.set(value), self.apply_filter()))
            self.selected_channel.set(channels[0])
        self.apply_filter()

    def on_filter_change(self, *args):
        self._build_filter_panel()
        self.apply_filter()

    def on_edge_filter_change(self, event=None):
        self._create_dynamic_panel(self.edge_detection_data, 'selected_edge_filter', self.edge_parameter_frame, self.edge_param_widgets, 'edge')
        self.apply_filter()
        
    def on_edge_toggle(self):
        is_enabled = self.edge_enabled.get()
        edge_state = 'normal' if is_enabled else 'disabled'
        mask_gen_state = 'disabled' if is_enabled else 'normal'

        for child in self.edge_controls_container.winfo_children():
            try: child.config(state=edge_state)
            except tk.TclError: pass
            
        for child in self.filter_frame.winfo_children():
            try: child.config(state=mask_gen_state)
            except tk.TclError: pass
            
        self.apply_filter()
        
    def on_morph_toggle(self):
        state = 'normal' if self.morph_enabled.get() else 'disabled'
        for child in self.morph_controls_container.winfo_children():
            try: child.config(state=state)
            except tk.TclError: pass
        self.apply_filter()

    def on_contour_toggle(self):
        state = 'normal' if self.contours_enabled.get() else 'disabled'
        for child in self.contour_controls_container.winfo_children():
            try: child.config(state=state)
            except tk.TclError: pass
        if not self.contours_enabled.get():
            self.object_count_text.set("Objects Found: --")
        self.apply_filter()

    def _update_filter_menu_states(self):
        is_enabled = self.channel_enabled.get()
        try:
            menu_name = self.filter_menubutton.cget('menu')
            if not menu_name: return
            filter_menu = self.filter_menubutton.nametowidget(menu_name)
        except tk.TclError: return

        for i, name in enumerate(self.filter_data.keys()):
            is_color_filter = self.filter_data[name]['type'] == 'color'
            if is_enabled and is_color_filter:
                filter_menu.entryconfigure(i, state='disabled')
            else:
                filter_menu.entryconfigure(i, state='normal')
    
    # --- UI BUILDER METHODS ---
    def _create_channel_controls(self):
        frame1 = self._create_param_row('Color Space', parent=self.channel_controls_container)
        space_menu = ttk.OptionMenu(frame1, self.selected_color_space, self.selected_color_space.get(), *self.channel_data.keys(), command=self.on_color_space_change)
        space_menu.grid(row=0, column=1, sticky='ew')
        
        frame2 = self._create_param_row('Channel', parent=self.channel_controls_container)
        self.channel_menu = ttk.OptionMenu(frame2, self.selected_channel, "")
        self.channel_menu.grid(row=0, column=1, sticky='ew')
        
    def _create_refinement_controls(self):
        self.edge_frame = ttk.LabelFrame(self.refinement_frame, text="Edge Detection (Overrides Mask Generation)", padding=5)
        self.edge_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(self.edge_frame, text="Enable", variable=self.edge_enabled, command=self.on_edge_toggle).pack(anchor='w')
        self.edge_controls_container = ttk.Frame(self.edge_frame)
        self.edge_controls_container.pack(fill=tk.X)
        self.edge_filter_menu = ttk.OptionMenu(self.edge_controls_container, self.selected_edge_filter, self.selected_edge_filter.get(), *self.edge_detection_data.keys(), command=self.on_edge_filter_change)
        self.edge_filter_menu.pack(fill=tk.X, pady=(5,0))
        self.edge_parameter_frame = ttk.Frame(self.edge_controls_container, padding=(0, 10, 0, 0))
        self.edge_parameter_frame.pack(fill=tk.X)

        self.morph_frame = ttk.LabelFrame(self.refinement_frame, text="Morphological Operations", padding=5)
        self.morph_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(self.morph_frame, text="Enable", variable=self.morph_enabled, command=self.on_morph_toggle).pack(anchor='w')
        self.morph_controls_container = ttk.Frame(self.morph_frame)
        self.morph_controls_container.pack(fill=tk.X)
        self._create_morph_controls()

    def _create_contour_controls(self, parent):
        contour_frame = ttk.LabelFrame(parent, text="7. Contour Analysis", padding="10")
        contour_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(contour_frame, text="Enable", variable=self.contours_enabled, command=self.on_contour_toggle).pack(anchor='w')
        self.contour_controls_container = ttk.Frame(contour_frame)
        self.contour_controls_container.pack(fill=tk.X)
        
        ttk.Checkbutton(self.contour_controls_container, text="Draw Contours on Image", variable=self.draw_contours, command=self.apply_filter).pack(anchor='w', pady=(5,0))
        
        min_area_frame = self._create_param_row("Min Area", parent=self.contour_controls_container)
        self._create_slider_and_entry(min_area_frame, self.contour_min_area, 0, 50000)
        
        max_area_frame = self._create_param_row("Max Area", parent=self.contour_controls_container)
        self._create_slider_and_entry(max_area_frame, self.contour_max_area, 0, 1000000)
        
        ttk.Label(self.contour_controls_container, textvariable=self.object_count_text, font=("Helvetica", 10, "bold")).pack(pady=5)

    def _create_display_controls(self, parent):
        display_frame = ttk.LabelFrame(parent, text="Display Options", padding="10")
        display_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        options = ["Final Result", "Binary Mask", "Enhanced Image", "Pre-processed Image", "Extracted Channel"]
        for option in options:
            rb = ttk.Radiobutton(display_frame, text=option, variable=self.display_mode, value=option, command=self.apply_filter)
            rb.pack(anchor='w', side=tk.LEFT, expand=True)
            if option == "Extracted Channel": self.channel_display_rb = rb
            if option == "Enhanced Image": self.enhanced_display_rb = rb

        hist_button = ttk.Button(display_frame, text="Show Histogram", command=self.show_histogram)
        hist_button.pack(anchor='e', expand=True)
        if not MATPLOTLIB_AVAILABLE: hist_button.config(state='disabled')

    def _build_filter_panel(self):
        config = self.filter_data.get(self.selected_filter.get(), {})
        for widget in self.param_widgets: widget.destroy()
        self.param_widgets.clear()
        for k in list(self.param_vars.keys()):
            if k.startswith('filter_') or k.startswith('color_') or k.startswith('otsu_'):
                if k in self.param_vars: del self.param_vars[k]
        if config.get('type') == 'color':
             for i, (channel, (min_val, max_val)) in enumerate(zip(config['channels'], config['ranges'])):
                frame = self._create_param_row(f"{channel} Min", "Max", parent=self.parameter_frame)
                self.param_widgets.append(frame)
                min_var = tk.IntVar(value=min_val); self.param_vars[f'color_{channel}_min'] = min_var
                slider1, entry1 = self._create_slider_and_entry(frame, min_var, min_val, max_val)
                self.param_widgets.extend([slider1, entry1])
                max_var = tk.IntVar(value=max_val); self.param_vars[f'color_{channel}_max'] = max_var
                slider2, entry2 = self._create_slider_and_entry(frame, max_var, min_val, max_val, column_offset=3)
                self.param_widgets.extend([slider2, entry2])
        elif config.get('type') == 'otsu':
            params = config['params']; p_name = 'Threshold Type'; p_data = params[p_name]
            frame1 = self._create_param_row(p_name, parent=self.parameter_frame); self.param_widgets.append(frame1)
            var = tk.StringVar(value=p_data['default']); self.param_vars[f'otsu_{p_name.replace(" ", "_")}'] = var
            menu = ttk.OptionMenu(frame1, var, p_data['default'], *p_data['options'], command=self.apply_filter)
            menu.grid(row=0, column=1, sticky='ew'); self.param_widgets.append(menu)
            frame2 = self._create_param_row("Calculated Threshold:", parent=self.parameter_frame); self.param_widgets.append(frame2)
            var = tk.StringVar(value="--")
            self.param_vars['otsu_Calculated_Threshold'] = var
            value_label = ttk.Label(frame2, textvariable=var, font=("Helvetica", 10, "bold"))
            value_label.grid(row=0, column=1)
        else:
             self._create_dynamic_panel(self.filter_data, 'selected_filter', self.parameter_frame, self.param_widgets, 'filter')

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
            
            if 'options' in p_data:
                var = tk.StringVar(value=p_data['default'])
                menu = ttk.OptionMenu(frame, var, p_data['default'], *p_data['options'], command=self.apply_filter)
                menu.grid(row=0, column=1, sticky='ew')
                widget_list.append(menu)
            else:
                var = tk.IntVar(value=p_data['default'])
                slider, entry = self._create_slider_and_entry(frame, var, p_data['range'][0], p_data['range'][1])
                widget_list.extend([slider, entry])
            self.param_vars[var_key] = var
            
    def _create_morph_controls(self):
        p_name = 'Morph Operation'; options = ['Dilate', 'Erode', 'Open', 'Close', 'Gradient', 'Top Hat', 'Black Hat']
        frame = self._create_param_row(p_name, parent=self.morph_controls_container); self.morph_param_widgets.append(frame)
        var = tk.StringVar(value=options[0]); self.param_vars[p_name] = var
        menu = ttk.OptionMenu(frame, var, options[0], *options, command=self.apply_filter)
        menu.grid(row=0, column=1, sticky='ew'); self.morph_param_widgets.append(menu)
        
        p_name = 'Morph Kernel Shape'; options = ['Rectangle', 'Ellipse', 'Cross']
        frame = self._create_param_row('Kernel Shape', parent=self.morph_controls_container); self.morph_param_widgets.append(frame)
        var = tk.StringVar(value=options[0]); self.param_vars[p_name] = var
        menu = ttk.OptionMenu(frame, var, options[0], *options, command=self.apply_filter)
        menu.grid(row=0, column=1, sticky='ew'); self.morph_param_widgets.append(menu)
        
        p_name = 'Morph Kernel Size'; frame = self._create_param_row('Kernel Size', parent=self.morph_controls_container); self.morph_param_widgets.append(frame)
        var = tk.IntVar(value=5); self.param_vars[p_name] = var
        slider, entry = self._create_slider_and_entry(frame, var, 1, 51); self.morph_param_widgets.extend([slider, entry])
        
        p_name = 'Morph Iterations'; frame = self._create_param_row('Iterations', parent=self.morph_controls_container); self.morph_param_widgets.append(frame)
        var = tk.IntVar(value=1); self.param_vars[p_name] = var
        slider, entry = self._create_slider_and_entry(frame, var, 1, 20); self.morph_param_widgets.extend([slider, entry])

    def apply_filter(self, event=None):
        if self.original_cv_image is None: return
        try:
            preprocessed_image = self._process_preprocessing(self.original_cv_image)
            enhanced_image = self._process_enhancement(preprocessed_image)
            freq_filtered_image = self._process_frequency_filter(enhanced_image)
            
            image_for_masking = freq_filtered_image
            extracted_channel = None
            if self.channel_enabled.get():
                image_for_masking = self._process_channel_extraction(freq_filtered_image)
                extracted_channel = image_for_masking.copy()
                if hasattr(self, 'channel_display_rb'): self.channel_display_rb.config(state='normal')
            else:
                if hasattr(self, 'channel_display_rb'): self.channel_display_rb.config(state='disabled')

            if self.selected_enhancement.get() != 'None' or self.selected_frequency_filter.get() != 'None':
                if hasattr(self, 'enhanced_display_rb'): self.enhanced_display_rb.config(state='normal')
            else:
                if hasattr(self, 'enhanced_display_rb'): self.enhanced_display_rb.config(state='disabled')

            if self.edge_enabled.get():
                gray_for_edge = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY) if len(enhanced_image.shape) == 3 else enhanced_image
                mask = self._process_edge_detection(gray_for_edge)
            else:
                mask = self._process_mask_generation(image_for_masking)
            
            if mask is None: return

            if self.morph_enabled.get():
                mask = self._process_morph_ops(mask)
            
            final_image = cv2.bitwise_and(preprocessed_image, preprocessed_image, mask=mask)
            if self.contours_enabled.get():
                final_image = self._process_contours(mask, final_image.copy())

            mode = self.display_mode.get()
            if mode == "Final Result": self.update_image_display(final_image)
            elif mode == "Binary Mask": self.update_image_display(mask)
            elif mode == "Enhanced Image": self.update_image_display(enhanced_image)
            elif mode == "Pre-processed Image": self.update_image_display(preprocessed_image)
            elif mode == "Extracted Channel" and extracted_channel is not None: self.update_image_display(extracted_channel)
            else: self.update_image_display(final_image)

        except (tk.TclError, KeyError, ValueError, IndexError) as e:
            pass

    def _process_preprocessing(self, image):
        preproc_type = self.selected_preproc.get()
        if preproc_type == 'None': return image
        if 'preproc_Kernel_Size' not in self.param_vars: return image
        ksize = int(self.param_vars.get('preproc_Kernel_Size', tk.IntVar(value=0)).get())
        if ksize > 0 and ksize % 2 == 0: ksize += 1
        if preproc_type == 'Gaussian Blur': return cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif preproc_type == 'Median Blur': return cv2.medianBlur(image, ksize)
        elif preproc_type == 'Bilateral Filter':
            d = int(self.param_vars['preproc_Diameter'].get()); sc = int(self.param_vars['preproc_Sigma_Color'].get()); ss = int(self.param_vars['preproc_Sigma_Space'].get())
            return cv2.bilateralFilter(image, d, sc, ss)
        return image

    def _process_enhancement(self, image):
        enhance_type = self.selected_enhancement.get()
        if enhance_type == 'None': return image

        if enhance_type == 'Histogram Equalization':
            if len(image.shape) == 2: return cv2.equalizeHist(image)
            img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
            return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

        elif enhance_type == 'CLAHE':
            if 'enhancement_Clip_Limit' not in self.param_vars: return image
            clip = float(self.param_vars['enhancement_Clip_Limit'].get())
            tile = int(self.param_vars['enhancement_Tile_Grid_Size'].get())
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            if len(image.shape) == 2: return clahe.apply(image)
            img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            img_ycrcb[:, :, 0] = clahe.apply(img_ycrcb[:, :, 0])
            return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

        elif enhance_type == 'Contrast Stretching':
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        elif enhance_type == 'Gamma Correction':
            if 'enhancement_Gamma_x100' not in self.param_vars: return image
            gamma = self.param_vars['enhancement_Gamma_x100'].get() / 100.0
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)

        elif enhance_type == 'Log Transform':
            max_val = np.max(image)
            if max_val == 0: return image
            c = 255 / np.log(1 + max_val)
            log_image = c * (np.log(image.astype(np.float32) + 1))
            return np.array(log_image, dtype=np.uint8)

        elif enhance_type == 'Single-Scale Retinex':
            if 'enhancement_Sigma' not in self.param_vars: return image
            sigma = self.param_vars['enhancement_Sigma'].get()
            img_float = image.astype(np.float32) + 1.0
            L = cv2.GaussianBlur(img_float, (0,0), sigma)
            R_log = np.log(img_float) - np.log(L)
            R = cv2.normalize(R_log, None, 0, 255, cv2.NORM_MINMAX)
            return R.astype(np.uint8)

        elif enhance_type == 'Unsharp Masking':
            if 'enhancement_Kernel_Size' not in self.param_vars: return image
            ksize = self.param_vars['enhancement_Kernel_Size'].get()
            if ksize > 0 and ksize % 2 == 0: ksize += 1
            alpha = self.param_vars['enhancement_Alpha_x10'].get() / 10.0
            blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
            return cv2.addWeighted(image, 1 + alpha, blurred, -alpha, 0)
            
        return image

    def _process_frequency_filter(self, image):
        filter_type = self.selected_frequency_filter.get()
        if filter_type == 'None': return image

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        rows, cols = gray.shape
        m_rows, m_cols = cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols)
        padded = cv2.copyMakeBorder(gray, 0, m_rows - rows, 0, m_cols - cols, cv2.BORDER_CONSTANT, value=0)
        
        planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
        complex_i = cv2.merge(planes)
        cv2.dft(complex_i, complex_i)
        
        dft_shift = np.fft.fftshift(complex_i)
        
        D0 = self.param_vars.get('frequency_Cutoff_Freq_(D0)', tk.IntVar(value=30)).get()
        n = self.param_vars.get('frequency_Order_(n)', tk.IntVar(value=2)).get()
        
        mask = self._create_frequency_filter_mask((m_rows, m_cols), filter_type, D0, n)
        mask_complex = cv2.merge([mask, mask])
        
        fshift = dft_shift * mask_complex
        
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        result = np.uint8(img_back)
        result = result[:rows, :cols]

        if len(image.shape) == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result

    def _create_frequency_filter_mask(self, shape, filter_type, D0, n=2):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.float32)
        
        for i in range(rows):
            for j in range(cols):
                D = np.sqrt((i - crow)**2 + (j - ccol)**2)
                
                if 'Low-Pass' in filter_type:
                    if filter_type == 'Ideal Low-Pass':
                        if D <= D0: mask[i, j] = 1
                    elif filter_type == 'Gaussian Low-Pass':
                        mask[i, j] = np.exp(-(D**2) / (2 * D0**2))
                    elif filter_type == 'Butterworth Low-Pass':
                        mask[i, j] = 1 / (1 + (D / D0)**(2*n))
                
                elif 'High-Pass' in filter_type:
                    if filter_type == 'Ideal High-Pass':
                        if D > D0: mask[i, j] = 1
                    elif filter_type == 'Gaussian High-Pass':
                        mask[i, j] = 1 - np.exp(-(D**2) / (2 * D0**2))
                    elif filter_type == 'Butterworth High-Pass':
                        if D != 0: mask[i, j] = 1 / (1 + (D0 / D)**(2*n))
        return mask

    def _process_channel_extraction(self, image):
        space_name = self.selected_color_space.get()
        channel_name = self.selected_channel.get()
        if not channel_name: return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        space_info = self.channel_data[space_name]
        
        if len(image.shape) == 3 and space_info['code'] is not None: 
            converted_image = cv2.cvtColor(image, space_info['code'])
        else: 
            converted_image = image

        if converted_image.ndim == 2: return converted_image
            
        channels = cv2.split(converted_image)
        channel_index = space_info['channels'].index(channel_name)
        return channels[channel_index]

    def _process_mask_generation(self, image_for_masking):
        filter_selection = self.selected_filter.get()
        filter_type = self.filter_data.get(filter_selection, {}).get('type')

        if image_for_masking.ndim == 3: gray = cv2.cvtColor(image_for_masking, cv2.COLOR_BGR2GRAY)
        else: gray = image_for_masking.copy()

        if filter_type == 'grayscale_range':
            if 'filter_Min_Value' not in self.param_vars: return gray
            min_v = int(self.param_vars['filter_Min_Value'].get()); max_v = int(self.param_vars['filter_Max_Value'].get())
            if min_v > max_v: min_v = max_v; self.param_vars['filter_Min_Value'].set(min_v)
            return cv2.inRange(gray, min_v, max_v)
        elif filter_type == 'adaptive_thresh':
            if 'filter_Block_Size' not in self.param_vars: return gray
            method = cv2.ADAPTIVE_THRESH_MEAN_C if self.param_vars['filter_Adaptive_Method'].get() == 'Mean C' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            type_f = cv2.THRESH_BINARY if self.param_vars['filter_Threshold_Type'].get() == 'Binary' else cv2.THRESH_BINARY_INV
            bsize = int(self.param_vars['filter_Block_Size'].get()); c = int(self.param_vars['filter_C_(Constant)'].get())
            if bsize % 2 == 0: bsize += 1; self.param_vars['filter_Block_Size'].set(bsize)
            return cv2.adaptiveThreshold(gray, 255, method, type_f, bsize, c)
        elif filter_type == 'otsu':
            if 'otsu_Threshold_Type' not in self.param_vars: return gray
            type_f = cv2.THRESH_BINARY if self.param_vars['otsu_Threshold_Type'].get() == 'Binary' else cv2.THRESH_BINARY_INV
            ret, mask = cv2.threshold(gray, 0, 255, type_f + cv2.THRESH_OTSU)
            if 'otsu_Calculated_Threshold' in self.param_vars: self.param_vars['otsu_Calculated_Threshold'].set(str(int(ret)))
            return mask
        elif filter_type == 'color':
            if image_for_masking.ndim == 2: h, w = image_for_masking.shape[:2]; return np.zeros((h, w), dtype=np.uint8)
            lower = np.array([self.param_vars[f'color_{c}_min'].get() for c in self.filter_data[filter_selection]['channels']])
            upper = np.array([self.param_vars[f'color_{c}_max'].get() for c in self.filter_data[filter_selection]['channels']])
            conv_map = {'RGB/BGR (Color Filter)': -1, 'HSV': cv2.COLOR_BGR2HSV, 'HLS': cv2.COLOR_BGR2HLS, 'Lab': cv2.COLOR_BGR2LAB, 'YCrCb': cv2.COLOR_BGR2YCrCb}
            code = conv_map[filter_selection]
            converted = cv2.cvtColor(image_for_masking, code) if code != -1 else image_for_masking
            return cv2.inRange(converted, lower, upper)
        else: h, w = image_for_masking.shape[:2]; return np.zeros((h, w), dtype=np.uint8)

    def _process_edge_detection(self, image_for_edge):
        if 'edge_Kernel_Size' not in self.param_vars: return image_for_edge
        edge_type = self.selected_edge_filter.get()
        if edge_type == 'Canny':
            t1 = int(self.param_vars['edge_Threshold_1'].get()); t2 = int(self.param_vars['edge_Threshold_2'].get())
            return cv2.Canny(image_for_edge, t1, t2)
        sobel_depth = cv2.CV_64F
        if edge_type == 'Sobel':
            ksize = int(self.param_vars['edge_Kernel_Size'].get());
            if ksize % 2 == 0: ksize += 1; self.param_vars['edge_Kernel_Size'].set(ksize)
            direction = self.param_vars['edge_Direction'].get()
            if direction == 'X': return cv2.convertScaleAbs(cv2.Sobel(image_for_edge, sobel_depth, 1, 0, ksize=ksize))
            if direction == 'Y': return cv2.convertScaleAbs(cv2.Sobel(image_for_edge, sobel_depth, 0, 1, ksize=ksize))
            grad_x = cv2.Sobel(image_for_edge, sobel_depth, 1, 0, ksize=ksize); grad_y = cv2.Sobel(image_for_edge, sobel_depth, 0, 1, ksize=ksize)
            return cv2.convertScaleAbs(np.sqrt(grad_x**2 + grad_y**2))
        kernels = {'Prewitt_X': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), 'Prewitt_Y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), 'Roberts_X': np.array([[1, 0], [0, -1]]), 'Roberts_Y': np.array([[0, 1], [-1, 0]])}
        if edge_type == 'Prewitt' or edge_type == 'Roberts':
            direction = self.param_vars[f'edge_Direction'].get()
            if direction == 'X': return cv2.convertScaleAbs(cv2.filter2D(image_for_edge, -1, kernels[f'{edge_type}_X']))
            if direction == 'Y': return cv2.convertScaleAbs(cv2.filter2D(image_for_edge, -1, kernels[f'{edge_type}_Y']))
            grad_x = cv2.filter2D(image_for_edge, -1, kernels[f'{edge_type}_X']); grad_y = cv2.filter2D(image_for_edge, -1, kernels[f'{edge_type}_Y'])
            return cv2.convertScaleAbs(np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2))
        return image_for_edge

    def _process_morph_ops(self, input_mask):
        op_str = self.param_vars['Morph Operation'].get(); shape_str = self.param_vars['Morph Kernel Shape'].get()
        k_size = int(self.param_vars['Morph Kernel Size'].get()); iterations = int(self.param_vars['Morph Iterations'].get())
        if k_size > 0 and k_size % 2 == 0: k_size += 1; self.param_vars['Morph Kernel Size'].set(k_size)
        shape_map = {'Rectangle': cv2.MORPH_RECT, 'Ellipse': cv2.MORPH_ELLIPSE, 'Cross': cv2.MORPH_CROSS}
        op_map = {'Dilate': cv2.MORPH_DILATE, 'Erode': cv2.MORPH_ERODE, 'Open': cv2.MORPH_OPEN, 'Close': cv2.MORPH_CLOSE, 'Gradient': cv2.MORPH_GRADIENT, 'Top Hat': cv2.MORPH_TOPHAT, 'Black Hat': cv2.MORPH_BLACKHAT}
        kernel = cv2.getStructuringElement(shape_map[shape_str], (k_size, k_size))
        return cv2.morphologyEx(input_mask, op_map[op_str], kernel, iterations=iterations)
    
    def _process_contours(self, mask, image_to_draw_on):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = self.contour_min_area.get()
        max_area = self.contour_max_area.get()
        
        filtered_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
        
        self.object_count_text.set(f"Objects Found: {len(filtered_contours)}")
        
        if self.draw_contours.get():
            cv2.drawContours(image_to_draw_on, filtered_contours, -1, (0, 255, 0), 2)
            
        return image_to_draw_on

    def _create_param_row(self, label1, label2=None, parent=None):
        if parent is None: parent = self.parameter_frame
        frame = ttk.Frame(parent); frame.pack(fill=tk.X, pady=2)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label1).grid(row=0, column=0, sticky='w', padx=5)
        if label2:
            frame.columnconfigure(4, weight=1)
            ttk.Label(frame, text=label2).grid(row=0, column=3, padx=(10, 0), sticky='w')
        return frame

    def _create_slider_and_entry(self, parent, variable, min_val, max_val, column_offset=0):
        def slider_command(value):
            variable.set(round(float(value))); self.apply_filter()
        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=variable, command=slider_command)
        slider.grid(row=0, column=column_offset + 1, sticky='ew', padx=5)
        entry = ttk.Entry(parent, textvariable=variable, width=7)
        entry.grid(row=0, column=column_offset + 2)
        entry.bind("<Return>", self.apply_filter)
        return slider, entry

    def update_image_display(self, cv_image):
        self.processed_cv_image = cv_image
        self.display_zoomed_image()
    
    def display_zoomed_image(self):
        if self.processed_cv_image is None: return

        h, w = self.processed_cv_image.shape[:2]
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)

        if new_w < 1 or new_h < 1: return

        zoomed_img = cv2.resize(self.processed_cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(zoomed_img.shape) == 2: rgb_image = cv2.cvtColor(zoomed_img, cv2.COLOR_GRAY2RGB)
        else: rgb_image = cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(rgb_image)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        
        self.canvas_right.itemconfig(self.image_on_canvas, image=self.tk_image)
        
        canvas_w = self.canvas_right.winfo_width()
        canvas_h = self.canvas_right.winfo_height()
        
        x_pos = max(0, (canvas_w - new_w) / 2)
        y_pos = max(0, (canvas_h - new_h) / 2)
        
        self.canvas_right.coords(self.image_on_canvas, x_pos, y_pos)
        self.canvas_right.config(scrollregion=self.canvas_right.bbox("all"))
        self.zoom_level_text.set(f"Zoom: {self.zoom_factor:.0%}")

    def zoom_in(self):
        self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.2)
        self.display_zoomed_image()

    def zoom_out(self):
        self.zoom_factor = max(self.min_zoom, self.zoom_factor / 1.2)
        self.display_zoomed_image()

    def fit_to_screen(self):
        if self.original_cv_image is None: return
        
        canvas_w = self.canvas_right.winfo_width()
        canvas_h = self.canvas_right.winfo_height()
        
        if canvas_w < 2 or canvas_h < 2:
            self.root.after(50, self.fit_to_screen)
            return

        img_h, img_w = self.original_cv_image.shape[:2]
        
        w_ratio = canvas_w / img_w
        h_ratio = canvas_h / img_h
        
        self.zoom_factor = min(w_ratio, h_ratio, 1.0) 
        self.apply_filter()

    def save_image(self):
        if self.processed_cv_image is None: messagebox.showwarning("No Image", "There is no image to save."); return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("BMP Image", "*.bmp"), ("All Files", "*.*")])
        if not filepath: return
        try:
            cv2.imwrite(filepath, self.processed_cv_image)
            messagebox.showinfo("Success", f"Image saved successfully to:\n{filepath}")
        except Exception as e: messagebox.showerror("Save Error", f"Could not save the image.\nError: {e}")

    def save_settings(self):
        settings = {}
        for var_name in dir(self):
            var = getattr(self, var_name)
            if isinstance(var, (tk.StringVar, tk.IntVar, tk.BooleanVar)):
                settings[var_name] = var.get()
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not filepath: return
        try:
            with open(filepath, 'w') as f: json.dump(settings, f, indent=4)
            messagebox.showinfo("Success", "Settings saved successfully.")
        except Exception as e: messagebox.showerror("Save Error", f"Could not save settings.\nError: {e}")

    def load_settings(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not filepath: return
        try:
            with open(filepath, 'r') as f: settings = json.load(f)
        except Exception as e: messagebox.showerror("Load Error", f"Could not load settings file.\nError: {e}"); return
        
        for var_name, value in settings.items():
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                if isinstance(var, (tk.StringVar, tk.IntVar, tk.BooleanVar)):
                    try: var.set(value)
                    except tk.TclError: pass
        
        self._initialize_ui_states()
        self.apply_filter()

    def show_histogram(self):
        if not MATPLOTLIB_AVAILABLE: messagebox.showerror("Missing Library", "Matplotlib is required for this feature."); return
        if self.original_cv_image is None: messagebox.showwarning("No Image", "Open an image first."); return
        
        try:
            preprocessed = self._process_preprocessing(self.original_cv_image)
            enhanced = self._process_enhancement(preprocessed)
            freq_filtered = self._process_frequency_filter(enhanced)
            
            image_for_hist = freq_filtered
            title = "Histogram of Frequency Filtered Image"
            if self.channel_enabled.get():
                image_for_hist = self._process_channel_extraction(freq_filtered)
                title = f"Histogram of '{self.selected_channel.get()}' Channel"

            if image_for_hist.ndim == 3:
                image_to_hist = cv2.cvtColor(image_for_hist, cv2.COLOR_BGR2GRAY)
            else:
                image_to_hist = image_for_hist
                
            hist = cv2.calcHist([image_to_hist], [0], None, [256], [0, 256])

            hist_window = tk.Toplevel(self.root); hist_window.title(title)
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(hist); ax.set_xlim([0, 256]); ax.set_xlabel("Pixel Intensity"); ax.set_ylabel("Pixel Count"); ax.grid()
            canvas = FigureCanvasTkAgg(fig, master=hist_window); canvas.draw(); canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        except Exception as e:
            messagebox.showerror("Histogram Error", f"Could not generate histogram.\nError: {e}")
            
    def show_frequency_spectrum(self):
        if not MATPLOTLIB_AVAILABLE: messagebox.showerror("Missing Library", "Matplotlib is required for this feature."); return
        if self.original_cv_image is None: messagebox.showwarning("No Image", "Open an image first."); return
        
        try:
            preprocessed = self._process_preprocessing(self.original_cv_image)
            enhanced = self._process_enhancement(preprocessed)
            
            if len(enhanced.shape) == 3:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            else:
                gray = enhanced

            rows, cols = gray.shape
            m_rows, m_cols = cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols)
            padded = cv2.copyMakeBorder(gray, 0, m_rows - rows, 0, m_cols - cols, cv2.BORDER_CONSTANT, value=0)
            
            dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            
            filter_type = self.selected_frequency_filter.get()
            D0 = self.param_vars.get('frequency_Cutoff_Freq_(D0)', tk.IntVar(value=30)).get()
            n = self.param_vars.get('frequency_Order_(n)', tk.IntVar(value=2)).get()
            filter_mask = self._create_frequency_filter_mask((m_rows, m_cols), filter_type, D0, n)
            
            spec_window = tk.Toplevel(self.root)
            spec_window.title("Frequency Spectrum and Filter")
            
            fig = Figure(figsize=(8, 4), dpi=100)
            ax1 = fig.add_subplot(121)
            ax1.imshow(magnitude_spectrum, cmap='gray')
            ax1.set_title('Magnitude Spectrum')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            ax2 = fig.add_subplot(122)
            ax2.imshow(filter_mask, cmap='gray')
            ax2.set_title(f'{filter_type} Mask')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            canvas = FigureCanvasTkAgg(fig, master=spec_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        except Exception as e:
            messagebox.showerror("Spectrum Error", f"Could not generate spectrum.\nError: {e}")

def main():
    """The main entry point for the application."""
    root = tk.Tk()
    # Add a modern theme
    style = ttk.Style(root)
    try:
        # 'clam' is a good cross-platform theme
        style.theme_use('clam')
    except tk.TclError:
        # Fallback if theme is not available
        pass
    app = AdvancedFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()