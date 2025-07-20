import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json

# Optional import for the histogram feature
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class AdvancedFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AlchemyCV - Vision 7 (Patched)")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        # --- Member Variables ---
        self.original_cv_image = None
        self.processed_cv_image = None
        self.tk_image = None
        self.pil_image_fullres = None # Full resolution PIL image for zooming
        self.intermediate_images = {} # Store intermediate pipeline images
        
        # --- QoL Variables ---
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 8.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.display_mode = tk.StringVar(value="Final Result")
        self.zoom_level_text = tk.StringVar(value="Zoom: 100%")

        # --- DATA STRUCTURES (Same as previous version) ---
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
        }
        self.morph_ops_data = {
            'Erode': {'var': tk.BooleanVar(value=False), 'kernel_var': tk.IntVar(value=3)},
            'Dilate': {'var': tk.BooleanVar(value=False), 'kernel_var': tk.IntVar(value=3)},
            'Open': {'var': tk.BooleanVar(value=False), 'kernel_var': tk.IntVar(value=3)},
            'Close': {'var': tk.BooleanVar(value=False), 'kernel_var': tk.IntVar(value=3)},
        }

        # --- TKinter Variables ---
        self.selected_preproc = tk.StringVar(value='None')
        self.selected_enhancement = tk.StringVar(value='None')
        self.channel_enabled = tk.BooleanVar(value=False)
        self.selected_color_space = tk.StringVar(value=list(self.channel_data.keys())[0])
        self.selected_channel = tk.StringVar()
        self.selected_filter = tk.StringVar(value=list(self.filter_data.keys())[0])
        self.param_vars = {}
        self.edge_enabled = tk.BooleanVar(value=False)
        self.selected_edge_filter = tk.StringVar(value='Canny')
        self.morph_enabled = tk.BooleanVar(value=False)
        self.contours_enabled = tk.BooleanVar(value=False)
        self.draw_contours = tk.BooleanVar(value=True)
        self.contour_min_area = tk.IntVar(value=50)
        self.object_count_text = tk.StringVar(value="Objects Found: --")
        
        # --- Widget Storage ---
        self.preproc_param_widgets, self.enhancement_param_widgets, self.param_widgets, self.edge_param_widgets, self.morph_param_widgets = [], [], [], [], []

        self._create_widgets()
        self._initialize_ui_states()
        self.selected_filter.trace_add('write', self.on_filter_change)

    def _create_widgets(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(controls_frame, width=480, stretch="never")
        
        right_container = ttk.Frame(main_pane)
        right_container.grid_rowconfigure(0, weight=1)
        right_container.grid_columnconfigure(0, weight=1)
        self.canvas_right = tk.Canvas(right_container, background="gray")
        self.canvas_right.grid(row=0, column=0, sticky="nsew")
        self.image_on_canvas = self.canvas_right.create_image(0, 0, anchor="center")
        main_pane.add(right_container)

        # --- Top Buttons ---
        top_button_frame = ttk.Frame(controls_frame)
        top_button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Button(top_button_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=2)
        self.save_button = ttk.Button(top_button_frame, text="Save Image", command=self.save_image, state='disabled')
        self.save_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(top_button_frame, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_button_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=2)

        # --- Canvas Tools (Zoom, etc.) ---
        canvas_tools_frame = ttk.Frame(right_container)
        canvas_tools_frame.grid(row=1, column=0, sticky='ew', pady=5)
        ttk.Button(canvas_tools_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        ttk.Button(canvas_tools_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        ttk.Button(canvas_tools_frame, text="Fit to Screen", command=self.fit_to_screen).pack(side=tk.LEFT, padx=5)
        ttk.Label(canvas_tools_frame, textvariable=self.zoom_level_text).pack(side=tk.LEFT, padx=10)
        ttk.Button(canvas_tools_frame, text="Show Histogram", command=self.show_histogram).pack(side=tk.RIGHT, padx=5)

        # --- Control Panels ---
        preproc_frame = ttk.LabelFrame(controls_frame, text="1. Pre-processing", padding="10"); preproc_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        enhancement_frame = ttk.LabelFrame(controls_frame, text="2. Image Enhancement", padding="10"); enhancement_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        channel_frame = ttk.LabelFrame(controls_frame, text="3. Channel Extractor", padding="10"); channel_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self.filter_frame = ttk.LabelFrame(controls_frame, text="4. Mask Generation", padding="10"); self.filter_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self.refinement_frame = ttk.LabelFrame(controls_frame, text="5. Mask Refinement", padding="10"); self.refinement_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self._create_contour_controls(controls_frame)
        self._create_display_options(controls_frame)

        # Populate panels
        ttk.OptionMenu(preproc_frame, self.selected_preproc, self.selected_preproc.get(), *self.preprocessing_data.keys(), command=self.on_preproc_change).pack(fill=tk.X)
        self.preproc_parameter_frame = ttk.Frame(preproc_frame, padding=(0, 10, 0, 0)); self.preproc_parameter_frame.pack(fill=tk.X)
        
        ttk.OptionMenu(enhancement_frame, self.selected_enhancement, self.selected_enhancement.get(), *self.enhancement_data.keys(), command=self.on_enhancement_change).pack(fill=tk.X)
        self.enhancement_parameter_frame = ttk.Frame(enhancement_frame, padding=(0, 10, 0, 0)); self.enhancement_parameter_frame.pack(fill=tk.X)
        
        ttk.Checkbutton(channel_frame, text="Enable", variable=self.channel_enabled, command=self.on_channel_viewer_toggle).pack(anchor='w')
        self.channel_controls_container = ttk.Frame(channel_frame); self.channel_controls_container.pack(fill=tk.X, pady=5)
        self._create_channel_controls()
        
        self.filter_menubutton = ttk.Menubutton(self.filter_frame, textvariable=self.selected_filter, direction='flush')
        filter_menu = tk.Menu(self.filter_menubutton, tearoff=False); self.filter_menubutton['menu'] = filter_menu
        for name in self.filter_data.keys(): filter_menu.add_radiobutton(label=name, variable=self.selected_filter)
        self.filter_menubutton.pack(fill=tk.X)
        self.parameter_frame = ttk.Frame(self.filter_frame, padding=(0, 10, 0, 0)); self.parameter_frame.pack(fill=tk.X)
        
        self._create_refinement_controls()
        
        # --- Bindings for Pan and Zoom ---
        self.canvas_right.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.canvas_right.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas_right.bind("<B1-Motion>", self._on_canvas_drag)

    def _initialize_ui_states(self):
        self.on_preproc_change()
        self.on_enhancement_change()
        self.on_color_space_change()
        self.on_filter_change()
        self.on_edge_filter_change()
        self.on_edge_toggle()
        self.on_morph_toggle()
        self.on_contour_toggle()

    def open_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif"), ("All files", "*.*")])
        if not filepath: return
        self.original_cv_image = cv2.imread(filepath)
        if self.original_cv_image is None:
            messagebox.showerror("Error", f"Failed to open image: {filepath}")
            return
        # Wait for canvas to be drawn to get correct dimensions for fit_to_screen
        self.root.after(100, self.fit_to_screen) 
        self.apply_filter()
        self.save_button.config(state='normal')

    def save_image(self):
        if self.intermediate_images.get("Final Result") is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if not filepath: return
        cv2.imwrite(filepath, self.intermediate_images["Final Result"])
        messagebox.showinfo("Success", f"Image saved to {filepath}")

    # --- UI Event Handlers ---
    def on_preproc_change(self, event=None): self._create_dynamic_panel(self.preprocessing_data, 'selected_preproc', self.preproc_parameter_frame, self.preproc_param_widgets, 'preproc'); self.apply_filter()
    def on_enhancement_change(self, event=None): self._create_dynamic_panel(self.enhancement_data, 'selected_enhancement', self.enhancement_parameter_frame, self.enhancement_param_widgets, 'enhancement'); self.apply_filter()
    def on_channel_viewer_toggle(self, event=None): self.update_widget_state(self.channel_controls_container, self.channel_enabled.get()); self.apply_filter()
    def on_color_space_change(self, event=None):
        space = self.selected_color_space.get(); channels = self.channel_data[space]['channels']
        if hasattr(self, 'channel_menu'):
            menu = self.channel_menu['menu']; menu.delete(0, 'end')
            for channel in channels: menu.add_command(label=channel, command=lambda value=channel: (self.selected_channel.set(value), self.apply_filter()))
            if channels: self.selected_channel.set(channels[0])
        self.apply_filter()
    def on_filter_change(self, *args): self._build_filter_panel(); self.apply_filter()
    def on_edge_toggle(self):
        is_enabled = self.edge_enabled.get()
        self.update_widget_state(self.edge_controls_container, is_enabled)
        self.update_widget_state(self.filter_frame, not is_enabled)
        self.apply_filter()
    def on_morph_toggle(self): self.update_widget_state(self.morph_controls_container, self.morph_enabled.get()); self.apply_filter()
    def on_contour_toggle(self): self.update_widget_state(self.contour_controls_container, self.contours_enabled.get()); self.apply_filter()
    def on_edge_filter_change(self, event=None): self._create_dynamic_panel(self.edge_detection_data, 'selected_edge_filter', self.edge_parameter_frame, self.edge_param_widgets, 'edge'); self.apply_filter()
    def on_display_mode_change(self, event=None): self._update_display_view()

    # --- Main Processing Pipeline ---
    def apply_filter(self, event=None):
        if self.original_cv_image is None: return
        try:
            self.intermediate_images.clear()
            self.intermediate_images['Original'] = self.original_cv_image.copy()

            preprocessed_image = self._process_preprocessing(self.original_cv_image)
            self.intermediate_images['Pre-processed'] = preprocessed_image

            enhanced_image = self._process_enhancement(preprocessed_image)
            self.intermediate_images['Enhanced'] = enhanced_image
            
            image_for_masking = enhanced_image
            if self.channel_enabled.get():
                image_for_masking = self._process_channel_extraction(enhanced_image)
            self.intermediate_images['Mask Input'] = image_for_masking
            
            if self.edge_enabled.get():
                gray_for_edge = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY) if len(enhanced_image.shape) == 3 else enhanced_image
                mask = self._process_edge_detection(gray_for_edge)
            else:
                mask = self._process_mask_generation(image_for_masking)
            
            if mask is None: return
            self.intermediate_images['Initial Mask'] = mask.copy()
            
            if self.morph_enabled.get():
                mask = self._process_morph_ops(mask)
            self.intermediate_images['Refined Mask'] = mask.copy()

            final_image = cv2.bitwise_and(self.original_cv_image, self.original_cv_image, mask=mask)
            
            if self.contours_enabled.get():
                final_image = self._process_contours(mask, final_image.copy())
            
            self.intermediate_images['Final Result'] = final_image
            self._update_display_view()
        except (tk.TclError, KeyError, ValueError, IndexError, AttributeError):
             pass

    # --- Processing Sub-routines ---
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
        except (KeyError, tk.TclError): return image.copy()
        return image.copy()

    def _process_enhancement(self, image):
        enhance_type = self.selected_enhancement.get()
        if enhance_type == 'None': return image
        try:
            if enhance_type == 'Histogram Equalization':
                if len(image.shape) == 2: return cv2.equalizeHist(image)
                img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb); img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0]); return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
            elif enhance_type == 'CLAHE':
                clip = float(self.param_vars['enhancement_Clip_Limit'].get()); tile = int(self.param_vars['enhancement_Tile_Grid_Size'].get())
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
                if len(image.shape) == 2: return clahe.apply(image)
                img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb); img_ycrcb[:, :, 0] = clahe.apply(img_ycrcb[:, :, 0]); return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
            elif enhance_type == 'Contrast Stretching':
                return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            elif enhance_type == 'Gamma Correction':
                gamma = self.param_vars['enhancement_Gamma_x100'].get() / 100.0; inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                return cv2.LUT(image, table)
        except (KeyError, tk.TclError): return image
        return image
    
    def _process_channel_extraction(self, image):
        space_name, channel_name = self.selected_color_space.get(), self.selected_channel.get()
        if not channel_name: return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        space_info = self.channel_data[space_name]
        converted_image = cv2.cvtColor(image, space_info['code']) if len(image.shape) == 3 and space_info['code'] is not None else image
        if converted_image.ndim == 2: return converted_image
        return cv2.split(converted_image)[space_info['channels'].index(channel_name)]

    def _process_mask_generation(self, image_for_masking):
        filter_selection = self.selected_filter.get()
        filter_config = self.filter_data.get(filter_selection, {})
        filter_type = filter_config.get('type')
        gray = cv2.cvtColor(image_for_masking, cv2.COLOR_BGR2GRAY) if image_for_masking.ndim == 3 else image_for_masking.copy()

        try:
            if filter_type == 'grayscale_range':
                min_v, max_v = self.param_vars['filter_Min_Value'].get(), self.param_vars['filter_Max_Value'].get()
                return cv2.inRange(gray, min_v, max_v)
            elif filter_type == 'adaptive_thresh':
                method = cv2.ADAPTIVE_THRESH_MEAN_C if self.param_vars['filter_Adaptive_Method'].get() == 'Mean C' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                type_f = cv2.THRESH_BINARY if self.param_vars['filter_Threshold_Type'].get() == 'Binary' else cv2.THRESH_BINARY_INV
                bsize, c = self.param_vars['filter_Block_Size'].get(), self.param_vars['filter_C_Constant'].get()
                if bsize > 1 and bsize % 2 == 0: bsize += 1
                return cv2.adaptiveThreshold(gray, 255, method, type_f, bsize, c)
            elif filter_type == "Otsu's Binarization":
                type_f = cv2.THRESH_BINARY if self.param_vars["otsu_Threshold_Type"].get() == 'Binary' else cv2.THRESH_BINARY_INV
                _, mask = cv2.threshold(gray, 0, 255, type_f + cv2.THRESH_OTSU)
                return mask
            elif filter_type == 'color':
                if image_for_masking.ndim == 2: return np.zeros_like(image_for_masking)
                lower = np.array([self.param_vars[f'color_{c}_min'].get() for c in filter_config['channels']])
                upper = np.array([self.param_vars[f'color_{c}_max'].get() for c in filter_config['channels']])
                conv_map = {'RGB/BGR (Color Filter)': -1, 'HSV': cv2.COLOR_BGR2HSV, 'HLS': cv2.COLOR_BGR2HLS, 'Lab': cv2.COLOR_BGR2LAB, 'YCrCb': cv2.COLOR_BGR2YCrCb}
                code = conv_map[filter_selection]
                converted = cv2.cvtColor(image_for_masking, code) if code != -1 else image_for_masking
                return cv2.inRange(converted, lower, upper)
        except (KeyError, tk.TclError): return np.zeros_like(gray)
        return np.zeros_like(gray)
    
    def _process_edge_detection(self, image_for_edge):
        edge_filter = self.selected_edge_filter.get()
        try:
            if edge_filter == 'Canny':
                t1, t2 = self.param_vars['edge_Threshold_1'].get(), self.param_vars['edge_Threshold_2'].get()
                return cv2.Canny(image_for_edge, t1, t2)
            elif edge_filter == 'Sobel':
                ksize = self.param_vars['edge_Kernel_Size'].get()
                if ksize > 0 and ksize % 2 == 0: ksize += 1
                direction = self.param_vars['edge_Direction'].get()
                sobelx = cv2.Sobel(image_for_edge, cv2.CV_64F, 1, 0, ksize=ksize)
                sobely = cv2.Sobel(image_for_edge, cv2.CV_64F, 0, 1, ksize=ksize)
                if direction == 'X': return cv2.convertScaleAbs(sobelx)
                if direction == 'Y': return cv2.convertScaleAbs(sobely)
                return cv2.convertScaleAbs(np.sqrt(sobelx**2 + sobely**2))
        except (KeyError, tk.TclError): return image_for_edge
        return image_for_edge

    def _process_morph_ops(self, input_mask):
        mask = input_mask.copy()
        ops = {'Erode': cv2.erode, 'Dilate': cv2.dilate, 'Open': cv2.MORPH_OPEN, 'Close': cv2.MORPH_CLOSE}
        for name, op_func in ops.items():
            if self.morph_ops_data[name]['var'].get():
                ksize = self.morph_ops_data[name]['kernel_var'].get()
                if ksize > 0:
                    kernel = np.ones((ksize, ksize), np.uint8)
                    if name in ['Erode', 'Dilate']: mask = op_func(mask, kernel, iterations=1)
                    else: mask = cv2.morphologyEx(mask, op_func, kernel)
        return mask

    def _process_contours(self, mask, image_to_draw_on):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.contour_min_area.get()]
        self.object_count_text.set(f"Objects Found: {len(filtered_contours)}")
        if self.draw_contours.get(): cv2.drawContours(image_to_draw_on, filtered_contours, -1, (0, 255, 0), 2)
        return image_to_draw_on

    # --- UI Builder Methods ---
    def _create_refinement_controls(self):
        edge_frame = ttk.LabelFrame(self.refinement_frame, text="Edge Detection (Overrides Mask Generation)", padding=5)
        edge_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(edge_frame, text="Enable", variable=self.edge_enabled, command=self.on_edge_toggle).pack(anchor='w')
        self.edge_controls_container = ttk.Frame(edge_frame); self.edge_controls_container.pack(fill=tk.X)
        ttk.OptionMenu(self.edge_controls_container, self.selected_edge_filter, self.selected_edge_filter.get(), *self.edge_detection_data.keys(), command=self.on_edge_filter_change).pack(fill=tk.X, pady=(5,0))
        self.edge_parameter_frame = ttk.Frame(self.edge_controls_container, padding=(0, 10, 0, 0)); self.edge_parameter_frame.pack(fill=tk.X)

        morph_frame = ttk.LabelFrame(self.refinement_frame, text="Morphological Operations", padding=5)
        morph_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(morph_frame, text="Enable", variable=self.morph_enabled, command=self.on_morph_toggle).pack(anchor='w')
        self.morph_controls_container = ttk.Frame(morph_frame); self.morph_controls_container.pack(fill=tk.X)
        self._create_morph_controls()

    def _create_contour_controls(self, parent):
        contour_frame = ttk.LabelFrame(parent, text="6. Contour Analysis", padding="10")
        contour_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(contour_frame, text="Enable", variable=self.contours_enabled, command=self.on_contour_toggle).pack(anchor='w')
        self.contour_controls_container = ttk.Frame(contour_frame); self.contour_controls_container.pack(fill=tk.X)
        ttk.Checkbutton(self.contour_controls_container, text="Draw Contours on Image", variable=self.draw_contours, command=self.apply_filter).pack(anchor='w', pady=(5,0))
        min_area_frame = self._create_param_row("Min Area", parent=self.contour_controls_container)
        self._create_slider_and_entry(min_area_frame, self.contour_min_area, 0, 50000)
        ttk.Label(self.contour_controls_container, textvariable=self.object_count_text, font=("Helvetica", 10, "bold")).pack(pady=5)

    def _create_display_options(self, parent):
        display_frame = ttk.LabelFrame(parent, text="7. Display Options", padding="10")
        display_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        options = ["Final Result", "Refined Mask", "Initial Mask", "Mask Input", "Enhanced", "Pre-processed", "Original"]
        for option in options:
            ttk.Radiobutton(display_frame, text=option, variable=self.display_mode, value=option, command=self.on_display_mode_change).pack(anchor='w')

    def _create_morph_controls(self):
        for name, data in self.morph_ops_data.items():
            frame = ttk.Frame(self.morph_controls_container); frame.pack(fill=tk.X, pady=2); frame.columnconfigure(2, weight=1)
            ttk.Checkbutton(frame, text=f"{name}:", variable=data['var'], command=self.apply_filter).grid(row=0, column=0, sticky='w')
            ttk.Label(frame, text="Kernel").grid(row=0, column=1, padx=(10, 5))
            slider, entry = self._create_slider_and_entry(frame, data['kernel_var'], 1, 31, column_offset=1)
            slider.grid_configure(row=0, column=2); entry.grid_configure(row=0, column=3); entry.config(width=5)

    def _create_channel_controls(self):
        frame1 = self._create_param_row('Color Space', parent=self.channel_controls_container)
        ttk.OptionMenu(frame1, self.selected_color_space, self.selected_color_space.get(), *self.channel_data.keys(), command=self.on_color_space_change).grid(row=0, column=1, sticky='ew')
        frame2 = self._create_param_row('Channel', parent=self.channel_controls_container)
        self.channel_menu = ttk.OptionMenu(frame2, self.selected_channel, ""); self.channel_menu.grid(row=0, column=1, sticky='ew')

    def _build_filter_panel(self):
        config = self.filter_data.get(self.selected_filter.get(), {})
        for widget in self.param_widgets: widget.destroy()
        self.param_widgets.clear()
        for k in list(self.param_vars.keys()):
            if k.startswith(('filter_', 'color_', 'otsu_')): 
                if k in self.param_vars: del self.param_vars[k]

        if config.get('type') == 'color':
             for i, (channel, (min_val, max_val)) in enumerate(zip(config.get('channels',[]), config.get('ranges',[]))):
                frame = self._create_param_row(f"{channel} Min", f"{channel} Max", parent=self.parameter_frame); self.param_widgets.append(frame)
                min_var, max_var = tk.IntVar(value=min_val), tk.IntVar(value=max_val)
                self.param_vars[f'color_{channel}_min'], self.param_vars[f'color_{channel}_max'] = min_var, max_var
                s1, e1 = self._create_slider_and_entry(frame, min_var, min_val, max_val, column_offset=0); self.param_widgets.extend([s1, e1])
                s2, e2 = self._create_slider_and_entry(frame, max_var, min_val, max_val, column_offset=3); self.param_widgets.extend([s2, e2])
        elif config.get('type') == 'otsu': self._create_dynamic_panel(self.filter_data, 'selected_filter', self.parameter_frame, self.param_widgets, 'otsu')
        else: self._create_dynamic_panel(self.filter_data, 'selected_filter', self.parameter_frame, self.param_widgets, 'filter')

    def _create_dynamic_panel(self, data, selector_var_name, parent_frame, widget_list, param_prefix):
        for widget in widget_list: widget.destroy()
        widget_list.clear()
        selection_name = getattr(self, selector_var_name).get()
        
        # --- FIXED ---
        config = data.get(selection_name, {})
        params = config.get('params', {})
        # --- END FIX ---

        prefix_to_clear = 'otsu' if param_prefix == 'otsu' else param_prefix
        for k in list(self.param_vars.keys()):
            if k.startswith(prefix_to_clear): 
                if k in self.param_vars: del self.param_vars[k]

        for p_name, p_data in params.items():
            frame = self._create_param_row(p_name, parent=parent_frame); widget_list.append(frame)
            var_key = f"{param_prefix}_{p_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            if 'options' in p_data:
                var = tk.StringVar(value=p_data['default'])
                menu = ttk.OptionMenu(frame, var, p_data['default'], *p_data['options'], command=self.apply_filter); menu.grid(row=0, column=1, sticky='ew'); widget_list.append(menu)
            else:
                var = tk.IntVar(value=p_data['default'])
                slider, entry = self._create_slider_and_entry(frame, var, p_data['range'][0], p_data['range'][1]); widget_list.extend([slider, entry])
            self.param_vars[var_key] = var

    def _create_param_row(self, label1, label2=None, parent=None):
        frame = ttk.Frame(parent if parent else self.parameter_frame); frame.pack(fill=tk.X, pady=2)
        frame.columnconfigure(1, weight=1); ttk.Label(frame, text=label1).grid(row=0, column=0, sticky='w', padx=5)
        if label2: frame.columnconfigure(4, weight=1); ttk.Label(frame, text=label2).grid(row=0, column=3, padx=(10, 0), sticky='w')
        return frame

    def _create_slider_and_entry(self, parent, variable, min_val, max_val, column_offset=0):
        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=variable, command=self.apply_filter); slider.grid(row=0, column=column_offset + 1, sticky='ew', padx=5)
        entry = ttk.Entry(parent, textvariable=variable, width=7); entry.grid(row=0, column=column_offset + 2); entry.bind("<Return>", self.apply_filter)
        return slider, entry

    def update_widget_state(self, container, is_enabled):
        state = 'normal' if is_enabled else 'disabled'
        for child in container.winfo_children():
            try: child.config(state=state)
            except tk.TclError: pass

    # --- NEW & UPDATED METHODS for QoL ---
    def _update_display_view(self):
        mode = self.display_mode.get()
        image_to_display = self.intermediate_images.get(mode)
        if image_to_display is not None:
            self.update_image_display(image_to_display)

    def update_image_display(self, cv_image):
        if cv_image is None: return
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB) if len(cv_image.shape) == 2 else cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.pil_image_fullres = Image.fromarray(rgb_image)
        self._apply_zoom_and_pan()

    def _apply_zoom_and_pan(self):
        if self.pil_image_fullres is None: return
        w, h = self.pil_image_fullres.size
        new_w, new_h = int(w * self.zoom_factor), int(h * self.zoom_factor)
        
        if new_w == 0 or new_h == 0: return # Prevent error on zero-size image
        
        resampling_filter = Image.Resampling.LANCZOS if self.zoom_factor > 1.0 else Image.Resampling.BOX
        resized_pil = self.pil_image_fullres.resize((new_w, new_h), resampling_filter)

        self.tk_image = ImageTk.PhotoImage(image=resized_pil)
        self.canvas_right.itemconfig(self.image_on_canvas, image=self.tk_image)
        canvas_w = self.canvas_right.winfo_width()
        canvas_h = self.canvas_right.winfo_height()
        self.canvas_right.coords(self.image_on_canvas, canvas_w/2, canvas_h/2)
        self.zoom_level_text.set(f"Zoom: {self.zoom_factor*100:.0f}%")

    def _on_ctrl_mousewheel(self, event):
        if event.delta > 0: self.zoom_in()
        else: self.zoom_out()

    def _on_canvas_press(self, event):
        self.canvas_right.scan_mark(event.x, event.y)

    def _on_canvas_drag(self, event):
        self.canvas_right.scan_dragto(event.x, event.y, gain=1)

    def zoom_in(self):
        self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
        self._apply_zoom_and_pan()

    def zoom_out(self):
        self.zoom_factor = max(self.min_zoom, self.zoom_factor / 1.1)
        self._apply_zoom_and_pan()

    def fit_to_screen(self):
        if self.pil_image_fullres is None: return
        canvas_w = self.canvas_right.winfo_width()
        canvas_h = self.canvas_right.winfo_height()
        if canvas_w < 50 or canvas_h < 50: return # Canvas not ready yet
        img_w, img_h = self.pil_image_fullres.size
        if img_w == 0 or img_h == 0: return

        w_ratio = canvas_w / img_w
        h_ratio = canvas_h / img_h
        self.zoom_factor = min(w_ratio, h_ratio, self.max_zoom) * 0.98 
        self._apply_zoom_and_pan()

    def save_settings(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not filepath: return
        
        settings = {
            'parameters': {key: var.get() for key, var in self.param_vars.items()},
            'morph_ops': {name: {'enabled': data['var'].get(), 'kernel': data['kernel_var'].get()} for name, data in self.morph_ops_data.items()},
            'selections': {
                'preproc': self.selected_preproc.get(), 'enhancement': self.selected_enhancement.get(),
                'color_space': self.selected_color_space.get(), 'channel': self.selected_channel.get(),
                'filter': self.selected_filter.get(), 'edge_filter': self.selected_edge_filter.get()
            },
            'toggles': {
                'channel': self.channel_enabled.get(), 'edge': self.edge_enabled.get(),
                'morph': self.morph_enabled.get(), 'contours': self.contours_enabled.get(),
                'draw_contours': self.draw_contours.get()
            },
            'contour_min_area': self.contour_min_area.get()
        }
        with open(filepath, 'w') as f: json.dump(settings, f, indent=4)
        messagebox.showinfo("Success", f"Settings saved to {filepath}")

    def load_settings(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not filepath: return
        try:
            with open(filepath, 'r') as f: settings = json.load(f)
            
            for key, val in settings['selections'].items(): getattr(self, f'selected_{key}').set(val)

            # Re-create panels first to ensure param_vars is populated
            self.on_preproc_change()
            self.on_enhancement_change()
            self._build_filter_panel()
            self.on_edge_filter_change()

            # Now set the loaded values
            for key, val in settings['parameters'].items():
                if key in self.param_vars: self.param_vars[key].set(val)
            for name, data in settings['morph_ops'].items():
                self.morph_ops_data[name]['var'].set(data['enabled'])
                self.morph_ops_data[name]['kernel_var'].set(data['kernel'])
            for key, val in settings['toggles'].items():
                var_name = 'draw_contours' if key == 'draw_contours' else f'{key}_enabled'
                getattr(self, var_name).set(val)
            self.contour_min_area.set(settings['contour_min_area'])

            # Refresh UI toggles
            self.on_color_space_change()
            self.on_edge_toggle()
            self.on_morph_toggle()
            self.on_contour_toggle()

            messagebox.showinfo("Success", "Settings loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {e}")
        finally:
            self.apply_filter()

    def show_histogram(self):
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Dependency Missing", "Matplotlib is required for the histogram feature.\nPlease install it via 'pip install matplotlib'")
            return
        
        image = self.intermediate_images.get(self.display_mode.get())
        if image is None:
            messagebox.showinfo("No Image", "Process an image first to see its histogram.")
            return

        hist_window = tk.Toplevel(self.root)
        hist_window.title(f"Histogram - {self.display_mode.get()}")
        
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        if len(image.shape) == 2: # Grayscale
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            ax.plot(hist, color='gray')
            ax.set_title("Intensity Histogram")
        else: # Color
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
            ax.set_title("Color Histogram")
        ax.set_xlim([0, 256])

        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedFilterApp(root)
    root.mainloop()