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

        # --- Member Variables ---
        self.original_cv_image = None
        self.processed_cv_image = None
        self.tk_image = None

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
            'Gradient': {'var': tk.BooleanVar(value=False), 'kernel_var': tk.IntVar(value=3)},
            'Top Hat': {'var': tk.BooleanVar(value=False), 'kernel_var': tk.IntVar(value=3)},
            'Black Hat': {'var': tk.BooleanVar(value=False), 'kernel_var': tk.IntVar(value=3)},
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
        self.preproc_param_widgets = []
        self.enhancement_param_widgets = []
        self.param_widgets = []
        self.edge_param_widgets, self.morph_param_widgets = [], []

        self._create_widgets()
        self._initialize_ui_states()
        self.selected_filter.trace_add('write', self.on_filter_change)

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
        
        enhancement_frame = ttk.LabelFrame(controls_frame, text="2. Image Enhancement", padding="10")
        enhancement_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.OptionMenu(enhancement_frame, self.selected_enhancement, self.selected_enhancement.get(), *self.enhancement_data.keys(), command=self.on_enhancement_change).pack(fill=tk.X)
        self.enhancement_parameter_frame = ttk.Frame(enhancement_frame, padding=(0, 10, 0, 0))
        self.enhancement_parameter_frame.pack(fill=tk.X)
        
        channel_frame = ttk.LabelFrame(controls_frame, text="3. Channel Extractor", padding="10")
        channel_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(channel_frame, text="Enable", variable=self.channel_enabled, command=self.on_channel_viewer_toggle).pack(anchor='w')
        self.channel_controls_container = ttk.Frame(channel_frame)
        self.channel_controls_container.pack(fill=tk.X, pady=5)
        self._create_channel_controls()
        
        self.filter_frame = ttk.LabelFrame(controls_frame, text="4. Mask Generation", padding="10")
        self.filter_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self.filter_menubutton = ttk.Menubutton(self.filter_frame, textvariable=self.selected_filter, direction='flush')
        filter_menu = tk.Menu(self.filter_menubutton, tearoff=False)
        for name in self.filter_data.keys():
            filter_menu.add_radiobutton(label=name, variable=self.selected_filter)
        self.filter_menubutton['menu'] = filter_menu
        self.filter_menubutton.pack(fill=tk.X)
        self.parameter_frame = ttk.Frame(self.filter_frame, padding=(0, 10, 0, 0))
        self.parameter_frame.pack(fill=tk.X)

        self.refinement_frame = ttk.LabelFrame(controls_frame, text="5. Mask Refinement", padding="10")
        self.refinement_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self._create_refinement_controls()
        
        self._create_contour_controls(controls_frame)

    def _initialize_ui_states(self):
        self.on_preproc_change()
        self.on_enhancement_change()
        self.on_color_space_change()
        self.on_filter_change()
        self.on_channel_viewer_toggle()
        self.on_edge_filter_change()
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
        self.apply_filter()
        self.save_button.config(state='normal')

    def save_image(self):
        if self.processed_cv_image is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if not filepath: return
        cv2.imwrite(filepath, self.processed_cv_image)
        messagebox.showinfo("Success", f"Image saved to {filepath}")

    # --- UI Event Handlers ---
    def on_preproc_change(self, event=None):
        self._create_dynamic_panel(self.preprocessing_data, 'selected_preproc', self.preproc_parameter_frame, self.preproc_param_widgets, 'preproc')
        self.apply_filter()

    def on_enhancement_change(self, event=None):
        self._create_dynamic_panel(self.enhancement_data, 'selected_enhancement', self.enhancement_parameter_frame, self.enhancement_param_widgets, 'enhancement')
        self.apply_filter()

    def on_channel_viewer_toggle(self, event=None):
        is_enabled = self.channel_enabled.get()
        channel_state = 'normal' if is_enabled else 'disabled'
        for child in self.channel_controls_container.winfo_children():
            try: child.config(state=channel_state)
            except tk.TclError: pass
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
        
    def on_edge_toggle(self):
        is_enabled = self.edge_enabled.get()
        state = 'normal' if is_enabled else 'disabled'
        for child in self.edge_controls_container.winfo_children():
            try: child.config(state=state)
            except tk.TclError: pass
        filter_state = 'disabled' if is_enabled else 'normal'
        for child in self.filter_frame.winfo_children():
            try: child.config(state=filter_state)
            except tk.TclError: pass
        self.apply_filter()

    def on_morph_toggle(self):
        is_enabled = self.morph_enabled.get()
        state = 'normal' if is_enabled else 'disabled'
        for child in self.morph_controls_container.winfo_children():
            try: child.config(state=state)
            except tk.TclError: pass
        self.apply_filter()

    def on_contour_toggle(self):
        is_enabled = self.contours_enabled.get()
        state = 'normal' if is_enabled else 'disabled'
        for child in self.contour_controls_container.winfo_children():
            try: child.config(state=state)
            except tk.TclError: pass
        self.apply_filter()

    def on_edge_filter_change(self, event=None):
        self._create_dynamic_panel(self.edge_detection_data, 'selected_edge_filter', self.edge_parameter_frame, self.edge_param_widgets, 'edge')
        self.apply_filter()

    # --- Main Processing Pipeline ---
    def apply_filter(self, event=None):
        if self.original_cv_image is None: return
        try:
            preprocessed_image = self._process_preprocessing(self.original_cv_image)
            enhanced_image = self._process_enhancement(preprocessed_image)
            
            image_for_masking = enhanced_image
            if self.channel_enabled.get():
                image_for_masking = self._process_channel_extraction(enhanced_image)
            
            if self.edge_enabled.get():
                gray_for_edge = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY) if len(enhanced_image.shape) == 3 else enhanced_image
                mask = self._process_edge_detection(gray_for_edge)
            else:
                mask = self._process_mask_generation(image_for_masking)
            
            if mask is None: return

            if self.morph_enabled.get():
                mask = self._process_morph_ops(mask)

            final_image = cv2.bitwise_and(self.original_cv_image, self.original_cv_image, mask=mask)
            
            if self.contours_enabled.get():
                final_image = self._process_contours(mask, final_image.copy())

            self.processed_cv_image = final_image
            self.update_image_display(self.processed_cv_image)
        except (tk.TclError, KeyError, ValueError, IndexError) as e:
            # This will prevent crashes from sliders being updated before an image is loaded
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
        space_name = self.selected_color_space.get()
        channel_name = self.selected_channel.get()
        if not channel_name: return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        space_info = self.channel_data[space_name]
        converted_image = cv2.cvtColor(image, space_info['code']) if len(image.shape) == 3 and space_info['code'] is not None else image
        if converted_image.ndim == 2: return converted_image
        channels = cv2.split(converted_image)
        channel_index = space_info['channels'].index(channel_name)
        return channels[channel_index]

    def _process_mask_generation(self, image_for_masking):
        filter_selection = self.selected_filter.get()
        filter_config = self.filter_data.get(filter_selection, {})
        filter_type = filter_config.get('type')
        if image_for_masking.ndim == 3: gray = cv2.cvtColor(image_for_masking, cv2.COLOR_BGR2GRAY)
        else: gray = image_for_masking.copy()

        try:
            if filter_type == 'grayscale_range':
                min_v = int(self.param_vars['filter_Min_Value'].get())
                max_v = int(self.param_vars['filter_Max_Value'].get())
                return cv2.inRange(gray, min_v, max_v)
            elif filter_type == 'adaptive_thresh':
                method = cv2.ADAPTIVE_THRESH_MEAN_C if self.param_vars['filter_Adaptive_Method'].get() == 'Mean C' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                type_f = cv2.THRESH_BINARY if self.param_vars['filter_Threshold_Type'].get() == 'Binary' else cv2.THRESH_BINARY_INV
                bsize = int(self.param_vars['filter_Block_Size'].get())
                c = int(self.param_vars['filter_C_(Constant)'].get())
                if bsize > 1 and bsize % 2 == 0: bsize += 1
                return cv2.adaptiveThreshold(gray, 255, method, type_f, bsize, c)
            elif filter_type == "Otsu's Binarization":
                type_f = cv2.THRESH_BINARY if self.param_vars["otsu_Threshold_Type"].get() == 'Binary' else cv2.THRESH_BINARY_INV
                ret, mask = cv2.threshold(gray, 0, 255, type_f + cv2.THRESH_OTSU)
                return mask
            elif filter_type == 'color':
                if image_for_masking.ndim == 2:
                    h, w = image_for_masking.shape[:2]
                    return np.zeros((h, w), dtype=np.uint8)
                lower = np.array([self.param_vars[f'color_{c}_min'].get() for c in filter_config['channels']])
                upper = np.array([self.param_vars[f'color_{c}_max'].get() for c in filter_config['channels']])
                conv_map = {'RGB/BGR (Color Filter)': -1, 'HSV': cv2.COLOR_BGR2HSV, 'HLS': cv2.COLOR_BGR2HLS, 'Lab': cv2.COLOR_BGR2LAB, 'YCrCb': cv2.COLOR_BGR2YCrCb}
                code = conv_map[filter_selection]
                converted = cv2.cvtColor(image_for_masking, code) if code != -1 else image_for_masking
                return cv2.inRange(converted, lower, upper)
        except (KeyError, tk.TclError): # Fallback on error
            h, w = image_for_masking.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        h, w = image_for_masking.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
        
    def _process_edge_detection(self, image_for_edge):
        edge_filter = self.selected_edge_filter.get()
        if edge_filter == 'Canny':
            t1 = self.param_vars['edge_Threshold_1'].get()
            t2 = self.param_vars['edge_Threshold_2'].get()
            return cv2.Canny(image_for_edge, t1, t2)
        elif edge_filter == 'Sobel':
            ksize = self.param_vars['edge_Kernel_Size'].get()
            if ksize > 0 and ksize % 2 == 0: ksize += 1
            direction = self.param_vars['edge_Direction'].get()

            sobelx = cv2.Sobel(image_for_edge, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(image_for_edge, cv2.CV_64F, 0, 1, ksize=ksize)
            
            if direction == 'X':
                return cv2.convertScaleAbs(sobelx)
            elif direction == 'Y':
                return cv2.convertScaleAbs(sobely)
            else: # Magnitude
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                return cv2.convertScaleAbs(magnitude)
        return image_for_edge

    def _process_morph_ops(self, input_mask):
        mask = input_mask.copy()
        ops = {
            'Erode': cv2.erode, 'Dilate': cv2.dilate, 'Open': cv2.MORPH_OPEN,
            'Close': cv2.MORPH_CLOSE, 'Gradient': cv2.MORPH_GRADIENT,
            'Top Hat': cv2.MORPH_TOPHAT, 'Black Hat': cv2.MORPH_BLACKHAT
        }
        for name, data in self.morph_ops_data.items():
            if data['var'].get():
                ksize = data['kernel_var'].get()
                if ksize > 0:
                    kernel = np.ones((ksize, ksize), np.uint8)
                    if name in ['Erode', 'Dilate']:
                        mask = ops[name](mask, kernel, iterations=1)
                    else:
                        mask = cv2.morphologyEx(mask, ops[name], kernel)
        return mask

    def _process_contours(self, mask, image_to_draw_on):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.contour_min_area.get()
        
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        self.object_count_text.set(f"Objects Found: {len(filtered_contours)}")
        
        if self.draw_contours.get():
            cv2.drawContours(image_to_draw_on, filtered_contours, -1, (0, 255, 0), 2)
            
        return image_to_draw_on

    # --- UI Builder Methods ---
    def _create_refinement_controls(self):
        edge_frame = ttk.LabelFrame(self.refinement_frame, text="Edge Detection (Overrides Mask Generation)", padding=5)
        edge_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(edge_frame, text="Enable", variable=self.edge_enabled, command=self.on_edge_toggle).pack(anchor='w')
        self.edge_controls_container = ttk.Frame(edge_frame)
        self.edge_controls_container.pack(fill=tk.X)
        ttk.OptionMenu(self.edge_controls_container, self.selected_edge_filter, self.selected_edge_filter.get(), *self.edge_detection_data.keys(), command=self.on_edge_filter_change).pack(fill=tk.X, pady=(5,0))
        self.edge_parameter_frame = ttk.Frame(self.edge_controls_container, padding=(0, 10, 0, 0))
        self.edge_parameter_frame.pack(fill=tk.X)

        morph_frame = ttk.LabelFrame(self.refinement_frame, text="Morphological Operations", padding=5)
        morph_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(morph_frame, text="Enable", variable=self.morph_enabled, command=self.on_morph_toggle).pack(anchor='w')
        self.morph_controls_container = ttk.Frame(morph_frame)
        self.morph_controls_container.pack(fill=tk.X)
        self._create_morph_controls()

    def _create_contour_controls(self, parent):
        contour_frame = ttk.LabelFrame(parent, text="6. Contour Analysis", padding="10")
        contour_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(contour_frame, text="Enable", variable=self.contours_enabled, command=self.on_contour_toggle).pack(anchor='w')
        self.contour_controls_container = ttk.Frame(contour_frame)
        self.contour_controls_container.pack(fill=tk.X)
        ttk.Checkbutton(self.contour_controls_container, text="Draw Contours on Image", variable=self.draw_contours, command=self.apply_filter).pack(anchor='w', pady=(5,0))
        min_area_frame = self._create_param_row("Min Area", parent=self.contour_controls_container)
        self._create_slider_and_entry(min_area_frame, self.contour_min_area, 0, 50000)
        ttk.Label(self.contour_controls_container, textvariable=self.object_count_text, font=("Helvetica", 10, "bold")).pack(pady=5)

    def _create_morph_controls(self):
        for name, data in self.morph_ops_data.items():
            frame = ttk.Frame(self.morph_controls_container)
            frame.pack(fill=tk.X, pady=2)
            frame.columnconfigure(2, weight=1)
            
            cb = ttk.Checkbutton(frame, text=f"{name}:", variable=data['var'], command=self.apply_filter)
            cb.grid(row=0, column=0, sticky='w')
            
            ttk.Label(frame, text="Kernel").grid(row=0, column=1, padx=(10, 5))
            
            slider, entry = self._create_slider_and_entry(frame, data['kernel_var'], 1, 31, column_offset=1)
            slider.grid_configure(row=0, column=2)
            entry.grid_configure(row=0, column=3)
            entry.config(width=5)

    def _create_channel_controls(self):
        frame1 = self._create_param_row('Color Space', parent=self.channel_controls_container)
        space_menu = ttk.OptionMenu(frame1, self.selected_color_space, self.selected_color_space.get(), *self.channel_data.keys(), command=self.on_color_space_change)
        space_menu.grid(row=0, column=1, sticky='ew')
        frame2 = self._create_param_row('Channel', parent=self.channel_controls_container)
        self.channel_menu = ttk.OptionMenu(frame2, self.selected_channel, "")
        self.channel_menu.grid(row=0, column=1, sticky='ew')

    def _build_filter_panel(self):
        config = self.filter_data.get(self.selected_filter.get(), {})
        for widget in self.param_widgets: widget.destroy()
        self.param_widgets.clear()
        for k in list(self.param_vars.keys()):
            if k.startswith('filter_') or k.startswith('color_') or k.startswith('otsu_'):
                if k in self.param_vars: del self.param_vars[k]
        if config.get('type') == 'color':
             for i, (channel, (min_val, max_val)) in enumerate(zip(config['channels'], config['ranges'])):
                frame = self._create_param_row(f"{channel} Min", f"{channel} Max", parent=self.parameter_frame)
                self.param_widgets.append(frame)
                min_var = tk.IntVar(value=min_val)
                self.param_vars[f'color_{channel}_min'] = min_var
                slider1, entry1 = self._create_slider_and_entry(frame, min_var, min_val, max_val, column_offset=0)
                self.param_widgets.extend([slider1, entry1])
                max_var = tk.IntVar(value=max_val)
                self.param_vars[f'color_{channel}_max'] = max_var
                slider2, entry2 = self._create_slider_and_entry(frame, max_var, min_val, max_val, column_offset=3)
                self.param_widgets.extend([slider2, entry2])
        elif config.get('type') == 'otsu':
            params = config['params']
            p_name = 'Threshold Type'
            p_data = params[p_name]
            frame1 = self._create_param_row(p_name, parent=self.parameter_frame)
            self.param_widgets.append(frame1)
            var = tk.StringVar(value=p_data['default'])
            self.param_vars[f'otsu_{p_name.replace(" ", "_")}'] = var
            menu = ttk.OptionMenu(frame1, var, p_data['default'], *p_data['options'], command=self.apply_filter)
            menu.grid(row=0, column=1, sticky='ew')
            self.param_widgets.append(menu)
        else:
             self._create_dynamic_panel(self.filter_data, 'selected_filter', self.parameter_frame, self.param_widgets, 'filter')

    def _create_dynamic_panel(self, data, selector_var_name, parent_frame, widget_list, param_prefix):
        for widget in widget_list: widget.destroy()
        widget_list.clear()
        selection_name = getattr(self, selector_var_name).get()
        config = data.get(selection_name, {})
        params = config.get('params', {})
        for k in list(self.param_vars.keys()):
            if k.startswith(param_prefix): del self.param_vars[k]
        for p_name, p_data in params.items():
            frame = self._create_param_row(p_name, parent=parent_frame)
            widget_list.append(frame)
            var_key = f"{param_prefix}_{p_name.replace(' ', '_').replace('(', '').replace(')', '')}"
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

    def _create_param_row(self, label1, label2=None, parent=None):
        if parent is None: parent = self.parameter_frame
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label1).grid(row=0, column=0, sticky='w', padx=5)
        if label2:
            frame.columnconfigure(4, weight=1)
            ttk.Label(frame, text=label2).grid(row=0, column=3, padx=(10, 0), sticky='w')
        return frame

    def _create_slider_and_entry(self, parent, variable, min_val, max_val, column_offset=0):
        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=variable, command=self.apply_filter)
        slider.grid(row=0, column=column_offset + 1, sticky='ew', padx=5)
        entry = ttk.Entry(parent, textvariable=variable, width=7)
        entry.grid(row=0, column=column_offset + 2)
        entry.bind("<Return>", self.apply_filter)
        return slider, entry

    def update_image_display(self, cv_image):
        if cv_image is None: return
        if len(cv_image.shape) == 2: rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else: rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        self.canvas_right.itemconfig(self.image_on_canvas, image=self.tk_image)
        self.canvas_right.config(scrollregion=self.canvas_right.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedFilterApp(root)
    root.mainloop()