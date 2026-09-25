
import os
import tempfile
import shutil
import zipfile
import re
import json
from datetime import datetime
from xml.sax.saxutils import escape
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from tkinter import ttk

import numpy as np
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def parse_dsc(path):
    params = {}
    if not path or not os.path.exists(path):
        return params

    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(None, 1)
            if len(parts) == 2:
                key, value = parts
                params[key.strip()] = value.strip()

    return params


def to_float(value):
    if value is None:
        return None
    value = str(value).strip().replace(",", ".")
    try:
        return float(value)
    except Exception:
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", value)
        return float(m.group(0)) if m else None


def to_int(value):
    v = to_float(value)
    return int(round(v)) if v is not None else None


def read_simple_txt(path):
    fields = []
    intensities = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if any(word in s.lower() for word in ["field", "index", "harm"]):
                continue
            if s.startswith("#") or s.startswith("*"):
                continue

            parts = s.replace(",", ".").split()
            if len(parts) < 3:
                continue

            try:
                float(parts[0])
                field = float(parts[1])
                intensity = float(parts[-1])
            except Exception:
                continue

            fields.append(field)
            intensities.append(intensity)

    if len(fields) < 2:
        raise ValueError("Aucune série champ/intensité exploitable n'a été trouvée dans le fichier TXT.")

    return fields, intensities



def read_bruker_dta(dta_path, dsc_path):
    """
    Decode a Bruker BES3T DTA spectrum using its companion DSC file.

    DSC fields used:
      XPTS, XMIN, XWID, BSEQ, IRFMT, IKKF.
    """
    if not dta_path or not os.path.exists(dta_path):
        raise ValueError("Fichier DTA manquant.")
    if not dsc_path or not os.path.exists(dsc_path):
        raise ValueError("Fichier DSC manquant pour décoder le DTA.")

    params = parse_dsc(dsc_path)

    try:
        xpts = int(float(str(params.get("XPTS", "")).replace(",", ".")))
    except Exception:
        raise ValueError("XPTS absent ou invalide dans le DSC.")

    if xpts < 2:
        raise ValueError("XPTS invalide dans le DSC.")

    ikkf = str(params.get("IKKF", "REAL")).strip().upper()
    if ikkf and ikkf != "REAL":
        raise ValueError(
            f"Format DTA IKKF={ikkf} non pris en charge "
            "(le programme attend un spectre REAL)."
        )

    irfmt = str(params.get("IRFMT", "D")).strip().upper()
    dtype_map = {
        "D": "f8",
        "F": "f4",
        "I": "i4",
        "S": "i2",
        "C": "i1",
    }
    if irfmt not in dtype_map:
        raise ValueError(f"IRFMT={irfmt} non pris en charge.")

    bseq = str(params.get("BSEQ", "BIG")).strip().upper()
    if bseq.startswith("BIG"):
        endian = ">"
    elif bseq.startswith("LIT"):
        endian = "<"
    else:
        raise ValueError(f"BSEQ={bseq} non reconnu dans le DSC.")

    dtype = np.dtype(endian + dtype_map[irfmt])

    expected_bytes = xpts * dtype.itemsize
    actual_bytes = os.path.getsize(dta_path)
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"DTA trop court : {actual_bytes} octets, "
            f"{expected_bytes} attendus d'après le DSC."
        )

    signal = np.fromfile(dta_path, dtype=dtype, count=xpts).astype(float)
    if signal.size != xpts:
        raise ValueError(
            f"Nombre de points DTA incohérent : {signal.size} lus, {xpts} attendus."
        )

    xmin = to_float(params.get("XMIN"))
    xwid = to_float(params.get("XWID"))
    if xmin is None or xwid is None:
        raise ValueError("XMIN ou XWID absent du DSC.")

    # Checked against the user's Bruker TXT exports:
    # first point = XMIN, last point = XMIN + XWID.
    field = np.linspace(float(xmin), float(xmin) + float(xwid), xpts)

    return field.tolist(), signal.tolist()



def read_bruker_kinetic_dataset(dta_path, dsc_path, ygf_path=None):
    """
    Decode a 2D Bruker BES3T kinetic dataset.

    DTA:
      flattened binary matrix containing YPTS spectra of XPTS points.

    DSC:
      defines XPTS, YPTS, XMIN, XWID, BSEQ, IRFMT, YTYP, YFMT,
      and metadata such as YNAM/YUNI and QValue.

    YGF:
      for an irregular Y grid (YTYP=IGD), contains one binary Y coordinate
      per spectrum. In the user's kinetic datasets this is the acquisition
      time in seconds.
    """
    if not dta_path or not os.path.exists(dta_path):
        raise ValueError("Fichier DTA du suivi manquant.")
    if not dsc_path or not os.path.exists(dsc_path):
        raise ValueError("Fichier DSC du suivi manquant.")

    params = parse_dsc(dsc_path)

    try:
        xpts = int(float(str(params.get("XPTS", "")).replace(",", ".")))
        ypts = int(float(str(params.get("YPTS", "")).replace(",", ".")))
    except Exception:
        raise ValueError("XPTS ou YPTS absent/invalide dans le DSC du suivi.")

    if xpts < 2 or ypts < 2:
        raise ValueError("Le DSC ne décrit pas un suivi 2D exploitable.")

    ikkf = str(params.get("IKKF", "REAL")).strip().upper()
    if ikkf and ikkf != "REAL":
        raise ValueError(
            f"IKKF={ikkf} non pris en charge pour ce suivi "
            "(le programme attend des données REAL)."
        )

    dtype_map = {
        "D": "f8",
        "F": "f4",
        "I": "i4",
        "S": "i2",
        "C": "i1",
    }

    irfmt = str(params.get("IRFMT", "D")).strip().upper()
    if irfmt not in dtype_map:
        raise ValueError(f"IRFMT={irfmt} non pris en charge.")

    bseq = str(params.get("BSEQ", "BIG")).strip().upper()
    if bseq.startswith("BIG"):
        endian = ">"
    elif bseq.startswith("LIT"):
        endian = "<"
    else:
        raise ValueError(f"BSEQ={bseq} non reconnu.")

    dtype = np.dtype(endian + dtype_map[irfmt])
    expected_points = xpts * ypts
    expected_bytes = expected_points * dtype.itemsize
    actual_bytes = os.path.getsize(dta_path)

    if actual_bytes < expected_bytes:
        raise ValueError(
            f"DTA trop court : {actual_bytes} octets, "
            f"{expected_bytes} attendus pour {ypts} × {xpts} points."
        )

    raw = np.fromfile(dta_path, dtype=dtype, count=expected_points).astype(float)
    if raw.size != expected_points:
        raise ValueError(
            f"Nombre de valeurs DTA incohérent : {raw.size} lues, "
            f"{expected_points} attendues."
        )

    spectra = raw.reshape((ypts, xpts))

    xmin = to_float(params.get("XMIN"))
    xwid = to_float(params.get("XWID"))
    if xmin is None or xwid is None:
        raise ValueError("XMIN ou XWID absent : axe de champ impossible à reconstruire.")

    field = np.linspace(float(xmin), float(xmin) + float(xwid), xpts)

    # Y axis / kinetic time.
    ytyp = str(params.get("YTYP", "")).strip().upper()
    yfmt = str(params.get("YFMT", "D")).strip().upper()
    times = None

    if ytyp == "IGD":
        if not ygf_path or not os.path.exists(ygf_path):
            raise ValueError(
                "Le DSC indique YTYP=IGD : le fichier YGF est nécessaire "
                "pour récupérer les temps du suivi."
            )

        if yfmt not in dtype_map:
            raise ValueError(f"YFMT={yfmt} non pris en charge.")

        ydtype = np.dtype(endian + dtype_map[yfmt])
        yvals = np.fromfile(ygf_path, dtype=ydtype, count=ypts).astype(float)

        if yvals.size != ypts:
            raise ValueError(
                f"YGF incohérent : {yvals.size} temps lus, {ypts} attendus."
            )

        times = yvals
    else:
        ymin = to_float(params.get("YMIN"))
        ywid = to_float(params.get("YWID"))

        if ymin is None or ywid is None:
            raise ValueError(
                "Axe temporel non reconstructible : YGF absent et YMIN/YWID manquants."
            )

        times = np.linspace(float(ymin), float(ymin) + float(ywid), ypts)

    return {
        "field": field,
        "times": np.asarray(times, dtype=float),
        "spectra": spectra,
        "params": params,
    }



def gauss_to_mt(value):
    """Convert magnetic field from gauss to millitesla for display."""
    return np.asarray(value, dtype=float) / 10.0



def microwave_frequency_hz_from_dsc(params):
    raw = params.get("MWFQ")
    if raw is None:
        return None
    try:
        value = float(str(raw).replace(",", ".").strip().strip("'\""))
    except Exception:
        return None
    if value <= 0:
        return None
    return value * 1e9 if value < 1e6 else value


def gauss_to_g(field_gauss, frequency_hz):
    b_tesla = np.asarray(field_gauss, dtype=float) * 1e-4
    if frequency_hz is None or frequency_hz <= 0:
        raise ValueError("Fréquence micro-onde MWFQ absente ou invalide.")
    mu_b_over_h = 13.99624555e9
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(frequency_hz) / (mu_b_over_h * b_tesla)


def g_to_gauss(g_value, frequency_hz):
    g_arr = np.asarray(g_value, dtype=float)
    mu_b_over_h = 13.99624555e9
    with np.errstate(divide="ignore", invalid="ignore"):
        b_tesla = float(frequency_hz) / (mu_b_over_h * g_arr)
    return b_tesla / 1e-4


def clean_stem(path):
    """Nom de base sans extension, normalisé pour comparer TXT et DSC."""
    return os.path.splitext(os.path.basename(path))[0].strip().lower()



def _excel_col_name(n):
    name = ""
    while n:
        n, rem = divmod(n - 1, 26)
        name = chr(65 + rem) + name
    return name


def write_simple_xlsx(path, rows):
    def cell_xml(row_idx, col_idx, value):
        ref = f"{_excel_col_name(col_idx)}{row_idx}"
        if value is None or value == "":
            return f'<c r="{ref}"/>'
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                if np.isfinite(value):
                    return f'<c r="{ref}"><v>{value}</v></c>'
            except Exception:
                pass
        txt = escape(str(value))
        return f'<c r="{ref}" t="inlineStr"><is><t>{txt}</t></is></c>'

    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = "".join(
            cell_xml(r_idx, c_idx, val)
            for c_idx, val in enumerate(row, start=1)
        )
        sheet_rows.append(f'<row r="{r_idx}">{cells}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>' + "".join(sheet_rows) + '</sheetData>'
        '</worksheet>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )

    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def export_table_dialog(parent, default_name, rows):
    path = filedialog.asksaveasfilename(
        parent=parent,
        title="Exporter les données",
        defaultextension=".txt",
        initialfile=default_name,
        filetypes=[
            ("Fichier texte tabulé", "*.txt"),
            ("Fichier Excel", "*.xlsx"),
        ]
    )
    if not path:
        return

    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xlsx":
            write_simple_xlsx(path, rows)
        else:
            if ext != ".txt":
                path += ".txt"
            with open(path, "w", encoding="utf-8", newline="") as f:
                for row in rows:
                    f.write("\t".join("" if v is None else str(v) for v in row) + "\n")

        messagebox.showinfo(
            "Export terminé",
            f"Données exportées avec succès :\n{path}",
            parent=parent
        )
    except Exception as e:
        messagebox.showerror(
            "Erreur d'export",
            str(e),
            parent=parent
        )



def parse_dsc_acquisition_datetime(path):
    """Return acquisition datetime from Bruker DSC DATE + TIME fields."""
    if not path:
        return None

    params = parse_dsc(path)
    date_text = str(params.get("DATE", "")).strip().strip("'\"")
    time_text = str(params.get("TIME", "")).strip().strip("'\"")

    if not date_text or not time_text:
        return None

    stamp = f"{date_text} {time_text}"

    # The supplied Bruker files use e.g. DATE 09/03/26 for 3 Sep 2026,
    # therefore MM/DD/YY is tried first.
    for fmt in (
        "%m/%d/%y %H:%M:%S",
        "%d/%m/%y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d.%m.%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(stamp, fmt)
        except Exception:
            pass

    return None


class RPEViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RPE Viewer")
        self.geometry("1220x780")
        self.minsize(930, 620)

        self.txt_path = None
        self.dsc_path = None
        self.dsc = {}
        self.field = []
        self.signal = []

        self.slider_guard = False

        self.multi_dta_paths = []
        self.multi_dsc_paths = []
        self.multi_pairs = []
        self.multi_analysis_window = None

        self.auto_dta_path = None
        self.auto_dsc_path = None
        self.auto_ygf_path = None
        self.auto_rows = []
        self.auto_analysis_window = None
        self.auto_temp_dir = None

        self.mn_dta_path = None
        self.mn_dsc_path = None
        self.mn_series_pairs = []
        self.mn_measurements = {}
        self.mn_project_path = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)

        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True)

        tab_visual = ttk.Frame(notebook)
        tab_multi = ttk.Frame(notebook)
        tab_auto = ttk.Frame(notebook)
        tab_mn = ttk.Frame(notebook)

        notebook.add(tab_visual, text="Visualiser")
        notebook.add(tab_multi, text="Multi-spectres")
        notebook.add(tab_auto, text="Suivi cinétique")
        notebook.add(tab_mn, text="Étalon interne Mn²⁺")

        # ============================================================
        # ONGLET 1 — VISUALISER
        # ============================================================
        visual_main = ttk.Frame(tab_visual, padding=8)
        visual_main.pack(fill="both", expand=True)

        visual_left = ttk.Frame(visual_main, width=310)
        visual_left.pack(side="left", fill="y", padx=(0, 8))
        visual_left.pack_propagate(False)

        visual_right = ttk.Frame(visual_main)
        visual_right.pack(side="right", fill="both", expand=True)

        import_box = tk.LabelFrame(
            visual_left, text="Visualiser un spectre", padx=10, pady=10,
            bg="#EAF3FF", bd=2, relief="groove"
        )
        import_box.pack(fill="x", pady=(0, 8))

        ttk.Button(import_box, text="Charger le fichier TXT", command=self.load_txt).pack(fill="x", pady=3)
        self.txt_label = tk.Label(
            import_box, text="TXT : aucun", wraplength=260, justify="left", anchor="w",
            bg="#EAF3FF", fg="#4A5568", font=("TkDefaultFont", 8)
        )
        self.txt_label.pack(fill="x", pady=(1, 5))

        ttk.Button(import_box, text="Charger le fichier DSC", command=self.load_dsc).pack(fill="x", pady=3)
        self.dsc_label = tk.Label(
            import_box, text="DSC : aucun", wraplength=260, justify="left", anchor="w",
            bg="#EAF3FF", fg="#4A5568", font=("TkDefaultFont", 8)
        )
        self.dsc_label.pack(fill="x", pady=(1, 5))

        ttk.Button(import_box, text="Afficher le spectre", command=self.try_display).pack(fill="x", pady=(6, 3))
        ttk.Button(import_box, text="Réinitialiser", command=self.reset).pack(fill="x", pady=3)

        ttk.Separator(import_box, orient="horizontal").pack(fill="x", pady=(7, 7))
        ttk.Label(import_box, text="Fenêtre du spectre", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")

        self.min_label_var = tk.StringVar(value="Champ min : —")
        ttk.Label(import_box, textvariable=self.min_label_var).pack(anchor="w", pady=(5, 0))
        self.min_slider = tk.Scale(
            import_box, from_=0, to=100, orient="horizontal", resolution=0.1,
            showvalue=False, command=self.on_min_slider, bg="#EAF3FF", highlightthickness=0
        )
        self.min_slider.pack(fill="x")

        self.max_label_var = tk.StringVar(value="Champ max : —")
        ttk.Label(import_box, textvariable=self.max_label_var).pack(anchor="w", pady=(5, 0))
        self.max_slider = tk.Scale(
            import_box, from_=0, to=100, orient="horizontal", resolution=0.1,
            showvalue=False, command=self.on_max_slider, bg="#EAF3FF", highlightthickness=0
        )
        self.max_slider.pack(fill="x")

        ttk.Button(import_box, text="Voir tout le spectre", command=self.reset_limits).pack(fill="x", pady=(7, 3))
        self.width_var = tk.StringVar(value="1220")
        self.height_var = tk.StringVar(value="780")
        ttk.Button(import_box, text="Redimensionner", command=self.resize_window_dialog).pack(fill="x", pady=(5, 2))

        info_box = ttk.LabelFrame(visual_left, text="Paramètres DSC", padding=10)
        info_box.pack(fill="both", expand=True)
        self.info_text = tk.Text(info_box, width=34, height=18, wrap="word", state="disabled")
        self.info_text.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Champ magnétique (mT)")
        self.ax.set_ylabel("Intensité RPE (a.u.)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=visual_right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = ttk.Frame(visual_right)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # ============================================================
        # ONGLET 2 — MULTI-SPECTRES
        # ============================================================
        multi_main = ttk.Frame(tab_multi, padding=12)
        multi_main.pack(fill="both", expand=True)
        multi_box = tk.LabelFrame(
            multi_main, text="Multi-spectres", padx=12, pady=12,
            bg="#FFF3E6", bd=2, relief="groove"
        )
        multi_box.pack(anchor="nw", fill="x", padx=4, pady=4)
        ttk.Label(
            multi_box, text="Charge les séries DTA et DSC, puis ouvre l'analyse multi-spectres. Le DTA est utilisé comme donnée brute de référence.",
            wraplength=720
        ).pack(anchor="w", pady=(0, 8))
        ttk.Button(multi_box, text="Charger multi DTA", command=self.load_multi_dta).pack(fill="x", pady=3)
        ttk.Button(multi_box, text="Charger multi DSC", command=self.load_multi_dsc).pack(fill="x", pady=3)
        ttk.Button(multi_box, text="Analyser multi-spectres", command=self.open_multi_window).pack(fill="x", pady=(8, 3))
        ttk.Button(multi_box, text="Réinitialiser", command=self.reset).pack(fill="x", pady=3)

        # ============================================================
        # ONGLET 3 — SUIVI CINÉTIQUE
        # ============================================================
        auto_main = ttk.Frame(tab_auto, padding=12)
        auto_main.pack(fill="both", expand=True)
        auto_box = tk.LabelFrame(
            auto_main, text="Suivi cinétique", padx=12, pady=12,
            bg="#EAF8EE", bd=2, relief="groove"
        )
        auto_box.pack(anchor="nw", fill="x", padx=4, pady=4)
        ttk.Label(
            auto_box,
            text=(
                "Charge le DTA, le DSC et le YGF du suivi. "
                "Le DTA contient les spectres bruts et le YGF les temps de mesure."
            ),
            wraplength=720
        ).pack(anchor="w", pady=(0, 8))

        ttk.Button(
            auto_box,
            text="Charger le DTA du suivi",
            command=self.load_auto_dta
        ).pack(fill="x", pady=3)

        self.auto_dta_label = tk.Label(
            auto_box, text="DTA : aucun", wraplength=700, justify="left", anchor="w",
            bg="#EAF8EE", fg="#415A49", font=("TkDefaultFont", 8)
        )
        self.auto_dta_label.pack(fill="x", pady=(1, 5))

        ttk.Button(
            auto_box,
            text="Charger le DSC du suivi",
            command=self.load_auto_dsc
        ).pack(fill="x", pady=3)

        self.auto_dsc_label = tk.Label(
            auto_box, text="DSC : aucun", wraplength=700, justify="left", anchor="w",
            bg="#EAF8EE", fg="#415A49", font=("TkDefaultFont", 8)
        )
        self.auto_dsc_label.pack(fill="x", pady=(1, 5))

        ttk.Button(
            auto_box,
            text="Charger le YGF du suivi",
            command=self.load_auto_ygf
        ).pack(fill="x", pady=3)

        self.auto_ygf_label = tk.Label(
            auto_box, text="YGF : aucun", wraplength=700, justify="left", anchor="w",
            bg="#EAF8EE", fg="#415A49", font=("TkDefaultFont", 8)
        )
        self.auto_ygf_label.pack(fill="x", pady=(1, 5))
        ttk.Button(auto_box, text="Analyser suivi cinétique", command=self.open_auto_window).pack(fill="x", pady=(8, 3))
        ttk.Button(auto_box, text="Réinitialiser", command=self.reset).pack(fill="x", pady=3)

        # ============================================================
        # ONGLET 4 — ÉTALON INTERNE Mn²⁺
        # ============================================================
        mn_main = ttk.Frame(tab_mn, padding=12)
        mn_main.pack(fill="both", expand=True)

        mn_box = ttk.LabelFrame(
            mn_main,
            text="Étalon interne Mn²⁺",
            padding=12
        )
        mn_box.pack(anchor="nw", fill="x", padx=4, pady=4)

        ttk.Label(
            mn_box,
            text=(
                "Analyse d'un spectre contenant le radical et l'étalon Mn²⁺ en capillaire. "
                "Sélectionne un pic isolé de Mn²⁺ puis une fenêtre contenant Mn + radical. "
                "Le calcul renvoie : (DI(Mn+radical) - DI(Mn)) / DI(Mn)."
            ),
            wraplength=820
        ).pack(anchor="w", pady=(0, 8))

        action_row = ttk.Frame(mn_box)
        action_row.pack(fill="x", pady=(0, 7))

        ttk.Button(
            action_row,
            text="Ajouter des DTA",
            command=self.load_mn_dta
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        ttk.Button(
            action_row,
            text="Charger une série",
            command=self.load_mn_series_project
        ).pack(side="left", fill="x", expand=True, padx=4)

        ttk.Button(
            action_row,
            text="Enregistrer la série",
            command=self.save_mn_series_project
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.mn_project_label = ttk.Label(
            mn_box,
            text="Projet : non enregistré",
            foreground="#555555"
        )
        self.mn_project_label.pack(anchor="w", pady=(0, 7))

        ttk.Label(
            mn_box,
            text=(
                "Les DSC de même nom sont associés automatiquement. Le temps affiché est le Δt "
                "par rapport à la mesure précédente, calculé avec DATE + TIME du DSC."
            ),
            wraplength=980
        ).pack(anchor="w", pady=(0, 7))

        series_wrap = ttk.Frame(mn_box)
        series_wrap.pack(fill="both", expand=True, pady=(3, 6))

        columns = (
            "spectre", "dsc", "time", "di_mn", "di_total", "di_radical", "ratio"
        )
        self.mn_series_tree = ttk.Treeview(
            series_wrap,
            columns=columns,
            show="headings",
            height=11,
            selectmode="browse"
        )
        headings = {
            "spectre": "Spectre",
            "dsc": "DSC",
            "time": "Temps depuis t0 (min)",
            "di_mn": "DI Mn",
            "di_total": "DI Total",
            "di_radical": "DI Radical",
            "ratio": "Ratio",
        }
        widths = {
            "spectre": 270,
            "dsc": 80,
            "time": 105,
            "di_mn": 105,
            "di_total": 105,
            "di_radical": 105,
            "ratio": 110,
        }
        for c in columns:
            self.mn_series_tree.heading(c, text=headings[c])
            self.mn_series_tree.column(c, width=widths[c], anchor="center")
        self.mn_series_tree.column("spectre", anchor="w")

        mn_scroll = ttk.Scrollbar(
            series_wrap, orient="vertical", command=self.mn_series_tree.yview
        )
        self.mn_series_tree.configure(yscrollcommand=mn_scroll.set)
        self.mn_series_tree.pack(side="left", fill="both", expand=True)
        mn_scroll.pack(side="right", fill="y")

        self.mn_series_tree.bind("<<TreeviewSelect>>", self.on_mn_series_select)
        self.mn_series_tree.bind(
            "<Double-1>",
            lambda _e: self.open_mn_internal_standard_window()
        )

        mn_buttons = ttk.Frame(mn_box)
        mn_buttons.pack(fill="x", pady=(4, 0))

        ttk.Button(
            mn_buttons,
            text="Analyser le spectre sélectionné",
            command=self.open_mn_internal_standard_window
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        ttk.Button(
            mn_buttons,
            text="Associer un DSC au spectre sélectionné",
            command=self.load_mn_dsc
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        ttk.Button(
            mn_box,
            text="Effacer la série et les mesures",
            command=self.reset_mn_internal_standard
        ).pack(fill="x", pady=(7, 3))

    # ---------------- IMPORT LOGIC ----------------
    def load_txt(self):
        path = filedialog.askopenfilename(
            title="Sélectionner le fichier TXT du spectre RPE",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return

        self.txt_path = path
        self.txt_label.config(text=f"TXT : {os.path.basename(path)}")

        # If a DSC is already loaded, validate immediately.
        if self.dsc_path:
            if not self.confirm_name_match():
                return

        matching_dsc = os.path.splitext(path)[0] + ".DSC"
        matching_dsc_lower = os.path.splitext(path)[0] + ".dsc"

        if not self.dsc_path:
            if os.path.exists(matching_dsc):
                self.dsc_path = matching_dsc
                self.dsc_label.config(text=f"DSC : {os.path.basename(matching_dsc)}")
                self.try_display()
            elif os.path.exists(matching_dsc_lower):
                self.dsc_path = matching_dsc_lower
                self.dsc_label.config(text=f"DSC : {os.path.basename(matching_dsc_lower)}")
                self.try_display()
        else:
            self.try_display()

    def load_dsc(self):
        path = filedialog.askopenfilename(
            title="Sélectionner le fichier DSC associé",
            filetypes=[("Fichiers DSC", "*.DSC *.dsc"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return

        old = self.dsc_path
        self.dsc_path = path
        self.dsc_label.config(text=f"DSC : {os.path.basename(path)}")

        if self.txt_path:
            if not self.confirm_name_match():
                # Restore previous DSC if user refuses mismatch
                self.dsc_path = old
                if old:
                    self.dsc_label.config(text=f"DSC : {os.path.basename(old)}")
                else:
                    self.dsc_label.config(text="DSC : aucun")
                return
            self.try_display()

    def confirm_name_match(self):
        if not self.txt_path or not self.dsc_path:
            return True

        txt_stem = clean_stem(self.txt_path)
        dsc_stem = clean_stem(self.dsc_path)

        if txt_stem == dsc_stem:
            return True

        return messagebox.askyesno(
            "Noms de fichiers différents",
            "Le fichier TXT et le fichier DSC n'ont pas exactement le même nom.\n\n"
            f"TXT : {os.path.basename(self.txt_path)}\n"
            f"DSC : {os.path.basename(self.dsc_path)}\n\n"
            "Veux-tu quand même utiliser cette paire de fichiers ?"
        )

    def try_display(self):
        if not self.txt_path or not self.dsc_path:
            messagebox.showwarning(
                "Fichiers manquants",
                "Charge d'abord un fichier TXT et un fichier DSC."
            )
            return

        if not self.confirm_name_match():
            return

        try:
            field, signal = read_simple_txt(self.txt_path)
            dsc_params = parse_dsc(self.dsc_path)
        except Exception as e:
            messagebox.showerror("Erreur d'import", str(e))
            return

        xpts = to_int(dsc_params.get("XPTS"))
        if xpts is not None and xpts != len(field):
            messagebox.showwarning(
                "Nombre de points",
                f"Le DSC indique XPTS = {xpts}, mais le TXT contient {len(field)} points exploitables.\n\n"
                "Le spectre sera affiché avec les points réellement présents dans le TXT."
            )

        self.field = field
        self.signal = signal
        self.dsc = dsc_params

        self.configure_sliders()
        self.update_info()
        self.plot_spectrum()

    # ---------------- SLIDERS ----------------

    # ---------------- MULTI IMPORT ----------------
    # ---------------- ÉTALON INTERNE Mn²⁺ ----------------
    def _mn_pair_datetime(self, pair):
        dsc = pair.get("dsc") if pair else None
        if not dsc or not os.path.exists(dsc):
            return None
        try:
            return parse_dsc_acquisition_datetime(dsc)
        except Exception:
            return None

    def sort_mn_series_pairs(self):
        selected = self.mn_dta_path
        self.mn_series_pairs.sort(
            key=lambda p: (
                self._mn_pair_datetime(p) is None,
                self._mn_pair_datetime(p) or datetime.max,
                os.path.basename(p.get("dta", "")).lower(),
            )
        )
        if selected:
            for p in self.mn_series_pairs:
                if p.get("dta") == selected:
                    self.mn_dta_path = p.get("dta")
                    self.mn_dsc_path = p.get("dsc")
                    break

    def refresh_mn_series_table(self):
        tree = getattr(self, "mn_series_tree", None)
        if tree is None:
            return

        self.sort_mn_series_pairs()
        selected_path = self.mn_dta_path
        tree.delete(*tree.get_children())

        # Reference time = first chronologically valid DSC acquisition.
        # All displayed times are relative to this t0, never to the previous row.
        valid_datetimes = [
            self._mn_pair_datetime(pair)
            for pair in self.mn_series_pairs
            if self._mn_pair_datetime(pair) is not None
        ]
        t0_dt = min(valid_datetimes) if valid_datetimes else None

        selected_iid = None

        for idx, pair in enumerate(self.mn_series_pairs):
            dta = pair.get("dta")
            dsc = pair.get("dsc")
            key = os.path.abspath(dta) if dta else ""
            saved = self.mn_measurements.get(key, {})
            acquisition_dt = self._mn_pair_datetime(pair)

            if acquisition_dt is None or t0_dt is None:
                time_text = "—"
            else:
                dt_min = (acquisition_dt - t0_dt).total_seconds() / 60.0
                time_text = f"{dt_min:.3f}"

            dsc_text = "OK" if (dsc and os.path.exists(dsc)) else "manquant"
            values = (
                os.path.basename(dta) if dta else "—",
                dsc_text,
                time_text,
                f"{saved['di_mn']:.6g}" if saved.get("di_mn") is not None else "—",
                f"{saved['di_total']:.6g}" if saved.get("di_total") is not None else "—",
                f"{saved['di_radical']:.6g}" if saved.get("di_radical") is not None else "—",
                f"{saved['ratio']:.6g}" if saved.get("ratio") is not None else "—",
            )
            iid = str(idx)
            tree.insert("", "end", iid=iid, values=values)
            if dta == selected_path:
                selected_iid = iid

        if selected_iid is not None:
            tree.selection_set(selected_iid)
            tree.focus(selected_iid)
            tree.see(selected_iid)

        if hasattr(self, "mn_project_label"):
            if self.mn_project_path:
                self.mn_project_label.config(
                    text=f"Projet : {os.path.basename(self.mn_project_path)}"
                )
            else:
                self.mn_project_label.config(text="Projet : non enregistré")

    def on_mn_series_select(self, _event=None):
        tree = getattr(self, "mn_series_tree", None)
        if tree is None:
            return
        selection = tree.selection()
        if not selection:
            return
        try:
            idx = int(selection[0])
            pair = self.mn_series_pairs[idx]
        except Exception:
            return
        self.mn_dta_path = pair.get("dta")
        self.mn_dsc_path = pair.get("dsc")

    def _serialize_mn_measurements(self):
        clean = {}
        for key, value in self.mn_measurements.items():
            item = {}
            for k, v in value.items():
                if isinstance(v, tuple):
                    item[k] = list(v)
                elif isinstance(v, np.generic):
                    item[k] = v.item()
                else:
                    item[k] = v
            clean[key] = item
        return clean

    def save_mn_series_project(self, silent=False):
        if not self.mn_project_path:
            path = filedialog.asksaveasfilename(
                title="Enregistrer la série Mn²⁺",
                defaultextension=".mnseries.json",
                filetypes=[
                    ("Projet Mn²⁺ RPE", "*.mnseries.json"),
                    ("Fichier JSON", "*.json"),
                ]
            )
            if not path:
                return False
            self.mn_project_path = path

        payload = {
            "format": "RPE_Viewer_MnSeries",
            "version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "pairs": [
                {"dta": p.get("dta"), "dsc": p.get("dsc")}
                for p in self.mn_series_pairs
            ],
            "measurements": self._serialize_mn_measurements(),
            "selected_dta": self.mn_dta_path,
        }

        try:
            with open(self.mn_project_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if not silent:
                messagebox.showerror("Enregistrement impossible", str(e))
            return False

        self.refresh_mn_series_table()
        if not silent:
            messagebox.showinfo(
                "Série enregistrée",
                f"Projet enregistré :\n{self.mn_project_path}"
            )
        return True

    def autosave_mn_series_project(self):
        if self.mn_project_path:
            self.save_mn_series_project(silent=True)

    def load_mn_series_project(self):
        path = filedialog.askopenfilename(
            title="Charger une série Mn²⁺",
            filetypes=[
                ("Projet Mn²⁺ RPE", "*.mnseries.json *.json"),
                ("Tous les fichiers", "*.*"),
            ]
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("format") != "RPE_Viewer_MnSeries":
                raise ValueError("Ce fichier n'est pas un projet Mn²⁺ RPE Viewer.")

            pairs = payload.get("pairs", [])
            measurements = payload.get("measurements", {})
            if not isinstance(pairs, list) or not isinstance(measurements, dict):
                raise ValueError("Structure de projet invalide.")

            self.mn_series_pairs = [
                {"dta": p.get("dta"), "dsc": p.get("dsc")}
                for p in pairs if p.get("dta")
            ]
            self.mn_measurements = measurements
            self.mn_project_path = path

            selected = payload.get("selected_dta")
            if selected and any(p.get("dta") == selected for p in self.mn_series_pairs):
                self.mn_dta_path = selected
                self.mn_dsc_path = next(
                    p.get("dsc") for p in self.mn_series_pairs if p.get("dta") == selected
                )
            elif self.mn_series_pairs:
                self.mn_dta_path = self.mn_series_pairs[0].get("dta")
                self.mn_dsc_path = self.mn_series_pairs[0].get("dsc")
            else:
                self.mn_dta_path = None
                self.mn_dsc_path = None

            self.refresh_mn_series_table()
        except Exception as e:
            messagebox.showerror("Chargement impossible", str(e))

    def load_mn_dta(self):
        paths = filedialog.askopenfilenames(
            title="Ajouter des DTA contenant Mn²⁺ + radical",
            filetypes=[("Fichiers Bruker DTA", "*.DTA *.dta"), ("Tous les fichiers", "*.*")]
        )
        if not paths:
            return

        known = {
            os.path.abspath(p.get("dta"))
            for p in self.mn_series_pairs if p.get("dta")
        }

        for path in paths:
            apath = os.path.abspath(path)
            if apath in known:
                continue

            stem = os.path.splitext(path)[0]
            companion = None
            for candidate in (stem + ".DSC", stem + ".dsc"):
                if os.path.exists(candidate):
                    companion = candidate
                    break

            self.mn_series_pairs.append({"dta": path, "dsc": companion})
            known.add(apath)

        self.sort_mn_series_pairs()
        if self.mn_series_pairs and not self.mn_dta_path:
            self.mn_dta_path = self.mn_series_pairs[0]["dta"]
            self.mn_dsc_path = self.mn_series_pairs[0].get("dsc")

        self.refresh_mn_series_table()
        self.autosave_mn_series_project()

    def load_mn_dsc(self):
        if not self.mn_series_pairs:
            messagebox.showwarning("Aucun spectre", "Ajoute d'abord au moins un DTA.")
            return

        path = filedialog.askopenfilename(
            title="Sélectionner le DSC associé",
            filetypes=[("Fichiers DSC", "*.DSC *.dsc"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return

        target = None
        if self.mn_dta_path:
            for pair in self.mn_series_pairs:
                if pair.get("dta") == self.mn_dta_path:
                    target = pair
                    break
        if target is None:
            target = self.mn_series_pairs[0]

        target["dsc"] = path
        self.mn_dta_path = target["dta"]
        self.mn_dsc_path = path
        self.refresh_mn_series_table()
        self.autosave_mn_series_project()

    def reset_mn_internal_standard(self):
        self.mn_dta_path = None
        self.mn_dsc_path = None
        self.mn_series_pairs = []
        self.mn_measurements = {}
        self.mn_project_path = None
        self.refresh_mn_series_table()

    def _compute_global_integral_profile(self, field_g, signal, edge_percent=5.0):
        x = np.asarray(field_g, dtype=float)
        y = np.asarray(signal, dtype=float)

        if x.size != y.size or x.size < 10:
            raise ValueError("Spectre invalide pour l'intégration globale.")

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        pct = min(20.0, max(1.0, float(edge_percent)))
        n_edge = max(3, int(round(x.size * pct / 100.0)))
        n_edge = min(n_edge, max(3, x.size // 4))

        xl = float(np.mean(x[:n_edge]))
        xr = float(np.mean(x[-n_edge:]))
        yl = float(np.mean(y[:n_edge]))
        yr = float(np.mean(y[-n_edge:]))

        if xr == xl:
            deriv_base = np.full_like(y, 0.5 * (yl + yr))
        else:
            deriv_base = yl + (yr - yl) * (x - xl) / (xr - xl)

        ycorr = y - deriv_base
        dx = np.diff(x)

        first_raw = np.zeros_like(x)
        first_raw[1:] = np.cumsum((ycorr[:-1] + ycorr[1:]) * 0.5 * dx)

        fl = float(np.mean(first_raw[:n_edge]))
        fr = float(np.mean(first_raw[-n_edge:]))

        if xr == xl:
            first_base = np.full_like(first_raw, 0.5 * (fl + fr))
        else:
            first_base = fl + (fr - fl) * (x - xl) / (xr - xl)

        first_corr = first_raw - first_base

        second = np.zeros_like(x)
        second[1:] = np.cumsum((first_corr[:-1] + first_corr[1:]) * 0.5 * dx)

        return {
            "field_g": x,
            "raw": y,
            "corrected": ycorr,
            "first": first_corr,
            "second_curve": second,
            "edge_percent": pct,
        }

    def _global_window_double_integral(self, profile, window_g):
        if profile is None or window_g is None:
            raise ValueError("Profil global ou fenêtre manquante.")

        a, b = window_g
        a, b = min(float(a), float(b)), max(float(a), float(b))

        x = np.asarray(profile["field_g"], dtype=float)
        second = np.asarray(profile["second_curve"], dtype=float)

        a = max(a, float(x[0]))
        b = min(b, float(x[-1]))
        if b <= a:
            raise ValueError("Fenêtre vide.")

        s_a = float(np.interp(a, x, second))
        s_b = float(np.interp(b, x, second))

        mask = (x >= a) & (x <= b)
        xw = x[mask]
        if xw.size == 0 or xw[0] > a:
            xw = np.insert(xw, 0, a)
        if xw[-1] < b:
            xw = np.append(xw, b)

        return {
            "field": xw,
            "raw": np.interp(xw, x, profile["raw"]),
            "corrected": np.interp(xw, x, profile["corrected"]),
            "first": np.interp(xw, x, profile["first"]),
            "second_curve": np.interp(xw, x, second) - s_a,
            "double_integral": s_b - s_a,
        }


    def open_mn_internal_standard_window(self):
        if self.mn_series_pairs and not self.mn_dta_path:
            self.mn_dta_path = self.mn_series_pairs[0].get("dta")
            self.mn_dsc_path = self.mn_series_pairs[0].get("dsc")

        if not self.mn_dta_path or not self.mn_dsc_path:
            messagebox.showwarning(
                "Fichiers manquants",
                "Charge d'abord le DTA et le DSC du spectre Mn²⁺ + radical."
            )
            return

        try:
            field_g, signal = read_bruker_dta(
                self.mn_dta_path,
                self.mn_dsc_path
            )
        except Exception as e:
            messagebox.showerror(
                "Lecture DTA/DSC impossible",
                str(e)
            )
            return

        x_g = np.asarray(field_g, dtype=float)
        y = np.asarray(signal, dtype=float)

        dsc_params_mn = parse_dsc(self.mn_dsc_path)
        mw_frequency_hz = microwave_frequency_hz_from_dsc(dsc_params_mn)
        if mw_frequency_hz is None:
            messagebox.showerror(
                "Fréquence micro-onde absente",
                "MWFQ n'a pas pu être lu dans le DSC. Le facteur g ne peut pas être calculé."
            )
            return

        x_gfactor = gauss_to_g(x_g, mw_frequency_hz)

        if x_gfactor.size < 10 or y.size != x_gfactor.size:
            messagebox.showerror(
                "Spectre invalide",
                "Le spectre ne contient pas suffisamment de points."
            )
            return

        win = tk.Toplevel(self)
        win.title("Étalon interne Mn²⁺ — analyse pas à pas")
        win.geometry("1320x860")
        win.minsize(1020, 680)

        main = ttk.Frame(win, padding=8)
        main.pack(fill="both", expand=True)

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        control_outer = ttk.Frame(main, width=385)
        control_outer.pack(side="right", fill="y")
        control_outer.pack_propagate(False)

        controls_canvas = tk.Canvas(
            control_outer,
            width=360,
            highlightthickness=0
        )
        controls_scrollbar = ttk.Scrollbar(
            control_outer,
            orient="vertical",
            command=controls_canvas.yview
        )
        controls_canvas.configure(
            yscrollcommand=controls_scrollbar.set
        )

        controls_scrollbar.pack(side="right", fill="y")
        controls_canvas.pack(side="left", fill="both", expand=True)

        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw"
        )

        def sync_controls_scroll(_event=None):
            controls.update_idletasks()
            controls_canvas.configure(
                scrollregion=controls_canvas.bbox("all")
            )

        def fit_controls_width(event):
            controls_canvas.itemconfigure(
                controls_window,
                width=event.width
            )

        def controls_mousewheel(event):
            if event.delta:
                controls_canvas.yview_scroll(
                    int(-event.delta / 120),
                    "units"
                )

        def bind_controls_wheel(_event=None):
            controls_canvas.bind_all(
                "<MouseWheel>",
                controls_mousewheel
            )

        def unbind_controls_wheel(_event=None):
            controls_canvas.unbind_all("<MouseWheel>")

        controls.bind("<Configure>", sync_controls_scroll)
        controls_canvas.bind("<Configure>", fit_controls_width)
        controls_canvas.bind("<Enter>", bind_controls_wheel)
        controls_canvas.bind("<Leave>", unbind_controls_wheel)

        fig = Figure(figsize=(8.7, 6.1), dpi=100)
        ax = fig.add_subplot(111)
        signal_line, = ax.plot(x_gfactor, y, linewidth=1.1)
        ax.set_xlabel("g")
        ax.set_ylabel("Intensité RPE (a.u.)")
        ax.set_title(os.path.basename(self.mn_dta_path))
        ax.grid(False)

        canv = FigureCanvasTkAgg(fig, master=plot_frame)
        canv.draw()
        canv.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canv, toolbar_frame)
        toolbar.update()

        xmin_full = float(np.min(x_gfactor))
        xmax_full = float(np.max(x_gfactor))
        ymin_full = float(np.min(y))
        ymax_full = float(np.max(y))
        yspan_full = max(ymax_full - ymin_full, 1e-12)
        full_ylim = (
            ymin_full - 0.05 * yspan_full,
            ymax_full + 0.05 * yspan_full
        )
        ax.set_xlim(xmin_full, xmax_full)
        ax.set_ylim(*full_ylim)

        # Stored analysis state. Windows are kept in mT for display,
        # converted to gauss only at integration time.
        state = {
            "mn_window_mt": None,
            "total_window_mt": None,
            "mn_result": None,
            "total_result": None,
            "mn_lines": [],
            "total_lines": [],
            "active_adjust": None,
            "drag_target": None,
            "period_window_mt": None,
            "period_lines": [],
            "period_width_mt": None,
            "period_locked": False,
            "locked_drag_anchor": None,
            "locked_drag_origin": None,
        }

        # Throttled live double integration while a window is moved.
        live_update_job = None
        live_update_delay_ms = 70

        # Optional popup showing derivative / first integral / cumulative second integral.
        integration_view = {
            "window": None,
            "canvas": None,
            "axes": None,
            "source_var": None,
            "di_var": None,
        }

        baseline_enabled_var = tk.BooleanVar(value=False)
        baseline_edge_percent_var = tk.DoubleVar(value=5.0)

        global_profile = self._compute_global_integral_profile(
            x_g,
            y,
            edge_percent=float(baseline_edge_percent_var.get())
        )

        show_zero_line_var = tk.BooleanVar(value=True)
        zero_line_artist = None

        # ============================================================
        # Série de spectres
        # ============================================================
        series_box = ttk.LabelFrame(
            controls,
            text="Série de spectres",
            padding=8
        )
        series_box.pack(fill="x", pady=(0, 8))

        series_names = [
            os.path.basename(p.get("dta", ""))
            for p in self.mn_series_pairs
            if p.get("dta")
        ]
        current_series_index = 0
        for i, p in enumerate(self.mn_series_pairs):
            if p.get("dta") == self.mn_dta_path:
                current_series_index = i
                break

        current_series_var = tk.StringVar(
            value=(
                os.path.basename(self.mn_dta_path)
                if self.mn_dta_path else ""
            )
        )

        series_combo = ttk.Combobox(
            series_box,
            textvariable=current_series_var,
            values=series_names,
            state="readonly",
            width=34
        )
        series_combo.pack(fill="x")

        save_status_var = tk.StringVar(value="Mesures non enregistrées")
        ttk.Label(
            series_box,
            textvariable=save_status_var,
            foreground="#555555"
        ).pack(anchor="w", pady=(5, 0))

        nav_series_row = ttk.Frame(series_box)
        nav_series_row.pack(fill="x", pady=(6, 0))

        ttk.Button(
            nav_series_row,
            text="◀ Précédent",
            command=lambda: switch_mn_series(-1)
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))

        ttk.Button(
            nav_series_row,
            text="Suivant ▶",
            command=lambda: switch_mn_series(1)
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        manual_box = ttk.LabelFrame(
            controls,
            text="Valeurs DI à appliquer",
            padding=8
        )
        manual_box.pack(fill="x", pady=(0, 8))

        manual_mn_var = tk.StringVar(value="")
        manual_total_var = tk.StringVar(value="")

        manual_row1 = ttk.Frame(manual_box)
        manual_row1.pack(fill="x")
        ttk.Label(manual_row1, text="DI Mn", width=10).pack(side="left")
        ttk.Entry(manual_row1, textvariable=manual_mn_var).pack(
            side="left", fill="x", expand=True
        )

        manual_row2 = ttk.Frame(manual_box)
        manual_row2.pack(fill="x", pady=(5, 0))
        ttk.Label(manual_row2, text="DI Total", width=10).pack(side="left")
        ttk.Entry(manual_row2, textvariable=manual_total_var).pack(
            side="left", fill="x", expand=True
        )

        ttk.Button(
            manual_box,
            text="Appliquer au tableau Mn²⁺",
            command=lambda: apply_manual_measurements()
        ).pack(fill="x", pady=(7, 0))

        ttk.Label(
            manual_box,
            text="Les valeurs calculées par les fenêtres remplissent automatiquement ces cases. Tu peux aussi les modifier à la main avant d'appliquer.",
            wraplength=325,
            foreground="#555555"
        ).pack(anchor="w", pady=(5, 0))

        view_var = tk.StringVar()
        direct_mode_var = tk.StringVar(value="Mode : navigation normale")

        ttk.Button(
            controls,
            text="Effacer les repères",
            command=lambda: reset_selections()
        ).pack(fill="x", pady=(0, 8))

        viewer_button_box = ttk.LabelFrame(
            controls,
            text="Visualisation de l'intégration",
            padding=8
        )
        viewer_button_box.pack(fill="x", pady=(0, 8))

        ttk.Button(
            viewer_button_box,
            text="Voir spectre + intégrales",
            command=lambda: open_integration_viewer()
        ).pack(fill="x")

        ttk.Label(
            viewer_button_box,
            text=(
                "Affiche le spectre dérivé, la 1re intégrale et "
                "la 2e intégrale cumulative."
            ),
            wraplength=325,
            foreground="#555555"
        ).pack(anchor="w", pady=(5, 0))

        period_box = ttk.LabelFrame(
            controls,
            text="Pic Mn²⁺",
            padding=8
        )
        period_box.pack(fill="x", pady=(0, 8))

        ttk.Label(
            period_box,
            text=(
                "Place simplement les deux bornes autour du pic Mn²⁺, "
                "puis calcule la double intégrale."
            ),
            wraplength=325,
            justify="left"
        ).pack(anchor="w")

        period_var = tk.StringVar(value="Bornes : non définies")
        period_lock_var = tk.StringVar(value="")

        ttk.Label(
            period_box,
            textvariable=period_var,
            font=("TkDefaultFont", 9, "bold")
        ).pack(anchor="w", pady=(7, 0))

        period_row1 = ttk.Frame(period_box)
        period_row1.pack(fill="x", pady=(7, 0))

        ttk.Button(
            period_row1,
            text="Régler pic",
            command=lambda: start_simple_mn_peak_adjust()
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        ttk.Button(
            period_row1,
            text="Calculer DI Mn",
            command=lambda: calculate_simple_mn_peak()
        ).pack(side="left", fill="x", expand=True)

        signal_box = ttk.LabelFrame(
            controls,
            text="Baseline / repère zéro",
            padding=8
        )
        signal_box.pack(fill="x", pady=(0, 8))

        top_signal_row = ttk.Frame(signal_box)
        top_signal_row.pack(fill="x")

        ttk.Checkbutton(
            top_signal_row,
            text="Corriger la baseline",
            variable=baseline_enabled_var
        ).pack(side="left")

        ttk.Label(
            top_signal_row,
            text="Bords %"
        ).pack(side="left", padx=(10, 3))

        baseline_spin = tk.Spinbox(
            top_signal_row,
            from_=1.0,
            to=20.0,
            increment=0.5,
            textvariable=baseline_edge_percent_var,
            width=5
        )
        baseline_spin.pack(side="left")

        ttk.Label(
            signal_box,
            text=(
                "La correction soustrait une baseline linéaire estimée à partir "
                "des moyennes des extrémités du spectre."
            ),
            wraplength=325,
            foreground="#555555"
        ).pack(anchor="w", pady=(5, 7))

        ttk.Separator(
            signal_box,
            orient="horizontal"
        ).pack(fill="x", pady=(2, 6))

        ttk.Checkbutton(
            signal_box,
            text="Afficher la ligne y = 0",
            variable=show_zero_line_var
        ).pack(anchor="w")

        ttk.Label(
            signal_box,
            text=(
                "La ligne horizontale y = 0 sert uniquement de repère visuel "
                "pour caler le signal Mn. Elle n'intervient pas dans le calcul."
            ),
            wraplength=325,
            foreground="#555555"
        ).pack(anchor="w", pady=(5, 0))

        live_box = ttk.LabelFrame(
            controls,
            text="Double intégrale en temps réel",
            padding=8
        )
        live_box.pack(fill="x", pady=(0, 8))

        live_view_var = tk.StringVar(value="Vue actuelle : —")
        live_di_var = tk.StringVar(value="DI de la vue : —")
        live_mn_var = tk.StringVar(value="DI Mn : —")
        live_total_var = tk.StringVar(value="DI Mn + radical : —")
        live_ratio_var = tk.StringVar(value="Rapport : —")

        ttk.Label(
            live_box,
            textvariable=live_view_var
        ).pack(anchor="w")

        ttk.Label(
            live_box,
            textvariable=live_di_var,
            font=("TkDefaultFont", 11, "bold")
        ).pack(anchor="w", pady=(4, 0))

        ttk.Separator(
            live_box,
            orient="horizontal"
        ).pack(fill="x", pady=7)

        ttk.Label(
            live_box,
            textvariable=live_mn_var
        ).pack(anchor="w")

        ttk.Label(
            live_box,
            textvariable=live_total_var
        ).pack(anchor="w", pady=(3, 0))

        ttk.Label(
            live_box,
            textvariable=live_ratio_var,
            font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", pady=(4, 0))

        ttk.Button(
            live_box,
            text="Prendre la vue actuelle comme Mn²⁺ + radical",
            command=lambda: capture_total()
        ).pack(fill="x", pady=(8, 0))

        ttk.Label(
            live_box,
            text="Mode additif : baseline et intégrales calculées une seule fois sur le spectre complet.",
            wraplength=325,
            foreground="#444444"
        ).pack(anchor="w", pady=(0, 5))

        ttk.Label(
            live_box,
            text=(
                "La DI de la vue se met à jour lorsque tu zoomes ou déplaces "
                "la zone X. Les valeurs Mn et total correspondent aux fenêtres mémorisées."
            ),
            wraplength=325,
            foreground="#555555"
        ).pack(anchor="w", pady=(6, 0))

        # Stored text values used by the calculation logic.
        mn_window_var = tk.StringVar(value="Fenêtre Mn : non définie")
        mn_di_var = tk.StringVar(value="Double intégrale Mn : —")
        total_window_var = tk.StringVar(value="Fenêtre totale : non définie")
        total_di_var = tk.StringVar(value="Double intégrale Mn + radical : —")

        # ============================================================
        # Final result
        # ============================================================
        result_box = ttk.LabelFrame(
            controls,
            text="Résultat",
            padding=8
        )
        result_box.pack(fill="x", pady=(0, 8))

        radical_di_var = tk.StringVar(value="DI radical : —")
        ratio_var = tk.StringVar(value="(Total − Mn) / Mn : —")

        ttk.Label(
            result_box,
            textvariable=radical_di_var
        ).pack(anchor="w")

        ttk.Label(
            result_box,
            textvariable=ratio_var,
            font=("TkDefaultFont", 11, "bold")
        ).pack(anchor="w", pady=(5, 0))

        ttk.Label(
            result_box,
            text="Rapport = [DI(Mn + radical) − DI(Mn)] / DI(Mn)",
            wraplength=325
        ).pack(anchor="w", pady=(6, 0))

        # ============================================================
        # Helpers
        # ============================================================
        def rebuild_global_profile():
            nonlocal global_profile
            try:
                pct = float(baseline_edge_percent_var.get())
            except Exception:
                pct = 5.0
            global_profile = self._compute_global_integral_profile(
                x_g, y, edge_percent=pct
            )

        def displayed_signal():
            if not baseline_enabled_var.get():
                return y.copy()
            return np.interp(
                x_g,
                global_profile["field_g"],
                global_profile["corrected"]
            )

        def refresh_signal_aids():
            nonlocal zero_line_artist

            # Update displayed signal with or without visual baseline correction.
            signal_line.set_ydata(displayed_signal())

            # Recreate zero line so it always remains clearly visible on top.
            if zero_line_artist is not None:
                try:
                    zero_line_artist.remove()
                except Exception:
                    pass
                zero_line_artist = None

            if show_zero_line_var.get():
                zero_line_artist = ax.axhline(
                    0.0,
                    color="#555555",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.85,
                    zorder=5
                )

            canv.draw_idle()

        def current_xlim_sorted():
            a, b = ax.get_xlim()
            return (min(float(a), float(b)), max(float(a), float(b)))

        def current_ylim_sorted():
            a, b = ax.get_ylim()
            return (min(float(a), float(b)), max(float(a), float(b)))

        def update_view_label(_ax=None):
            xa, xb = current_xlim_sorted()
            ya, yb = current_ylim_sorted()
            view_var.set(
                f"Vue actuelle : g = {xa:.6f}–{xb:.6f}\n"
                f"Y = {ya:.4g}–{yb:.4g}"
            )

        def on_xlim_changed(_ax=None):
            update_view_label()
            schedule_live_di()

        ax.callbacks.connect("xlim_changed", on_xlim_changed)
        ax.callbacks.connect("ylim_changed", update_view_label)

        def remove_lines(key):
            for artist in state[key]:
                try:
                    artist.remove()
                except Exception:
                    pass
            state[key] = []

        def draw_window_lines(window_mt, kind):
            if kind == "mn":
                key = "mn_lines"
                color = "#1f5aa6"
                linestyle = "-"
            elif kind == "total":
                key = "total_lines"
                color = "#444444"
                linestyle = "--"
            else:
                key = "period_lines"
                color = "#777777"
                linestyle = ":"

            remove_lines(key)

            a, b = window_mt
            active = state.get("active_adjust") == kind
            if kind == "mn" and state.get("active_adjust") == "locked_mn":
                active = True
            linewidth = 2.2 if active else 1.4

            left = ax.axvline(
                a,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth
            )
            right = ax.axvline(
                b,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth
            )
            state[key] = [left, right]
            canv.draw_idle()

        def redraw_saved_windows():
            if state["mn_window_mt"] is not None:
                draw_window_lines(state["mn_window_mt"], "mn")
            else:
                remove_lines("mn_lines")

            if state["total_window_mt"] is not None:
                draw_window_lines(state["total_window_mt"], "total")
            else:
                remove_lines("total_lines")

            if state["period_window_mt"] is not None:
                draw_window_lines(state["period_window_mt"], "period")
            else:
                remove_lines("period_lines")

            canv.draw_idle()

        def auto_y_current_x():
            xa, xb = current_xlim_sorted()
            mask = (x_gfactor >= xa) & (x_gfactor <= xb)
            yy = displayed_signal()[mask]
            yy = yy[np.isfinite(yy)]

            if yy.size == 0:
                return

            lo = float(np.min(yy))
            hi = float(np.max(yy))
            span = hi - lo
            pad = 0.10 * span if span > 0 else max(abs(hi) * 0.10, 1.0)
            ax.set_ylim(lo - pad, hi + pad)
            canv.draw_idle()

        def full_view():
            ax.set_xlim(xmin_full, xmax_full)
            ax.set_ylim(*full_ylim)
            canv.draw_idle()

        def zoom_saved(window_mt):
            if window_mt is None:
                return
            a, b = window_mt
            width = max(b - a, (xmax_full - xmin_full) / 1000.0)
            margin = 0.20 * width
            ax.set_xlim(
                max(xmin_full, a - margin),
                min(xmax_full, b + margin)
            )
            auto_y_current_x()
            canv.draw_idle()

        def recalc_final():
            mn_result = state["mn_result"]
            total_result = state["total_result"]

            if mn_result is None or total_result is None:
                radical_di_var.set("DI radical : —")
                ratio_var.set("(Total − Mn) / Mn : —")
                return

            di_mn = float(mn_result["double_integral"])
            di_total = float(total_result["double_integral"])
            di_radical = di_total - di_mn

            radical_di_var.set(
                f"DI radical = total − Mn : {di_radical:.8g}"
            )

            if abs(di_mn) <= np.finfo(float).eps:
                ratio_var.set(
                    "(Total − Mn) / Mn : impossible (DI Mn ≈ 0)"
                )
            else:
                ratio = di_radical / di_mn
                ratio_var.set(
                    f"(Total − Mn) / Mn = {ratio:.8g}"
                )

        def update_period_label():
            window = state.get("period_window_mt")
            if window is None:
                if state.get("mn_window_mt") is not None:
                    a, b = state["mn_window_mt"]
                    period_var.set(
                        f"Bornes utilisées : g = {a:.6f}–{b:.6f}"
                    )
                else:
                    period_var.set("Bornes : non définies")
                return

            a, b = window
            period_var.set(
                f"Bornes pic Mn²⁺ : g = {a:.6f}–{b:.6f}"
            )

        def start_simple_mn_peak_adjust():
            # Reuse the previous Mn bounds if they already exist.
            if state.get("period_window_mt") is None:
                if state.get("mn_window_mt") is not None:
                    state["period_window_mt"] = tuple(state["mn_window_mt"])
                else:
                    a, b = current_xlim_sorted()
                    state["period_window_mt"] = (a, b)

            set_active_adjust("period")
            update_period_label()
            redraw_saved_windows()

        def calculate_simple_mn_peak():
            window = state.get("period_window_mt")
            if window is None:
                messagebox.showwarning(
                    "Bornes non définies",
                    "Clique d'abord sur « Régler pic » et place les deux bornes.",
                    parent=win
                )
                return

            a, b = window
            state["mn_window_mt"] = (a, b)

            # The editable grey boundaries become the validated Mn window.
            state["period_window_mt"] = None
            state["period_locked"] = False
            state["period_width_mt"] = None
            state["active_adjust"] = None
            state["drag_target"] = None
            direct_mode_var.set("Mode : navigation normale")

            recalculate_kind("mn", show_error=True)
            update_period_label()
            refresh_live_di()
            redraw_saved_windows()

        def capture_period_from_view():
            a, b = current_xlim_sorted()
            if b <= a:
                return
            state["period_window_mt"] = (a, b)
            state["period_locked"] = False
            state["period_width_mt"] = None
            period_lock_var.set("")
            update_period_label()
            redraw_saved_windows()

        def lock_period():
            window = state.get("period_window_mt")
            if window is None:
                messagebox.showwarning(
                    "Période non définie",
                    "Définis d'abord la période pic-à-pic.",
                    parent=win
                )
                return
            a, b = window
            width = b - a
            if not np.isfinite(width) or width <= 0:
                return
            state["period_width_mt"] = float(width)
            state["period_locked"] = True
            period_lock_var.set(
                f"Largeur verrouillée : Δg = {width:.6f}"
            )
            set_active_adjust(None)

        def unlock_period():
            state["period_locked"] = False
            state["period_width_mt"] = None
            state["locked_drag_anchor"] = None
            state["locked_drag_origin"] = None
            period_lock_var.set("")
            if state.get("active_adjust") == "locked_mn":
                set_active_adjust(None)

        def build_locked_window(center):
            width = state.get("period_width_mt")
            if not state.get("period_locked") or width is None:
                messagebox.showwarning(
                    "Période non verrouillée",
                    "Verrouille d'abord la période pic-à-pic.",
                    parent=win
                )
                return None

            half = 0.5 * width
            a = center - half
            b = center + half

            if a < xmin_full:
                b += xmin_full - a
                a = xmin_full
            if b > xmax_full:
                a -= b - xmax_full
                b = xmax_full

            a = max(xmin_full, a)
            b = min(xmax_full, b)
            return (a, b)

        def center_locked_mn_on_view():
            xa, xb = current_xlim_sorted()
            window = build_locked_window(0.5 * (xa + xb))
            if window is None:
                return
            state["mn_window_mt"] = window
            a, b = window
            mn_window_var.set(
                f"Fenêtre Mn verrouillée : g = {a:.6f}–{b:.6f}   Δg = {b-a:.6f}"
            )
            recalculate_kind("mn", show_error=True)
            redraw_saved_windows()

        def window_from_current_view(kind):
            a, b = current_xlim_sorted()
            if b <= a:
                return

            if kind == "mn":
                state["mn_window_mt"] = (a, b)
                mn_window_var.set(f"Fenêtre Mn : g = {a:.6f}–{b:.6f}")
            else:
                state["total_window_mt"] = (a, b)
                total_window_var.set(f"Fenêtre totale : g = {a:.6f}–{b:.6f}")

            redraw_saved_windows()

        def set_active_adjust(kind):
            if kind == "period":
                if state["period_window_mt"] is None:
                    capture_period_from_view()
                state["active_adjust"] = "period"
                state["drag_target"] = None
                direct_mode_var.set(
                    "Mode : régler pic Mn²⁺ — déplacer les deux bornes"
                )
                redraw_saved_windows()
                return

            if kind == "locked_mn":
                if not state.get("period_locked"):
                    messagebox.showwarning(
                        "Période non verrouillée",
                        "Verrouille d'abord la période pic-à-pic.",
                        parent=win
                    )
                    return
                if state["mn_window_mt"] is None:
                    center_locked_mn_on_view()
                    if state["mn_window_mt"] is None:
                        return
                state["active_adjust"] = "locked_mn"
                state["drag_target"] = None
                direct_mode_var.set(
                    "Mode : fenêtre Mn verrouillée — glisser la fenêtre entière gauche/droite"
                )
                redraw_saved_windows()
                return

            if kind == "mn" and state.get("period_locked"):
                # While locked, individual Mn boundaries must not change the width.
                set_active_adjust("locked_mn")
                return

            if kind == "mn" and state["mn_window_mt"] is None:
                window_from_current_view("mn")
            elif kind == "total" and state["total_window_mt"] is None:
                window_from_current_view("total")

            state["active_adjust"] = kind
            state["drag_target"] = None

            if kind == "mn":
                direct_mode_var.set(
                    "Mode : réglage Mn²⁺ — déplacer les traits bleus"
                )
            elif kind == "total":
                direct_mode_var.set(
                    "Mode : réglage Mn²⁺ + radical — déplacer les traits gris"
                )
            else:
                direct_mode_var.set("Mode : navigation normale")

            redraw_saved_windows()
            refresh_live_di()

        def recalculate_kind(kind, show_error=False):
            window_mt = (
                state["mn_window_mt"]
                if kind == "mn"
                else state["total_window_mt"]
            )
            if window_mt is None:
                return None

            a, b = window_mt
            try:
                b1 = float(g_to_gauss(a, mw_frequency_hz))
                b2 = float(g_to_gauss(b, mw_frequency_hz))
                result = self._global_window_double_integral(
                    global_profile,
                    (min(b1, b2), max(b1, b2))
                )
            except Exception as e:
                if show_error:
                    messagebox.showerror(
                        "Double intégration impossible",
                        str(e),
                        parent=win
                    )
                return None

            if kind == "mn":
                state["mn_result"] = result
                mn_window_var.set(f"Fenêtre Mn : g = {a:.6f}–{b:.6f}")
                mn_di_var.set(
                    f"Double intégrale Mn : {result['double_integral']:.8g}"
                )
                manual_mn_var.set(f"{result['double_integral']:.12g}")
            else:
                state["total_result"] = result
                total_window_var.set(
                    f"Fenêtre totale : g = {a:.6f}–{b:.6f}"
                )
                total_di_var.set(
                    f"Double intégrale Mn + radical : {result['double_integral']:.8g}"
                )
                manual_total_var.set(f"{result['double_integral']:.12g}")

            recalc_final()
            update_integration_viewer()
            return result

        def nearest_active_boundary(event):
            kind = state.get("active_adjust")
            if kind not in ("mn", "total", "period"):
                return None
            if event.inaxes != ax or event.x is None:
                return None

            if kind == "mn":
                window_mt = state["mn_window_mt"]
            elif kind == "total":
                window_mt = state["total_window_mt"]
            else:
                window_mt = state["period_window_mt"]

            if window_mt is None:
                return None

            distances = []
            for side, xval in (("left", window_mt[0]), ("right", window_mt[1])):
                xpix = ax.transData.transform((xval, 0))[0]
                distances.append((abs(float(event.x) - float(xpix)), side))

            distance, side = min(distances, key=lambda item: item[0])
            if distance <= 12.0:
                return kind, side
            return None

        def on_mouse_press(event):
            try:
                if toolbar.mode:
                    return
            except Exception:
                pass

            if event.inaxes != ax or event.xdata is None:
                return

            if state.get("active_adjust") == "locked_mn":
                state["locked_drag_anchor"] = float(event.xdata)
                state["locked_drag_origin"] = tuple(state["mn_window_mt"])
                state["drag_target"] = ("locked_mn", "window")
                return

            target = nearest_active_boundary(event)
            if target is not None:
                state["drag_target"] = target

        def on_mouse_motion(event):
            target = state.get("drag_target")
            if target is None or event.inaxes != ax or event.xdata is None:
                return

            kind, side = target

            if kind == "locked_mn":
                anchor = state.get("locked_drag_anchor")
                origin = state.get("locked_drag_origin")
                if anchor is None or origin is None:
                    return
                delta = float(event.xdata) - anchor
                a0, b0 = origin
                width = b0 - a0
                a = a0 + delta
                b = b0 + delta

                if a < xmin_full:
                    a = xmin_full
                    b = a + width
                if b > xmax_full:
                    b = xmax_full
                    a = b - width

                state["mn_window_mt"] = (a, b)
                mn_window_var.set(
                    f"Fenêtre Mn verrouillée : {a:.4f}–{b:.4f} mT   largeur = {width:.4f} mT"
                )
                lines = state["mn_lines"]
                if len(lines) == 2:
                    lines[0].set_xdata([a, a])
                    lines[1].set_xdata([b, b])
                schedule_live_di()
                canv.draw_idle()
                return

            value = min(xmax_full, max(xmin_full, float(event.xdata)))
            min_sep = max(
                (xmax_full - xmin_full) / max(len(x_gfactor) - 1, 1),
                1e-8
            )

            if kind == "mn":
                a, b = state["mn_window_mt"]
                lines = state["mn_lines"]
            elif kind == "total":
                a, b = state["total_window_mt"]
                lines = state["total_lines"]
            else:
                a, b = state["period_window_mt"]
                lines = state["period_lines"]

            if side == "left":
                a = min(value, b - min_sep)
            else:
                b = max(value, a + min_sep)

            if kind == "mn":
                state["mn_window_mt"] = (a, b)
                mn_window_var.set(f"Fenêtre Mn : g = {a:.6f}–{b:.6f}")
            elif kind == "total":
                state["total_window_mt"] = (a, b)
                total_window_var.set(
                    f"Fenêtre totale : g = {a:.6f}–{b:.6f}"
                )
            else:
                state["period_window_mt"] = (a, b)
                state["period_locked"] = False
                state["period_width_mt"] = None
                update_period_label()

            if len(lines) == 2:
                idx = 0 if side == "left" else 1
                xnew = a if side == "left" else b
                lines[idx].set_xdata([xnew, xnew])

            schedule_live_di()
            canv.draw_idle()

        def on_mouse_release(event):
            target = state.get("drag_target")
            if target is None:
                return

            kind, _side = target
            state["drag_target"] = None

            if kind == "locked_mn":
                state["locked_drag_anchor"] = None
                state["locked_drag_origin"] = None
                recalculate_kind("mn", show_error=True)
                redraw_saved_windows()
                return

            if kind == "period":
                update_period_label()
                redraw_saved_windows()
                return

            recalculate_kind(kind, show_error=True)
            refresh_live_di()
            redraw_saved_windows()

        canv.mpl_connect("button_press_event", on_mouse_press)
        canv.mpl_connect("motion_notify_event", on_mouse_motion)
        canv.mpl_connect("button_release_event", on_mouse_release)

        def selected_integration_window_mt():
            source_var = integration_view.get("source_var")
            source = source_var.get() if source_var is not None else "Vue actuelle"

            if source == "Fenêtre Mn":
                return state.get("mn_window_mt")
            if source == "Fenêtre Mn + radical":
                return state.get("total_window_mt")

            return current_xlim_sorted()

        def update_integration_viewer():
            viewer = integration_view.get("window")
            canvas_view = integration_view.get("canvas")
            axes_view = integration_view.get("axes")
            di_text_var = integration_view.get("di_var")

            if (
                viewer is None
                or not viewer.winfo_exists()
                or canvas_view is None
                or axes_view is None
            ):
                return

            window_mt = selected_integration_window_mt()
            if window_mt is None:
                for axis in axes_view:
                    axis.clear()
                axes_view[0].set_title("Fenêtre non définie")
                if di_text_var is not None:
                    di_text_var.set("Double intégrale : —")
                canvas_view.draw_idle()
                return

            a_mt, b_mt = window_mt
            if not np.isfinite(a_mt) or not np.isfinite(b_mt) or a_mt >= b_mt:
                return

            try:
                b1 = float(g_to_gauss(a_mt, mw_frequency_hz))
                b2 = float(g_to_gauss(b_mt, mw_frequency_hz))
                result = self._global_window_double_integral(
                    global_profile,
                    (min(b1, b2), max(b1, b2))
                )
            except Exception as e:
                for axis in axes_view:
                    axis.clear()
                axes_view[0].set_title(f"Intégration impossible : {e}")
                if di_text_var is not None:
                    di_text_var.set("Double intégrale : —")
                canvas_view.draw_idle()
                return

            x_result_mt = gauss_to_g(result["field"], mw_frequency_hz)

            ax_raw, ax_first, ax_second = axes_view

            ax_raw.clear()
            ax_first.clear()
            ax_second.clear()

            # 1) Original derivative in the selected window + baseline-corrected derivative.
            ax_raw.plot(
                x_result_mt,
                result["raw"],
                linewidth=0.9,
                alpha=0.45,
                label="Signal brut"
            )
            ax_raw.plot(
                x_result_mt,
                result["corrected"],
                linewidth=1.2,
                label="Après correction baseline"
            )
            ax_raw.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.55)
            ax_raw.set_ylabel("Signal RPE")
            ax_raw.set_title(
                f"Spectre dérivé — g = {a_mt:.6f} à {b_mt:.6f}"
            )
            ax_raw.legend(loc="best", fontsize=8)
            ax_raw.grid(True, linestyle=":", linewidth=0.45, alpha=0.3)

            # 2) Corrected first integral: absorption-like profile.
            ax_first.plot(
                x_result_mt,
                result["first"],
                linewidth=1.2
            )
            ax_first.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.55)
            ax_first.set_ylabel("1re intégrale")
            ax_first.set_title("Première intégrale (profil d'absorption corrigé)")
            ax_first.grid(True, linestyle=":", linewidth=0.45, alpha=0.3)

            # 3) Cumulative second integral.
            second_curve = np.asarray(result["second_curve"], dtype=float)
            ax_second.plot(
                x_result_mt,
                second_curve,
                linewidth=1.3
            )
            ax_second.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.55)
            ax_second.set_xlabel("g")
            ax_second.set_ylabel("2e intégrale cumulée")
            ax_second.set_title(
                f"Deuxième intégrale cumulative — valeur finale = {result['double_integral']:.8g}"
            )
            ax_second.grid(True, linestyle=":", linewidth=0.45, alpha=0.3)

            if second_curve.size:
                ax_second.plot(
                    [x_result_mt[-1]],
                    [second_curve[-1]],
                    marker="o",
                    markersize=5
                )

            if di_text_var is not None:
                di_text_var.set(
                    f"Double intégrale finale : {result['double_integral']:.8g}"
                )

            try:
                viewer_fig = canvas_view.figure
                viewer_fig.tight_layout()
            except Exception:
                pass

            canvas_view.draw_idle()

        def open_integration_viewer():
            viewer = integration_view.get("window")
            if viewer is not None and viewer.winfo_exists():
                viewer.lift()
                update_integration_viewer()
                return

            viewer = tk.Toplevel(win)
            viewer.title("Visualisation de la double intégration")
            viewer.geometry("900x850")
            viewer.minsize(760, 650)

            outer_view = ttk.Frame(viewer, padding=8)
            outer_view.pack(fill="both", expand=True)

            top_controls = ttk.Frame(outer_view)
            top_controls.pack(fill="x", pady=(0, 8))

            ttk.Label(
                top_controls,
                text="Fenêtre affichée :"
            ).pack(side="left")

            source_var = tk.StringVar(value="Vue actuelle")
            source_combo = ttk.Combobox(
                top_controls,
                textvariable=source_var,
                values=[
                    "Vue actuelle",
                    "Fenêtre Mn",
                    "Fenêtre Mn + radical"
                ],
                state="readonly",
                width=22
            )
            source_combo.pack(side="left", padx=(6, 10))

            di_text_var = tk.StringVar(value="Double intégrale : —")
            ttk.Label(
                top_controls,
                textvariable=di_text_var,
                font=("TkDefaultFont", 10, "bold")
            ).pack(side="left")

            fig_view = Figure(figsize=(8.2, 7.4), dpi=100)
            ax_raw = fig_view.add_subplot(311)
            ax_first = fig_view.add_subplot(312)
            ax_second = fig_view.add_subplot(313)

            canvas_view = FigureCanvasTkAgg(
                fig_view,
                master=outer_view
            )
            canvas_view.draw()
            canvas_view.get_tk_widget().pack(fill="both", expand=True)

            toolbar_view_frame = ttk.Frame(outer_view)
            toolbar_view_frame.pack(fill="x")
            toolbar_view = NavigationToolbar2Tk(
                canvas_view,
                toolbar_view_frame
            )
            toolbar_view.update()

            integration_view["window"] = viewer
            integration_view["canvas"] = canvas_view
            integration_view["axes"] = (ax_raw, ax_first, ax_second)
            integration_view["source_var"] = source_var
            integration_view["di_var"] = di_text_var

            source_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e: update_integration_viewer()
            )

            def close_viewer():
                integration_view["window"] = None
                integration_view["canvas"] = None
                integration_view["axes"] = None
                integration_view["source_var"] = None
                integration_view["di_var"] = None
                viewer.destroy()

            viewer.protocol("WM_DELETE_WINDOW", close_viewer)

            update_integration_viewer()

        def compute_window_di(window_gfactor):
            if window_gfactor is None:
                return None

            ga, gb = window_gfactor
            if not np.isfinite(ga) or not np.isfinite(gb) or ga == gb:
                return None

            try:
                b1 = float(g_to_gauss(ga, mw_frequency_hz))
                b2 = float(g_to_gauss(gb, mw_frequency_hz))
                result = self._global_window_double_integral(
                    global_profile,
                    (min(b1, b2), max(b1, b2))
                )
                return float(result["double_integral"])
            except Exception:
                return None

        def refresh_live_di():
            """
            Live DI is based on the currently visible X-range.
            This works with Matplotlib zoom/pan and does not depend on
            any specific adjustment mode.
            """
            xa, xb = current_xlim_sorted()
            live_view_var.set(
                f"Vue actuelle : g = {xa:.6f}–{xb:.6f}"
            )

            view_di = compute_window_di((xa, xb))
            if view_di is None:
                live_di_var.set("DI de la vue : —")
            else:
                live_di_var.set(
                    f"DI de la vue : {view_di:.8g}"
                )

            di_mn = compute_window_di(state["mn_window_mt"])
            di_total = compute_window_di(state["total_window_mt"])

            if di_mn is None:
                live_mn_var.set("DI Mn : —")
            else:
                live_mn_var.set(f"DI Mn : {di_mn:.8g}")

            if di_total is None:
                live_total_var.set("DI Mn + radical : —")
            else:
                live_total_var.set(
                    f"DI Mn + radical : {di_total:.8g}"
                )

            if (
                di_mn is not None
                and di_total is not None
                and abs(di_mn) > np.finfo(float).eps
            ):
                ratio_live = (di_total - di_mn) / di_mn
                live_ratio_var.set(
                    f"(Total − Mn) / Mn = {ratio_live:.8g}"
                )
            else:
                live_ratio_var.set("Rapport : —")

            update_integration_viewer()

        def schedule_live_di():
            """
            Debounce X-range/mouse events so the interface remains fluid.
            """
            nonlocal live_update_job

            if live_update_job is not None:
                return

            def _run():
                nonlocal live_update_job
                live_update_job = None
                refresh_live_di()

            live_update_job = win.after(
                live_update_delay_ms,
                _run
            )

        def capture_mn():
            if state.get("period_locked"):
                center_locked_mn_on_view()
            else:
                window_from_current_view("mn")
                recalculate_kind("mn", show_error=True)

        def capture_total():
            window_from_current_view("total")
            recalculate_kind("total", show_error=True)
            refresh_live_di()

        def store_current_mn_measurement(di_mn, di_total):
            di_mn = float(di_mn)
            di_total = float(di_total)
            if abs(di_mn) <= np.finfo(float).eps:
                messagebox.showerror(
                    "DI Mn nulle",
                    "Impossible de calculer le rapport avec une DI Mn nulle.",
                    parent=win
                )
                return False

            di_radical = di_total - di_mn
            ratio = di_radical / di_mn

            key = os.path.abspath(self.mn_dta_path)
            self.mn_measurements[key] = {
                "dta": self.mn_dta_path,
                "dsc": self.mn_dsc_path,
                "mw_frequency_hz": mw_frequency_hz,
                "di_mn": di_mn,
                "di_total": di_total,
                "di_radical": di_radical,
                "ratio": ratio,
                "mn_window_g": tuple(state["mn_window_mt"]) if state.get("mn_window_mt") else None,
                "total_window_g": tuple(state["total_window_mt"]) if state.get("total_window_mt") else None,
                "period_window_g": tuple(state["period_window_mt"]) if state.get("period_window_mt") else None,
                "period_width_g": state.get("period_width_mt"),
                "period_locked": bool(state.get("period_locked")),
            }

            radical_di_var.set(f"DI radical : {di_radical:.8g}")
            ratio_var.set(f"(Total − Mn) / Mn : {ratio:.8g}")
            self.refresh_mn_series_table()
            self.autosave_mn_series_project()
            save_status_var.set("Mesures appliquées ✓")
            return True

        def apply_manual_measurements():
            try:
                di_mn = float(manual_mn_var.get().strip().replace(",", "."))
                di_total = float(manual_total_var.get().strip().replace(",", "."))
            except Exception:
                messagebox.showerror(
                    "Valeurs invalides",
                    "Entre une valeur numérique pour DI Mn et DI Total.",
                    parent=win
                )
                return False
            return store_current_mn_measurement(di_mn, di_total)

        def save_current_mn_measurement():
            mn_res = state.get("mn_result")
            total_res = state.get("total_result")
            if mn_res is None or total_res is None:
                return apply_manual_measurements()
            manual_mn_var.set(f"{float(mn_res['double_integral']):.12g}")
            manual_total_var.set(f"{float(total_res['double_integral']):.12g}")
            return apply_manual_measurements()

        def switch_mn_series(delta=0, absolute_index=None):
            if not self.mn_series_pairs:
                return

            current_idx = 0
            for i, pair in enumerate(self.mn_series_pairs):
                if pair.get("dta") == self.mn_dta_path:
                    current_idx = i
                    break

            if absolute_index is not None:
                new_idx = int(absolute_index)
            else:
                new_idx = current_idx + int(delta)

            new_idx = max(0, min(len(self.mn_series_pairs) - 1, new_idx))
            if new_idx == current_idx:
                return

            pair = self.mn_series_pairs[new_idx]
            if not pair.get("dsc"):
                messagebox.showwarning(
                    "DSC manquant",
                    "Ce spectre n'a pas encore de DSC associé.",
                    parent=win
                )
                return

            self.mn_dta_path = pair.get("dta")
            self.mn_dsc_path = pair.get("dsc")
            self.refresh_mn_series_table()
            win.destroy()
            self.open_mn_internal_standard_window()

        def on_series_combo_selected(_event=None):
            name = current_series_var.get()
            for idx, pair in enumerate(self.mn_series_pairs):
                if os.path.basename(pair.get("dta", "")) == name:
                    switch_mn_series(absolute_index=idx)
                    return

        series_combo.bind("<<ComboboxSelected>>", on_series_combo_selected)

        def reset_selections():
            # IMPORTANT: keep the current X/Y view exactly as it is.
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            state["mn_window_mt"] = None
            state["total_window_mt"] = None
            state["mn_result"] = None
            state["total_result"] = None
            state["active_adjust"] = None
            state["drag_target"] = None
            state["period_window_mt"] = None
            state["period_width_mt"] = None
            state["period_locked"] = False
            state["locked_drag_anchor"] = None
            state["locked_drag_origin"] = None

            direct_mode_var.set("Mode : navigation normale")
            period_var.set("Bornes : non définies")
            period_lock_var.set("")

            remove_lines("mn_lines")
            remove_lines("total_lines")
            remove_lines("period_lines")

            mn_window_var.set("Fenêtre Mn : non définie")
            mn_di_var.set("Double intégrale Mn : —")
            total_window_var.set("Fenêtre totale : non définie")
            total_di_var.set("Double intégrale Mn + radical : —")
            radical_di_var.set("DI radical : —")
            ratio_var.set("(Total − Mn) / Mn : —")
            manual_mn_var.set("")
            manual_total_var.set("")

            ax.set_xlim(current_xlim)
            ax.set_ylim(current_ylim)

            refresh_live_di()
            update_integration_viewer()
            canv.draw_idle()

        # ============================================================
        # Buttons
        # ============================================================
        baseline_enabled_var.trace_add(
            "write",
            lambda *_: refresh_signal_aids()
        )
        def on_global_baseline_changed(*_):
            rebuild_global_profile()
            refresh_signal_aids()
            refresh_live_di()
            update_integration_viewer()

        baseline_edge_percent_var.trace_add(
            "write",
            on_global_baseline_changed
        )
        show_zero_line_var.trace_add(
            "write",
            lambda *_: refresh_signal_aids()
        )

        saved_measurement = self.mn_measurements.get(
            os.path.abspath(self.mn_dta_path)
        )
        if saved_measurement:
            if saved_measurement.get("di_mn") is not None:
                manual_mn_var.set(f"{float(saved_measurement['di_mn']):.12g}")
            if saved_measurement.get("di_total") is not None:
                manual_total_var.set(f"{float(saved_measurement['di_total']):.12g}")
            if saved_measurement.get("di_radical") is not None:
                radical_di_var.set(f"DI radical : {float(saved_measurement['di_radical']):.8g}")
            if saved_measurement.get("ratio") is not None:
                ratio_var.set(f"(Total − Mn) / Mn : {float(saved_measurement['ratio']):.8g}")
            if saved_measurement.get("mn_window_g"):
                state["mn_window_mt"] = tuple(saved_measurement["mn_window_g"])
            if saved_measurement.get("total_window_g"):
                state["total_window_mt"] = tuple(saved_measurement["total_window_g"])
            if saved_measurement.get("period_window_g"):
                state["period_window_mt"] = tuple(saved_measurement["period_window_g"])
            state["period_width_mt"] = None
            state["period_locked"] = False

            if state["mn_window_mt"] is not None:
                recalculate_kind("mn", show_error=False)
            if state["total_window_mt"] is not None:
                recalculate_kind("total", show_error=False)
            update_period_label()
            redraw_saved_windows()
            save_status_var.set("Mesures enregistrées ✓")

        update_view_label()
        refresh_signal_aids()
        refresh_live_di()


    def load_multi_dta(self):
        paths = filedialog.askopenfilenames(
            title="Sélectionner plusieurs fichiers DTA",
            filetypes=[("Fichiers Bruker DTA", "*.DTA *.dta"), ("Tous les fichiers", "*.*")]
        )
        if not paths:
            return

        self.multi_dta_paths = list(paths)
        self.build_multi_pairs()

    def load_multi_dsc(self):
        paths = filedialog.askopenfilenames(
            title="Sélectionner plusieurs fichiers DSC",
            filetypes=[("Fichiers DSC", "*.DSC *.dsc"), ("Tous les fichiers", "*.*")]
        )
        if not paths:
            return

        self.multi_dsc_paths = list(paths)
        self.build_multi_pairs()

    def build_multi_pairs(self):
        dta_map = {clean_stem(p): p for p in self.multi_dta_paths}
        dsc_map = {clean_stem(p): p for p in self.multi_dsc_paths}

        previous = {
            row.get("key"): row
            for row in getattr(self, "multi_pairs", [])
            if row.get("key")
        }

        all_keys = sorted(set(dta_map) | set(dsc_map))
        rows = []

        for key in all_keys:
            dta = dta_map.get(key)
            dsc = dsc_map.get(key)
            old = previous.get(key, {})

            acquisition_dt = parse_dsc_acquisition_datetime(dsc) if dsc else None

            status = "ok"
            data_error = None

            if dta is None or dsc is None:
                status = "warning"
            else:
                try:
                    field, signal = read_bruker_dta(dta, dsc)
                    if len(field) != len(signal):
                        raise ValueError("Axe et signal de tailles différentes.")
                except Exception as e:
                    status = "warning"
                    data_error = str(e)

            rows.append({
                "key": key,
                "dta": dta,
                "dsc": dsc,
                "status": status,
                "data_error": data_error,
                "acquisition_dt": acquisition_dt,
                "time_min": None,
                "double_integral": old.get("double_integral"),
                "include": old.get("include", True)
            })

        valid_dts = [
            row["acquisition_dt"]
            for row in rows
            if row.get("status") == "ok" and row.get("acquisition_dt") is not None
        ]

        if valid_dts:
            t0 = min(valid_dts)
            for row in rows:
                dt = row.get("acquisition_dt")
                if dt is not None:
                    row["time_min"] = (dt - t0).total_seconds() / 60.0

            rows.sort(
                key=lambda row: (
                    row.get("acquisition_dt") is None,
                    row.get("acquisition_dt") or datetime.max
                )
            )

        self.multi_pairs = rows

    def _compute_double_integral_arrays(self, field, signal, analysis_window):
        if analysis_window is None:
            raise ValueError("Fenêtre d'analyse manquante.")

        x = np.asarray(field, dtype=float)
        y = np.asarray(signal, dtype=float)

        if x.size != y.size or x.size < 5:
            raise ValueError("Signal invalide ou nombre de points insuffisant.")

        bmin, bmax = analysis_window
        mask = (x >= bmin) & (x <= bmax)

        xw = x[mask]
        yw = y[mask]

        if len(xw) < 5:
            raise ValueError(
                "La fenêtre d'analyse contient trop peu de points pour effectuer une double intégration."
            )

        order = np.argsort(xw)
        xw = xw[order]
        yw = yw[order]

        if xw[-1] == xw[0]:
            raise ValueError("Fenêtre de champ invalide.")

        baseline = yw[0] + (yw[-1] - yw[0]) * (xw - xw[0]) / (xw[-1] - xw[0])
        ycorr = yw - baseline

        dx = np.diff(xw)
        first = np.zeros_like(xw, dtype=float)
        first[1:] = np.cumsum((ycorr[:-1] + ycorr[1:]) * 0.5 * dx)

        first_baseline = (
            first[0]
            + (first[-1] - first[0]) * (xw - xw[0]) / (xw[-1] - xw[0])
        )
        first_corr = first - first_baseline

        second_value = float(np.trapezoid(first_corr, xw))

        second_curve = np.zeros_like(xw, dtype=float)
        second_curve[1:] = np.cumsum(
            (first_corr[:-1] + first_corr[1:]) * 0.5 * dx
        )

        return {
            "field": xw,
            "raw": yw,
            "corrected": ycorr,
            "first": first_corr,
            "second_curve": second_curve,
            "double_integral": second_value,
        }

    def compute_double_integral(self, txt_path, analysis_window):
        # Kept for Visualiser / Suivi cinétique.
        if not txt_path:
            raise ValueError("Fichier TXT manquant.")
        field, signal = read_simple_txt(txt_path)
        return self._compute_double_integral_arrays(field, signal, analysis_window)

    def compute_double_integral_dta(self, dta_path, dsc_path, analysis_window):
        # Multi-spectra reference calculation.
        field, signal = read_bruker_dta(dta_path, dsc_path)
        return self._compute_double_integral_arrays(field, signal, analysis_window)

    def compute_double_integral_dta_g(self, dta_path, dsc_path, analysis_window_g):
        """Double integration with the user window defined in spectroscopic g."""
        params = parse_dsc(dsc_path)
        freq = microwave_frequency_hz_from_dsc(params)
        if freq is None:
            raise ValueError("MWFQ absent du DSC : impossible de convertir la fenêtre g.")
        g1, g2 = analysis_window_g
        b1 = float(g_to_gauss(g1, freq))
        b2 = float(g_to_gauss(g2, freq))
        return self.compute_double_integral_dta(
            dta_path, dsc_path, (min(b1, b2), max(b1, b2))
        )

    def compute_double_integral_txt_g(self, txt_path, dsc_path, analysis_window_g):
        """Double integration of a temporary TXT with the window defined in g."""
        params = parse_dsc(dsc_path)
        freq = microwave_frequency_hz_from_dsc(params)
        if freq is None:
            raise ValueError("MWFQ absent du DSC : impossible de convertir la fenêtre g.")
        g1, g2 = analysis_window_g
        b1 = float(g_to_gauss(g1, freq))
        b2 = float(g_to_gauss(g2, freq))
        return self.compute_double_integral(
            txt_path, (min(b1, b2), max(b1, b2))
        )

    def open_spectra_overlay_window(
        self,
        parent,
        spectra_items,
        title="Superposition des spectres",
        analysis_window=None,
        x_axis="mT"
    ):
        """
        Configurable publication-style overlay editor.

        spectra_items: list of dicts containing
            label, field, signal, included (optional), time_min (optional)
        """
        if not spectra_items:
            messagebox.showwarning(
                "Aucun spectre",
                "Aucun spectre exploitable n'est disponible pour la superposition.",
                parent=parent
            )
            return

        ow = tk.Toplevel(parent)
        ow.title(title)
        ow.geometry("1480x860")
        ow.minsize(1080, 680)

        main = ttk.Frame(ow, padding=8)
        main.pack(fill="both", expand=True)

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        control_outer = ttk.LabelFrame(
            main,
            text="Mise en forme de la figure",
            padding=6
        )
        control_outer.pack(side="right", fill="y")

        ctrl_canvas = tk.Canvas(
            control_outer,
            width=460,
            highlightthickness=0
        )
        ctrl_scroll = ttk.Scrollbar(
            control_outer,
            orient="vertical",
            command=ctrl_canvas.yview
        )
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)

        ctrl_scroll.pack(side="right", fill="y")
        ctrl_canvas.pack(side="left", fill="both", expand=True)

        controls = ttk.Frame(ctrl_canvas)
        ctrl_window = ctrl_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw"
        )

        def sync_controls(event=None):
            controls.update_idletasks()
            ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))

        def fit_control_width(event):
            ctrl_canvas.itemconfigure(ctrl_window, width=event.width)

        controls.bind("<Configure>", sync_controls)
        ctrl_canvas.bind("<Configure>", fit_control_width)

        fig = Figure(figsize=(8.8, 6.5), dpi=100)
        ax = fig.add_subplot(111)

        canv = FigureCanvasTkAgg(fig, master=plot_frame)
        canv.draw()
        canv.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canv, toolbar_frame)
        toolbar.update()

        # ============================================================
        # VARIABLES GÉNÉRALES
        # ============================================================
        title_var = tk.StringVar(value=title)
        xlabel_var = tk.StringVar(value=("g" if x_axis == "g" else "Champ magnétique (mT)"))
        ylabel_var = tk.StringVar(value="Intensité RPE (a.u.)")

        font_family_var = tk.StringVar(value="Arial")
        title_size_var = tk.DoubleVar(value=14)
        axis_label_size_var = tk.DoubleVar(value=12)
        tick_size_var = tk.DoubleVar(value=10)

        figure_width_cm_var = tk.DoubleVar(value=18.0)
        figure_height_cm_var = tk.DoubleVar(value=13.0)
        export_dpi_var = tk.IntVar(value=300)
        transparent_var = tk.BooleanVar(value=False)
        tight_layout_var = tk.BooleanVar(value=True)

        legend_var = tk.BooleanVar(value=len(spectra_items) <= 12)
        legend_title_var = tk.StringVar(value="")
        legend_size_var = tk.DoubleVar(value=9)
        legend_title_size_var = tk.DoubleVar(value=10)
        legend_frame_var = tk.BooleanVar(value=False)
        legend_pos_var = tk.StringVar(value="Automatique")

        use_window_var = tk.BooleanVar(value=analysis_window is not None)
        manual_xlim_var = tk.BooleanVar(value=False)
        manual_ylim_var = tk.BooleanVar(value=False)
        xmin_var = tk.StringVar(value="")
        xmax_var = tk.StringVar(value="")
        ymin_var = tk.StringVar(value="")
        ymax_var = tk.StringVar(value="")

        grid_var = tk.BooleanVar(value=True)
        grid_minor_var = tk.BooleanVar(value=False)
        grid_style_var = tk.StringVar(value="Pointillés")
        grid_width_var = tk.DoubleVar(value=0.5)
        grid_alpha_var = tk.DoubleVar(value=0.35)

        spine_width_var = tk.DoubleVar(value=1.0)
        show_top_spine_var = tk.BooleanVar(value=True)
        show_right_spine_var = tk.BooleanVar(value=True)
        tick_direction_var = tk.StringVar(value="Extérieur")
        tick_length_var = tk.DoubleVar(value=4.0)
        tick_width_var = tk.DoubleVar(value=1.0)

        axes_bg_var = tk.StringVar(value="#FFFFFF")
        figure_bg_var = tk.StringVar(value="#FFFFFF")

        # ============================================================
        # TEXTE / POLICE
        # ============================================================
        text_box = ttk.LabelFrame(controls, text="Titre, axes et police", padding=7)
        text_box.pack(fill="x", pady=(0, 8))

        ttk.Label(text_box, text="Titre").grid(row=0, column=0, sticky="w")
        ttk.Entry(text_box, textvariable=title_var).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(6, 0)
        )

        ttk.Label(text_box, text="Axe X").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(text_box, textvariable=xlabel_var).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(6, 0), pady=(5, 0)
        )

        ttk.Label(text_box, text="Axe Y").grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(text_box, textvariable=ylabel_var).grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=(6, 0), pady=(5, 0)
        )

        ttk.Label(text_box, text="Police").grid(row=3, column=0, sticky="w", pady=(5, 0))
        font_combo = ttk.Combobox(
            text_box,
            textvariable=font_family_var,
            values=[
                "Arial",
                "Calibri",
                "Times New Roman",
                "Cambria",
                "DejaVu Sans",
                "DejaVu Serif",
                "Courier New",
            ],
            state="normal",
            width=18
        )
        font_combo.grid(row=3, column=1, sticky="ew", padx=(6, 10), pady=(5, 0))

        ttk.Label(text_box, text="Titre").grid(row=3, column=2, sticky="e", pady=(5, 0))
        tk.Spinbox(
            text_box, from_=6, to=40, increment=1,
            textvariable=title_size_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=3, column=3, sticky="w", padx=(4, 0), pady=(5, 0))

        ttk.Label(text_box, text="Labels axes").grid(row=4, column=0, sticky="w", pady=(5, 0))
        tk.Spinbox(
            text_box, from_=6, to=30, increment=1,
            textvariable=axis_label_size_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=4, column=1, sticky="w", padx=(6, 10), pady=(5, 0))

        ttk.Label(text_box, text="Ticks").grid(row=4, column=2, sticky="e", pady=(5, 0))
        tk.Spinbox(
            text_box, from_=5, to=24, increment=1,
            textvariable=tick_size_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=4, column=3, sticky="w", padx=(4, 0), pady=(5, 0))

        for c in range(4):
            text_box.grid_columnconfigure(c, weight=1 if c == 1 else 0)

        # ============================================================
        # LIMITES DES AXES
        # ============================================================
        axis_box = ttk.LabelFrame(controls, text="Limites des axes", padding=7)
        axis_box.pack(fill="x", pady=(0, 8))

        window_cb = ttk.Checkbutton(
            axis_box,
            text="Limiter X à la fenêtre d'analyse",
            variable=use_window_var
        )
        window_cb.grid(row=0, column=0, columnspan=4, sticky="w")
        if analysis_window is None:
            window_cb.config(state="disabled")

        ttk.Checkbutton(
            axis_box,
            text="Limites X manuelles",
            variable=manual_xlim_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

        ttk.Label(axis_box, text="X min").grid(row=2, column=0, sticky="w")
        ttk.Entry(axis_box, textvariable=xmin_var, width=10).grid(
            row=2, column=1, sticky="ew", padx=(4, 8)
        )
        ttk.Label(axis_box, text="X max").grid(row=2, column=2, sticky="w")
        ttk.Entry(axis_box, textvariable=xmax_var, width=10).grid(
            row=2, column=3, sticky="ew", padx=(4, 0)
        )

        ttk.Checkbutton(
            axis_box,
            text="Limites Y manuelles",
            variable=manual_ylim_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(axis_box, text="Y min").grid(row=4, column=0, sticky="w")
        ttk.Entry(axis_box, textvariable=ymin_var, width=10).grid(
            row=4, column=1, sticky="ew", padx=(4, 8)
        )
        ttk.Label(axis_box, text="Y max").grid(row=4, column=2, sticky="w")
        ttk.Entry(axis_box, textvariable=ymax_var, width=10).grid(
            row=4, column=3, sticky="ew", padx=(4, 0)
        )

        for c in (1, 3):
            axis_box.grid_columnconfigure(c, weight=1)

        # ============================================================
        # GRILLE / AXES / TICKS
        # ============================================================
        grid_box = ttk.LabelFrame(controls, text="Grille, bordures et ticks", padding=7)
        grid_box.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(
            grid_box, text="Afficher la grille", variable=grid_var
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Checkbutton(
            grid_box, text="Grille mineure", variable=grid_minor_var
        ).grid(row=0, column=2, columnspan=2, sticky="w")

        ttk.Label(grid_box, text="Style grille").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Combobox(
            grid_box,
            textvariable=grid_style_var,
            values=["Trait plein", "Tirets", "Pointillés", "Tiret-point"],
            state="readonly",
            width=12
        ).grid(row=1, column=1, sticky="ew", padx=(4, 8), pady=(5, 0))

        ttk.Label(grid_box, text="Épaisseur").grid(row=1, column=2, sticky="w", pady=(5, 0))
        tk.Spinbox(
            grid_box, from_=0.1, to=3.0, increment=0.1,
            textvariable=grid_width_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=1, column=3, sticky="w", padx=(4, 0), pady=(5, 0))

        ttk.Label(grid_box, text="Opacité grille").grid(row=2, column=0, sticky="w", pady=(5, 0))
        tk.Spinbox(
            grid_box, from_=0.0, to=1.0, increment=0.05,
            textvariable=grid_alpha_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=2, column=1, sticky="w", padx=(4, 8), pady=(5, 0))

        ttk.Label(grid_box, text="Épaisseur axes").grid(row=2, column=2, sticky="w", pady=(5, 0))
        tk.Spinbox(
            grid_box, from_=0.2, to=5.0, increment=0.1,
            textvariable=spine_width_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=2, column=3, sticky="w", padx=(4, 0), pady=(5, 0))

        ttk.Checkbutton(
            grid_box, text="Bordure haute", variable=show_top_spine_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

        ttk.Checkbutton(
            grid_box, text="Bordure droite", variable=show_right_spine_var
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=(5, 0))

        ttk.Label(grid_box, text="Direction ticks").grid(row=4, column=0, sticky="w", pady=(5, 0))
        ttk.Combobox(
            grid_box,
            textvariable=tick_direction_var,
            values=["Extérieur", "Intérieur", "Les deux"],
            state="readonly",
            width=12
        ).grid(row=4, column=1, sticky="ew", padx=(4, 8), pady=(5, 0))

        ttk.Label(grid_box, text="Longueur").grid(row=4, column=2, sticky="w", pady=(5, 0))
        tk.Spinbox(
            grid_box, from_=0, to=15, increment=0.5,
            textvariable=tick_length_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=4, column=3, sticky="w", padx=(4, 0), pady=(5, 0))

        ttk.Label(grid_box, text="Épaisseur ticks").grid(row=5, column=0, sticky="w", pady=(5, 0))
        tk.Spinbox(
            grid_box, from_=0.2, to=5.0, increment=0.1,
            textvariable=tick_width_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=5, column=1, sticky="w", padx=(4, 8), pady=(5, 0))

        for c in (1, 3):
            grid_box.grid_columnconfigure(c, weight=1)

        # ============================================================
        # FONDS
        # ============================================================
        bg_box = ttk.LabelFrame(controls, text="Fond", padding=7)
        bg_box.pack(fill="x", pady=(0, 8))

        ttk.Label(bg_box, text="Fond du graphique").grid(row=0, column=0, sticky="w")
        axes_bg_btn = tk.Button(
            bg_box, text="Choisir", bg=axes_bg_var.get(),
            activebackground=axes_bg_var.get(), width=8
        )
        axes_bg_btn.grid(row=0, column=1, sticky="w", padx=(6, 15))

        ttk.Label(bg_box, text="Fond de la figure").grid(row=0, column=2, sticky="w")
        figure_bg_btn = tk.Button(
            bg_box, text="Choisir", bg=figure_bg_var.get(),
            activebackground=figure_bg_var.get(), width=8
        )
        figure_bg_btn.grid(row=0, column=3, sticky="w", padx=(6, 0))

        def choose_bg(var, button, window_title):
            result = colorchooser.askcolor(
                color=var.get(),
                parent=ow,
                title=window_title
            )
            if result and result[1]:
                var.set(result[1])
                button.config(bg=result[1], activebackground=result[1])
                refresh_plot()

        axes_bg_btn.config(
            command=lambda: choose_bg(
                axes_bg_var, axes_bg_btn, "Couleur de fond du graphique"
            )
        )
        figure_bg_btn.config(
            command=lambda: choose_bg(
                figure_bg_var, figure_bg_btn, "Couleur de fond de la figure"
            )
        )

        # ============================================================
        # LÉGENDE
        # ============================================================
        legend_box = ttk.LabelFrame(controls, text="Légende", padding=7)
        legend_box.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(
            legend_box, text="Afficher la légende", variable=legend_var
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Checkbutton(
            legend_box, text="Cadre de légende", variable=legend_frame_var
        ).grid(row=0, column=2, columnspan=2, sticky="w")

        ttk.Label(legend_box, text="Position").grid(row=1, column=0, sticky="w", pady=(5, 0))
        legend_pos_combo = ttk.Combobox(
            legend_box,
            textvariable=legend_pos_var,
            values=[
                "Automatique",
                "Haut droite",
                "Haut gauche",
                "Bas droite",
                "Bas gauche",
                "Centre droite",
                "Centre gauche",
                "Haut centre",
                "Bas centre",
            ],
            state="readonly",
            width=16
        )
        legend_pos_combo.grid(row=1, column=1, sticky="ew", padx=(4, 8), pady=(5, 0))

        ttk.Label(legend_box, text="Titre").grid(row=1, column=2, sticky="w", pady=(5, 0))
        ttk.Entry(
            legend_box, textvariable=legend_title_var
        ).grid(row=1, column=3, sticky="ew", padx=(4, 0), pady=(5, 0))

        ttk.Label(legend_box, text="Taille texte").grid(row=2, column=0, sticky="w", pady=(5, 0))
        tk.Spinbox(
            legend_box, from_=5, to=24, increment=1,
            textvariable=legend_size_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=2, column=1, sticky="w", padx=(4, 8), pady=(5, 0))

        ttk.Label(legend_box, text="Taille titre").grid(row=2, column=2, sticky="w", pady=(5, 0))
        tk.Spinbox(
            legend_box, from_=5, to=24, increment=1,
            textvariable=legend_title_size_var, width=5,
            command=lambda: refresh_plot()
        ).grid(row=2, column=3, sticky="w", padx=(4, 0), pady=(5, 0))

        for c in (1, 3):
            legend_box.grid_columnconfigure(c, weight=1)

        # ============================================================
        # DÉGRADÉ
        # ============================================================
        gradient_box = ttk.LabelFrame(controls, text="Dégradé de couleurs", padding=7)
        gradient_box.pack(fill="x", pady=(0, 8))

        ttk.Label(gradient_box, text="Palette").grid(row=0, column=0, sticky="w")
        gradient_var = tk.StringVar(value="Viridis")
        gradient_combo = ttk.Combobox(
            gradient_box,
            textvariable=gradient_var,
            values=[
                "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
                "Turbo", "Coolwarm", "Blues", "Greens", "Reds",
                "Purples", "Gris", "Personnalisé",
            ],
            state="readonly",
            width=18
        )
        gradient_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        reverse_gradient_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            gradient_box,
            text="Inverser le dégradé",
            variable=reverse_gradient_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

        custom_start = tk.StringVar(value="#440154")
        custom_end = tk.StringVar(value="#FDE725")

        custom_row = ttk.Frame(gradient_box)
        custom_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        ttk.Label(custom_row, text="Personnalisé :").pack(side="left")

        start_btn = tk.Button(
            custom_row,
            text="Début",
            bg=custom_start.get(),
            activebackground=custom_start.get(),
            width=7
        )
        start_btn.pack(side="left", padx=(6, 4))

        end_btn = tk.Button(
            custom_row,
            text="Fin",
            bg=custom_end.get(),
            activebackground=custom_end.get(),
            width=7
        )
        end_btn.pack(side="left")

        gradient_box.grid_columnconfigure(1, weight=1)

        # ============================================================
        # COURBES
        # ============================================================
        style_map = {
            "Trait plein": "-",
            "Tirets": "--",
            "Pointillés": ":",
            "Tiret-point": "-.",
        }

        mpl_cmaps = {
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno",
            "Magma": "magma",
            "Cividis": "cividis",
            "Turbo": "turbo",
            "Coolwarm": "coolwarm",
            "Blues": "Blues",
            "Greens": "Greens",
            "Reds": "Reds",
            "Purples": "Purples",
            "Gris": "Greys",
        }

        cmap = matplotlib.colormaps.get_cmap("viridis")
        n_items = max(len(spectra_items) - 1, 1)
        curve_states = []

        curves_box = ttk.LabelFrame(controls, text="Courbes individuelles", padding=7)
        curves_box.pack(fill="x", pady=(0, 8))

        for idx, item in enumerate(spectra_items):
            if x_axis == "g":
                freq = item.get("frequency_hz")
                if freq is None:
                    raise ValueError("Fréquence MWFQ manquante pour une courbe en g.")
                field = gauss_to_g(item["field"], freq)
            else:
                field = gauss_to_mt(item["field"])
            signal = np.asarray(item["signal"], dtype=float)

            color = matplotlib.colors.to_hex(cmap(idx / n_items))
            visible_default = bool(item.get("included", True))
            default_label = item.get("label", f"Spectre {idx + 1}")

            line, = ax.plot(
                field,
                signal,
                linewidth=1.2,
                linestyle="-",
                color=color,
                visible=visible_default,
                label=default_label
            )

            state = {
                "line": line,
                "visible": tk.BooleanVar(value=visible_default),
                "color": tk.StringVar(value=color),
                "style": tk.StringVar(value="Trait plein"),
                "width": tk.DoubleVar(value=1.2),
                "legend_label": tk.StringVar(value=default_label),
            }
            curve_states.append(state)

            row = ttk.LabelFrame(
                curves_box,
                text=default_label,
                padding=6
            )
            row.pack(fill="x", pady=3)

            first_line = ttk.Frame(row)
            first_line.pack(fill="x")

            ttk.Checkbutton(
                first_line,
                text="Afficher",
                variable=state["visible"]
            ).pack(side="left")

            time_min = item.get("time_min")
            if time_min is not None:
                ttk.Label(
                    first_line,
                    text=f"t = {float(time_min):.3f} min"
                ).pack(side="right")

            legend_line = ttk.Frame(row)
            legend_line.pack(fill="x", pady=(5, 0))
            ttk.Label(legend_line, text="Légende").pack(side="left")
            ttk.Entry(
                legend_line,
                textvariable=state["legend_label"]
            ).pack(side="left", fill="x", expand=True, padx=(6, 0))

            second_line = ttk.Frame(row)
            second_line.pack(fill="x", pady=(5, 0))

            ttk.Label(second_line, text="Couleur").pack(side="left")

            color_button = tk.Button(
                second_line,
                text="   ",
                bg=color,
                activebackground=color,
                width=3,
                relief="solid",
                bd=1
            )
            color_button.pack(side="left", padx=(5, 10))
            state["color_button"] = color_button

            ttk.Label(second_line, text="Style").pack(side="left")
            style_combo = ttk.Combobox(
                second_line,
                textvariable=state["style"],
                values=list(style_map.keys()),
                state="readonly",
                width=12
            )
            style_combo.pack(side="left", padx=(5, 10))

            ttk.Label(second_line, text="Épaisseur").pack(side="left")
            width_spin = tk.Spinbox(
                second_line,
                from_=0.3,
                to=5.0,
                increment=0.1,
                textvariable=state["width"],
                width=5
            )
            width_spin.pack(side="left", padx=(5, 0))

            def choose_color(st=state, btn=color_button):
                result = colorchooser.askcolor(
                    color=st["color"].get(),
                    parent=ow,
                    title="Choisir la couleur de la courbe"
                )
                if result and result[1]:
                    st["color"].set(result[1])
                    btn.config(
                        bg=result[1],
                        activebackground=result[1]
                    )
                    refresh_plot()

            color_button.config(command=choose_color)

            state["visible"].trace_add("write", lambda *_: refresh_plot())
            state["legend_label"].trace_add("write", lambda *_: refresh_plot())
            style_combo.bind("<<ComboboxSelected>>", lambda _e: refresh_plot())
            width_spin.config(command=lambda: refresh_plot())
            width_spin.bind("<Return>", lambda _e: refresh_plot())
            width_spin.bind("<FocusOut>", lambda _e: refresh_plot())

        # ============================================================
        # EXPORT
        # ============================================================
        export_box = ttk.LabelFrame(controls, text="Dimensions et export PNG", padding=7)
        export_box.pack(fill="x", pady=(0, 8))

        ttk.Label(export_box, text="Largeur (cm)").grid(row=0, column=0, sticky="w")
        tk.Spinbox(
            export_box, from_=5, to=50, increment=0.5,
            textvariable=figure_width_cm_var, width=7
        ).grid(row=0, column=1, sticky="w", padx=(4, 10))

        ttk.Label(export_box, text="Hauteur (cm)").grid(row=0, column=2, sticky="w")
        tk.Spinbox(
            export_box, from_=5, to=40, increment=0.5,
            textvariable=figure_height_cm_var, width=7
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))

        ttk.Label(export_box, text="DPI").grid(row=1, column=0, sticky="w", pady=(5, 0))
        tk.Spinbox(
            export_box, from_=72, to=1200, increment=25,
            textvariable=export_dpi_var, width=7
        ).grid(row=1, column=1, sticky="w", padx=(4, 10), pady=(5, 0))

        ttk.Checkbutton(
            export_box,
            text="Fond transparent",
            variable=transparent_var
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(5, 0))

        ttk.Checkbutton(
            export_box,
            text="Rogner au contenu",
            variable=tight_layout_var
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(5, 0))

        # ============================================================
        # FONCTIONS DE STYLE
        # ============================================================
        legend_loc_map = {
            "Automatique": "best",
            "Haut droite": "upper right",
            "Haut gauche": "upper left",
            "Bas droite": "lower right",
            "Bas gauche": "lower left",
            "Centre droite": "center right",
            "Centre gauche": "center left",
            "Haut centre": "upper center",
            "Bas centre": "lower center",
        }

        tick_direction_map = {
            "Extérieur": "out",
            "Intérieur": "in",
            "Les deux": "inout",
        }

        def safe_float(var, default):
            try:
                return float(var.get())
            except Exception:
                return default

        def parse_manual_limit(var):
            txt = str(var.get()).strip().replace(",", ".")
            if txt == "":
                return None
            return float(txt)

        def choose_custom_start():
            result = colorchooser.askcolor(
                color=custom_start.get(),
                parent=ow,
                title="Couleur de début du dégradé"
            )
            if result and result[1]:
                custom_start.set(result[1])
                start_btn.config(bg=result[1], activebackground=result[1])

        def choose_custom_end():
            result = colorchooser.askcolor(
                color=custom_end.get(),
                parent=ow,
                title="Couleur de fin du dégradé"
            )
            if result and result[1]:
                custom_end.set(result[1])
                end_btn.config(bg=result[1], activebackground=result[1])

        start_btn.config(command=choose_custom_start)
        end_btn.config(command=choose_custom_end)

        def interpolate_hex(c1, c2, t):
            rgb1 = np.array(matplotlib.colors.to_rgb(c1), dtype=float)
            rgb2 = np.array(matplotlib.colors.to_rgb(c2), dtype=float)
            rgb = rgb1 * (1.0 - t) + rgb2 * t
            return matplotlib.colors.to_hex(rgb)

        def apply_gradient():
            n = len(curve_states)
            if n == 0:
                return

            palette_name = gradient_var.get()
            reverse = reverse_gradient_var.get()

            for idx, state in enumerate(curve_states):
                t = idx / max(n - 1, 1)
                if reverse:
                    t = 1.0 - t

                if palette_name == "Personnalisé":
                    color = interpolate_hex(
                        custom_start.get(),
                        custom_end.get(),
                        t
                    )
                else:
                    cmap_name = mpl_cmaps.get(palette_name, "viridis")
                    cmap_obj = matplotlib.colormaps.get_cmap(cmap_name)
                    color = matplotlib.colors.to_hex(cmap_obj(t))

                state["color"].set(color)
                state["color_button"].config(
                    bg=color,
                    activebackground=color
                )

            refresh_plot()

        ttk.Button(
            gradient_box,
            text="Appliquer le dégradé à toutes les courbes",
            command=apply_gradient
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(7, 0))

        def set_all_visible(value):
            for state in curve_states:
                state["visible"].set(bool(value))
            refresh_plot()

        visible_buttons = ttk.Frame(curves_box)
        visible_buttons.pack(fill="x", pady=(4, 0))

        ttk.Button(
            visible_buttons,
            text="Tout afficher",
            command=lambda: set_all_visible(True)
        ).pack(side="left", padx=(0, 5))

        ttk.Button(
            visible_buttons,
            text="Tout masquer",
            command=lambda: set_all_visible(False)
        ).pack(side="left")

        def get_visible_lines():
            return [
                state["line"]
                for state in curve_states
                if state["visible"].get()
            ]

        def autoscale_from_visible():
            visible_lines = get_visible_lines()
            if not visible_lines:
                return

            all_x = np.concatenate([
                np.asarray(line.get_xdata(), dtype=float)
                for line in visible_lines
            ])
            all_y = np.concatenate([
                np.asarray(line.get_ydata(), dtype=float)
                for line in visible_lines
            ])

            finite_x = all_x[np.isfinite(all_x)]
            finite_y = all_y[np.isfinite(all_y)]

            if finite_x.size:
                ax.set_xlim(float(np.min(finite_x)), float(np.max(finite_x)))

            if finite_y.size:
                ymin = float(np.min(finite_y))
                ymax = float(np.max(finite_y))
                span = ymax - ymin
                pad = 0.05 * span if span > 0 else max(abs(ymax) * 0.05, 1.0)
                ax.set_ylim(ymin - pad, ymax + pad)

        def refresh_plot(*_):
            family = font_family_var.get().strip() or "Arial"

            for state in curve_states:
                line = state["line"]
                line.set_visible(state["visible"].get())
                line.set_color(state["color"].get())
                line.set_linestyle(style_map.get(state["style"].get(), "-"))
                line.set_label(state["legend_label"].get())
                line.set_linewidth(max(0.1, safe_float(state["width"], 1.2)))

            ax.set_title(
                title_var.get(),
                fontsize=safe_float(title_size_var, 14),
                fontfamily=family
            )
            ax.set_xlabel(
                xlabel_var.get(),
                fontsize=safe_float(axis_label_size_var, 12),
                fontfamily=family
            )
            ax.set_ylabel(
                ylabel_var.get(),
                fontsize=safe_float(axis_label_size_var, 12),
                fontfamily=family
            )

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily(family)
                label.set_fontsize(safe_float(tick_size_var, 10))

            # X/Y limits
            autoscale_from_visible()

            if analysis_window is not None and use_window_var.get():
                if x_axis == "g":
                    ax.set_xlim(float(analysis_window[0]), float(analysis_window[1]))
                else:
                    ax.set_xlim(*(float(analysis_window[0]) / 10.0, float(analysis_window[1]) / 10.0))

            if manual_xlim_var.get():
                try:
                    xmin = parse_manual_limit(xmin_var)
                    xmax = parse_manual_limit(xmax_var)
                    current = ax.get_xlim()
                    ax.set_xlim(
                        xmin if xmin is not None else current[0],
                        xmax if xmax is not None else current[1]
                    )
                except Exception:
                    pass

            if manual_ylim_var.get():
                try:
                    ymin = parse_manual_limit(ymin_var)
                    ymax = parse_manual_limit(ymax_var)
                    current = ax.get_ylim()
                    ax.set_ylim(
                        ymin if ymin is not None else current[0],
                        ymax if ymax is not None else current[1]
                    )
                except Exception:
                    pass

            # Spines
            spine_width = max(0.1, safe_float(spine_width_var, 1.0))
            for name, spine in ax.spines.items():
                spine.set_linewidth(spine_width)
                if name == "top":
                    spine.set_visible(show_top_spine_var.get())
                elif name == "right":
                    spine.set_visible(show_right_spine_var.get())
                else:
                    spine.set_visible(True)

            ax.tick_params(
                axis="both",
                which="major",
                direction=tick_direction_map.get(tick_direction_var.get(), "out"),
                length=max(0.0, safe_float(tick_length_var, 4.0)),
                width=max(0.1, safe_float(tick_width_var, 1.0)),
                labelsize=safe_float(tick_size_var, 10),
            )

            if grid_minor_var.get():
                ax.minorticks_on()
            else:
                ax.minorticks_off()

            grid_style = style_map.get(grid_style_var.get(), ":")
            if grid_var.get():
                ax.grid(
                    True,
                    which="major",
                    linestyle=grid_style,
                    linewidth=max(0.05, safe_float(grid_width_var, 0.5)),
                    alpha=min(1.0, max(0.0, safe_float(grid_alpha_var, 0.35)))
                )
                if grid_minor_var.get():
                    ax.grid(
                        True,
                        which="minor",
                        linestyle=grid_style,
                        linewidth=max(0.05, safe_float(grid_width_var, 0.5) * 0.7),
                        alpha=min(1.0, max(0.0, safe_float(grid_alpha_var, 0.35) * 0.6))
                    )
            else:
                ax.grid(False, which="both")

            ax.set_facecolor(axes_bg_var.get())
            fig.patch.set_facecolor(figure_bg_var.get())

            old_legend = ax.get_legend()
            if old_legend is not None:
                old_legend.remove()

            if legend_var.get():
                handles = get_visible_lines()
                if handles:
                    title_text = legend_title_var.get().strip()
                    leg = ax.legend(
                        handles=handles,
                        fontsize=safe_float(legend_size_var, 9),
                        loc=legend_loc_map.get(legend_pos_var.get(), "best"),
                        title=title_text if title_text else None,
                        frameon=legend_frame_var.get()
                    )
                    for txt in leg.get_texts():
                        txt.set_fontfamily(family)
                        txt.set_fontsize(safe_float(legend_size_var, 9))
                    if leg.get_title() is not None:
                        leg.get_title().set_fontfamily(family)
                        leg.get_title().set_fontsize(
                            safe_float(legend_title_size_var, 10)
                        )

            try:
                if tight_layout_var.get():
                    fig.tight_layout()
            except Exception:
                pass

            canv.draw_idle()

        def export_png():
            path = filedialog.asksaveasfilename(
                parent=ow,
                title="Exporter la superposition en PNG",
                defaultextension=".png",
                filetypes=[("Image PNG", "*.png")]
            )
            if not path:
                return

            try:
                refresh_plot()

                width_cm = max(1.0, safe_float(figure_width_cm_var, 18.0))
                height_cm = max(1.0, safe_float(figure_height_cm_var, 13.0))
                dpi = max(50, int(safe_float(export_dpi_var, 300)))

                old_size = fig.get_size_inches().copy()
                fig.set_size_inches(width_cm / 2.54, height_cm / 2.54)

                if tight_layout_var.get():
                    try:
                        fig.tight_layout()
                    except Exception:
                        pass

                fig.savefig(
                    path,
                    dpi=dpi,
                    transparent=transparent_var.get(),
                    bbox_inches="tight" if tight_layout_var.get() else None,
                    facecolor=fig.get_facecolor(),
                    edgecolor="none"
                )

                fig.set_size_inches(old_size)
                canv.draw_idle()

                messagebox.showinfo(
                    "Export terminé",
                    f"Image PNG exportée avec succès :\n{path}",
                    parent=ow
                )
            except Exception as e:
                messagebox.showerror(
                    "Erreur d'export PNG",
                    str(e),
                    parent=ow
                )

        export_buttons = ttk.Frame(export_box)
        export_buttons.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        ttk.Button(
            export_buttons,
            text="Appliquer la mise en forme",
            command=refresh_plot
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            export_buttons,
            text="Exporter PNG",
            command=export_png
        ).pack(side="right")

        # ============================================================
        # RESET STYLE
        # ============================================================
        def reset_style():
            title_var.set(title)
            xlabel_var.set("Champ magnétique (mT)")
            ylabel_var.set("Intensité RPE (a.u.)")
            font_family_var.set("Arial")
            title_size_var.set(14)
            axis_label_size_var.set(12)
            tick_size_var.set(10)

            legend_var.set(len(spectra_items) <= 12)
            legend_title_var.set("")
            legend_size_var.set(9)
            legend_title_size_var.set(10)
            legend_frame_var.set(False)
            legend_pos_var.set("Automatique")

            use_window_var.set(analysis_window is not None)
            manual_xlim_var.set(False)
            manual_ylim_var.set(False)
            xmin_var.set("")
            xmax_var.set("")
            ymin_var.set("")
            ymax_var.set("")

            grid_var.set(True)
            grid_minor_var.set(False)
            grid_style_var.set("Pointillés")
            grid_width_var.set(0.5)
            grid_alpha_var.set(0.35)

            spine_width_var.set(1.0)
            show_top_spine_var.set(True)
            show_right_spine_var.set(True)
            tick_direction_var.set("Extérieur")
            tick_length_var.set(4.0)
            tick_width_var.set(1.0)

            axes_bg_var.set("#FFFFFF")
            figure_bg_var.set("#FFFFFF")
            axes_bg_btn.config(bg="#FFFFFF", activebackground="#FFFFFF")
            figure_bg_btn.config(bg="#FFFFFF", activebackground="#FFFFFF")

            figure_width_cm_var.set(18.0)
            figure_height_cm_var.set(13.0)
            export_dpi_var.set(300)
            transparent_var.set(False)
            tight_layout_var.set(True)

            for idx, state in enumerate(curve_states):
                state["visible"].set(bool(spectra_items[idx].get("included", True)))
                state["style"].set("Trait plein")
                state["width"].set(1.2)
                state["legend_label"].set(
                    spectra_items[idx].get("label", f"Spectre {idx + 1}")
                )

            gradient_var.set("Viridis")
            reverse_gradient_var.set(False)
            apply_gradient()
            refresh_plot()

        ttk.Button(
            export_box,
            text="Réinitialiser le style",
            command=reset_style
        ).grid(row=4, column=0, columnspan=4, sticky="ew", pady=(7, 0))

        # ============================================================
        # LIVE UPDATE HOOKS
        # ============================================================
        for var in (
            title_var, xlabel_var, ylabel_var, font_family_var,
            manual_xlim_var, manual_ylim_var,
            grid_var, grid_minor_var,
            show_top_spine_var, show_right_spine_var,
            legend_var, legend_title_var, legend_frame_var,
            use_window_var
        ):
            var.trace_add("write", lambda *_: refresh_plot())

        legend_pos_combo.bind("<<ComboboxSelected>>", lambda _e: refresh_plot())
        font_combo.bind("<<ComboboxSelected>>", lambda _e: refresh_plot())

        for var in (
            xmin_var, xmax_var, ymin_var, ymax_var
        ):
            var.trace_add("write", lambda *_: refresh_plot())

        refresh_plot()

    def open_multi_window(self):
        self.build_multi_pairs()

        if not self.multi_dta_paths and not self.multi_dsc_paths:
            messagebox.showwarning(
                "Fichiers manquants",
                "Charge d'abord les fichiers multi DTA et DSC."
            )
            return

        win = tk.Toplevel(self)
        win.title("Analyse multi spectre")
        win.geometry("1220x700")
        win.minsize(980, 560)

        outer = ttk.Frame(win, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text=f"Analyse multi spectre — {len(self.multi_pairs)} ligne(s)",
            font=("TkDefaultFont", 12, "bold")
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            outer,
            text="Association automatique DTA ↔ DSC par nom. Le DTA est utilisé comme signal brut de référence. "
                 "Le temps est calculé automatiquement à partir de DATE + TIME de chaque DSC, "
                 "puis la première mesure est définie à t = 0.",
            wraplength=1160
        ).pack(anchor="w", pady=(0, 10))

        control_bar = ttk.Frame(outer)
        control_bar.pack(fill="x", pady=(0, 8))

        ttk.Label(control_bar, text="Horodatages DSC :").pack(side="left")

        refresh_button = ttk.Button(
            control_bar,
            text="Actualiser depuis les DSC"
        )
        refresh_button.pack(side="left", padx=(6, 16))

        self.multi_window_label = ttk.Label(
            control_bar,
            text="Fenêtre d'analyse : non définie",
            font=("TkDefaultFont", 9, "bold")
        )
        self.multi_window_label.pack(side="left")

        selection_bar_multi = ttk.Frame(outer)
        selection_bar_multi.pack(fill="x", pady=(0, 8))

        # -------- stable table container with both scrollbars --------
        table_wrap = ttk.Frame(outer)
        table_wrap.pack(fill="both", expand=True)

        canvas = tk.Canvas(table_wrap, highlightthickness=0)
        yscroll = ttk.Scrollbar(table_wrap, orient="vertical", command=canvas.yview)
        xscroll = ttk.Scrollbar(table_wrap, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        table = ttk.Frame(canvas)
        table_window = canvas.create_window((0, 0), window=table, anchor="nw")

        # Fixed logical widths: they do not stretch with window resizing.
        COL_W = {
            0: 280,
            1: 280,
            2: 105,
            3: 100,
            4: 190,
            5: 110,
            6: 155,
        }

        def sync_table(event=None):
            table.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

        table.bind("<Configure>", sync_table)

        def read_times_into_rows(show_errors=True):
            missing = [
                i + 1
                for i, row in enumerate(self.multi_pairs)
                if row.get("status") == "ok" and row.get("acquisition_dt") is None
            ]

            if missing and show_errors:
                messagebox.showwarning(
                    "Horodatage DSC manquant",
                    "Impossible de lire DATE et/ou TIME dans certains DSC.\n"
                    "Lignes concernées : " + ", ".join(map(str, missing))
                )

            return not missing

        def open_single_spectrum(row):
            if not row.get("dta") or not row.get("dsc"):
                messagebox.showwarning(
                    "Spectre indisponible",
                    "Une paire DTA/DSC complète est nécessaire pour afficher ce spectre."
                )
                return

            try:
                field, signal = read_bruker_dta(row["dta"], row["dsc"])
                dsc_params = parse_dsc(row["dsc"])
            except Exception as e:
                messagebox.showerror("Erreur d'ouverture", str(e))
                return

            sw = tk.Toplevel(win)
            sw.title(os.path.basename(row["dta"]))
            sw.geometry("900x620")
            sw.minsize(700, 500)

            frame = ttk.Frame(sw, padding=8)
            frame.pack(fill="both", expand=True)

            fig = Figure(figsize=(8, 5.5), dpi=100)
            ax = fig.add_subplot(111)
            freq = microwave_frequency_hz_from_dsc(dsc_params)
            if freq is None:
                messagebox.showerror("MWFQ absent", "Impossible d'afficher ce spectre en g.", parent=sw)
                sw.destroy()
                return
            field_disp = gauss_to_g(field, freq)
            ax.plot(field_disp, signal, linewidth=1.2)
            ax.set_xlabel("g")
            ax.set_ylabel("Intensité RPE (a.u.)")
            ax.set_title(os.path.basename(row["dta"]))

            if self.multi_analysis_window is not None:
                ax.set_xlim(*self.multi_analysis_window)

            fig.tight_layout()

            canvas_spec = FigureCanvasTkAgg(fig, master=frame)
            canvas_spec.draw()
            canvas_spec.get_tk_widget().pack(fill="both", expand=True)

            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill="x")
            toolbar = NavigationToolbar2Tk(canvas_spec, toolbar_frame)
            toolbar.update()

            info_lines = []
            if row.get("time_min") is not None:
                info_lines.append(f"Temps : {row['time_min']:.3f} min")
            if row.get("acquisition_dt") is not None:
                info_lines.append(
                    f"Heure DSC : {row['acquisition_dt'].strftime('%H:%M:%S')}"
                )
            if dsc_params.get("XPTS"):
                info_lines.append(f"Points : {dsc_params['XPTS']}")
            if self.multi_analysis_window is not None:
                info_lines.append(
                    f"Fenêtre : g = {self.multi_analysis_window[0]:.6f}–{self.multi_analysis_window[1]:.6f}"
                )

            if info_lines:
                ttk.Label(
                    frame,
                    text="   |   ".join(info_lines)
                ).pack(anchor="w", pady=(4, 0))

        def define_analysis_window(row):
            if not row.get("dta") or not row.get("dsc"):
                messagebox.showwarning(
                    "Spectre indisponible",
                    "Impossible de définir une fenêtre sans paire DTA/DSC."
                )
                return

            try:
                field, signal = read_bruker_dta(row["dta"], row["dsc"])
                params_ref = parse_dsc(row["dsc"])
                freq_ref = microwave_frequency_hz_from_dsc(params_ref)
                if freq_ref is None:
                    raise ValueError("MWFQ absent du DSC.")
            except Exception as e:
                messagebox.showerror("Erreur d'ouverture", str(e))
                return

            fw = tk.Toplevel(win)
            fw.title("Définir la fenêtre d'analyse")
            fw.geometry("940x690")
            fw.minsize(760, 560)

            topf = ttk.Frame(fw, padding=10)
            topf.pack(fill="both", expand=True)

            ttk.Label(
                topf,
                text="Définis la fenêtre sur ce spectre de référence. "
                     "Après validation, elle sera identique pour tous les spectres.",
                wraplength=880
            ).pack(anchor="w", pady=(0, 8))

            fig = Figure(figsize=(8, 5), dpi=100)
            ax = fig.add_subplot(111)
            field_mt = gauss_to_g(field, freq_ref)
            ax.plot(field_mt, signal, linewidth=1.2)
            ax.set_xlabel("g")
            ax.set_ylabel("Intensité RPE (a.u.)")
            ax.set_title(os.path.basename(row["dta"]))

            xmin_data = float(np.min(field_mt))
            xmax_data = float(np.max(field_mt))

            if self.multi_analysis_window is None:
                init_min, init_max = xmin_data, xmax_data
            else:
                init_min = max(xmin_data, self.multi_analysis_window[0])
                init_max = min(xmax_data, self.multi_analysis_window[1])

            ax.set_xlim(init_min, init_max)
            fig.tight_layout()

            canv = FigureCanvasTkAgg(fig, master=topf)
            canv.draw()
            canv.get_tk_widget().pack(fill="both", expand=True)

            controls = ttk.Frame(topf)
            controls.pack(fill="x", pady=(8, 0))

            min_var = tk.DoubleVar(value=init_min)
            max_var = tk.DoubleVar(value=init_max)

            step = (xmax_data - xmin_data) / max(len(field) - 1, 1)
            if step <= 0:
                step = 0.01

            min_text = tk.StringVar(value=f"Min : {init_min:.6f} g")
            max_text = tk.StringVar(value=f"Max : {init_max:.6f} g")

            ttk.Label(controls, textvariable=min_text).pack(anchor="w")
            min_scale = tk.Scale(
                controls,
                from_=xmin_data,
                to=xmax_data,
                resolution=step,
                orient="horizontal",
                variable=min_var,
                showvalue=False
            )
            min_scale.pack(fill="x")

            ttk.Label(controls, textvariable=max_text).pack(anchor="w", pady=(6, 0))
            max_scale = tk.Scale(
                controls,
                from_=xmin_data,
                to=xmax_data,
                resolution=step,
                orient="horizontal",
                variable=max_var,
                showvalue=False
            )
            max_scale.pack(fill="x")

            guard = {"active": False}

            def update_window(*_):
                if guard["active"]:
                    return

                a = min_var.get()
                b = max_var.get()

                if a >= b:
                    guard["active"] = True
                    if a >= b:
                        if a == min_scale.get():
                            min_var.set(max(xmin_data, b - step))
                        else:
                            max_var.set(min(xmax_data, a + step))
                    guard["active"] = False
                    a = min_var.get()
                    b = max_var.get()

                min_text.set(f"Min : {a:.6f} g")
                max_text.set(f"Max : {b:.6f} g")
                ax.set_xlim(a, b)
                canv.draw_idle()

            min_var.trace_add("write", update_window)
            max_var.trace_add("write", update_window)

            buttons = ttk.Frame(topf)
            buttons.pack(fill="x", pady=(10, 0))

            def validate_window():
                a = float(min_var.get())
                b = float(max_var.get())

                if a >= b:
                    messagebox.showerror(
                        "Fenêtre invalide",
                        "La borne minimale doit être inférieure à la borne maximale."
                    )
                    return

                # Window is stored in g; each DSC converts it back to B for integration.
                self.multi_analysis_window = (a, b)
                self.multi_window_label.config(
                    text=f"Fenêtre d'analyse : g = {a:.6f}–{b:.6f}"
                )

                # Same global window shown on every row.
                for rr in self.multi_pairs:
                    rr["analysis_window"] = self.multi_analysis_window
                    rr["double_integral"] = None

                calculate_all_double_integrals(show_messages=False)
                draw_table()
                fw.destroy()

            ttk.Button(
                buttons,
                text="Valider cette fenêtre pour tous les spectres",
                command=validate_window
            ).pack(side="right")

        def calculate_all_double_integrals(show_messages=True):
            if self.multi_analysis_window is None:
                if show_messages:
                    messagebox.showwarning(
                        "Fenêtre non définie",
                        "Définis d'abord une fenêtre d'analyse commune."
                    )
                return False

            errors = []
            calculated = 0

            for idx, row in enumerate(self.multi_pairs, start=1):
                if not row.get("dta") or not row.get("dsc"):
                    row["double_integral"] = None
                    errors.append((idx, "Paire DTA/DSC incomplète"))
                    continue

                try:
                    result = self.compute_double_integral_dta_g(
                        row["dta"],
                        row["dsc"],
                        self.multi_analysis_window
                    )
                    row["double_integral"] = result["double_integral"]
                    calculated += 1
                except Exception as e:
                    row["double_integral"] = None
                    errors.append((idx, str(e)))

            if show_messages:
                if errors:
                    preview = "\n".join(
                        f"Ligne {i} : {msg}" for i, msg in errors[:8]
                    )
                    if len(errors) > 8:
                        preview += "\n..."
                    messagebox.showwarning(
                        "Double intégration",
                        f"{calculated} spectre(s) calculé(s).\n\n"
                        f"Problèmes rencontrés :\n{preview}"
                    )
                else:
                    messagebox.showinfo(
                        "Double intégration",
                        f"Double intégration calculée pour {calculated} spectre(s)."
                    )

            return calculated > 0

        def open_integration_detail(row):
            if not row.get("dta") or not row.get("dsc"):
                messagebox.showwarning(
                    "Spectre indisponible",
                    "Une paire DTA/DSC complète est nécessaire pour cette intégration."
                )
                return

            if self.multi_analysis_window is None:
                messagebox.showwarning(
                    "Fenêtre non définie",
                    "Définis d'abord la fenêtre d'analyse."
                )
                return

            try:
                result = self.compute_double_integral_dta(
                    row["dta"],
                    row["dsc"],
                    self.multi_analysis_window
                )
                row["double_integral"] = result["double_integral"]
            except Exception as e:
                messagebox.showerror("Erreur d'intégration", str(e))
                return

            iw = tk.Toplevel(win)
            iw.title("Double intégration — " + os.path.basename(row["dta"]))
            iw.geometry("940x760")
            iw.minsize(760, 620)

            frame = ttk.Frame(iw, padding=8)
            frame.pack(fill="both", expand=True)

            fig = Figure(figsize=(8, 7), dpi=100)
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            x = result["field"]
            params_detail = parse_dsc(row["dsc"])
            freq_detail = microwave_frequency_hz_from_dsc(params_detail)
            x_disp = gauss_to_g(x, freq_detail)

            ax1.plot(x_disp, result["raw"], linewidth=1.0, label="Signal brut")
            ax1.plot(x_disp, result["corrected"], linewidth=1.1, label="Signal corrigé")
            ax1.set_ylabel("Signal RPE")
            ax1.legend(loc="best", fontsize=8)
            ax1.set_title(os.path.basename(row["dta"]))

            ax2.plot(x_disp, result["first"], linewidth=1.1)
            ax2.set_ylabel("1re intégrale")

            ax3.plot(x_disp, result["second_curve"], linewidth=1.1)
            ax3.set_xlabel("g")
            ax3.set_ylabel("2e intégrale")

            fig.tight_layout()

            canv = FigureCanvasTkAgg(fig, master=frame)
            canv.draw()
            canv.get_tk_widget().pack(fill="both", expand=True)

            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill="x")
            toolbar = NavigationToolbar2Tk(canv, toolbar_frame)
            toolbar.update()

            ttk.Label(
                frame,
                text=(
                    f"Fenêtre : g = {self.multi_analysis_window[0]:.6f}–"
                    f"{self.multi_analysis_window[1]:.6f}   |   "
                    f"Double intégrale : {result['double_integral']:.6g}"
                ),
                font=("TkDefaultFont", 10, "bold")
            ).pack(anchor="w", pady=(5, 0))

        def collect_kinetic_data():
            if not read_times_into_rows(show_errors=True):
                return None

            if self.multi_analysis_window is None:
                messagebox.showwarning(
                    "Fenêtre non définie",
                    "Définis d'abord la fenêtre d'analyse commune."
                )
                return None

            if not calculate_all_double_integrals(show_messages=False):
                messagebox.showerror(
                    "Double intégration",
                    "Aucune double intégrale exploitable n'a pu être calculée."
                )
                return None

            valid = []
            for row in self.multi_pairs:
                if not row.get("include", True):
                    continue

                t = row.get("time_min", "")
                di = row.get("double_integral")
                if t == "" or di is None:
                    continue

                try:
                    t = float(t)
                    di = abs(float(di))
                except Exception:
                    continue

                if not np.isfinite(di) or di == 0:
                    continue

                if np.isfinite(t):
                    valid.append((t, di, row))

            if len(valid) < 3:
                messagebox.showwarning(
                    "Points insuffisants",
                    "Il faut au moins 3 spectres avec un temps et une double intégrale non nulle."
                )
                return None

            valid.sort(key=lambda z: z[0])
            times = np.asarray([v[0] for v in valid], dtype=float)
            integrals = np.asarray([v[1] for v in valid], dtype=float)
            return times, integrals, valid

        def open_normalized_kinetics():
            collected = collect_kinetic_data()
            if collected is None:
                return

            times, integrals, valid = collected
            ynorm = 100.0 * integrals / integrals[0]
            t0 = times[0]
            trel = times - t0

            kw = tk.Toplevel(win)
            kw.title("Cinétique RPE normalisée")
            kw.geometry("980x790")
            kw.minsize(800, 640)

            frame = ttk.Frame(kw, padding=10)
            frame.pack(fill="both", expand=True)

            fit_options = ttk.LabelFrame(frame, text="Options du fit", padding=8)
            fit_options.pack(fill="x", pady=(0, 8))

            plateau_zero_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                fit_options,
                text="Imposer le plateau à 0 (C = 0)",
                variable=plateau_zero_var
            ).pack(side="left")

            controls = ttk.LabelFrame(frame, text="Extrapolation du fit", padding=8)
            controls.pack(fill="x", pady=(0, 8))

            export_bar = ttk.Frame(frame)
            export_bar.pack(fill="x", pady=(0, 8))

            fig = Figure(figsize=(8.7, 5.8), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(times, ynorm, s=36, label="Double intégrale normalisée")
            fit_line, = ax.plot([], [], linewidth=1.6, label="Fit exponentiel")
            ax.axvline(
                times[-1],
                linestyle=":",
                linewidth=1.0,
                alpha=0.6,
                label="Dernier point mesuré"
            )
            ax.set_xlabel("Temps (min)")
            ax.set_ylabel("Double intégrale normalisée (%)")
            ax.set_title("Cinétique RPE")
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
            ax.legend(loc="best")

            stats_artist = ax.text(
                0.98, 0.97, "",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
            )

            last_time = float(times[-1])
            first_time = float(times[0])
            measured_span = max(last_time - first_time, 1.0)
            max_extrap = first_time + 5.0 * measured_span
            if max_extrap <= last_time:
                max_extrap = last_time + max(measured_span, 10.0)

            extrap_var = tk.DoubleVar(value=last_time)
            extrap_label = tk.StringVar(value=f"Fin d'affichage du fit : {last_time:.2f} min")
            ttk.Label(controls, textvariable=extrap_label).pack(anchor="w")
            tk.Scale(
                controls,
                from_=last_time,
                to=max_extrap,
                resolution=max(measured_span / 500.0, 0.01),
                orient="horizontal",
                variable=extrap_var,
                showvalue=False
            ).pack(fill="x")

            canv = FigureCanvasTkAgg(fig, master=frame)
            canv.draw()
            canv.get_tk_widget().pack(fill="both", expand=True)

            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill="x")
            toolbar = NavigationToolbar2Tk(canv, toolbar_frame)
            toolbar.update()

            fit_state = {}

            def compute_fit():
                try:
                    if plateau_zero_var.get():
                        def model(t, A, k):
                            return A * np.exp(-k * t)

                        p0 = [100.0, 1.0 / max(float(np.ptp(trel)), 1e-6)]
                        popt, _ = curve_fit(
                            model,
                            trel,
                            ynorm,
                            p0=p0,
                            bounds=([0.0, 0.0], [np.inf, np.inf]),
                            maxfev=30000
                        )
                        A, k = [float(v) for v in popt]
                        C = 0.0
                        yfit_pts = model(trel, *popt)
                        fit_state["model"] = model
                        fit_state["popt"] = popt
                        model_name = "y = A·exp(-k·t), C = 0"
                    else:
                        def model(t, A, k, C):
                            return C + A * np.exp(-k * t)

                        y_min = float(np.min(ynorm))
                        span_t = max(float(np.ptp(trel)), 1e-6)
                        p0 = [
                            max(100.0 - y_min, 1.0),
                            1.0 / span_t,
                            max(y_min, 0.0),
                        ]
                        popt, _ = curve_fit(
                            model,
                            trel,
                            ynorm,
                            p0=p0,
                            bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
                            maxfev=30000
                        )
                        A, k, C = [float(v) for v in popt]
                        yfit_pts = model(trel, *popt)
                        fit_state["model"] = model
                        fit_state["popt"] = popt
                        model_name = "y = C + A·exp(-k·t)"

                    ss_res = float(np.sum((ynorm - yfit_pts) ** 2))
                    ss_tot = float(np.sum((ynorm - np.mean(ynorm)) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                    half_life = np.log(2.0) / k if k > 0 else float("inf")

                    fit_state.update({
                        "A": A, "k": k, "C": C,
                        "r2": r2, "half_life": half_life,
                        "model_name": model_name,
                    })

                    stats_artist.set_text(
                        f"Modèle : {model_name}\n"
                        f"A = {A:.4g} %\n"
                        f"C = {C:.4g} %\n"
                        f"k = {k:.6g} min⁻¹\n"
                        f"t½ = {half_life:.4g} min\n"
                        f"R² = {r2:.5f}"
                    )
                    update_extrapolation()
                except Exception as e:
                    messagebox.showerror("Fit exponentiel impossible", str(e), parent=kw)

            def update_extrapolation(*_):
                if "model" not in fit_state:
                    return
                end_abs = max(float(extrap_var.get()), last_time)
                end_rel = end_abs - t0
                dense_rel = np.linspace(0.0, max(end_rel, 1e-9), 700)
                dense_abs = dense_rel + t0
                dense_fit = fit_state["model"](dense_rel, *fit_state["popt"])
                fit_line.set_data(dense_abs, dense_fit)
                extrap_label.set(f"Fin d'affichage du fit : {end_abs:.2f} min")
                ax.relim()
                ax.autoscale_view()
                ax.set_xlim(first_time, end_abs + max(0.5, 0.03 * max(end_abs-first_time, 1)))
                canv.draw_idle()

            plateau_zero_var.trace_add("write", lambda *_: compute_fit())
            extrap_var.trace_add("write", update_extrapolation)

            def export_exponential_data():
                if "model" not in fit_state:
                    return

                end_abs = max(float(extrap_var.get()), last_time)
                end_rel = end_abs - t0
                dense_rel_export = np.linspace(0.0, max(end_rel, 1e-9), 700)
                dense_abs_export = dense_rel_export + t0
                dense_fit_export = fit_state["model"](dense_rel_export, *fit_state["popt"])
                fit_at_exp = fit_state["model"](trel, *fit_state["popt"])

                max_len = max(len(times), len(dense_abs_export))
                rows_export = [
                    ["Cinétique RPE normalisée - fit exponentiel"],
                    ["Fenêtre d'analyse min (g)", float(self.multi_analysis_window[0])],
                    ["Fenêtre d'analyse max (g)", float(self.multi_analysis_window[1])],
                    ["Modèle", fit_state["model_name"]],
                    ["A (%)", fit_state["A"]],
                    ["k (min^-1)", fit_state["k"]],
                    ["C (%)", fit_state["C"]],
                    ["t1/2 (min)", fit_state["half_life"]],
                    ["R2", fit_state["r2"]],
                    ["Fin extrapolation (min)", end_abs],
                    [],
                    [
                        "Temps expérimental (min)",
                        "Double intégrale |DI|",
                        "DI normalisée (%)",
                        "Fit aux temps expérimentaux (%)",
                        "Temps courbe fit (min)",
                        "Fit exponentiel (%)"
                    ]
                ]

                for i in range(max_len):
                    rows_export.append([
                        float(times[i]) if i < len(times) else "",
                        float(integrals[i]) if i < len(integrals) else "",
                        float(ynorm[i]) if i < len(ynorm) else "",
                        float(fit_at_exp[i]) if i < len(fit_at_exp) else "",
                        float(dense_abs_export[i]) if i < len(dense_abs_export) else "",
                        float(dense_fit_export[i]) if i < len(dense_fit_export) else "",
                    ])

                export_table_dialog(
                    kw,
                    "cinetique_RPE_fit_exponentiel.txt",
                    rows_export
                )

            ttk.Button(
                export_bar,
                text="Exporter les données (TXT / Excel)",
                command=export_exponential_data
            ).pack(side="right")

            compute_fit()

        def open_order_determination():
            collected = collect_kinetic_data()
            if collected is None:
                return

            times, integrals, valid = collected
            ratio = integrals / integrals[0]
            t0 = float(times[0])
            trel = times - t0

            ow = tk.Toplevel(win)
            ow.title("Détermination de l'ordre")
            ow.geometry("1080x820")
            ow.minsize(880, 680)

            frame = ttk.Frame(ow, padding=10)
            frame.pack(fill="both", expand=True)

            compare_box = ttk.LabelFrame(
                frame, text="Comparaison automatique des modèles directs", padding=8
            )
            compare_box.pack(fill="x", pady=(0, 8))

            headers_cmp = ["Modèle", "R² direct", "RMSE", "AICc", "ΔAICc"]
            for c, h in enumerate(headers_cmp):
                ttk.Label(
                    compare_box, text=h, font=("TkDefaultFont", 9, "bold"), anchor="center"
                ).grid(row=0, column=c, sticky="ew", padx=6, pady=3)

            cmp_labels = {}
            for r, name in enumerate(["Ordre 0", "Ordre 1", "Ordre 2"], start=1):
                ttk.Label(compare_box, text=name).grid(row=r, column=0, sticky="w", padx=6, pady=3)
                cmp_labels[name] = []
                for c in range(1, 5):
                    lab = ttk.Label(compare_box, text="—", anchor="center")
                    lab.grid(row=r, column=c, sticky="ew", padx=6, pady=3)
                    cmp_labels[name].append(lab)

            for c in range(5):
                compare_box.grid_columnconfigure(c, weight=1)

            conclusion_var = tk.StringVar(value="")
            ttk.Label(
                compare_box, textvariable=conclusion_var,
                font=("TkDefaultFont", 9, "bold"), wraplength=980
            ).grid(row=4, column=0, columnspan=5, sticky="w", padx=6, pady=(8, 3))

            controls = ttk.LabelFrame(frame, text="Linéarisation affichée", padding=8)
            controls.pack(fill="x", pady=(0, 8))
            ttk.Label(controls, text="Ordre :").pack(side="left", padx=(0, 8))

            order_var = tk.StringVar(value="Ordre 1")
            combo = ttk.Combobox(
                controls, textvariable=order_var,
                values=["Ordre 0", "Ordre 1", "Ordre 2"],
                state="readonly", width=12
            )
            combo.pack(side="left")

            export_bar = ttk.Frame(frame)
            export_bar.pack(fill="x", pady=(0, 8))

            fig = Figure(figsize=(9.2, 5.8), dpi=100)
            ax = fig.add_subplot(111)
            canv = FigureCanvasTkAgg(fig, master=frame)
            canv.draw()
            canv.get_tk_widget().pack(fill="both", expand=True)

            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill="x")
            NavigationToolbar2Tk(canv, toolbar_frame).update()

            state = {}

            def direct_metrics():
                y = ratio.astype(float)
                n = len(y)
                span = max(float(np.ptp(trel)), 1e-9)

                def m0(t, k):
                    return 1.0 - k * t

                def m1(t, k):
                    return np.exp(-k * t)

                def m2(t, k):
                    return 1.0 / (1.0 + k * t)

                models = {"Ordre 0": m0, "Ordre 1": m1, "Ordre 2": m2}
                results = {}

                for name, model in models.items():
                    try:
                        popt, _ = curve_fit(
                            model, trel, y, p0=[1.0 / span],
                            bounds=([0.0], [np.inf]), maxfev=30000
                        )
                        pred = model(trel, *popt)
                        residuals = y - pred
                        rss = float(np.sum(residuals ** 2))
                        rmse = float(np.sqrt(rss / n))
                        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")

                        p = 1
                        rss_safe = max(rss, np.finfo(float).tiny)
                        aic = n * np.log(rss_safe / n) + 2 * p
                        aicc = (
                            aic + (2 * p * (p + 1)) / (n - p - 1)
                            if n > p + 1 else float("inf")
                        )

                        results[name] = {
                            "r2": r2, "rmse": rmse, "aicc": float(aicc),
                            "k": float(popt[0]), "pred": pred, "residuals": residuals
                        }
                    except Exception:
                        results[name] = {
                            "r2": float("nan"), "rmse": float("nan"),
                            "aicc": float("inf"), "k": float("nan"),
                            "pred": np.full_like(y, np.nan),
                            "residuals": np.full_like(y, np.nan)
                        }

                finite = [v["aicc"] for v in results.values() if np.isfinite(v["aicc"])]
                best = min(finite) if finite else float("inf")
                for res in results.values():
                    res["delta_aicc"] = (
                        res["aicc"] - best
                        if np.isfinite(res["aicc"]) and np.isfinite(best)
                        else float("inf")
                    )
                return results

            direct_results = direct_metrics()

            for name, res in direct_results.items():
                values = [
                    f'{res["r2"]:.5f}' if np.isfinite(res["r2"]) else "—",
                    f'{res["rmse"]:.5g}' if np.isfinite(res["rmse"]) else "—",
                    f'{res["aicc"]:.3f}' if np.isfinite(res["aicc"]) else "—",
                    f'{res["delta_aicc"]:.3f}' if np.isfinite(res["delta_aicc"]) else "—",
                ]
                for lab, value in zip(cmp_labels[name], values):
                    lab.config(text=value)

            ranked = sorted(
                [(name, res["delta_aicc"]) for name, res in direct_results.items()
                 if np.isfinite(res["delta_aicc"])],
                key=lambda x: x[1]
            )

            if not ranked:
                conclusion = "Comparaison automatique impossible avec ces données."
            elif len(ranked) == 1:
                conclusion = f"Modèle favorisé : {ranked[0][0]}."
            else:
                best_name = ranked[0][0]
                second_name, second_delta = ranked[1]
                if second_delta < 2.0:
                    conclusion = (
                        f"Modèles non discriminables : {best_name} et {second_name} "
                        f"(ΔAICc = {second_delta:.2f} < 2). "
                        "Les données ne permettent pas d'attribuer solidement un ordre unique."
                    )
                elif second_delta < 4.0:
                    conclusion = (
                        f"{best_name} est favorisé, mais {second_name} reste plausible "
                        f"(ΔAICc = {second_delta:.2f})."
                    )
                else:
                    conclusion = (
                        f"Modèle favorisé : {best_name} "
                        f"(prochain modèle : ΔAICc = {second_delta:.2f})."
                    )

            conclusion_var.set(conclusion)

            def transform(name):
                if name == "Ordre 0":
                    return ratio, "DI/DI₀", "Ordre 0 : DI/DI₀ = a·t + b", lambda s: -s
                if name == "Ordre 2":
                    return 1.0 / ratio, "1/(DI/DI₀)", "Ordre 2 : 1/(DI/DI₀) = a·t + b", lambda s: s
                return np.log(ratio), "ln(DI/DI₀)", "Ordre 1 : ln(DI/DI₀) = a·t + b", lambda s: -s

            def update_order(*_):
                y, ylabel, equation, kfun = transform(order_var.get())
                slope, intercept = np.polyfit(trel, y, 1)
                fit_exp = slope * trel + intercept
                residuals = y - fit_exp
                rss = float(np.sum(residuals ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2_lin = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
                rmse_lin = float(np.sqrt(np.mean(residuals ** 2)))
                kval = float(kfun(float(slope)))

                dense_rel = np.linspace(float(np.min(trel)), float(np.max(trel)), 500)
                dense_abs = dense_rel + t0
                dense_fit = slope * dense_rel + intercept

                direct = direct_results[order_var.get()]

                ax.clear()
                ax.scatter(times, y, s=36, label=ylabel)
                ax.plot(dense_abs, dense_fit, linewidth=1.6, label="Fit linéaire")
                ax.set_xlabel("Temps (min)")
                ax.set_ylabel(ylabel)
                ax.set_title("Détermination de l'ordre cinétique")
                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
                ax.legend(loc="best")
                ax.text(
                    0.98, 0.97,
                    f"{equation}\n"
                    f"R² linéarisé = {r2_lin:.5f}\n"
                    f"RMSE linéarisé = {rmse_lin:.5g}\n"
                    f"R² direct = {direct['r2']:.5f}\n"
                    f"RMSE direct = {direct['rmse']:.5g}\n"
                    f"AICc = {direct['aicc']:.3f}   ΔAICc = {direct['delta_aicc']:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.7,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
                )
                fig.tight_layout()
                canv.draw_idle()

                state.update({
                    "order_name": order_var.get(), "y": y, "ylabel": ylabel,
                    "slope": float(slope), "intercept": float(intercept),
                    "k_linearized": kval, "r2_linearized": float(r2_lin),
                    "rmse_linearized": rmse_lin, "dense_abs": dense_abs,
                    "dense_fit": dense_fit, "fit_exp": fit_exp,
                    "residuals": residuals, "direct": direct
                })

            combo.bind("<<ComboboxSelected>>", update_order)

            def export_order():
                if not state:
                    return
                direct = state["direct"]
                max_len = max(len(times), len(state["dense_abs"]))
                out = [
                    ["Détermination de l'ordre cinétique RPE"],
                    ["Ordre affiché", state["order_name"]],
                    ["Transformation", state["ylabel"]],
                    ["Fenêtre min (g)", self.multi_analysis_window[0]],
                    ["Fenêtre max (g)", self.multi_analysis_window[1]],
                    ["R2 linéarisé", state["r2_linearized"]],
                    ["RMSE linéarisé", state["rmse_linearized"]],
                    ["Pente", state["slope"]],
                    ["Intercept", state["intercept"]],
                    ["k apparent linéarisé", state["k_linearized"]],
                    ["R2 direct", direct["r2"]],
                    ["RMSE direct", direct["rmse"]],
                    ["AICc direct", direct["aicc"]],
                    ["Delta AICc", direct["delta_aicc"]],
                    ["k direct", direct["k"]],
                    ["Conclusion", conclusion_var.get()],
                    [],
                    ["Comparaison des trois modèles directs"],
                    ["Modèle", "R2 direct", "RMSE", "AICc", "Delta AICc", "k direct"],
                ]
                for name in ["Ordre 0", "Ordre 1", "Ordre 2"]:
                    res = direct_results[name]
                    out.append([name, res["r2"], res["rmse"], res["aicc"], res["delta_aicc"], res["k"]])

                out.extend([
                    [],
                    ["Temps exp (min)", "Double intégrale |DI|", "DI/DI0",
                     state["ylabel"], "Fit linéarisé", "Résidu linéarisé",
                     "Temps fit (min)", "Fit linéaire"]
                ])

                for i in range(max_len):
                    out.append([
                        float(times[i]) if i < len(times) else "",
                        float(integrals[i]) if i < len(integrals) else "",
                        float(ratio[i]) if i < len(ratio) else "",
                        float(state["y"][i]) if i < len(state["y"]) else "",
                        float(state["fit_exp"][i]) if i < len(state["fit_exp"]) else "",
                        float(state["residuals"][i]) if i < len(state["residuals"]) else "",
                        float(state["dense_abs"][i]) if i < len(state["dense_abs"]) else "",
                        float(state["dense_fit"][i]) if i < len(state["dense_fit"]) else "",
                    ])

                export_table_dialog(ow, "determination_ordre_RPE.txt", out)

            ttk.Button(
                export_bar, text="Exporter les données (TXT / Excel)", command=export_order
            ).pack(side="right")

            update_order()

        def set_all_multi_included(value):
            for row in self.multi_pairs:
                row["include"] = bool(value)
            draw_table()

        ttk.Button(
            selection_bar_multi,
            text="Tout cocher",
            command=lambda: set_all_multi_included(True)
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            selection_bar_multi,
            text="Tout décocher",
            command=lambda: set_all_multi_included(False)
        ).pack(side="left")

        def draw_table():
            for widget in table.winfo_children():
                widget.destroy()

            headers = ["DTA", "DSC", "Statut", "Temps DSC", "Fenêtre", "Inclure", "Double intégrale"]
            for c, h in enumerate(headers):
                header = ttk.Label(
                    table,
                    text=h,
                    font=("TkDefaultFont", 9, "bold"),
                    anchor="center",
                    width=1
                )
                header.grid(row=0, column=c, sticky="nsew", padx=2, pady=4)
                table.grid_columnconfigure(c, minsize=COL_W[c], weight=0)

            for r, row in enumerate(self.multi_pairs, start=1):
                dta_name = os.path.basename(row["dta"]) if row["dta"] else "—"
                dsc_name = os.path.basename(row["dsc"]) if row["dsc"] else "—"

                if row["dta"]:
                    dta_btn = tk.Button(
                        table,
                        text=dta_name,
                        relief="flat",
                        bd=0,
                        anchor="w",
                        fg="#1F5AA6",
                        cursor="hand2",
                        font=("TkDefaultFont", 9, "underline"),
                        command=lambda rr=row: open_single_spectrum(rr)
                    )
                    dta_btn.grid(row=r, column=0, sticky="ew", padx=4, pady=3)
                else:
                    ttk.Label(table, text="—", anchor="w").grid(
                        row=r, column=0, sticky="ew", padx=4, pady=3
                    )

                ttk.Label(table, text=dsc_name, anchor="w").grid(
                    row=r, column=1, sticky="ew", padx=4, pady=3
                )

                if row["status"] == "ok":
                    status_text = "● OK"
                    status_fg = "#1B8A3A"
                else:
                    status_text = "● À vérifier"
                    status_fg = "#C77D00"

                tk.Label(
                    table,
                    text=status_text,
                    fg=status_fg,
                    font=("TkDefaultFont", 9, "bold")
                ).grid(row=r, column=2, padx=6, pady=3)

                acquisition_dt = row.get("acquisition_dt")
                elapsed = row.get("time_min")

                if acquisition_dt is None or elapsed is None:
                    time_text = "—"
                else:
                    time_text = f"{elapsed:.3f} min\n{acquisition_dt.strftime('%H:%M:%S')}"

                ttk.Label(
                    table,
                    text=time_text,
                    anchor="center",
                    justify="center"
                ).grid(row=r, column=3, padx=6, pady=3)

                if self.multi_analysis_window is None:
                    window_text = "Définir"
                else:
                    window_text = (
                        f"{self.multi_analysis_window[0] / 10.0:.3f}–"
                        f"{self.multi_analysis_window[1]:.6f} g"
                    )

                ttk.Button(
                    table,
                    text=window_text,
                    command=lambda rr=row: define_analysis_window(rr)
                ).grid(row=r, column=4, sticky="ew", padx=6, pady=3)

                def toggle_include(rr=row):
                    rr["include"] = not rr.get("include", True)
                    draw_table()

                included = row.get("include", True)
                include_text = "● Inclus" if included else "● Exclu"
                include_fg = "#1B8A3A" if included else "#B3261E"

                tk.Button(
                    table,
                    text=include_text,
                    relief="flat",
                    bd=0,
                    fg=include_fg,
                    cursor="hand2",
                    font=("TkDefaultFont", 9, "bold"),
                    command=toggle_include
                ).grid(row=r, column=5, sticky="ew", padx=6, pady=3)

                if row.get("double_integral") is None:
                    di_text = "—"
                elif abs(float(row["double_integral"])) == 0:
                    di_text = "0 (ignoré)"
                else:
                    di_text = f"{row['double_integral']:.6g}"

                if row.get("dta") and row.get("dsc") and self.multi_analysis_window is not None:
                    tk.Button(
                        table,
                        text=di_text,
                        relief="flat",
                        bd=0,
                        fg="#6B3FA0",
                        cursor="hand2",
                        font=("TkDefaultFont", 9, "underline"),
                        command=lambda rr=row: open_integration_detail(rr)
                    ).grid(row=r, column=6, sticky="ew", padx=6, pady=3)
                else:
                    ttk.Label(
                        table,
                        text=di_text,
                        anchor="center"
                    ).grid(row=r, column=6, sticky="ew", padx=6, pady=3)

            sync_table()

        def sort_by_time():
            # Preserve current inclusion state, reread DATE/TIME from all DSC,
            # recompute elapsed times and redraw chronologically.
            self.build_multi_pairs()
            draw_table()

        refresh_button.config(command=sort_by_time)

        def validate_multi():
            if not read_times_into_rows(show_errors=True):
                return

            bad_pairs = [
                i + 1
                for i, row in enumerate(self.multi_pairs)
                if row["status"] != "ok"
            ]

            if bad_pairs:
                messagebox.showwarning(
                    "Associations à vérifier",
                    "Certaines lignes n'ont pas une paire DTA/DSC complète ou décodable.\n"
                    "Lignes concernées : " + ", ".join(map(str, bad_pairs))
                )
                return

            if self.multi_analysis_window is None:
                messagebox.showwarning(
                    "Fenêtre non définie",
                    "Définis une fenêtre d'analyse sur l'un des spectres avant de continuer."
                )
                return

            messagebox.showinfo(
                "Multi spectre",
                "Associations, horodatages DSC et fenêtre d'analyse vérifiés."
            )

        draw_table()

        def open_multi_overlay():
            items = []
            errors = []

            for idx, row in enumerate(self.multi_pairs, start=1):
                if not row.get("dta") or not row.get("dsc"):
                    continue

                try:
                    field, signal = read_bruker_dta(
                        row["dta"],
                        row["dsc"]
                    )
                except Exception as e:
                    errors.append(f"Ligne {idx} : {e}")
                    continue

                label = os.path.splitext(
                    os.path.basename(row["dta"])
                )[0]

                params_overlay = parse_dsc(row["dsc"])
                freq_overlay = microwave_frequency_hz_from_dsc(params_overlay)
                items.append({
                    "label": label,
                    "field": field,
                    "signal": signal,
                    "frequency_hz": freq_overlay,
                    "time_min": row.get("time_min"),
                    "included": row.get("include", True),
                })

            if not items:
                messagebox.showwarning(
                    "Superposition impossible",
                    "Aucune paire DTA/DSC exploitable n'est disponible.",
                    parent=win
                )
                return

            if errors:
                messagebox.showwarning(
                    "Certains spectres ont été ignorés",
                    "\n".join(errors[:8]) + ("\n..." if len(errors) > 8 else ""),
                    parent=win
                )

            self.open_spectra_overlay_window(
                win,
                items,
                title="Superposition — Multi-spectres",
                analysis_window=self.multi_analysis_window,
                x_axis="g"
            )

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))

        ttk.Button(
            footer,
            text="Superposer les spectres",
            command=open_multi_overlay
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            footer,
            text="Cinétique normalisée + fit exponentiel",
            command=open_normalized_kinetics
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            footer,
            text="Détermination de l'ordre",
            command=open_order_determination
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            footer,
            text="Calculer les doubles intégrales",
            command=lambda: (
                calculate_all_double_integrals(show_messages=True)
                and draw_table()
            )
        ).pack(side="left")

        ttk.Button(
            footer,
            text="Vérifier associations, heures DSC et fenêtre",
            command=validate_multi
        ).pack(side="right")

    # ---------------- SUIVI AUTOMATIQUE ----------------
    def load_auto_dta(self):
        path = filedialog.askopenfilename(
            title="Sélectionner le DTA du suivi RPE",
            filetypes=[("Fichiers Bruker DTA", "*.DTA *.dta"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return

        self.auto_dta_path = path
        self.auto_dta_label.config(text=f"DTA : {os.path.basename(path)}")

        # Ergonomic auto-detection of companion DSC/YGF with the same stem.
        stem = os.path.splitext(path)[0]

        if not self.auto_dsc_path:
            for candidate in (stem + ".DSC", stem + ".dsc"):
                if os.path.exists(candidate):
                    self.auto_dsc_path = candidate
                    self.auto_dsc_label.config(text=f"DSC : {os.path.basename(candidate)}")
                    break

        if not self.auto_ygf_path:
            for candidate in (stem + ".YGF", stem + ".ygf"):
                if os.path.exists(candidate):
                    self.auto_ygf_path = candidate
                    self.auto_ygf_label.config(text=f"YGF : {os.path.basename(candidate)}")
                    break

    def load_auto_dsc(self):
        path = filedialog.askopenfilename(
            title="Sélectionner le DSC associé au suivi RPE",
            filetypes=[("Fichiers DSC", "*.DSC *.dsc"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return

        self.auto_dsc_path = path
        self.auto_dsc_label.config(text=f"DSC : {os.path.basename(path)}")

    def load_auto_ygf(self):
        path = filedialog.askopenfilename(
            title="Sélectionner le YGF associé au suivi RPE",
            filetypes=[("Fichiers Bruker YGF", "*.YGF *.ygf"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return

        self.auto_ygf_path = path
        self.auto_ygf_label.config(text=f"YGF : {os.path.basename(path)}")

    def prepare_auto_rows(self):
        if not self.auto_dta_path or not self.auto_dsc_path:
            raise ValueError("Charge d'abord le DTA et le DSC du suivi.")

        dataset = read_bruker_kinetic_dataset(
            self.auto_dta_path,
            self.auto_dsc_path,
            self.auto_ygf_path
        )

        field = dataset["field"]
        times = dataset["times"]
        spectra = dataset["spectra"]

        if spectra.shape[0] != len(times):
            raise ValueError(
                "Nombre de spectres DTA et nombre de temps YGF incompatibles."
            )

        if self.auto_temp_dir and os.path.isdir(self.auto_temp_dir):
            try:
                shutil.rmtree(self.auto_temp_dir)
            except Exception:
                pass

        self.auto_temp_dir = tempfile.mkdtemp(prefix="rpe_auto_dta_")
        self.auto_rows = []

        base = os.path.splitext(os.path.basename(self.auto_dta_path))[0]

        # Internal simple TXT files are generated only as a temporary transport
        # format for the existing plotting/integration code. All values originate
        # directly from the raw DTA, never from a user-exported TXT.
        for idx in range(spectra.shape[0]):
            time_s = float(times[idx])
            signal = spectra[idx]

            temp_txt = os.path.join(
                self.auto_temp_dir,
                f"{base}_scan_{idx + 1:03d}.txt"
            )

            with open(temp_txt, "w", encoding="utf-8") as f:
                f.write("index\tField [G]\t1st Harm 0deg Abs\n")
                for j, (b, y) in enumerate(zip(field, signal)):
                    f.write(f"{j}\t{float(b):.15g}\t{float(y):.15g}\n")

            self.auto_rows.append({
                "txt": temp_txt,
                "dsc": self.auto_dsc_path,
                "time_min": time_s / 60.0,
                "time_s": time_s,
                "double_integral": None,
                "include": True,
                "display_name": f"Spectre {idx + 1}"
            })

        self.auto_rows.sort(key=lambda r: r["time_min"])

    def open_auto_window(self):
        try:
            self.prepare_auto_rows()
        except Exception as e:
            messagebox.showerror("Suivi automatique", str(e))
            return

        rows = self.auto_rows
        win = tk.Toplevel(self)
        win.title("Analyse suivi automatique")
        win.geometry("1100x700")
        win.minsize(900, 560)

        outer = ttk.Frame(win, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text=f"Analyse suivi automatique — {len(rows)} spectre(s)",
            font=("TkDefaultFont", 12, "bold")
        ).pack(anchor="w", pady=(0, 8))

        dsc_params = parse_dsc(self.auto_dsc_path)
        auto_frequency_hz = microwave_frequency_hz_from_dsc(dsc_params)
        if auto_frequency_hz is None:
            messagebox.showerror("MWFQ absent", "Impossible de travailler en g sans MWFQ dans le DSC.", parent=win)
            win.destroy()
            return
        q_value = dsc_params.get("QValue", "—")
        y_unit = str(dsc_params.get("YUNI", "'s'")).strip("'\"")

        ttk.Label(
            outer,
            text=(
                "Spectres lus directement dans le DTA ; temps lus dans le YGF. "
                f"XPTS : {dsc_params.get('XPTS', '—')}   |   "
                f"YPTS : {dsc_params.get('YPTS', '—')}   |   "
                f"Unité Y : {y_unit or '—'}   |   "
                f"Q : {q_value}"
            ),
            wraplength=1030
        ).pack(anchor="w", pady=(0, 10))

        self.auto_window_label = ttk.Label(
            outer,
            text="Fenêtre d'analyse : non définie",
            font=("TkDefaultFont", 9, "bold")
        )
        self.auto_window_label.pack(anchor="w", pady=(0, 8))

        selection_bar = ttk.Frame(outer)
        selection_bar.pack(fill="x", pady=(0, 8))

        table_wrap = ttk.Frame(outer)
        table_wrap.pack(fill="both", expand=True)

        canvas = tk.Canvas(table_wrap, highlightthickness=0)
        yscroll = ttk.Scrollbar(table_wrap, orient="vertical", command=canvas.yview)
        xscroll = ttk.Scrollbar(table_wrap, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        table = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=table, anchor="nw")

        COL_W = {0: 260, 1: 140, 2: 210, 3: 120, 4: 170}

        def sync_table(event=None):
            table.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

        table.bind("<Configure>", sync_table)

        def calculate_all(show_messages=True):
            if self.auto_analysis_window is None:
                if show_messages:
                    messagebox.showwarning("Fenêtre non définie", "Définis d'abord la fenêtre commune.", parent=win)
                return False

            errors = []
            count = 0
            for i, row in enumerate(rows, start=1):
                try:
                    result = self.compute_double_integral_txt_g(row["txt"], self.auto_dsc_path, self.auto_analysis_window)
                    row["double_integral"] = result["double_integral"]
                    count += 1
                except Exception as e:
                    row["double_integral"] = None
                    errors.append((i, str(e)))

            if show_messages:
                if errors:
                    messagebox.showwarning(
                        "Double intégration",
                        f"{count} spectre(s) calculé(s), {len(errors)} erreur(s).",
                        parent=win
                    )
                else:
                    messagebox.showinfo(
                        "Double intégration",
                        f"Double intégration calculée pour {count} spectre(s).",
                        parent=win
                    )
            return count > 0

        def open_spectrum(row):
            field, signal = read_simple_txt(row["txt"])
            sw = tk.Toplevel(win)
            sw.title(f"{row['display_name']} — {row['time_min']:.3f} min")
            sw.geometry("900x620")

            frame = ttk.Frame(sw, padding=8)
            frame.pack(fill="both", expand=True)

            fig = Figure(figsize=(8, 5.5), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(gauss_to_g(field, auto_frequency_hz), signal, linewidth=1.2)
            ax.set_xlabel("g")
            ax.set_ylabel("Intensité RPE (a.u.)")
            ax.set_title(f"{row['display_name']} — t = {row['time_min']:.3f} min")
            if self.auto_analysis_window is not None:
                ax.set_xlim(*self.auto_analysis_window)
            fig.tight_layout()

            canv = FigureCanvasTkAgg(fig, master=frame)
            canv.draw()
            canv.get_tk_widget().pack(fill="both", expand=True)

            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill="x")
            toolbar = NavigationToolbar2Tk(canv, toolbar_frame)
            toolbar.update()

        def define_window(row):
            field, signal = read_simple_txt(row["txt"])

            fw = tk.Toplevel(win)
            fw.title("Définir la fenêtre d'analyse")
            fw.geometry("940x690")

            frame = ttk.Frame(fw, padding=10)
            frame.pack(fill="both", expand=True)

            fig = Figure(figsize=(8, 5), dpi=100)
            ax = fig.add_subplot(111)
            field_mt = gauss_to_g(field, auto_frequency_hz)
            ax.plot(field_mt, signal, linewidth=1.2)
            ax.set_xlabel("g")
            ax.set_ylabel("Intensité RPE (a.u.)")
            ax.set_title(f"{row['display_name']} — t = {row['time_min']:.3f} min")

            xmin_data, xmax_data = float(np.min(field_mt)), float(np.max(field_mt))
            if self.auto_analysis_window is None:
                init_min, init_max = xmin_data, xmax_data
            else:
                init_min = self.auto_analysis_window[0]
                init_max = self.auto_analysis_window[1]

            ax.set_xlim(init_min, init_max)
            fig.tight_layout()

            canv = FigureCanvasTkAgg(fig, master=frame)
            canv.draw()
            canv.get_tk_widget().pack(fill="both", expand=True)

            controls = ttk.Frame(frame)
            controls.pack(fill="x", pady=(8, 0))

            step = max((xmax_data - xmin_data) / max(len(field)-1, 1), 1e-6)
            min_var = tk.DoubleVar(value=init_min)
            max_var = tk.DoubleVar(value=init_max)
            min_lab = tk.StringVar(value=f"Min : {init_min:.6f} g")
            max_lab = tk.StringVar(value=f"Max : {init_max:.6f} g")

            ttk.Label(controls, textvariable=min_lab).pack(anchor="w")
            tk.Scale(
                controls, from_=xmin_data, to=xmax_data, resolution=step,
                orient="horizontal", variable=min_var, showvalue=False
            ).pack(fill="x")

            ttk.Label(controls, textvariable=max_lab).pack(anchor="w", pady=(6, 0))
            tk.Scale(
                controls, from_=xmin_data, to=xmax_data, resolution=step,
                orient="horizontal", variable=max_var, showvalue=False
            ).pack(fill="x")

            def update(*_):
                a, b = float(min_var.get()), float(max_var.get())
                if a < b:
                    min_lab.set(f"Min : {a:.6f} g")
                    max_lab.set(f"Max : {b:.6f} g")
                    ax.set_xlim(a, b)
                    canv.draw_idle()

            min_var.trace_add("write", update)
            max_var.trace_add("write", update)

            def validate():
                a, b = float(min_var.get()), float(max_var.get())
                if a >= b:
                    messagebox.showerror("Fenêtre invalide", "Min doit être inférieur à Max.", parent=fw)
                    return

                self.auto_analysis_window = (a, b)
                self.auto_window_label.config(text=f"Fenêtre d'analyse : g = {a:.6f}–{b:.6f}")
                for rr in rows:
                    rr["double_integral"] = None
                calculate_all(show_messages=False)
                draw_table()
                fw.destroy()

            ttk.Button(
                frame, text="Valider cette fenêtre pour tout le suivi",
                command=validate
            ).pack(anchor="e", pady=(10, 0))

        def integration_detail(row):
            if self.auto_analysis_window is None:
                return

            result = self.compute_double_integral_txt_g(row["txt"], self.auto_dsc_path, self.auto_analysis_window)
            row["double_integral"] = result["double_integral"]

            iw = tk.Toplevel(win)
            iw.title(f"Double intégration — {row['display_name']}")
            iw.geometry("940x760")

            frame = ttk.Frame(iw, padding=8)
            frame.pack(fill="both", expand=True)

            fig = Figure(figsize=(8, 7), dpi=100)
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            x = result["field"]
            x_disp = gauss_to_g(x, auto_frequency_hz)
            ax1.plot(x_disp, result["raw"], linewidth=1.0, label="Signal brut")
            ax1.plot(x_disp, result["corrected"], linewidth=1.1, label="Signal corrigé")
            ax1.legend(loc="best", fontsize=8)
            ax1.set_ylabel("Signal RPE")
            ax1.set_title(f"{row['display_name']} — t = {row['time_min']:.3f} min")

            ax2.plot(x_disp, result["first"], linewidth=1.1)
            ax2.set_ylabel("1re intégrale")

            ax3.plot(x_disp, result["second_curve"], linewidth=1.1)
            ax3.set_xlabel("g")
            ax3.set_ylabel("2e intégrale")

            fig.tight_layout()
            canv = FigureCanvasTkAgg(fig, master=frame)
            canv.draw()
            canv.get_tk_widget().pack(fill="both", expand=True)

            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill="x")
            toolbar = NavigationToolbar2Tk(canv, toolbar_frame)
            toolbar.update()

        def collect_data():
            if self.auto_analysis_window is None:
                messagebox.showwarning("Fenêtre non définie", "Définis d'abord la fenêtre commune.", parent=win)
                return None
            if not calculate_all(show_messages=False):
                return None

            valid = []
            for row in rows:
                if not row.get("include", True):
                    continue

                di = row.get("double_integral")
                if di is None:
                    continue

                val = abs(float(di))
                t = float(row["time_min"])

                # A scan with a strictly null double integral is treated as
                # an unrecorded / empty scan and is never included in kinetics.
                if not np.isfinite(val) or val == 0:
                    continue

                if np.isfinite(t):
                    valid.append((t, val))

            if len(valid) < 3:
                messagebox.showwarning(
                    "Points insuffisants",
                    "Il faut au moins 3 points cochés avec une double intégrale non nulle.",
                    parent=win
                )
                return None

            valid.sort(key=lambda z: z[0])
            return (
                np.asarray([v[0] for v in valid], dtype=float),
                np.asarray([v[1] for v in valid], dtype=float)
            )

        def exp_window():
            collected = collect_data()
            if collected is None:
                return

            times, integrals = collected
            ynorm = 100.0 * integrals / integrals[0]
            t0 = times[0]
            trel = times - t0

            kw = tk.Toplevel(win)
            kw.title("Cinétique RPE normalisée — suivi automatique")
            kw.geometry("980x790")

            frame = ttk.Frame(kw, padding=10)
            frame.pack(fill="both", expand=True)

            fit_options = ttk.LabelFrame(frame, text="Options du fit", padding=8)
            fit_options.pack(fill="x", pady=(0, 8))

            plateau_zero_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                fit_options,
                text="Imposer le plateau à 0 (C = 0)",
                variable=plateau_zero_var
            ).pack(side="left")

            controls = ttk.LabelFrame(frame, text="Extrapolation du fit", padding=8)
            controls.pack(fill="x", pady=(0,8))

            export_bar = ttk.Frame(frame)
            export_bar.pack(fill="x", pady=(0,8))

            fig = Figure(figsize=(8.7,5.8), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(times, ynorm, s=36, label="Double intégrale normalisée")
            fit_line, = ax.plot([], [], linewidth=1.6, label="Fit exponentiel")
            ax.axvline(times[-1], linestyle=":", linewidth=1.0, alpha=0.6, label="Dernier point mesuré")
            ax.set_xlabel("Temps (min)")
            ax.set_ylabel("Double intégrale normalisée (%)")
            ax.set_title("Cinétique RPE — suivi automatique")
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
            ax.legend(loc="best")

            stats_artist = ax.text(
                0.98,0.97,"",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
            )

            last_time = float(times[-1])
            first_time = float(times[0])
            span = max(last_time-first_time, 1.0)
            max_extrap = first_time + 5.0*span
            if max_extrap <= last_time:
                max_extrap = last_time + max(span,10.0)

            extrap_var = tk.DoubleVar(value=last_time)
            extrap_lab = tk.StringVar(value=f"Fin d'affichage du fit : {last_time:.2f} min")
            ttk.Label(controls, textvariable=extrap_lab).pack(anchor="w")
            tk.Scale(
                controls, from_=last_time, to=max_extrap,
                resolution=max(span/500.0,0.01), orient="horizontal",
                variable=extrap_var, showvalue=False
            ).pack(fill="x")

            canv = FigureCanvasTkAgg(fig, master=frame)
            canv.draw()
            canv.get_tk_widget().pack(fill="both", expand=True)
            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill="x")
            NavigationToolbar2Tk(canv, toolbar_frame).update()

            fit_state = {}

            def compute_fit():
                try:
                    if plateau_zero_var.get():
                        def model(t, A, k):
                            return A*np.exp(-k*t)

                        popt, _ = curve_fit(
                            model, trel, ynorm,
                            p0=[100.0, 1.0/max(float(np.ptp(trel)),1e-6)],
                            bounds=([0.0,0.0],[np.inf,np.inf]),
                            maxfev=30000
                        )
                        A,k = [float(v) for v in popt]
                        C = 0.0
                        fit_exp = model(trel,*popt)
                        model_name = "y = A·exp(-k·t), C = 0"
                    else:
                        def model(t,A,k,C):
                            return C + A*np.exp(-k*t)

                        p0 = [
                            max(100.0-float(np.min(ynorm)),1.0),
                            1.0/max(float(np.ptp(trel)),1e-6),
                            max(float(np.min(ynorm)),0.0)
                        ]
                        popt,_ = curve_fit(
                            model,trel,ynorm,p0=p0,
                            bounds=([0.0,0.0,-np.inf],[np.inf,np.inf,np.inf]),
                            maxfev=30000
                        )
                        A,k,C = [float(v) for v in popt]
                        fit_exp = model(trel,*popt)
                        model_name = "y = C + A·exp(-k·t)"

                    ss_res = float(np.sum((ynorm-fit_exp)**2))
                    ss_tot = float(np.sum((ynorm-np.mean(ynorm))**2))
                    r2 = 1.0-ss_res/ss_tot if ss_tot>0 else float("nan")
                    half = np.log(2.0)/k if k>0 else float("inf")

                    fit_state.update({
                        "model":model,"popt":popt,"A":A,"k":k,"C":C,
                        "r2":r2,"half":half,"model_name":model_name
                    })
                    stats_artist.set_text(
                        f"Modèle : {model_name}\nA = {A:.4g} %\nC = {C:.4g} %\n"
                        f"k = {k:.6g} min⁻¹\nt½ = {half:.4g} min\nR² = {r2:.5f}"
                    )
                    update()
                except Exception as e:
                    messagebox.showerror("Fit exponentiel impossible",str(e),parent=kw)

            def update(*_):
                if "model" not in fit_state:
                    return
                end_abs = max(float(extrap_var.get()), last_time)
                dense_rel = np.linspace(0.0, max(end_abs-t0, 1e-9), 700)
                dense_abs = dense_rel+t0
                dense_fit = fit_state["model"](dense_rel,*fit_state["popt"])
                fit_line.set_data(dense_abs,dense_fit)
                extrap_lab.set(f"Fin d'affichage du fit : {end_abs:.2f} min")
                ax.relim(); ax.autoscale_view()
                ax.set_xlim(first_time, end_abs + max(0.5,0.03*max(end_abs-first_time,1)))
                canv.draw_idle()

            plateau_zero_var.trace_add("write", lambda *_: compute_fit())
            extrap_var.trace_add("write", update)

            def export_exp():
                if "model" not in fit_state:
                    return
                end_abs = max(float(extrap_var.get()), last_time)
                dense_rel = np.linspace(0.0,max(end_abs-t0,1e-9),700)
                dense_abs = dense_rel+t0
                dense_fit = fit_state["model"](dense_rel,*fit_state["popt"])
                fit_exp = fit_state["model"](trel,*fit_state["popt"])

                m = max(len(times),len(dense_abs))
                out = [
                    ["Suivi automatique RPE - fit exponentiel"],
                    ["Fenêtre min (g)", self.auto_analysis_window[0]],
                    ["Fenêtre max (g)", self.auto_analysis_window[1]],
                    ["Modèle",fit_state["model_name"]],
                    ["A (%)",fit_state["A"]],["k (min^-1)",fit_state["k"]],
                    ["C (%)",fit_state["C"]],["t1/2 (min)",fit_state["half"]],
                    ["R2",fit_state["r2"]],[],
                    ["Temps exp (min)","Double intégrale |DI|","DI normalisée (%)",
                     "Fit aux temps exp (%)","Temps fit (min)","Fit exponentiel (%)"]
                ]
                for i in range(m):
                    out.append([
                        float(times[i]) if i<len(times) else "",
                        float(integrals[i]) if i<len(integrals) else "",
                        float(ynorm[i]) if i<len(ynorm) else "",
                        float(fit_exp[i]) if i<len(fit_exp) else "",
                        float(dense_abs[i]) if i<len(dense_abs) else "",
                        float(dense_fit[i]) if i<len(dense_fit) else "",
                    ])
                export_table_dialog(kw,"suivi_auto_fit_exponentiel.txt",out)

            ttk.Button(
                export_bar,text="Exporter les données (TXT / Excel)",command=export_exp
            ).pack(side="right")

            compute_fit()

        def order_window():
            collected = collect_data()
            if collected is None:
                return

            times, integrals = collected
            ratio = integrals / integrals[0]
            t0 = float(times[0])
            trel = times - t0

            ow = tk.Toplevel(win)
            ow.title("Détermination de l'ordre — suivi automatique")
            ow.geometry("1080x820")
            ow.minsize(880, 680)

            frame = ttk.Frame(ow, padding=10)
            frame.pack(fill="both", expand=True)

            compare_box = ttk.LabelFrame(
                frame, text="Comparaison automatique des modèles directs", padding=8
            )
            compare_box.pack(fill="x", pady=(0, 8))

            headers_cmp = ["Modèle", "R² direct", "RMSE", "AICc", "ΔAICc"]
            for c, h in enumerate(headers_cmp):
                ttk.Label(
                    compare_box, text=h, font=("TkDefaultFont", 9, "bold"), anchor="center"
                ).grid(row=0, column=c, sticky="ew", padx=6, pady=3)

            cmp_labels = {}
            for r, name in enumerate(["Ordre 0", "Ordre 1", "Ordre 2"], start=1):
                ttk.Label(compare_box, text=name).grid(row=r, column=0, sticky="w", padx=6, pady=3)
                cmp_labels[name] = []
                for c in range(1, 5):
                    lab = ttk.Label(compare_box, text="—", anchor="center")
                    lab.grid(row=r, column=c, sticky="ew", padx=6, pady=3)
                    cmp_labels[name].append(lab)

            for c in range(5):
                compare_box.grid_columnconfigure(c, weight=1)

            conclusion_var = tk.StringVar(value="")
            ttk.Label(
                compare_box, textvariable=conclusion_var,
                font=("TkDefaultFont", 9, "bold"), wraplength=980
            ).grid(row=4, column=0, columnspan=5, sticky="w", padx=6, pady=(8, 3))

            controls = ttk.LabelFrame(frame, text="Linéarisation affichée", padding=8)
            controls.pack(fill="x", pady=(0, 8))
            ttk.Label(controls, text="Ordre :").pack(side="left", padx=(0, 8))

            order_var = tk.StringVar(value="Ordre 1")
            combo = ttk.Combobox(
                controls, textvariable=order_var,
                values=["Ordre 0", "Ordre 1", "Ordre 2"],
                state="readonly", width=12
            )
            combo.pack(side="left")

            export_bar = ttk.Frame(frame)
            export_bar.pack(fill="x", pady=(0, 8))

            fig = Figure(figsize=(9.2, 5.8), dpi=100)
            ax = fig.add_subplot(111)
            canv = FigureCanvasTkAgg(fig, master=frame)
            canv.draw()
            canv.get_tk_widget().pack(fill="both", expand=True)

            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill="x")
            NavigationToolbar2Tk(canv, toolbar_frame).update()

            state = {}

            def direct_metrics():
                y = ratio.astype(float)
                n = len(y)
                span = max(float(np.ptp(trel)), 1e-9)

                def m0(t, k):
                    return 1.0 - k * t

                def m1(t, k):
                    return np.exp(-k * t)

                def m2(t, k):
                    return 1.0 / (1.0 + k * t)

                models = {"Ordre 0": m0, "Ordre 1": m1, "Ordre 2": m2}
                results = {}

                for name, model in models.items():
                    try:
                        popt, _ = curve_fit(
                            model, trel, y, p0=[1.0 / span],
                            bounds=([0.0], [np.inf]), maxfev=30000
                        )
                        pred = model(trel, *popt)
                        residuals = y - pred
                        rss = float(np.sum(residuals ** 2))
                        rmse = float(np.sqrt(rss / n))
                        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")

                        p = 1
                        rss_safe = max(rss, np.finfo(float).tiny)
                        aic = n * np.log(rss_safe / n) + 2 * p
                        aicc = (
                            aic + (2 * p * (p + 1)) / (n - p - 1)
                            if n > p + 1 else float("inf")
                        )

                        results[name] = {
                            "r2": r2, "rmse": rmse, "aicc": float(aicc),
                            "k": float(popt[0]), "pred": pred, "residuals": residuals
                        }
                    except Exception:
                        results[name] = {
                            "r2": float("nan"), "rmse": float("nan"),
                            "aicc": float("inf"), "k": float("nan"),
                            "pred": np.full_like(y, np.nan),
                            "residuals": np.full_like(y, np.nan)
                        }

                finite = [v["aicc"] for v in results.values() if np.isfinite(v["aicc"])]
                best = min(finite) if finite else float("inf")
                for res in results.values():
                    res["delta_aicc"] = (
                        res["aicc"] - best
                        if np.isfinite(res["aicc"]) and np.isfinite(best)
                        else float("inf")
                    )
                return results

            direct_results = direct_metrics()

            for name, res in direct_results.items():
                values = [
                    f'{res["r2"]:.5f}' if np.isfinite(res["r2"]) else "—",
                    f'{res["rmse"]:.5g}' if np.isfinite(res["rmse"]) else "—",
                    f'{res["aicc"]:.3f}' if np.isfinite(res["aicc"]) else "—",
                    f'{res["delta_aicc"]:.3f}' if np.isfinite(res["delta_aicc"]) else "—",
                ]
                for lab, value in zip(cmp_labels[name], values):
                    lab.config(text=value)

            ranked = sorted(
                [(name, res["delta_aicc"]) for name, res in direct_results.items()
                 if np.isfinite(res["delta_aicc"])],
                key=lambda x: x[1]
            )

            if not ranked:
                conclusion = "Comparaison automatique impossible avec ces données."
            elif len(ranked) == 1:
                conclusion = f"Modèle favorisé : {ranked[0][0]}."
            else:
                best_name = ranked[0][0]
                second_name, second_delta = ranked[1]
                if second_delta < 2.0:
                    conclusion = (
                        f"Modèles non discriminables : {best_name} et {second_name} "
                        f"(ΔAICc = {second_delta:.2f} < 2). "
                        "Les données ne permettent pas d'attribuer solidement un ordre unique."
                    )
                elif second_delta < 4.0:
                    conclusion = (
                        f"{best_name} est favorisé, mais {second_name} reste plausible "
                        f"(ΔAICc = {second_delta:.2f})."
                    )
                else:
                    conclusion = (
                        f"Modèle favorisé : {best_name} "
                        f"(prochain modèle : ΔAICc = {second_delta:.2f})."
                    )

            conclusion_var.set(conclusion)

            def transform(name):
                if name == "Ordre 0":
                    return ratio, "DI/DI₀", "Ordre 0 : DI/DI₀ = a·t + b", lambda s: -s
                if name == "Ordre 2":
                    return 1.0 / ratio, "1/(DI/DI₀)", "Ordre 2 : 1/(DI/DI₀) = a·t + b", lambda s: s
                return np.log(ratio), "ln(DI/DI₀)", "Ordre 1 : ln(DI/DI₀) = a·t + b", lambda s: -s

            def update_order(*_):
                y, ylabel, equation, kfun = transform(order_var.get())
                slope, intercept = np.polyfit(trel, y, 1)
                fit_exp = slope * trel + intercept
                residuals = y - fit_exp
                rss = float(np.sum(residuals ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2_lin = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
                rmse_lin = float(np.sqrt(np.mean(residuals ** 2)))
                kval = float(kfun(float(slope)))

                dense_rel = np.linspace(float(np.min(trel)), float(np.max(trel)), 500)
                dense_abs = dense_rel + t0
                dense_fit = slope * dense_rel + intercept

                direct = direct_results[order_var.get()]

                ax.clear()
                ax.scatter(times, y, s=36, label=ylabel)
                ax.plot(dense_abs, dense_fit, linewidth=1.6, label="Fit linéaire")
                ax.set_xlabel("Temps (min)")
                ax.set_ylabel(ylabel)
                ax.set_title("Détermination de l'ordre cinétique")
                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
                ax.legend(loc="best")
                ax.text(
                    0.98, 0.97,
                    f"{equation}\n"
                    f"R² linéarisé = {r2_lin:.5f}\n"
                    f"RMSE linéarisé = {rmse_lin:.5g}\n"
                    f"R² direct = {direct['r2']:.5f}\n"
                    f"RMSE direct = {direct['rmse']:.5g}\n"
                    f"AICc = {direct['aicc']:.3f}   ΔAICc = {direct['delta_aicc']:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.7,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
                )
                fig.tight_layout()
                canv.draw_idle()

                state.update({
                    "order_name": order_var.get(), "y": y, "ylabel": ylabel,
                    "slope": float(slope), "intercept": float(intercept),
                    "k_linearized": kval, "r2_linearized": float(r2_lin),
                    "rmse_linearized": rmse_lin, "dense_abs": dense_abs,
                    "dense_fit": dense_fit, "fit_exp": fit_exp,
                    "residuals": residuals, "direct": direct
                })

            combo.bind("<<ComboboxSelected>>", update_order)

            def export_order():
                if not state:
                    return
                direct = state["direct"]
                max_len = max(len(times), len(state["dense_abs"]))
                out = [
                    ["Détermination de l'ordre cinétique RPE"],
                    ["Ordre affiché", state["order_name"]],
                    ["Transformation", state["ylabel"]],
                    ["Fenêtre min (g)", self.auto_analysis_window[0]],
                    ["Fenêtre max (g)", self.auto_analysis_window[1]],
                    ["R2 linéarisé", state["r2_linearized"]],
                    ["RMSE linéarisé", state["rmse_linearized"]],
                    ["Pente", state["slope"]],
                    ["Intercept", state["intercept"]],
                    ["k apparent linéarisé", state["k_linearized"]],
                    ["R2 direct", direct["r2"]],
                    ["RMSE direct", direct["rmse"]],
                    ["AICc direct", direct["aicc"]],
                    ["Delta AICc", direct["delta_aicc"]],
                    ["k direct", direct["k"]],
                    ["Conclusion", conclusion_var.get()],
                    [],
                    ["Comparaison des trois modèles directs"],
                    ["Modèle", "R2 direct", "RMSE", "AICc", "Delta AICc", "k direct"],
                ]
                for name in ["Ordre 0", "Ordre 1", "Ordre 2"]:
                    res = direct_results[name]
                    out.append([name, res["r2"], res["rmse"], res["aicc"], res["delta_aicc"], res["k"]])

                out.extend([
                    [],
                    ["Temps exp (min)", "Double intégrale |DI|", "DI/DI0",
                     state["ylabel"], "Fit linéarisé", "Résidu linéarisé",
                     "Temps fit (min)", "Fit linéaire"]
                ])

                for i in range(max_len):
                    out.append([
                        float(times[i]) if i < len(times) else "",
                        float(integrals[i]) if i < len(integrals) else "",
                        float(ratio[i]) if i < len(ratio) else "",
                        float(state["y"][i]) if i < len(state["y"]) else "",
                        float(state["fit_exp"][i]) if i < len(state["fit_exp"]) else "",
                        float(state["residuals"][i]) if i < len(state["residuals"]) else "",
                        float(state["dense_abs"][i]) if i < len(state["dense_abs"]) else "",
                        float(state["dense_fit"][i]) if i < len(state["dense_fit"]) else "",
                    ])

                export_table_dialog(ow, "determination_ordre_RPE.txt", out)

            ttk.Button(
                export_bar, text="Exporter les données (TXT / Excel)", command=export_order
            ).pack(side="right")

            update_order()

        def draw_table():
            for widget in table.winfo_children():
                widget.destroy()

            headers = ["Spectre","Temps (min)","Fenêtre","Inclure","Double intégrale"]
            for c,h in enumerate(headers):
                ttk.Label(table,text=h,font=("TkDefaultFont",9,"bold"),anchor="center").grid(
                    row=0,column=c,sticky="nsew",padx=3,pady=4
                )
                table.grid_columnconfigure(c,minsize=COL_W[c],weight=0)

            for r,row in enumerate(rows,start=1):
                tk.Button(
                    table,text=row["display_name"],relief="flat",bd=0,anchor="w",
                    fg="#1F5AA6",cursor="hand2",font=("TkDefaultFont",9,"underline"),
                    command=lambda rr=row: open_spectrum(rr)
                ).grid(row=r,column=0,sticky="ew",padx=4,pady=3)

                ttk.Label(table,text=f"{row['time_min']:.3f}",anchor="center").grid(
                    row=r,column=1,sticky="ew",padx=4,pady=3
                )

                wtxt = "Définir" if self.auto_analysis_window is None else (
                    f"{self.auto_analysis_window[0]:.6f}–{self.auto_analysis_window[1]:.6f} g"
                )
                ttk.Button(table,text=wtxt,command=lambda rr=row: define_window(rr)).grid(
                    row=r,column=2,sticky="ew",padx=6,pady=3
                )

                def toggle_include(rr=row):
                    rr["include"] = not rr.get("include", True)
                    draw_table()

                included = row.get("include", True)
                include_text = "● Inclus" if included else "● Exclu"
                include_fg = "#1B8A3A" if included else "#B3261E"

                tk.Button(
                    table,
                    text=include_text,
                    relief="flat",
                    bd=0,
                    fg=include_fg,
                    cursor="hand2",
                    font=("TkDefaultFont",9,"bold"),
                    command=toggle_include
                ).grid(row=r,column=3,sticky="ew",padx=6,pady=3)

                if row.get("double_integral") is None:
                    ditxt = "—"
                elif abs(float(row["double_integral"])) == 0:
                    ditxt = "0 (ignoré)"
                else:
                    ditxt = f"{row['double_integral']:.6g}"
                if self.auto_analysis_window is not None:
                    tk.Button(
                        table,text=ditxt,relief="flat",bd=0,fg="#6B3FA0",cursor="hand2",
                        font=("TkDefaultFont",9,"underline"),
                        command=lambda rr=row: integration_detail(rr)
                    ).grid(row=r,column=4,sticky="ew",padx=6,pady=3)
                else:
                    ttk.Label(table,text=ditxt,anchor="center").grid(
                        row=r,column=4,sticky="ew",padx=6,pady=3
                    )

            sync_table()

        draw_table()

        def open_auto_overlay():
            items = []
            errors = []

            for idx, row in enumerate(rows, start=1):
                try:
                    field, signal = read_simple_txt(row["txt"])
                except Exception as e:
                    errors.append(f"Spectre {idx} : {e}")
                    continue

                items.append({
                    "label": row.get("display_name", f"Spectre {idx}"),
                    "field": field,
                    "signal": signal,
                    "frequency_hz": auto_frequency_hz,
                    "time_min": row.get("time_min"),
                    "included": row.get("include", True),
                })

            if not items:
                messagebox.showwarning(
                    "Superposition impossible",
                    "Aucun spectre du suivi n'est exploitable.",
                    parent=win
                )
                return

            if errors:
                messagebox.showwarning(
                    "Certains spectres ont été ignorés",
                    "\n".join(errors[:8]) + ("\n..." if len(errors) > 8 else ""),
                    parent=win
                )

            self.open_spectra_overlay_window(
                win,
                items,
                title="Superposition — Suivi cinétique",
                analysis_window=self.auto_analysis_window,
                x_axis="g"
            )

        footer = ttk.Frame(outer)
        footer.pack(fill="x",pady=(10,0))
        ttk.Button(
            footer,
            text="Superposer les spectres",
            command=open_auto_overlay
        ).pack(side="left", padx=(0,8))
        ttk.Button(footer,text="Cinétique normalisée + fit exponentiel",command=exp_window).pack(side="left",padx=(0,8))
        ttk.Button(footer,text="Détermination de l'ordre",command=order_window).pack(side="left",padx=(0,8))
        ttk.Button(
            footer,text="Calculer les doubles intégrales",
            command=lambda: (calculate_all(show_messages=True) and draw_table())
        ).pack(side="left")

    def configure_sliders(self):
        if not self.field:
            return

        field_mt = gauss_to_mt(self.field)
        xmin = float(np.min(field_mt))
        xmax = float(np.max(field_mt))

        # resolution chosen from actual point spacing, displayed in mT
        if len(field_mt) > 1:
            sorted_unique = sorted(set(float(v) for v in field_mt))
            diffs = [b-a for a, b in zip(sorted_unique[:-1], sorted_unique[1:]) if b > a]
            res = min(diffs) if diffs else (xmax-xmin)/1000
        else:
            res = (xmax-xmin)/1000 if xmax > xmin else 0.01

        res = max(res, 1e-6)

        self.slider_guard = True
        for slider in (self.min_slider, self.max_slider):
            slider.configure(from_=xmin, to=xmax, resolution=res)

        self.min_slider.set(xmin)
        self.max_slider.set(xmax)
        self.slider_guard = False

        self.update_slider_labels()

    def on_min_slider(self, _=None):
        if self.slider_guard or not self.field:
            return

        minv = float(self.min_slider.get())
        maxv = float(self.max_slider.get())

        if minv >= maxv:
            self.slider_guard = True
            self.min_slider.set(maxv)
            self.slider_guard = False
            return

        self.update_slider_labels()
        self.apply_slider_limits()

    def on_max_slider(self, _=None):
        if self.slider_guard or not self.field:
            return

        minv = float(self.min_slider.get())
        maxv = float(self.max_slider.get())

        if maxv <= minv:
            self.slider_guard = True
            self.max_slider.set(minv)
            self.slider_guard = False
            return

        self.update_slider_labels()
        self.apply_slider_limits()

    def update_slider_labels(self):
        if not self.field:
            self.min_label_var.set("Champ min : —")
            self.max_label_var.set("Champ max : —")
            return
        self.min_label_var.set(f"Champ min : {self.min_slider.get():.3f} mT")
        self.max_label_var.set(f"Champ max : {self.max_slider.get():.3f} mT")

    def apply_slider_limits(self):
        if not self.field:
            return

        xmin = float(self.min_slider.get())
        xmax = float(self.max_slider.get())

        if xmin < xmax:
            self.ax.set_xlim(xmin, xmax)
            self.canvas.draw_idle()

    def reset_limits(self):
        if not self.field:
            return

        field_mt = gauss_to_mt(self.field)
        xmin = float(np.min(field_mt))
        xmax = float(np.max(field_mt))

        self.slider_guard = True
        self.min_slider.set(xmin)
        self.max_slider.set(xmax)
        self.slider_guard = False

        self.update_slider_labels()
        self.ax.set_xlim(xmin, xmax)
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw_idle()

    # ---------------- GRAPH / INFO ----------------
    def plot_spectrum(self):
        self.ax.clear()
        self.ax.plot(gauss_to_mt(self.field), self.signal, linewidth=1.2)
        self.ax.set_xlabel("Champ magnétique (mT)")
        self.ax.set_ylabel("Intensité RPE (a.u.)")
        self.ax.set_title(os.path.basename(self.txt_path) if self.txt_path else "Spectre RPE")
        self.ax.set_xlim(float(self.min_slider.get()), float(self.max_slider.get()))

        info_lines = []

        if self.dsc.get("XPTS"):
            info_lines.append(f"Points : {self.dsc.get('XPTS')}")

        if self.dsc.get("MWFQ"):
            try:
                freq = float(str(self.dsc.get("MWFQ")).replace(",", "."))
                if abs(freq) > 1e6:
                    info_lines.append(f"νMW : {freq/1e9:.6f} GHz")
                else:
                    info_lines.append(f"νMW : {freq:g}")
            except Exception:
                info_lines.append(f"νMW : {self.dsc.get('MWFQ')}")

        if self.dsc.get("AVGS"):
            info_lines.append(f"Scans : {self.dsc.get('AVGS')}")

        if self.dsc.get("RMA"):
            info_lines.append(f"Mod. : {self.dsc.get('RMA')}")

        if self.dsc.get("RCAG"):
            info_lines.append(f"Gain : {self.dsc.get('RCAG')}")

        if self.dsc.get("MPD"):
            info_lines.append(f"Attén. : {self.dsc.get('MPD')}")
        elif self.dsc.get("MP"):
            info_lines.append(f"Puissance : {self.dsc.get('MP')}")

        if self.dsc.get("SWT"):
            info_lines.append(f"Balayage : {self.dsc.get('SWT')}")

        if info_lines:
            self.ax.text(
                0.02, 0.98,
                "\n".join(info_lines),
                transform=self.ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="0.75",
                    alpha=0.85
                )
            )

        self.ax.grid(False)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update_info(self):
        keys = [
            ("XPTS", "Nombre de points"),
            ("MWFQ", "Fréquence micro-onde"),
            ("AVGS", "Nombre de scans"),
            ("RCAG", "Gain"),
            ("RMA", "Modulation"),
            ("MP", "Puissance micro-onde"),
            ("MPD", "Atténuation"),
            ("SWT", "Temps de balayage"),
        ]

        lines = [f"Points TXT : {len(self.field)}", ""]

        if "XMIN" in self.dsc:
            try:
                lines.append(
                    f"Champ initial : {float(str(self.dsc['XMIN']).replace(',', '.')) / 10.0:.3f} mT"
                )
            except Exception:
                lines.append(f"Champ initial : {self.dsc['XMIN']}")

        if "XWID" in self.dsc:
            try:
                lines.append(
                    f"Largeur de champ : {float(str(self.dsc['XWID']).replace(',', '.')) / 10.0:.3f} mT"
                )
            except Exception:
                lines.append(f"Largeur de champ : {self.dsc['XWID']}")

        for key, label in keys:
            if key in self.dsc:
                lines.append(f"{label} : {self.dsc[key]}")

        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "\n".join(lines))
        self.info_text.config(state="disabled")

    def resize_window_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("Redimensionner la fenêtre")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Largeur (px)").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        w_var = tk.StringVar(value=str(self.winfo_width()))
        ttk.Entry(frame, textvariable=w_var, width=12).grid(row=0, column=1, pady=4)

        ttk.Label(frame, text="Hauteur (px)").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        h_var = tk.StringVar(value=str(self.winfo_height()))
        ttk.Entry(frame, textvariable=h_var, width=12).grid(row=1, column=1, pady=4)

        def apply_size():
            try:
                w = int(float(w_var.get().replace(",", ".")))
                h = int(float(h_var.get().replace(",", ".")))
                if w < 700 or h < 450:
                    raise ValueError
                self.width_var.set(str(w))
                self.height_var.set(str(h))
                self.geometry(f"{w}x{h}")
                dialog.destroy()
            except Exception:
                messagebox.showerror(
                    "Taille invalide",
                    "Utilise au minimum 700 px de largeur et 450 px de hauteur.",
                    parent=dialog
                )

        ttk.Button(frame, text="Appliquer", command=apply_size).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )

    def resize_window(self):
        try:
            w = int(float(self.width_var.get().replace(",", ".")))
            h = int(float(self.height_var.get().replace(",", ".")))
            if w < 700 or h < 450:
                raise ValueError
            self.geometry(f"{w}x{h}")
        except Exception:
            messagebox.showerror(
                "Taille invalide",
                "Utilise au minimum 700 px de largeur et 450 px de hauteur."
            )

    def reset(self):
        self.txt_path = None
        self.dsc_path = None
        self.dsc = {}
        self.field = []
        self.signal = []


        self.txt_label.config(text="TXT : aucun")
        self.dsc_label.config(text="DSC : aucun")

        self.multi_dta_paths = []
        self.multi_dsc_paths = []
        self.multi_pairs = []

        self.auto_dta_path = None
        self.auto_dsc_path = None
        self.auto_ygf_path = None
        self.auto_rows = []
        self.auto_analysis_window = None
        self.auto_dta_label.config(text="DTA : aucun")
        self.auto_dsc_label.config(text="DSC : aucun")
        self.auto_ygf_label.config(text="YGF : aucun")

        self.mn_dta_path = None
        self.mn_dsc_path = None
        self.mn_series_pairs = []
        self.mn_measurements = {}
        self.mn_project_path = None
        if hasattr(self, "mn_series_tree"):
            self.refresh_mn_series_table()
        if self.auto_temp_dir and os.path.isdir(self.auto_temp_dir):
            try:
                shutil.rmtree(self.auto_temp_dir)
            except Exception:
                pass
        self.auto_temp_dir = None

        self.min_label_var.set("Champ min : —")
        self.max_label_var.set("Champ max : —")

        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.config(state="disabled")

        self.ax.clear()
        self.ax.set_xlabel("Champ magnétique (mT)")
        self.ax.set_ylabel("Intensité RPE (a.u.)")
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = RPEViewer()
    app.mainloop()
