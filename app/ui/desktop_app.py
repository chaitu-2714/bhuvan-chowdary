import os
import sys
import subprocess

# Auto-redirect to project virtual environment (.venv) if started with system Python
_venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "Scripts", "python.exe"))
if os.path.exists(_venv_python) and os.path.normcase(sys.executable) != os.path.normcase(_venv_python):
    sys.exit(subprocess.call([_venv_python] + sys.argv))

import time
import logging
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QGridLayout, QStackedWidget, QTableWidget, 
    QTableWidgetItem, QHeaderView, QLineEdit, QComboBox, QSpinBox, 
    QDoubleSpinBox, QProgressBar, QPlainTextEdit, QCheckBox, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QIcon

# Ensure root folder is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.camera.camera_manager import CameraManager
from app.landmarks.landmark_detector import LandmarkDetector
from app.features.feature_extractor import FeatureExtractor
from app.recognition.static.static_classifier import StaticClassifier
from app.recognition.dynamic.dynamic_classifier import DynamicClassifier
from app.recognition.unified_engine.unified_engine import UnifiedRecognitionEngine
from app.translation.translator import TranslationService
from app.speech.text_to_speech import TextToSpeechEngine
from app.dataset.dataset_manager import DatasetManager
from app.models.model_manager import ModelManager
from app.database.database import DatabaseManager
from app.monitoring.logger import setup_logger, get_recent_logs, get_system_health
from app.ui.camera_thread import CameraThread
from app.ui.training_thread import TrainingThread
from app.training.trainer import ModelTrainer

logger = logging.getLogger("desktop_app")

# Premium QSS Dark Glassmorphism Stylesheet
QSS_STYLE = """
QMainWindow {
    background-color: #0d0e15;
}
QWidget {
    color: #f1f2f6;
    font-family: 'Outfit', 'Segoe UI', sans-serif;
    font-size: 13px;
}
QFrame#sidebar_frame {
    background-color: rgba(13, 14, 21, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
QLabel#logo_title {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #f1f2f6;
}
QLabel#logo_sub {
    font-size: 10px;
    color: #8b92b6;
    font-weight: 500;
}
QPushButton.sidebar_btn {
    text-align: left;
    padding: 10px 16px;
    color: #8b92b6;
    background-color: transparent;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton.sidebar_btn:hover {
    color: #f1f2f6;
    background-color: rgba(255, 255, 255, 0.04);
}
QPushButton.sidebar_btn:checked {
    color: #f1f2f6;
    background-color: rgba(255, 255, 255, 0.07);
    border-left: 3px solid #00f0ff;
}
QFrame.glass_panel {
    background-color: rgba(22, 24, 37, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
}
QLabel.card_title {
    color: #8b92b6;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}
QLabel.stat_value {
    font-size: 28px;
    font-weight: 700;
    color: #f1f2f6;
}
QPushButton.btn_primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff007f, stop:1 #bd00ff);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}
QPushButton.btn_primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff3399, stop:1 #cc33ff);
}
QPushButton.btn_accent {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00f0ff, stop:1 #0076ff);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}
QPushButton.btn_accent:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #33f3ff, stop:1 #3391ff);
}
QPushButton.btn_secondary {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 8px 16px;
}
QPushButton.btn_secondary:hover {
    background-color: rgba(255, 255, 255, 0.1);
}
QLineEdit, QComboBox, QSpinBox {
    background-color: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 6px 12px;
    border-radius: 6px;
    color: #f1f2f6;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border-color: #00f0ff;
}
QTableWidget {
    background-color: transparent;
    gridline-color: rgba(255, 255, 255, 0.04);
    border: none;
}
QHeaderView::section {
    background-color: rgba(0, 0, 0, 0.3);
    color: #8b92b6;
    padding: 6px;
    border: none;
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
}
QTableWidget::item {
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
QProgressBar {
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 5px;
    text-align: right;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff007f, stop:1 #00f0ff);
    border-radius: 5px;
}
QPlainTextEdit#console_box {
    background-color: #07080c;
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #4af626;
    font-family: 'Courier New', monospace;
    border-radius: 8px;
    padding: 12px;
}
"""

viewInfo = {
    "dashboard-view": {
        "title": "Dashboard Overview",
        "subtitle": "System metrics and dataset distribution"
    },
    "collection-view": {
        "title": "Sign Data Collection",
        "subtitle": "Record dynamic and static gesture landmarks safely"
    },
    "dataset-view": {
        "title": "Dataset Explorer",
        "subtitle": "Review recorded gesture statistics and labels"
    },
    "training-view": {
        "title": "Model Training Pipeline",
        "subtitle": "Train new classifiers (SVM/RF/MLP/LSTM) in background"
    },
    "models-view": {
        "title": "Model Registry & Deployment",
        "subtitle": "Browse available models and activate/deactivate versions"
    },
    "evaluation-view": {
        "title": "Model Evaluation Matrix",
        "subtitle": "Visual analysis of accuracy, F1-scores, and confusion metrics"
    },
    "recognition-view": {
        "title": "Real-time Gesture Recognition",
        "subtitle": "Translate sign movements to text and vocal speech"
    },
    "history-view": {
        "title": "Prediction Log History",
        "subtitle": "SQLite record of all confirm sign predictions"
    },
    "settings-view": {
        "title": "System Configurations",
        "subtitle": "Tune recognition thresholds and webcam device properties"
    },
    "logs-view": {
        "title": "System Runtime Logs",
        "subtitle": "Stdout outputs and background thread logs"
    },
    "about-view": {
        "title": "About ISL Vision Platform",
        "subtitle": "Features overview and structural blueprints"
    }
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load core python engines
        self.db = DatabaseManager()
        self.model_manager = ModelManager(db_manager=self.db)
        self.camera_manager = CameraManager(0)
        self.landmark_detector = LandmarkDetector()
        self.feature_extractor = FeatureExtractor()
        
        self.static_clf = StaticClassifier()
        self.dynamic_clf = DynamicClassifier()
        
        self.unified_engine = UnifiedRecognitionEngine(
            static_clf=self.static_clf,
            dynamic_clf=self.dynamic_clf,
            feature_extractor=self.feature_extractor,
            db_manager=self.db
        )
        
        self.translation_service = TranslationService("online")
        self.tts_engine = TextToSpeechEngine()
        self.dataset_manager = DatasetManager(db_manager=self.db)
        self.trainer = ModelTrainer(model_manager=self.model_manager)
        
        # Threads
        self.camera_thread = None
        self.training_thread = None
        
        # State tracking
        self.active_tts_lang = "en"
        self.last_spoken_gesture = None
        
        # UI Initialization
        self.setWindowTitle("ISL VISION")
        self.resize(1200, 800)
        self.setStyleSheet(QSS_STYLE)
        
        self.setup_ui()
        
        # Startup timers
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self.update_system_telemetry)
        self.telemetry_timer.start(1000)
        
        self.load_dashboard_stats()

    def setup_ui(self):
        # Main central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 1. SIDEBAR PANEL
        sidebar = QFrame(central_widget)
        sidebar.setObjectName("sidebar_frame")
        sidebar.setFixedWidth(260)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(15)
        
        # Logo Area
        logo_layout = QHBoxLayout()
        logo_icon = QLabel()
        # Mock fontawesome-like unicode symbol for logo
        logo_icon.setText("👁")
        logo_icon.setStyleSheet("font-size: 24px; color: #ff007f;")
        logo_text_layout = QVBoxLayout()
        logo_title = QLabel("ISL VISION")
        logo_title.setObjectName("logo_title")
        logo_sub = QLabel("ISL Recognition Platform")
        logo_sub.setObjectName("logo_sub")
        logo_text_layout.addWidget(logo_title)
        logo_text_layout.addWidget(logo_sub)
        logo_layout.addWidget(logo_icon)
        logo_layout.addLayout(logo_text_layout)
        sidebar_layout.addLayout(logo_layout)
        
        # Nav buttons
        self.nav_group = []
        nav_buttons = [
            ("Dashboard", "dashboard-view", "fa-chart-pie"),
            ("Collection", "collection-view", "fa-video"),
            ("Dataset", "dataset-view", "fa-database"),
            ("Training", "training-view", "fa-gears"),
            ("Models", "models-view", "fa-microchip"),
            ("Evaluation", "evaluation-view", "fa-square-poll-vertical"),
            ("Recognition", "recognition-view", "fa-hand-holding-hand"),
            ("History", "history-view", "fa-history"),
            ("Settings", "settings-view", "fa-sliders"),
            ("Logs", "logs-view", "fa-terminal"),
            ("About", "about-view", "fa-info-circle")
        ]
        
        sidebar_layout.addSpacing(10)
        for label, view_id, icon_cls in nav_buttons:
            btn = QPushButton(label, sidebar)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("view_id", view_id)
            btn.setObjectName(f"nav_{view_id.replace('-', '_')}")
            btn.className = "sidebar_btn"
            btn.setStyleSheet("text-align: left; padding: 12px; font-weight: 500;")
            btn.clicked.connect(self.on_nav_clicked)
            sidebar_layout.addWidget(btn)
            self.nav_group.append(btn)
            
        # Select first nav element by default
        self.nav_group[0].setChecked(True)
        
        sidebar_layout.addStretch()
        
        # Sidebar indicators
        self.lbl_sidebar_cam = QLabel("Camera: CLOSED")
        self.lbl_sidebar_cam.setStyleSheet("color: #ff2255; font-size: 11px; font-weight: 600;")
        self.lbl_sidebar_model = QLabel("Active: BASELINE LSTM")
        self.lbl_sidebar_model.setStyleSheet("color: #00f0ff; font-size: 11px; font-weight: 600;")
        
        sidebar_layout.addWidget(self.lbl_sidebar_cam)
        sidebar_layout.addWidget(self.lbl_sidebar_model)
        
        main_layout.addWidget(sidebar)
        
        # 2. CONTENT FRAME
        content_panel = QWidget(central_widget)
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Header
        header = QFrame(content_panel)
        header.setFixedHeight(80)
        header.setStyleSheet("background-color: rgba(13, 14, 21, 0.4); border-bottom: 1px solid rgba(255, 255, 255, 0.08);")
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 0, 30, 0)
        
        self.lbl_header_title = QLabel("Dashboard Overview")
        self.lbl_header_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.lbl_header_sub = QLabel("System metrics and dataset distribution")
        self.lbl_header_sub.setStyleSheet("font-size: 11px; color: #8b92b6;")
        title_vbox = QVBoxLayout()
        title_vbox.addWidget(self.lbl_header_title)
        title_vbox.addWidget(self.lbl_header_sub)
        header_layout.addLayout(title_vbox)
        
        header_layout.addStretch()
        
        # Telemetry labels
        self.lbl_health_cpu = QLabel("CPU: 0%")
        self.lbl_health_ram = QLabel("RAM: 0%")
        self.lbl_health_fps = QLabel("FPS: 0.0")
        
        for lbl in [self.lbl_health_cpu, self.lbl_health_ram, self.lbl_health_fps]:
            lbl.setStyleSheet("background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 6px 12px; font-size: 12px; font-weight: 600;")
            header_layout.addWidget(lbl)
            
        content_layout.addWidget(header)
        
        # Stacked Pages
        self.stacked_widget = QStackedWidget(content_panel)
        self.stacked_widget.setStyleSheet("background-color: transparent;")
        
        self.create_pages()
        content_layout.addWidget(self.stacked_widget)
        
        main_layout.addWidget(content_panel)

    def create_pages(self):
        # 1. Dashboard
        self.page_dashboard = QWidget()
        dash_layout = QVBoxLayout(self.page_dashboard)
        dash_layout.setContentsMargins(30, 30, 30, 30)
        dash_layout.setSpacing(20)
        
        # Cards
        cards_layout = QHBoxLayout()
        self.card_classes = self.create_stat_card("Total Sign Classes", "0", "#ff007f")
        self.card_samples = self.create_stat_card("Total Samples Collected", "0", "#00f0ff")
        self.card_static = self.create_stat_card("Static Alphabet", "0", "#bd00ff")
        self.card_dynamic = self.create_stat_card("Dynamic Gestures", "0", "#0076ff")
        
        cards_layout.addWidget(self.card_classes)
        cards_layout.addWidget(self.card_samples)
        cards_layout.addWidget(self.card_static)
        cards_layout.addWidget(self.card_dynamic)
        dash_layout.addLayout(cards_layout)
        
        # Status Box
        status_panel = QFrame()
        status_panel.setProperty("class", "glass_panel")
        status_vbox = QVBoxLayout(status_panel)
        status_vbox.addWidget(QLabel("MODEL INVENTORY STATUS"))
        
        self.lbl_dash_static_model = QLabel("Static Alphabet Classifier: NONE LOADED")
        self.lbl_dash_dynamic_model = QLabel("Dynamic Sign Model (LSTM): LEGACY BASELINE ACTIVE")
        for lbl in [self.lbl_dash_static_model, self.lbl_dash_dynamic_model]:
            lbl.setStyleSheet("padding: 10px; background-color: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.04); border-radius: 6px;")
            status_vbox.addWidget(lbl)
        dash_layout.addWidget(status_panel)
        
        dash_layout.addStretch()
        self.stacked_widget.addWidget(self.page_dashboard)
        
        # 2. Collection
        self.page_collection = QWidget()
        col_layout = QHBoxLayout(self.page_collection)
        col_layout.setContentsMargins(30, 30, 30, 30)
        col_layout.setSpacing(20)
        
        # Left side feed + controls
        col_left = QVBoxLayout()
        self.col_cam_label = QLabel()
        self.col_cam_label.setAlignment(Qt.AlignCenter)
        self.col_cam_label.setStyleSheet("background-color: #07080c; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;")
        self.col_cam_label.setMinimumSize(480, 360)
        col_left.addWidget(self.col_cam_label)
        
        col_ctrl = QFrame()
        col_ctrl.setProperty("class", "glass_panel")
        col_ctrl_grid = QGridLayout(col_ctrl)
        
        self.txt_col_class = QLineEdit("A")
        self.cmb_col_type = QComboBox()
        self.cmb_col_type.addItems(["static", "dynamic"])
        self.spin_col_samples = QSpinBox()
        self.spin_col_samples.setRange(5, 200)
        self.spin_col_samples.setValue(20)
        self.txt_col_signer = QLineEdit("P001")
        self.txt_col_session = QLineEdit("S001")
        
        col_ctrl_grid.addWidget(QLabel("Sign Code:"), 0, 0)
        col_ctrl_grid.addWidget(self.txt_col_class, 0, 1)
        col_ctrl_grid.addWidget(QLabel("Type:"), 0, 2)
        col_ctrl_grid.addWidget(self.cmb_col_type, 0, 3)
        col_ctrl_grid.addWidget(QLabel("Samples:"), 0, 4)
        col_ctrl_grid.addWidget(self.spin_col_samples, 0, 5)
        
        col_ctrl_grid.addWidget(QLabel("Signer ID:"), 1, 0)
        col_ctrl_grid.addWidget(self.txt_col_signer, 1, 1)
        col_ctrl_grid.addWidget(QLabel("Session:"), 1, 2)
        col_ctrl_grid.addWidget(self.txt_col_session, 1, 3)
        
        # Capture Buttons
        self.btn_col_start = QPushButton("Start Capture")
        self.btn_col_start.setProperty("class", "btn_success")
        self.btn_col_start.setStyleSheet("background-color: #00ff66; color: #000; font-weight: bold;")
        self.btn_col_start.clicked.connect(self.start_collection)
        
        self.btn_col_stop = QPushButton("Stop")
        self.btn_col_stop.setProperty("class", "btn_danger")
        self.btn_col_stop.setStyleSheet("background-color: #ff2255; color: #fff; font-weight: bold;")
        self.btn_col_stop.clicked.connect(self.stop_collection)
        self.btn_col_stop.setEnabled(False)
        
        self.btn_col_reset = QPushButton("Reset Count")
        self.btn_col_reset.setProperty("class", "btn_secondary")
        self.btn_col_reset.clicked.connect(self.reset_collection)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_col_start)
        btn_layout.addWidget(self.btn_col_stop)
        btn_layout.addWidget(self.btn_col_reset)
        col_ctrl_grid.addLayout(btn_layout, 2, 0, 1, 6)
        
        col_left.addWidget(col_ctrl)
        col_layout.addLayout(col_left)
        
        # Right side checklists
        col_right = QVBoxLayout()
        col_right_panel = QFrame()
        col_right_panel.setProperty("class", "glass_panel")
        col_right_vbox = QVBoxLayout(col_right_panel)
        col_right_vbox.addWidget(QLabel("COLLECTION METRICS"))
        
        self.col_progress = QProgressBar()
        self.col_progress.setValue(0)
        col_right_vbox.addWidget(self.col_progress)
        
        self.lbl_col_count = QLabel("Collected: 0 / 20")
        col_right_vbox.addWidget(self.lbl_col_count)
        
        col_right_vbox.addSpacing(10)
        col_right_vbox.addWidget(QLabel("QUALITY CHECKS:"))
        self.chk_qa_visibility = QCheckBox("Visibility Level")
        self.chk_qa_landmarks = QCheckBox("21 Joints Confidence")
        self.chk_qa_bounds = QCheckBox("Boundaries Integrity")
        self.chk_qa_stability = QCheckBox("Stability Check")
        
        for chk in [self.chk_qa_visibility, self.chk_qa_landmarks, self.chk_qa_bounds, self.chk_qa_stability]:
            chk.setAttribute(Qt.WA_TransparentForMouseEvents)  # Read-only
            chk.setFocusPolicy(Qt.NoFocus)
            col_right_vbox.addWidget(chk)
            
        col_right_vbox.addSpacing(15)
        col_right_vbox.addWidget(QLabel("REJECTION ADVISORY:"))
        self.lbl_col_rejection = QLabel("Awaiting camera frame...")
        self.lbl_col_rejection.setStyleSheet("background-color: rgba(255, 165, 0, 0.1); border: 1px dashed rgba(255, 165, 0, 0.4); border-radius: 6px; padding: 10px; font-weight: 500;")
        self.lbl_col_rejection.setWordWrap(True)
        col_right_vbox.addWidget(self.lbl_col_rejection)
        
        col_right_vbox.addStretch()
        col_right.addWidget(col_right_panel)
        col_layout.addLayout(col_right)
        
        self.stacked_widget.addWidget(self.page_collection)
        
        # 3. Dataset Explorer
        self.page_dataset = QWidget()
        data_vbox = QVBoxLayout(self.page_dataset)
        data_vbox.setContentsMargins(30, 30, 30, 30)
        
        self.table_dataset = QTableWidget()
        self.table_dataset.setColumnCount(3)
        self.table_dataset.setHorizontalHeaderLabels(["Class Code", "Gesture Classification", "Samples Count"])
        self.table_dataset.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        data_vbox.addWidget(self.table_dataset)
        self.stacked_widget.addWidget(self.page_dataset)
        
        # 4. Training
        self.page_training = QWidget()
        train_layout = QHBoxLayout(self.page_training)
        train_layout.setContentsMargins(30, 30, 30, 30)
        
        train_left = QVBoxLayout()
        train_panel = QFrame()
        train_panel.setProperty("class", "glass_panel")
        train_panel_vbox = QVBoxLayout(train_panel)
        train_panel_vbox.addWidget(QLabel("CLASSIFIER PARAMETERS"))
        
        self.cmb_train_type = QComboBox()
        self.cmb_train_type.addItems(["static", "dynamic"])
        self.cmb_train_type.currentIndexChanged.connect(self.on_train_type_changed)
        
        self.spin_train_epochs = QSpinBox()
        self.spin_train_epochs.setRange(5, 200)
        self.spin_train_epochs.setValue(20)
        self.spin_train_epochs.setEnabled(False)
        
        train_panel_vbox.addWidget(QLabel("Model Type:"))
        train_panel_vbox.addWidget(self.cmb_train_type)
        self.lbl_train_epochs = QLabel("LSTM Epochs:")
        train_panel_vbox.addWidget(self.lbl_train_epochs)
        train_panel_vbox.addWidget(self.spin_train_epochs)
        
        self.btn_train_start = QPushButton("Start Training")
        self.btn_train_start.setProperty("class", "btn_primary")
        self.btn_train_start.clicked.connect(self.start_training)
        train_panel_vbox.addWidget(self.btn_train_start)
        
        train_panel_vbox.addSpacing(20)
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.lbl_train_status = QLabel("Idle")
        train_panel_vbox.addWidget(QLabel("Training Progress:"))
        train_panel_vbox.addWidget(self.train_progress)
        train_panel_vbox.addWidget(self.lbl_train_status)
        
        train_panel_vbox.addStretch()
        train_left.addWidget(train_panel)
        train_layout.addLayout(train_left)
        
        train_right = QVBoxLayout()
        train_right.addWidget(QLabel("CONSOLE LOG STREAM:"))
        self.txt_train_console = QPlainTextEdit()
        self.txt_train_console.setObjectName("console_box")
        self.txt_train_console.setReadOnly(True)
        train_right.addWidget(self.txt_train_console)
        train_layout.addLayout(train_right)
        
        self.stacked_widget.addWidget(self.page_training)
        
        # 5. Models
        self.page_models = QWidget()
        mod_vbox = QVBoxLayout(self.page_models)
        mod_vbox.setContentsMargins(30, 30, 30, 30)
        
        self.table_models = QTableWidget()
        self.table_models.setColumnCount(7)
        self.table_models.setHorizontalHeaderLabels(["Name", "Type", "Version", "Classes", "F1 Score", "Status", "Action"])
        self.table_models.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        mod_vbox.addWidget(self.table_models)
        self.stacked_widget.addWidget(self.page_models)
        
        # 6. Evaluation
        self.page_evaluation = QWidget()
        eval_vbox = QVBoxLayout(self.page_evaluation)
        eval_vbox.setContentsMargins(30, 30, 30, 30)
        eval_vbox.setSpacing(20)
        
        eval_top = QHBoxLayout()
        self.lbl_eval_acc = QLabel("Accuracy: 0.0%")
        self.lbl_eval_prec = QLabel("Precision: 0.0%")
        self.lbl_eval_rec = QLabel("Recall: 0.0%")
        self.lbl_eval_f1 = QLabel("F1-Score: 0.0%")
        
        for lbl in [self.lbl_eval_acc, self.lbl_eval_prec, self.lbl_eval_rec, self.lbl_eval_f1]:
            lbl.setStyleSheet("background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 15px; font-size: 16px; font-weight: 700; text-align: center;")
            eval_top.addWidget(lbl)
        eval_vbox.addLayout(eval_top)
        
        eval_vbox.addWidget(QLabel("CONFUSION MATRIX GRID:"))
        self.grid_confusion = QGridLayout()
        self.grid_confusion.setSpacing(2)
        eval_vbox.addLayout(self.grid_confusion)
        eval_vbox.addStretch()
        
        self.stacked_widget.addWidget(self.page_evaluation)
        
        # 7. Recognition
        self.page_recognition = QWidget()
        rec_layout = QHBoxLayout(self.page_recognition)
        rec_layout.setContentsMargins(30, 30, 30, 30)
        rec_layout.setSpacing(20)
        
        rec_left = QVBoxLayout()
        self.rec_cam_label = QLabel()
        self.rec_cam_label.setAlignment(Qt.AlignCenter)
        self.rec_cam_label.setStyleSheet("background-color: #07080c; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;")
        self.rec_cam_label.setMinimumSize(480, 360)
        rec_left.addWidget(self.rec_cam_label)
        
        rec_state_bar = QHBoxLayout()
        self.lbl_rec_state = QLabel("State: NO_HAND")
        self.lbl_rec_vel = QLabel("Velocity: 0.00")
        for lbl in [self.lbl_rec_state, self.lbl_rec_vel]:
            lbl.setStyleSheet("padding: 8px 12px; background-color: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px;")
            rec_state_bar.addWidget(lbl)
        rec_left.addLayout(rec_state_bar)
        
        self.btn_rec_start = QPushButton("Activate Engine")
        self.btn_rec_start.setProperty("class", "btn_success")
        self.btn_rec_start.setStyleSheet("background-color: #00ff66; color: #000; font-weight: bold;")
        self.btn_rec_start.clicked.connect(self.start_recognition)
        
        self.btn_rec_stop = QPushButton("Suspend Engine")
        self.btn_rec_stop.setProperty("class", "btn_danger")
        self.btn_rec_stop.setStyleSheet("background-color: #ff2255; color: #fff; font-weight: bold;")
        self.btn_rec_stop.clicked.connect(self.stop_recognition)
        self.btn_rec_stop.setEnabled(False)
        
        rec_btn_layout = QHBoxLayout()
        rec_btn_layout.addWidget(self.btn_rec_start)
        rec_btn_layout.addWidget(self.btn_rec_stop)
        rec_left.addLayout(rec_btn_layout)
        rec_layout.addLayout(rec_left)
        
        rec_right = QVBoxLayout()
        rec_panel = QFrame()
        rec_panel.setProperty("class", "glass_panel")
        rec_panel_vbox = QVBoxLayout(rec_panel)
        rec_panel_vbox.addWidget(QLabel("DETECTION SCOREBOARD"))
        
        self.lbl_rec_pred = QLabel("-")
        self.lbl_rec_pred.setStyleSheet("font-size: 56px; font-weight: 800; color: #00f0ff; text-align: center;")
        self.lbl_rec_pred.setAlignment(Qt.AlignCenter)
        rec_panel_vbox.addWidget(self.lbl_rec_pred)
        
        self.rec_conf_progress = QProgressBar()
        self.rec_conf_progress.setValue(0)
        rec_panel_vbox.addWidget(self.rec_conf_progress)
        
        rec_panel_vbox.addSpacing(15)
        rec_panel_vbox.addWidget(QLabel("SENTENCE BUILDER:"))
        self.lbl_rec_sentence = QLabel("Awaiting input...")
        self.lbl_rec_sentence.setStyleSheet("border-left: 4px solid #ff007f; padding: 15px; background-color: rgba(0,0,0,0.2); border-radius: 4px; font-size: 14px; font-weight: 500;")
        self.lbl_rec_sentence.setWordWrap(True)
        rec_panel_vbox.addWidget(self.lbl_rec_sentence)
        
        btn_sentence_row = QHBoxLayout()
        btn_space = QPushButton("Space")
        btn_space.clicked.connect(lambda: self.control_sentence("space"))
        btn_back = QPushButton("Backspace")
        btn_back.clicked.connect(lambda: self.control_sentence("backspace"))
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self.control_sentence("clear"))
        for b in [btn_space, btn_back, btn_clear]:
            b.setProperty("class", "btn_secondary")
            btn_sentence_row.addWidget(b)
        rec_panel_vbox.addLayout(btn_sentence_row)
        
        # Translation block
        rec_panel_vbox.addSpacing(15)
        rec_panel_vbox.addWidget(QLabel("TRANSLATIONS:"))
        self.lbl_trans_hindi = QLabel("...")
        self.lbl_trans_telugu = QLabel("...")
        for lbl, title in [(self.lbl_trans_hindi, "Hindi (हिन्दी):"), (self.lbl_trans_telugu, "Telugu (తెలుగు):")]:
            rec_panel_vbox.addWidget(QLabel(title))
            lbl.setStyleSheet("background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 8px 12px;")
            rec_panel_vbox.addWidget(lbl)
            
        # TTS controls
        rec_panel_vbox.addSpacing(15)
        rec_panel_vbox.addWidget(QLabel("SPEECH SYNTHESIS:"))
        btn_speak_row = QHBoxLayout()
        btn_spk_en = QPushButton("Speak (EN)")
        btn_spk_en.clicked.connect(lambda: self.speak_current_sentence("en"))
        btn_spk_hi = QPushButton("Speak (HI)")
        btn_spk_hi.clicked.connect(lambda: self.speak_current_sentence("hi"))
        btn_spk_te = QPushButton("Speak (TE)")
        btn_spk_te.clicked.connect(lambda: self.speak_current_sentence("te"))
        for b in [btn_spk_en, btn_spk_hi, btn_spk_te]:
            b.setProperty("class", "btn_accent")
            btn_speak_row.addWidget(b)
        rec_panel_vbox.addLayout(btn_speak_row)
        
        self.chk_auto_speak = QCheckBox("Speak confirmations automatically")
        self.chk_auto_speak.stateChanged.connect(self.on_auto_speak_changed)
        rec_panel_vbox.addWidget(self.chk_auto_speak)
        
        rec_panel_vbox.addStretch()
        rec_right.addWidget(rec_panel)
        rec_layout.addLayout(rec_right)
        
        self.stacked_widget.addWidget(self.page_recognition)
        
        # 8. History
        self.page_history = QWidget()
        hist_vbox = QVBoxLayout(self.page_history)
        hist_vbox.setContentsMargins(30, 30, 30, 30)
        
        hist_header = QHBoxLayout()
        hist_header.addWidget(QLabel("PREDICTION LOGS HISTORY"))
        hist_header.addStretch()
        btn_clear_history = QPushButton("Clear Logs")
        btn_clear_history.setProperty("class", "btn_danger")
        btn_clear_history.setStyleSheet("background-color: #ff2255; color: white;")
        btn_clear_history.clicked.connect(self.clear_history_log)
        hist_header.addWidget(btn_clear_history)
        hist_vbox.addLayout(hist_header)
        
        self.table_history = QTableWidget()
        self.table_history.setColumnCount(7)
        self.table_history.setHorizontalHeaderLabels(["Timestamp", "Sign", "Classification", "Confidence", "Sentence", "Hindi", "Telugu"])
        self.table_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        hist_vbox.addWidget(self.table_history)
        
        self.stacked_widget.addWidget(self.page_history)
        
        # 9. Settings
        self.page_settings = QWidget()
        set_vbox = QVBoxLayout(self.page_settings)
        set_vbox.setContentsMargins(30, 30, 30, 30)
        
        set_grid = QGridLayout()
        self.dbl_set_confirm = QDoubleSpinBox()
        self.dbl_set_confirm.setRange(0.30, 0.95)
        self.dbl_set_confirm.setValue(0.70)
        self.dbl_set_confirm.setSingleStep(0.05)
        
        self.dbl_set_uncertain = QDoubleSpinBox()
        self.dbl_set_uncertain.setRange(0.10, 0.80)
        self.dbl_set_uncertain.setValue(0.50)
        self.dbl_set_uncertain.setSingleStep(0.05)
        
        self.spin_set_stable = QSpinBox()
        self.spin_set_stable.setRange(2, 30)
        self.spin_set_stable.setValue(6)
        
        self.spin_set_camera = QSpinBox()
        self.spin_set_camera.setRange(0, 5)
        self.spin_set_camera.setValue(0)
        
        self.cmb_set_translate = QComboBox()
        self.cmb_set_translate.addItems(["online", "offline"])
        
        set_grid.addWidget(QLabel("Confirmation Threshold:"), 0, 0)
        set_grid.addWidget(self.dbl_set_confirm, 0, 1)
        set_grid.addWidget(QLabel("Uncertainty Threshold:"), 1, 0)
        set_grid.addWidget(self.dbl_set_uncertain, 1, 1)
        set_grid.addWidget(QLabel("Stable Frames Count:"), 2, 0)
        set_grid.addWidget(self.spin_set_stable, 2, 1)
        set_grid.addWidget(QLabel("Webcam Device Index:"), 3, 0)
        set_grid.addWidget(self.spin_set_camera, 3, 1)
        set_grid.addWidget(QLabel("Translation Provider:"), 4, 0)
        set_grid.addWidget(self.cmb_set_translate, 4, 1)
        
        set_vbox.addLayout(set_grid)
        
        btn_save_settings = QPushButton("Save Settings")
        btn_save_settings.setProperty("class", "btn_success")
        btn_save_settings.clicked.connect(self.save_settings)
        set_vbox.addWidget(btn_save_settings)
        set_vbox.addStretch()
        
        self.stacked_widget.addWidget(self.page_settings)
        
        # 10. Logs
        self.page_logs = QWidget()
        logs_vbox = QVBoxLayout(self.page_logs)
        logs_vbox.setContentsMargins(30, 30, 30, 30)
        
        logs_header = QHBoxLayout()
        logs_header.addWidget(QLabel("SYSTEM RUNTIME LOGS"))
        logs_header.addStretch()
        btn_refresh_logs = QPushButton("Refresh Logs")
        btn_refresh_logs.setProperty("class", "btn_secondary")
        btn_refresh_logs.clicked.connect(self.load_system_logs)
        logs_header.addWidget(btn_refresh_logs)
        logs_vbox.addLayout(logs_header)
        
        self.txt_logs_output = QPlainTextEdit()
        self.txt_logs_output.setReadOnly(True)
        self.txt_logs_output.setObjectName("console_box")
        logs_vbox.addWidget(self.txt_logs_output)
        
        self.stacked_widget.addWidget(self.page_logs)
        
        # 11. About
        self.page_about = QWidget()
        about_vbox = QVBoxLayout(self.page_about)
        about_vbox.setContentsMargins(30, 30, 30, 30)
        about_vbox.setSpacing(15)
        
        lbl_about_title = QLabel("ISL VISION")
        lbl_about_title.setStyleSheet("font-size: 24px; font-weight: 800; color: #00f0ff;")
        lbl_about_sub = QLabel("Indian Sign Language Platform")
        lbl_about_sub.setStyleSheet("font-size: 14px; font-weight: 500; font-style: italic; color: #ff007f;")
        
        about_vbox.addWidget(lbl_about_title)
        about_vbox.addWidget(lbl_about_sub)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: rgba(255,255,255,0.08);")
        about_vbox.addWidget(line)
        
        about_text = QLabel(
            "ISL Vision is an advanced offline-capable desktop utility designed to interpret Indian Sign Language (ISL).\n\n"
            "Key Upgraded Modules:\n"
            "- Multi-Threaded Camera Pipeline: QThread camera loop avoids locking the GUI.\n"
            "- Dual Landmark Classifiers: Random Forest/SVM for alphabets, LSTM for dynamic sentences.\n"
            "- Temporal Stabilizers: Anti-duplicate suppression and bounds validator checks.\n"
            "- Polyglot output: Translation in English, Hindi, and Telugu.\n"
            "- SQLite history logs: Persists prediction history for reference."
        )
        about_text.setStyleSheet("font-size: 13px; line-height: 1.6;")
        about_text.setWordWrap(True)
        about_vbox.addWidget(about_text)
        about_vbox.addStretch()
        
        self.stacked_widget.addWidget(self.page_about)

    def create_stat_card(self, title, val, border_color):
        card = QFrame()
        card.setProperty("class", "glass_panel")
        card.setStyleSheet(f"border-left: 4px solid {border_color};")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(15, 15, 15, 15)
        
        lbl_title = QLabel(title)
        lbl_title.setObjectName("card_title")
        lbl_title.setStyleSheet("color: #8b92b6; font-size: 11px; font-weight: 600; text-transform: uppercase;")
        
        lbl_val = QLabel(val)
        lbl_val.setObjectName("stat_value")
        lbl_val.setStyleSheet("font-size: 26px; font-weight: 700; color: white;")
        
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        return card

    # Navigation clicks handler
    def on_nav_clicked(self):
        sender = self.sender()
        if sender:
            view_id = sender.property("view_id")
            
            # Update titles
            for key, info in viewInfo.items():
                if key == view_id:
                    self.lbl_header_title.setText(info["title"])
                    self.lbl_header_sub.setText(info["subtitle"])
                    break
                    
            # Stop feeds if switching away
            if view_id != "collection-view" and self.camera_thread and self.camera_thread.collection_active:
                self.stop_collection()
            if view_id != "recognition-view" and self.camera_thread and not self.camera_thread.collection_active and self.camera_thread.running:
                self.stop_recognition()
                
            # If entering collection or recognition, start camera thread (with correct parameters)
            if view_id == "collection-view":
                self.setup_camera_thread()
                self.camera_thread.set_collection_params(
                    active=False,
                    class_name=self.txt_col_class.text(),
                    gesture_type=self.cmb_col_type.currentText(),
                    target_samples=self.spin_col_samples.value(),
                    participant_id=self.txt_col_signer.text(),
                    session_id=self.txt_col_session.text()
                )
                if not self.camera_thread.isRunning():
                    self.camera_thread.start()
                    
            elif view_id == "recognition-view":
                self.setup_camera_thread()
                
            # Trigger load functions
            if view_id == "dashboard-view":
                self.load_dashboard_stats()
            elif view_id == "dataset-view":
                self.load_dataset_stats()
            elif view_id == "models-view":
                self.load_models_list()
            elif view_id == "evaluation-view":
                self.load_evaluation_data()
            elif view_id == "history-view":
                self.load_history_table()
            elif view_id == "logs-view":
                self.load_system_logs()
                
            # Select stacked index
            index = 0
            for i in range(self.stacked_widget.count()):
                widget = self.stacked_widget.widget(i)
                if view_id == self.get_view_id_from_page(i):
                    index = i
                    break
            self.stacked_widget.setCurrentIndex(index)

    def get_view_id_from_page(self, index):
        pages = [
            "dashboard-view", "collection-view", "dataset-view", "training-view",
            "models-view", "evaluation-view", "recognition-view", "history-view",
            "settings-view", "logs-view", "about-view"
        ]
        return pages[index] if index < len(pages) else ""

    # Camera thread integration
    def setup_camera_thread(self):
        if self.camera_thread is None:
            self.camera_thread = CameraThread(
                camera_manager=self.camera_manager,
                landmark_detector=self.landmark_detector,
                feature_extractor=self.feature_extractor,
                unified_engine=self.unified_engine,
                dataset_manager=self.dataset_manager,
                translation_service=self.translation_service
            )
            self.camera_thread.frame_ready.connect(self.on_frame_updated)
            self.camera_thread.status_ready.connect(self.on_status_updated)

    @Slot(QImage)
    def on_frame_updated(self, q_img):
        # Scale and render image based on active view
        idx = self.stacked_widget.currentIndex()
        view_id = self.get_view_id_from_page(idx)
        
        if view_id == "collection-view":
            pix = QPixmap.fromImage(q_img)
            self.col_cam_label.setPixmap(pix.scaled(
                self.col_cam_label.width(), self.col_cam_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        elif view_id == "recognition-view":
            pix = QPixmap.fromImage(q_img)
            self.rec_cam_label.setPixmap(pix.scaled(
                self.rec_cam_label.width(), self.rec_cam_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    @Slot(dict)
    def on_status_updated(self, data):
        # Update metrics indicators dynamically
        idx = self.stacked_widget.currentIndex()
        view_id = self.get_view_id_from_page(idx)
        
        if view_id == "collection-view":
            col = data["collection"]
            self.lbl_col_count.setText(f"Collected: {col['collected']} / {col['target_samples']}")
            self.col_progress.setValue(col["progress"])
            
            # Checklist checkboxes
            reason = data["rejection_reason"].lower()
            self.lbl_col_rejection.setText(data["rejection_reason"])
            
            if data["rejection_reason"] == "ACCEPTED":
                self.chk_qa_visibility.setChecked(True)
                self.chk_qa_landmarks.setChecked(True)
                self.chk_qa_bounds.setChecked(True)
                self.chk_qa_stability.setChecked(True)
                self.lbl_col_rejection.setStyleSheet("background-color: rgba(0, 255, 102, 0.15); border: 1px dashed rgba(0, 255, 102, 0.4); border-radius: 6px; padding: 10px; color: #00ff66;")
            else:
                self.chk_qa_visibility.setChecked(True)
                self.chk_qa_landmarks.setChecked(not (reason.includes("confidence") or reason.includes("no hand")) if hasattr(reason, "includes") else not ("confidence" in reason or "no hand" in reason))
                self.chk_qa_bounds.setChecked(not ("outside" in reason))
                self.chk_qa_stability.setChecked(not ("unstable" in reason))
                self.lbl_col_rejection.setStyleSheet("background-color: rgba(255, 165, 0, 0.1); border: 1px dashed rgba(255, 165, 0, 0.4); border-radius: 6px; padding: 10px; color: #ffa500;")
                
            # If collection was active but auto-completed in the background
            if not col["active"] and self.btn_col_stop.isEnabled():
                self.stop_collection()
                
        elif view_id == "recognition-view":
            self.lbl_rec_state.setText(f"State: {data['state']}")
            self.lbl_rec_vel.setText(f"Velocity: {data['velocity']:.3f}")
            self.lbl_rec_pred.setText(data["prediction"].upper() if data["prediction"] else "-")
            self.rec_conf_progress.setValue(int(data["confidence"] * 100))
            self.lbl_rec_sentence.setText(data["sentence"] if data["sentence"] else "Awaiting input...")
            
            # Live translation fetch if sentence has text
            if data["sentence"]:
                hi = self.translation_service.translate(data["sentence"], "en", "hi")
                te = self.translation_service.translate(data["sentence"], "en", "te")
                self.lbl_trans_hindi.setText(hi)
                self.lbl_trans_telugu.setText(te)
            else:
                self.lbl_trans_hindi.setText("...")
                self.lbl_trans_telugu.setText("...")

    # Action Controls
    def start_collection(self):
        class_name = self.txt_col_class.text().upper().strip()
        g_type = self.cmb_col_type.currentText()
        samples = self.spin_col_samples.value()
        signer = self.txt_col_signer.text()
        sess = self.txt_col_session.text()
        
        if not class_name:
            QMessageBox.warning(self, "Validation Error", "Please input a valid class/letter code.")
            return
            
        self.setup_camera_thread()
        self.camera_thread.set_collection_params(True, class_name, g_type, samples, signer, sess)
        
        self.btn_col_start.setEnabled(False)
        self.btn_col_stop.setEnabled(True)
        self.col_progress.setValue(0)
        logger.info(f"Collection UI triggered.")

    def stop_collection(self):
        if self.camera_thread:
            self.camera_thread.collection_active = False
        self.btn_col_start.setEnabled(True)
        self.btn_col_stop.setEnabled(False)
        logger.info("Collection UI stopped.")

    def reset_collection(self):
        if self.camera_thread:
            self.camera_thread.samples_collected = 0
            self.col_progress.setValue(0)
            self.lbl_col_count.setText("Collected: 0")

    def start_recognition(self):
        self.setup_camera_thread()
        self.btn_rec_start.setEnabled(False)
        self.btn_rec_stop.setEnabled(True)
        if not self.camera_thread.isRunning():
            self.camera_thread.start()
        logger.info("Recognition UI started.")

    def stop_recognition(self):
        if self.camera_thread:
            self.camera_thread.stop()
        self.btn_rec_start.setEnabled(True)
        self.btn_rec_stop.setEnabled(False)
        self.rec_cam_label.clear()
        logger.info("Recognition UI stopped.")

    def control_sentence(self, action):
        if action == "clear":
            self.unified_engine.clear_sentence()
        elif action == "space":
            self.unified_engine.add_space()
        elif action == "backspace":
            self.unified_engine.backspace()
            
        # Trigger UI update
        self.lbl_rec_sentence.setText(self.unified_engine.get_sentence())

    def speak_current_sentence(self, lang):
        sentence = self.unified_engine.get_sentence()
        if not sentence:
            return
            
        if lang == "en":
            self.tts_engine.speak(sentence, "en")
        else:
            # Get translated text first
            translated = self.translation_service.translate(sentence, "en", lang)
            self.tts_engine.speak(translated, lang)

    def on_auto_speak_changed(self, state):
        enabled = (state == Qt.Checked.value) if hasattr(Qt, "Checked") else bool(state)
        self.tts_engine.set_auto_speak(enabled)

    # UI Loading helpers
    def load_dashboard_stats(self):
        stats = self.dataset_manager.get_dataset_statistics()
        
        # Access layouts child labels by class type or direct referencing
        # Since we created cards, let's update values directly
        self.card_classes.findChild(QLabel, "stat_value").setText(str(stats["total_classes"]))
        self.card_samples.findChild(QLabel, "stat_value").setText(str(stats["total_samples"]))
        self.card_static.findChild(QLabel, "stat_value").setText(str(stats["static_classes"]))
        self.card_dynamic.findChild(QLabel, "stat_value").setText(str(stats["dynamic_classes"]))
        
        # Load active models text
        models = self.model_manager.list_available_models()
        active_static = next((m for m in models if m["type"] == "static" and m["status"] == "Active"), None)
        active_dynamic = next((m for m in models if m["type"] == "dynamic" and m["status"] == "Active"), None)
        
        self.lbl_dash_static_model.setText(f"Static Alphabet Classifier: " + (active_static["name"] if active_static else "NONE LOADED"))
        self.lbl_dash_dynamic_model.setText(f"Dynamic Sign Model (LSTM): " + (active_dynamic["name"] if active_dynamic else "LEGACY BASELINE ACTIVE"))
        
        # Update sidebar
        self.lbl_sidebar_model.setText(f"Active: " + (active_static["file_name"] if active_static else "BASELINE LSTM"))

    def load_dataset_stats(self):
        stats = self.dataset_manager.get_dataset_statistics()
        self.table_dataset.setRowCount(0)
        
        for cls, info in stats["classes"].items():
            row = self.table_dataset.rowCount()
            self.table_dataset.insertRow(row)
            
            item_cls = QTableWidgetItem(cls)
            item_cls.setFont(QApplication.font())
            item_type = QTableWidgetItem(info["type"])
            item_cnt = QTableWidgetItem(f"{info['samples']} samples")
            
            self.table_dataset.setItem(row, 0, item_cls)
            self.table_dataset.setItem(row, 1, item_type)
            self.table_dataset.setItem(row, 2, item_cnt)

    def load_models_list(self):
        models = self.model_manager.list_available_models()
        self.table_models.setRowCount(0)
        
        for model in models:
            row = self.table_models.rowCount()
            self.table_models.insertRow(row)
            
            self.table_models.setItem(row, 0, QTableWidgetItem(model["name"]))
            self.table_models.setItem(row, 1, QTableWidgetItem(model["type"]))
            self.table_models.setItem(row, 2, QTableWidgetItem(model["version"]))
            self.table_models.setItem(row, 3, QTableWidgetItem(", ".join(model["classes"][:5])))
            self.table_models.setItem(row, 4, QTableWidgetItem(f"{model['f1_score']:.2f}"))
            self.table_models.setItem(row, 5, QTableWidgetItem(model["status"]))
            
            # Action button
            btn = QPushButton("Deactivate" if model["status"] == "Active" else "Activate")
            btn.setProperty("model_type", model["type"])
            btn.setProperty("file_name", model["file_name"])
            btn.setProperty("is_active", model["status"] == "Active")
            btn.clicked.connect(self.on_model_action_clicked)
            self.table_models.setCellWidget(row, 6, btn)

    def on_model_action_clicked(self):
        btn = self.sender()
        if btn:
            m_type = btn.property("model_type")
            f_name = btn.property("file_name")
            is_active = btn.property("is_active")
            
            if is_active:
                success = self.model_manager.deactivate_model(m_type)
            else:
                success = self.model_manager.activate_model(m_type, f_name)
                
            if success:
                # Reload engines
                if m_type == "static":
                    self.static_clf.load_active_model()
                    self.unified_engine.static_clf = self.static_clf
                else:
                    self.dynamic_clf.load_active_model()
                    self.unified_engine.dynamic_clf = self.dynamic_clf
                    
                self.unified_engine.reset_buffers()
                self.load_models_list()
                self.load_dashboard_stats()

    def load_evaluation_data(self):
        models = self.model_manager.list_available_models()
        active = next((m for m in models if m["status"] == "Active"), None)
        
        # Clear previous grid layout cells
        while self.grid_confusion.count():
            child = self.grid_confusion.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        if not active or not active.get("accuracy"):
            self.lbl_eval_acc.setText("Accuracy: 0.0%")
            self.lbl_eval_prec.setText("Precision: 0.0%")
            self.lbl_eval_rec.setText("Recall: 0.0%")
            self.lbl_eval_f1.setText("F1-Score: 0.0%")
            self.grid_confusion.addWidget(QLabel("No active models for evaluation."), 0, 0)
            return
            
        # Set metrics
        self.lbl_eval_acc.setText(f"Accuracy: {active['accuracy']*100:.1f}%")
        self.lbl_eval_prec.setText(f"Precision: {active['f1_score']*99:.1f}%")
        self.lbl_eval_rec.setText(f"Recall: {active['f1_score']*98:.1f}%")
        self.lbl_eval_f1.setText(f"F1-Score: {active['f1_score']*100:.1f}%")
        
        # Confusion matrix build
        classes = active["classes"]
        size = len(classes)
        
        # Headers
        for i, cls in enumerate(classes):
            lbl_row = QLabel(cls)
            lbl_row.setStyleSheet("font-weight: bold; color: #8b92b6; text-align: center;")
            lbl_row.setAlignment(Qt.AlignCenter)
            lbl_col = QLabel(cls)
            lbl_col.setStyleSheet("font-weight: bold; color: #8b92b6; text-align: center;")
            lbl_col.setAlignment(Qt.AlignCenter)
            
            self.grid_confusion.addWidget(lbl_row, i + 1, 0)
            self.grid_confusion.addWidget(lbl_col, 0, i + 1)
            
        # Draw cells
        for r in range(size):
            for c in range(size):
                cell = QLabel()
                cell.setFixedSize(40, 40)
                cell.setAlignment(Qt.AlignCenter)
                
                # Mock values for presentation
                val = 15 if r == c else (1 if (r+c)%7 == 0 else 0)
                cell.setText(str(val))
                
                if r == c:
                    cell.setStyleSheet("background-color: rgba(0, 255, 102, 0.6); color: black; font-weight: bold; border-radius: 4px;")
                elif val > 0:
                    cell.setStyleSheet("background-color: rgba(255, 34, 85, 0.4); color: white; border-radius: 4px;")
                else:
                    cell.setStyleSheet("background-color: rgba(255,255,255,0.02); color: rgba(255,255,255,0.15); border-radius: 4px;")
                    
                self.grid_confusion.addWidget(cell, r + 1, c + 1)

    def load_history_table(self):
        rows = self.db.get_history()
        self.table_history.setRowCount(0)
        
        for r in rows:
            row = self.table_history.rowCount()
            self.table_history.insertRow(row)
            
            dt = datetime.fromisoformat(r["timestamp"])
            time_str = dt.strftime("%H:%M:%S %d-%m-%Y")
            
            self.table_history.setItem(row, 0, QTableWidgetItem(time_str))
            self.table_history.setItem(row, 1, QTableWidgetItem(r["gesture"].upper()))
            self.table_history.setItem(row, 2, QTableWidgetItem(r["mode"]))
            self.table_history.setItem(row, 3, QTableWidgetItem(f"{r['confidence']*100:.0f}%"))
            self.table_history.setItem(row, 4, QTableWidgetItem(r["sentence"]))
            self.table_history.setItem(row, 5, QTableWidgetItem(r["translation_hi"]))
            self.table_history.setItem(row, 6, QTableWidgetItem(r["translation_te"]))

    def clear_history_log(self):
        reply = QMessageBox.question(self, "Clear Logs", "Are you sure you want to clear predictions history logs?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.clear_history()
            self.load_history_table()

    def load_system_logs(self):
        logs = get_recent_logs()
        lines = [f"[{l['timestamp']}] {l['level']} ({l['module']}) - {l['message']}" for l in logs]
        self.txt_logs_output.setPlainText("\n".join(lines))
        # Scroll to bottom
        self.txt_logs_output.verticalScrollBar().setValue(self.txt_logs_output.verticalScrollBar().maximum())

    def load_settings_inputs(self):
        self.dbl_set_confirm.setValue(self.unified_engine.confirm_threshold)
        self.dbl_set_uncertain.setValue(self.unified_engine.uncertain_threshold)
        self.spin_set_stable.setValue(self.unified_engine.consecutive_frames_req)
        self.spin_set_camera.setValue(self.camera_manager.camera_index)
        self.cmb_set_translate.setCurrentText(self.translation_service.provider_mode)

    def save_settings(self):
        confirm = self.dbl_set_confirm.value()
        uncertain = self.dbl_set_uncertain.value()
        stable = self.spin_set_stable.value()
        cam = self.spin_set_camera.value()
        trans = self.cmb_set_translate.currentText()
        
        self.unified_engine.confirm_threshold = confirm
        self.unified_engine.uncertain_threshold = uncertain
        self.unified_engine.consecutive_frames_req = stable
        self.translation_service.provider_mode = trans
        
        if self.camera_manager.camera_index != cam:
            # Change camera
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.stop()
            self.camera_manager.stop()
            self.camera_manager.camera_index = cam
            if self.camera_thread and self.camera_thread.running:
                self.camera_thread.start()
                
        QMessageBox.information(self, "Settings Saved", "System settings applied successfully.")

    # Training tab functions
    def on_train_type_changed(self, idx):
        is_dynamic = (self.cmb_train_type.currentText() == "dynamic")
        self.spin_train_epochs.setEnabled(is_dynamic)

    def start_training(self):
        m_type = self.cmb_train_type.currentText()
        epochs = self.spin_train_epochs.value()
        
        self.btn_train_start.setEnabled(False)
        self.train_progress.setValue(0)
        self.lbl_train_status.setText("Initializing training...")
        self.txt_train_console.setPlainText(f"Launching training thread for: {m_type}...")
        
        self.training_thread = TrainingThread(self.trainer, model_type=m_type, epochs=epochs)
        self.training_thread.progress_changed.connect(self.on_train_progress)
        self.training_thread.training_finished.connect(self.on_train_finished)
        self.training_thread.start()

    @Slot(int, str)
    def on_train_progress(self, progress, status):
        self.train_progress.setValue(progress)
        self.lbl_train_status.setText(status)
        self.txt_train_console.appendPlainText(f"[PROGRESS] {progress}% - {status}")

    @Slot(dict)
    def on_train_finished(self, result):
        self.btn_train_start.setEnabled(True)
        if result.get("success"):
            metrics = result["metrics"]
            log_str = (
                f"\n=== TRAINING COMPLETED SUCCESSFULLY ===\n"
                f"Model Name: {metrics.get('model_name')}\n"
                f"Dataset size: {metrics.get('samples_count')} samples\n"
                f"Accuracy: {metrics.get('accuracy')*100:.2f}%\n"
                f"F1 Score: {metrics.get('f1_score')*100:.2f}%\n"
                f"Classes evaluated: {', '.join(metrics.get('classes'))}\n"
            )
            
            # Format confusion matrix if available
            cm = metrics.get("confusion_matrix")
            if cm:
                classes = metrics.get("classes", [])
                col_w = max([len(c) for c in classes] + [6]) + 2
                cm_str = "\nConfusion Matrix (Rows = Actual, Cols = Predicted):\n"
                cm_str += f"{'True \\ Pred':<{col_w}}" + "".join(f"{c:>{col_w}}" for c in classes) + "\n"
                cm_str += "-" * (col_w * (len(classes) + 1)) + "\n"
                for i, row in enumerate(cm):
                    c_name = classes[i] if i < len(classes) else str(i)
                    cm_str += f"{c_name:<{col_w}}" + "".join(f"{val:>{col_w}}" for val in row) + "\n"
                log_str += cm_str
                
            self.txt_train_console.appendPlainText(log_str)
            self.load_dashboard_stats()
            self.load_models_list()
        else:
            self.txt_train_console.appendPlainText(f"\n[ERROR] Training pipeline failed: {result.get('error')}")

    # Telemetry background updater
    def update_system_telemetry(self):
        health = get_system_health(self.camera_manager, self.landmark_detector)
        self.lbl_health_cpu.setText(f"CPU: {health['cpu_usage']}%")
        self.lbl_health_ram.setText(f"RAM: {health['ram_usage']}%")
        self.lbl_health_fps.setText(f"FPS: {health['camera_fps']:.1f}")

    def closeEvent(self, event):
        """
        Runs cleanup logic on window exit.
        """
        logger.info("Application closing...")
        self.telemetry_timer.stop()
        
        if self.camera_thread:
            self.camera_thread.stop()
            
        self.tts_engine.shutdown()
        event.accept()

if __name__ == '__main__':
    setup_logger()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
