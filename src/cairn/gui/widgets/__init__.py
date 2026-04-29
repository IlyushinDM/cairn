"""Виджеты CAIRN GUI."""
from cairn.gui.widgets.sidebar import Sidebar, DataSourceSection
from cairn.gui.widgets.data_tab import DataTab
from cairn.gui.widgets.training_tab import TrainingTab, TrainingWorker
from cairn.gui.widgets.results_tab import ResultsTab
from cairn.gui.widgets.explanation_tab import ExplanationTab
from cairn.gui.widgets.settings_dialog import SettingsDialog

__all__ = [
    "Sidebar", "DataSourceSection",
    "DataTab", "TrainingTab", "TrainingWorker",
    "ResultsTab", "ExplanationTab", "SettingsDialog",
]
