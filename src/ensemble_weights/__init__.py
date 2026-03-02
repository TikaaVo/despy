"""
ensemble_weights — Dynamic Ensemble Selection library.

Metrics
-------
Pass a metric name string:

    KNNDWS(task='classification', metric='log_loss', mode='min')

Or import a metric function directly:

    from ensemble_weights.metrics import log_loss, mae

    KNNDWS(task='classification', metric=log_loss, mode='min')

Available built-in metrics:
    Scalar predictions (pass predict() output):
        'mae', 'mse', 'rmse', 'accuracy'

    Probability predictions (pass predict_proba() output):
        'log_loss', 'prob_correct'
"""

from ensemble_weights.des.knndws   import KNNDWS
from ensemble_weights.des.ola      import OLA
from ensemble_weights.des.knorau   import KNORAU
from ensemble_weights.des.knorae   import KNORAE
from ensemble_weights.des.knoraiu import KNORAIU
from ensemble_weights.router       import DynamicRouter
from ensemble_weights._config      import SPEED_PRESETS, list_presets
from ensemble_weights.analysis     import analyze

__all__ = [
    'KNNDWS',
    'OLA',
    'KNORAU',
    'KNORAE',
    'KNORAIU',
    'DynamicRouter',
    'SPEED_PRESETS',
    'list_presets',
    'analyze',
]