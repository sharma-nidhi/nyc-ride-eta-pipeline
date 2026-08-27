import sys
import warnings
print('python', sys.executable)
# Show all warnings
warnings.simplefilter('default')

try:
    import seaborn as sns
    print('seaborn', sns.__version__)
except Exception as e:
    print('seaborn import error:', e)

try:
    import xgboost as xgb
    print('xgboost', xgb.__version__)
except Exception as e:
    print('xgboost import error:', e)
