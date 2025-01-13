import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# csvファイルのパスを受け取り、データフレームの上位5行を表示する関数
def show_head(csv_path):
    df = pd.read_csv(csv_path)
    return df.head()

# csvファイルのパスを受け取り、データフレームの情報を表示する関数
def show_info(csv_path):
    df = pd.read_csv(csv_path)
    return df.info()