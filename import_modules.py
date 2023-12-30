import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import re
from fuzzywuzzy import fuzz
from urllib.parse import quote
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer