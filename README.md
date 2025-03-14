# 🌍 geedownload  

**`geedownload`** is a Python package designed to facilitate the downloading of **Landsat** and **Sentinel** imagery from **Google Earth Engine (GEE)**.

## 📥 Installation  

To install the package directly from GitHub, run:  

```bash
pip install git+https://github.com/Coastal-Research-Collaborative/geedownload.git
```
## 🚀 Usage

1️⃣ Import the package
```python
import geedownload
```
2️⃣ Define a Site, Coordinates, and Timeframe
```python
sitename = 'georgiajekyllisland'

coords = [
    [-81.41396967622494, 31.035661672924554],
    [-81.40667406770443, 31.035661672924554],
    [-81.40667406770443, 31.053126706868298],
    [-81.41396967622494, 31.053126706868298],
    [-81.41396967622494, 31.035661672924554]
]

start_date = '2024-07-01'
end_date = '2024-08-30'
```
3️⃣ Download Imagery
```python


geedownload.retrieve_imagery(
    sitename=sitename, 
    start_date=start_date,
    end_date=end_date,
    data_dir='path where imagery will be downloaded',  # Specify where to save images
    polygon=coords
)
```
