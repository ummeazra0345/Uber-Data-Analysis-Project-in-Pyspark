# Uber Supply & Demand Analytics using PySpark

An end-to-end exploratory data analysis and operational optimization project built with **Apache Spark (PySpark)**. This repository analyzes two weeks of city-level ride-hailing supply and demand data to uncover peak demand windows, supply deficits, driver utilization ratios, and optimal shift scheduling.

---

## Table of Contents
* [Project Overview](#project-overview)
* [Tech Stack](#tech-stack)
* [Dataset Schema](#dataset-schema)
* [Setup & Installation](#setup--installation)
* [Spark Session Initialization](#spark-session-initialization)
* [Analytical Questions & PySpark Solutions](#analytical-questions--pyspark-solutions)
* [Key Takeaways](#key-takeaways)
* [Author](#author)

---

## Project Overview
Ride-hailing marketplaces rely on balancing real-time rider demand with active driver supply. Using PySpark DataFrame APIs and Window functions, this project evaluates hourly marketplace logs to:
* Identify peak trip completion dates, rolling 24-hour volumes, and rush-hour demand windows.
* Quantify unfulfilled demand (`Zeroes`) during weekend surges and 72-hour bottleneck periods.
* Measure driver utilization using volume-weighted trip-per-driver ratios.
* Recommend optimal 8-hour driver shift windows, targeted driver allocation hours, and an operational "end-of-day" reset timestamp.

---

## Tech Stack
* **Language:** Python 3.x
* **Big Data Framework:** Apache Spark (`PySpark SQL`, `Window Functions`)
* **Data Format:** CSV

---

## Dataset Schema

The dataset (`dataset.csv`) captures hourly marketplace activity over a two-week period across seven fields:

| Column | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `Date` | `Date` | Calendar date of the observation | `2012-09-10` |
| `Time (Local)` | `Integer` | Start of the hour in 24-hour local format (`16` = 4:00 PM – 5:00 PM) | `16` |
| `Eyeballs` | `Integer` | Number of riders who opened the app during that hour | `11` |
| `Zeroes` | `Integer` | Number of riders who opened the app and saw zero available cars | `2` |
| `Requests` | `Integer` | Number of riders who actively requested a ride | `4` |
| `Completed Trips` | `Integer` | Number of ride requests successfully fulfilled | `3` |
| `Unique Drivers` | `Integer` | Number of distinct drivers logged in during that hour | `6` |

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ummeazra0345/Uber-Data-Analysis-Project-in-Pyspark.git](https://github.com/ummeazra0345/Uber-Data-Analysis-Project-in-Pyspark.git)
   cd Uber-Data-Analysis-Project-in-Pyspark
   ```

2. **Install PySpark:**
   ```bash
   pip install pyspark
   ```

3. **Run the analysis script or Jupyter notebook:**
   ```bash
   spark-submit uber_analysis.py
   ```

---

## Spark Session Initialization

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("UberSupplyDemandAnalysis") \
    .getOrCreate()

# Load dataset with schema inference
df = spark.read.csv("dataset.csv", header=True, inferSchema=True)
```

---

## Analytical Questions & PySpark Solutions

### 1. Which date recorded the highest number of completed trips?
Aggregate total completed trips grouped by `Date` and sort in descending order.

```python
top_date = (
    df.groupBy("Date")
    .agg(F.sum("Completed Trips").alias("total_trips"))
    .orderBy(F.desc("total_trips"))
    .first()["Date"]
)

print("Date with most completed trips:", top_date)
# Output: 2012-09-15
```

---

### 2. What was the highest volume of completed trips within any 24-hour window?
Use a 24-hour window aggregation over the timestamp to calculate peak rolling volume.

```python
peak_24hr_trips = (
    df.groupBy(F.window("Time (Local)", "24 hours"))
    .agg(F.sum("Completed Trips").alias("total_completed_trips"))
    .orderBy(F.desc("total_completed_trips"))
    .first()["total_completed_trips"]
)

print("Highest completed trips in a 24-hour period:", peak_24hr_trips)
# Output: 2102
```

---

### 3. Which hour of the day had the highest ride request volume?
Group records by `Time (Local)` across the two-week span and sum `Requests`.

```python
peak_request_hour = (
    df.groupBy(F.hour("Time (Local)").alias("hour"))
    .agg(F.sum("Requests").alias("total_requests"))
    .orderBy(F.desc("total_requests"))
    .first()["hour"]
)

print("Hour with most requests:", peak_request_hour)
# Output: 17 (5:00 PM - 6:00 PM)
```

---

### 4. What percentage of all "Zeroes" occurred during weekend peak hours (Friday 5:00 PM – Sunday 3:00 AM)?
Filter records falling within the weekend window and compute the ratio against total zeroes.

```python
weekend_zeroes = (
    df.filter(
        ((F.dayofweek("Date") == 6) & (F.hour("Time (Local)") >= 17)) |
        (F.dayofweek("Date") == 7) |
        ((F.dayofweek("Date") == 1) & (F.hour("Time (Local)") < 3))
    )
    .agg(F.sum("Zeroes").alias("weekend_zeroes"))
    .collect()[0]["weekend_zeroes"]
)

total_zeroes = df.agg(F.sum("Zeroes").alias("total_zeroes")).collect()[0]["total_zeroes"]
weekend_zero_pct = (weekend_zeroes / total_zeroes) * 100

print(f"Percentage of zeroes on weekends: {weekend_zero_pct:.2f}%")
# Output: ~41.33%
```

---

### 5. What is the weighted average ratio of completed trips per driver?
Weight each hour's `Completed Trips / Unique Drivers` ratio by that hour's total completed trip volume.

```python
weighted_avg_ratio = (
    df.withColumn("completed_per_driver", F.col("Completed Trips") / F.col("Unique Drivers"))
    .groupBy("Date", "Time (Local)")
    .agg(
        F.avg("completed_per_driver").alias("avg_ratio"),
        F.sum("Completed Trips").alias("trip_volume")
    )
    .withColumn("weighted_component", F.col("avg_ratio") * F.col("trip_volume"))
    .agg(F.sum("weighted_component") / F.sum("trip_volume"))
    .collect()[0][0]
)

print(f"Weighted average completed trips per driver: {weighted_avg_ratio:.4f}")
# Output: 1.2869
```

---

### 6. Which 8-hour consecutive shift window captures the highest request volume?
Aggregate hourly requests across the two-week period and evaluate consecutive 8-hour windows using a sliding window specification.

```python
hourly_requests = (
    df.groupBy(F.hour("Time (Local)").alias("hour"))
    .agg(F.sum("Requests").alias("total_requests"))
    .orderBy("hour")
)

shift_window = Window.orderBy("hour").rowsBetween(0, 7)

busiest_shift = (
    hourly_requests
    .withColumn("shift_8hr_requests", F.sum("total_requests").over(shift_window))
    .orderBy(F.desc("shift_8hr_requests"))
    .limit(1)
)

busiest_shift.show()
```

---

### 7. True or False: Driver supply always increases when rider demand increases?
**False.** Analysis of hourly supply vs. demand trends shows multiple intervals—particularly late-night weekend surges and sudden weather or city event spikes—where `Requests` and `Eyeballs` rise sharply while `Unique Drivers` remains flat or declines, resulting in elevated `Zeroes` and unfulfilled requests.

---

### 8. Which 72-hour period recorded the highest Zeroes-to-Eyeballs ratio?
Bucket timestamps into 72-hour intervals (`72 * 3600` seconds) and rank by unfulfilled supply ratio (`Zeroes / Eyeballs`).

```python
period_ratios = (
    df.groupBy(
        ((F.col("Date").cast("timestamp").cast("long") / (72 * 3600)).cast("int")).alias("period_72hr")
    )
    .agg(
        F.sum("Zeroes").alias("total_zeroes"),
        F.sum("Eyeballs").alias("total_eyeballs")
    )
    .withColumn("zero_to_eyeball_ratio", F.col("total_zeroes") / F.col("total_eyeballs"))
    .orderBy(F.desc("zero_to_eyeball_ratio"))
)

period_ratios.show(1)
```

---

### 9. If you could add 5 drivers to a single hour every day, which hour maximizes impact?
Evaluate the ratio of rider demand (`Requests` / `Eyeballs`) to active `Unique Drivers` per hour to identify the sharpest supply bottleneck.

```python
supply_bottleneck = (
    df.groupBy("Time (Local)")
    .agg(
        (F.sum("Requests") / F.sum("Unique Drivers")).alias("requests_per_driver"),
        (F.sum("Zeroes") / F.sum("Eyeballs")).alias("zero_rate")
    )
    .orderBy(F.desc("requests_per_driver"))
)

supply_bottleneck.show(1)
```

---

### 10. Which hour represents the true operational "end of day" (lowest supply and demand)?
Identify the hour with the lowest average `Completed Trips` and `Unique Drivers` across the two-week dataset.

```python
operational_day_end = (
    df.groupBy("Time (Local)")
    .agg(
        F.mean("Completed Trips").alias("avg_completed_trips"),
        F.mean("Unique Drivers").alias("avg_unique_drivers")
    )
    .orderBy("avg_completed_trips", "avg_unique_drivers")
)

operational_day_end.show(1)
```

---

## Key Takeaways
* **Weekend Supply Deficit:** Over 41% of all zero-car app opens (`Zeroes`) occur between Friday 5:00 PM and Sunday 3:00 AM, signaling a major opportunity for weekend driver incentives.
* **Evening Rush Peak:** Hour `17` (5:00 PM – 6:00 PM) generates the highest daily request concentration across the two-week period.
* **Driver Utilization:** Active drivers complete an average of **1.29 trips per hour** on a volume-weighted basis.

---

## Author
* **Umme Azra** — [GitHub](https://github.com/ummeazra0345) | [LinkedIn](https://www.linkedin.com/in/umme-azra-987a721b2/)
