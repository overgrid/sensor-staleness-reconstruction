from staleness_pipeline.data_source import find_matching_columns, load_series_from_csv
from staleness_pipeline.detection import find_stuck_periods

#path = "data/hum_temp_wide.csv"
path = "data/ecbc3d63b0e4_last_30_days_mean_ecbc3d63b0e4_wide.csv"
columns = find_matching_columns(path, ["temperature", "humidity"])
print(columns)  # confirm it finds both

for col in columns:
    series = load_series_from_csv(path, column=col)
    periods = find_stuck_periods(series)
    print(series.name, "->", len(periods), "stuck periods")
