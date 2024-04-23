import pandas as pd


def merge_station_data(weather_data, station_data):
    """Merge weather data with station information."""
    return pd.merge(weather_data, station_data[["STATIONS_ID", "GEOBREITE", "GEOLAENGE", "STATIONSNAME"]],
                    how="inner", on="STATIONS_ID")


def calculate_mean_temperature_per_day(data):
    """Calculate mean temperature per day for each station."""
    data['MESS_DATUM'] = pd.to_datetime(data['MESS_DATUM'])
    data.set_index('MESS_DATUM', inplace=True)
    return data.groupby('STATIONS_ID').resample('D')['TT_TU'].mean().reset_index()
