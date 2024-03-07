import pandas as pd

# %%
# import the csv files from the storage folder
df_2019 = pd.read_csv(
    "/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices_data/energy-charts_Tägliche_Börsenstrompreise_in_Deutschland_2019.csv")
df_2020 = pd.read_csv(
    "/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices_data/energy-charts_Tägliche_Börsenstrompreise_in_Deutschland_2020.csv")
df_2021 = pd.read_csv(
    "/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices_data/energy-charts_Tägliche_Börsenstrompreise_in_Deutschland_2021.csv")
df_2022 = pd.read_csv(
    "/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices_data/energy-charts_Tägliche_Börsenstrompreise_in_Deutschland_2022.csv")
df_2023 = pd.read_csv(
    "/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices_data/energy-charts_Tägliche_Börsenstrompreise_in_Deutschland_2023.csv")

# merge the dataframes into one
df = pd.concat([df_2019, df_2020, df_2021, df_2022, df_2023])

# save the merged dataframe to a csv file
df.to_csv(
    "/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices_data/energy_prices_2019_2023.csv",
    index=False)
print("Data successfully merged and saved.")

# %%
df.head
# %%
