COPY (SELECT * FROM read_csv_auto('train.csv'))
TO 'train.parquet' (FORMAT PARQUET);

COPY (SELECT * FROM read_csv_auto('test.csv'))
TO 'test.parquet' (FORMAT PARQUET);

COPY (SELECT * FROM read_csv_auto('sample_submission.csv'))
TO 'sample_submission.parquet' (FORMAT PARQUET);

COPY (
  SELECT
    CAST(Soil_pH AS DOUBLE) AS Soil_pH,
    CAST(Soil_Moisture AS DOUBLE) AS Soil_Moisture,
    CAST(Organic_Carbon AS DOUBLE) AS Organic_Carbon,
    CAST(Electrical_Conductivity AS DOUBLE) AS Electrical_Conductivity,
    CAST(Temperature_C AS DOUBLE) AS Temperature_C,
    CAST(Humidity AS DOUBLE) AS Humidity,
    CAST(Rainfall_mm AS DOUBLE) AS Rainfall_mm,
    CAST(Sunlight_Hours AS DOUBLE) AS Sunlight_Hours,
    CAST(Wind_Speed_kmh AS DOUBLE) AS Wind_Speed_kmh,
    CAST(Field_Area_hectare AS DOUBLE) AS Field_Area_hectare,
    CAST(Mulching_Used AS INTEGER) AS Mulching_Used,
    CAST(Previous_Irrigation_mm AS DOUBLE) AS Previous_Irrigation_mm,
    CAST(Soil_Type = 'Clay' AS INTEGER) AS Soil_Type_Clay,
    CAST(Soil_Type = 'Loamy' AS INTEGER) AS Soil_Type_Loamy,
    CAST(Soil_Type = 'Sandy' AS INTEGER) AS Soil_Type_Sandy,
    CAST(Soil_Type = 'Silt' AS INTEGER) AS Soil_Type_Silt,
    CAST(Crop_Type = 'Cotton' AS INTEGER) AS Crop_Type_Cotton,
    CAST(Crop_Type = 'Maize' AS INTEGER) AS Crop_Type_Maize,
    CAST(Crop_Type = 'Potato' AS INTEGER) AS Crop_Type_Potato,
    CAST(Crop_Type = 'Rice' AS INTEGER) AS Crop_Type_Rice,
    CAST(Crop_Type = 'Sugarcane' AS INTEGER) AS Crop_Type_Sugarcane,
    CAST(Crop_Type = 'Wheat' AS INTEGER) AS Crop_Type_Wheat,
    CAST(Crop_Growth_Stage = 'Flowering' AS INTEGER) AS Crop_Growth_Stage_Flowering,
    CAST(Crop_Growth_Stage = 'Harvest' AS INTEGER) AS Crop_Growth_Stage_Harvest,
    CAST(Crop_Growth_Stage = 'Sowing' AS INTEGER) AS Crop_Growth_Stage_Sowing,
    CAST(Crop_Growth_Stage = 'Vegetative' AS INTEGER) AS Crop_Growth_Stage_Vegetative,
    CAST(Season = 'Kharif' AS INTEGER) AS Season_Kharif,
    CAST(Season = 'Rabi' AS INTEGER) AS Season_Rabi,
    CAST(Season = 'Zaid' AS INTEGER) AS Season_Zaid,
    CAST(Irrigation_Type = 'Canal' AS INTEGER) AS Irrigation_Type_Canal,
    CAST(Irrigation_Type = 'Drip' AS INTEGER) AS Irrigation_Type_Drip,
    CAST(Irrigation_Type = 'Rainfed' AS INTEGER) AS Irrigation_Type_Rainfed,
    CAST(Irrigation_Type = 'Sprinkler' AS INTEGER) AS Irrigation_Type_Sprinkler,
    CAST(Water_Source = 'Groundwater' AS INTEGER) AS Water_Source_Groundwater,
    CAST(Water_Source = 'Rainwater' AS INTEGER) AS Water_Source_Rainwater,
    CAST(Water_Source = 'Reservoir' AS INTEGER) AS Water_Source_Reservoir,
    CAST(Water_Source = 'River' AS INTEGER) AS Water_Source_River,
    CAST(Region = 'Central' AS INTEGER) AS Region_Central,
    CAST(Region = 'East' AS INTEGER) AS Region_East,
    CAST(Region = 'North' AS INTEGER) AS Region_North,
    CAST(Region = 'South' AS INTEGER) AS Region_South,
    CAST(Region = 'West' AS INTEGER) AS Region_West
  FROM read_csv_auto('train.csv')
  ORDER BY id
)
TO 'x_train.parquet' (FORMAT PARQUET);

COPY (
  SELECT
    CAST(Soil_pH AS DOUBLE) AS Soil_pH,
    CAST(Soil_Moisture AS DOUBLE) AS Soil_Moisture,
    CAST(Organic_Carbon AS DOUBLE) AS Organic_Carbon,
    CAST(Electrical_Conductivity AS DOUBLE) AS Electrical_Conductivity,
    CAST(Temperature_C AS DOUBLE) AS Temperature_C,
    CAST(Humidity AS DOUBLE) AS Humidity,
    CAST(Rainfall_mm AS DOUBLE) AS Rainfall_mm,
    CAST(Sunlight_Hours AS DOUBLE) AS Sunlight_Hours,
    CAST(Wind_Speed_kmh AS DOUBLE) AS Wind_Speed_kmh,
    CAST(Field_Area_hectare AS DOUBLE) AS Field_Area_hectare,
    CAST(Mulching_Used AS INTEGER) AS Mulching_Used,
    CAST(Previous_Irrigation_mm AS DOUBLE) AS Previous_Irrigation_mm,
    CAST(Soil_Type = 'Clay' AS INTEGER) AS Soil_Type_Clay,
    CAST(Soil_Type = 'Loamy' AS INTEGER) AS Soil_Type_Loamy,
    CAST(Soil_Type = 'Sandy' AS INTEGER) AS Soil_Type_Sandy,
    CAST(Soil_Type = 'Silt' AS INTEGER) AS Soil_Type_Silt,
    CAST(Crop_Type = 'Cotton' AS INTEGER) AS Crop_Type_Cotton,
    CAST(Crop_Type = 'Maize' AS INTEGER) AS Crop_Type_Maize,
    CAST(Crop_Type = 'Potato' AS INTEGER) AS Crop_Type_Potato,
    CAST(Crop_Type = 'Rice' AS INTEGER) AS Crop_Type_Rice,
    CAST(Crop_Type = 'Sugarcane' AS INTEGER) AS Crop_Type_Sugarcane,
    CAST(Crop_Type = 'Wheat' AS INTEGER) AS Crop_Type_Wheat,
    CAST(Crop_Growth_Stage = 'Flowering' AS INTEGER) AS Crop_Growth_Stage_Flowering,
    CAST(Crop_Growth_Stage = 'Harvest' AS INTEGER) AS Crop_Growth_Stage_Harvest,
    CAST(Crop_Growth_Stage = 'Sowing' AS INTEGER) AS Crop_Growth_Stage_Sowing,
    CAST(Crop_Growth_Stage = 'Vegetative' AS INTEGER) AS Crop_Growth_Stage_Vegetative,
    CAST(Season = 'Kharif' AS INTEGER) AS Season_Kharif,
    CAST(Season = 'Rabi' AS INTEGER) AS Season_Rabi,
    CAST(Season = 'Zaid' AS INTEGER) AS Season_Zaid,
    CAST(Irrigation_Type = 'Canal' AS INTEGER) AS Irrigation_Type_Canal,
    CAST(Irrigation_Type = 'Drip' AS INTEGER) AS Irrigation_Type_Drip,
    CAST(Irrigation_Type = 'Rainfed' AS INTEGER) AS Irrigation_Type_Rainfed,
    CAST(Irrigation_Type = 'Sprinkler' AS INTEGER) AS Irrigation_Type_Sprinkler,
    CAST(Water_Source = 'Groundwater' AS INTEGER) AS Water_Source_Groundwater,
    CAST(Water_Source = 'Rainwater' AS INTEGER) AS Water_Source_Rainwater,
    CAST(Water_Source = 'Reservoir' AS INTEGER) AS Water_Source_Reservoir,
    CAST(Water_Source = 'River' AS INTEGER) AS Water_Source_River,
    CAST(Region = 'Central' AS INTEGER) AS Region_Central,
    CAST(Region = 'East' AS INTEGER) AS Region_East,
    CAST(Region = 'North' AS INTEGER) AS Region_North,
    CAST(Region = 'South' AS INTEGER) AS Region_South,
    CAST(Region = 'West' AS INTEGER) AS Region_West
  FROM read_csv_auto('test.csv')
  ORDER BY id
)
TO 'x_test.parquet' (FORMAT PARQUET);

COPY (
  SELECT
    CASE Irrigation_Need
      WHEN 'Low' THEN 0
      WHEN 'Medium' THEN 1
      WHEN 'High' THEN 2
    END AS Irrigation_Need
  FROM read_csv_auto('train.csv')
  ORDER BY id
)
TO 'y_train.parquet' (FORMAT PARQUET);

COPY (
  SELECT id
  FROM read_csv_auto('test.csv')
  ORDER BY id
)
TO 'test_ids.parquet' (FORMAT PARQUET);
