import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# Load the dataset
file_path = "fuzzy1.csv"  # Replace with the actual dataset path
df = pd.read_csv(file_path)

# Define input variables
soil_moisture = ctrl.Antecedent(np.arange(0, 1001, 1), 'Soil Moisture')
temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'Temperature')
air_humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'Air Humidity')

# Define output variable
pump_status = ctrl.Consequent(np.arange(0, 2, 1), 'Pump Status')

# Define bell-shaped (Gaussian) membership functions for Soil Moisture
soil_moisture['Low'] = fuzz.gaussmf(soil_moisture.universe, 200, 100)
soil_moisture['Medium'] = fuzz.gaussmf(soil_moisture.universe, 550, 180)
soil_moisture['High'] = fuzz.gaussmf(soil_moisture.universe, 800, 100)
soil_moisture['Very High'] = fuzz.gaussmf(soil_moisture.universe, 1000, 50)

# Define bell-shaped membership functions for Temperature
temperature['Low'] = fuzz.gaussmf(temperature.universe, 10, 5)
temperature['Medium'] = fuzz.gaussmf(temperature.universe, 25, 5)
temperature['High'] = fuzz.gaussmf(temperature.universe, 35, 5)
temperature['Very High'] = fuzz.gaussmf(temperature.universe, 45, 5)

# Define bell-shaped membership functions for Air Humidity
air_humidity['Low'] = fuzz.gaussmf(air_humidity.universe, 25, 15)
air_humidity['Medium'] = fuzz.gaussmf(air_humidity.universe, 50, 10)
air_humidity['High'] = fuzz.gaussmf(air_humidity.universe, 80, 10)
air_humidity['Very High'] = fuzz.gaussmf(air_humidity.universe, 100, 5)

# Define output membership functions
pump_status['ON'] = fuzz.trimf(pump_status.universe, [0, 1, 1])
pump_status['OFF'] = fuzz.trimf(pump_status.universe, [0, 0, 1])

# Define fuzzy rules
rules = [
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['Very High'] & temperature['Medium'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['Very High'] & temperature['Low'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['Very High'] & temperature['High'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['Very High'] & temperature['Very High'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['High'] & temperature['Medium'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['High'] & temperature['High'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['High'] & temperature['Low'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['High'] & temperature['Very High'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['Low'] & temperature['High'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['Low'] & temperature['Medium'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['Low'] & temperature['Very High'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['Low'] & temperature['Low'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Medium'] & air_humidity['Medium'] & temperature['Low'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Low'] & temperature['High'] & air_humidity['Very High'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Low'] & temperature['High'] & air_humidity['Low'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Low'] & temperature['High'] & air_humidity['High'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Low'] & temperature['High'] & air_humidity['Medium'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Low'] & temperature['Medium'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Low'] & temperature['Low'], pump_status['ON']),
    ctrl.Rule(soil_moisture['Low'] & temperature['Very High'], pump_status['ON']),
    ctrl.Rule(soil_moisture['High'], pump_status['OFF']),
    ctrl.Rule(soil_moisture['Very High'], pump_status['OFF']),
]

# Create and simulate control system
pump_ctrl = ctrl.ControlSystem(rules)
pump_simulation = ctrl.ControlSystemSimulation(pump_ctrl)

# Define mapping for categorical values in the dataset
soil_moisture_map = {"Low": 200, "Medium": 550, "High": 800, "Very High": 1000}
temperature_map = {"Low": 10, "Medium": 25, "High": 35, "Very High": 45}
air_humidity_map = {"Low": 25, "Medium": 50, "High": 80, "Very High": 100}

# Compute output for each row in the dataset
output_results = []
for _, row in df.iterrows():
    # Convert categorical values to numerical
    soil_value = soil_moisture_map.get(row['Soil Moisture'], 550)
    temp_value = temperature_map.get(row['Temperature'], 25)
    humidity_value = air_humidity_map.get(row['Air Humidity'], 50)

    # Set input values
    pump_simulation.input['Soil Moisture'] = soil_value
    pump_simulation.input['Temperature'] = temp_value
    pump_simulation.input['Air Humidity'] = humidity_value

    # Compute the output
    pump_simulation.compute()
    pump_status_result = 'ON' if pump_simulation.output['Pump Status'] > 0.5 else 'OFF'
    output_results.append(pump_status_result)

# Append predictions to the dataset
df['Predicted Pump'] = output_results

# Save the dataset with predictions
output_file_path = "fuzzy_predictions.csv"
df.to_csv(output_file_path, index=False)

# Indicate that the output file has been saved
print(f"Predicted results saved to {output_file_path}")
